# Lab book — CP-FTMW spectrometer

Dated entries, newest last. Numbers quoted here are measured unless marked
otherwise; anything inferred or assumed is labelled as such.

---

## 2026-10-19 — 10 MHz reference commissioning · scope stale-frame bug · AWG markers working

**Instruments:** Tabor Proteus P9482D (fw 1.237.0), Tektronix MSO64B (FV 2.10.5.1825),
Wenzel 501-27501-11 10 MHz OCXO, Mini-Circuits ZX10R-2-183-S+ splitter.

**Wiring during the session:** Wenzel → ZX10R → arm 1 → scope rear Ref In (no
terminator); arm 2 → 6 dB pad → AWG REF IN. Scope AUX OUT → CH2. AWG CH2 → scope CH3.
AWG CH1 M1 → scope CH4.

### 1. Reference chain verified end to end

The scope was already locked (`ROSc:SOUrce? → EXTERNAL`, `ROSc:STATe? → LOCKED`).
The AWG has **no lock-status query** — `:ROSC:STAT?`, `:ROSC:LOCK?` and
`:INST:CHAS:CLOC:SOUR?` all time out; `:ROSC:SOUR?` only echoes what was accepted
and reads `EXT` even with nothing on REF IN. Verification therefore had to be by
measurement.

Measured AWG clock error against the reference-locked scope (500 MHz tone, 4–5
captures per point):

| AWG reference state | clock error |
|---|---|
| EXT, cable connected | −0.012 ± 0.028 ppm |
| INT (TCXO) | +0.007 … +0.044 ppm |
| **EXT, cable disconnected** | **+3.334 ± 0.030 ppm** |

The third row is the positive control and the key result. It proves two things at
once: the measurement really can see a wrong clock (3.3 ppm resolved at 0.03 ppm,
100:1), and `:ROSC:SOUR EXT` genuinely reaches the hardware rather than being
cosmetic.

**Why EXT and INT looked identical:** this unit's internal TCXO sits ~10–40 ppb
from the Wenzel — roughly 25–100× better than its ±1 ppm spec, which is a bound
over temperature and aging, not a typical offset. Locked and unlocked therefore
produce almost the same frequency, and no short beat test can separate them. This
cost most of the session.

**Operational rule:** never leave the reference set to EXT with no cable on REF IN.
That state is 3.3 ppm — nearly 100× worse than simply using the internal TCXO — and
nothing on screen indicates it.

### 2. Drift test — attempted, abandoned as unresolvable

`run_ref_drift.py` (interleaved INT/EXT so room temperature is common-mode, saves
csv/json/png). At 5 and 10 minutes both states sat **on** the ~0.03 ppm measurement
noise floor (wander/noise 0.66–1.04). Duration was never the limiting factor; the
noise floor was. Resolving ppb-level wander would need ~200 µs captures (100k
cycles) instead of 20 µs. Not pursued: the OCXO's real advantages here are phase
noise (20–30 dB) and bounded long-term drift, neither of which this test touches.

### 3. Firmware findings (Proteus)

- `:ROSC:FREQ` **defaults to 100M**, so `:ROSC:SOUR EXT` alone locks to nothing.
  Frequency must be sent first. Confirmed by readback on a fresh instrument.
- **`*RST` clears both** `:ROSC:SOUR` → INT and `:ROSC:FREQ` → 100M.
- **Entering 9 GS/s Direct mode clears the reference**; re-applying afterwards
  holds (0/3 kept when set before, 3/3 when set after). Not a limitation of the
  mode — just an ordering trap. Ruled out as causes: the sample-clock value,
  every individual SCPI command in the path, and a second `*RST`.
- `:ROSC:BYPass` **does not exist** on fw 1.237.0 — the query times out and queues
  `208, syntax error in scpi`. It appears only in Programming Manual rev 1.6.

Code changes: `set_reference_external()` sends FREQ before SOUR and raises on
readback mismatch; all seven reset sites funnel through `_reset_locked()`;
`_ensure_reference_locked()` re-asserts at the end of `_play()` and
`_duc_upload_ch()`. Verified 3/3 for 2.5 GS/s Direct, 9 GS/s Direct and 9 GS/s DUC.

### 4. Scope stale-frame trap — found, confirmed, guarded

`Scope.get_waveform()` read the buffer **without arming an acquisition**. When the
scope is not triggering it returns the byte-identical frame on every call, with no
error. Demonstrated with a NORMal trigger at an impossible level: 3 calls → 1
distinct waveform.

This is not a cosmetic bug. It produced a shot-to-shot phase coherence of exactly
**1.000 (circular sd −0.0°) for both the locked and unlocked AWG** — i.e. "coherent
averaging is perfect" — when every "shot" was one frame copied 25 times. **Any
averaging, coherence or shot-statistics result taken before this fix may be
flattered by it**, and the failure mode is always in the flattering direction.

Fix: `get_waveform(..., fresh=True)` arms one acquisition, polls `ACQuire:STATE?`
to completion, and **raises** on timeout rather than returning a stale frame.
Default stays `fresh=False` for live view.

### 5. Markers working — manual is wrong about the format

Implemented `AWG.marker_pulse()` / `marker_off()`. Getting a clean pulse required
correcting two documented claims:

- Manual Table 8.1 says bit0–3 = M1–M4 with bits 4–7 "N/A". In fact **each byte
  holds two marker points**: low nibble = M1–M4 for the first point, high nibble =
  the same four for the second. Evidence: 16 bytes of `0x01` → 16 *separate*
  0.89 ns pulses 1.8 ns apart; `0x03` and `0x05` identical (bit1 = M2, bit2 = M3);
  **`0x11` → one continuous pulse**. High value for marker *m* is
  `(1<<(m-1)) | (1<<(m-1+4))`.
- §8.3.1 and §8.3.2 give different points-per-byte for the 9 GS/s and 2.5 GS/s
  models. Measured, **one rule covers both: marker bytes = (`:TRAC:DEF` length)/4.**

| mode | segment | bytes | per byte | requested → measured |
|---|---|---|---|---|
| 2.5 GS/s Direct | 2048 pts | 512 | 1.6 ns | 81.9 → **81.6 ns** on an 819.2 ns period |
| 9 GS/s DUC | 8192 | 2048 | 1.8 ns | 364 → **364.4 ns** on a 3.641 µs period |

In DUC this is one marker point per complex baseband sample (0.889 ns = 8 DAC
samples), consistent with the known 8-sample granularity.

Also established:

- **Markers play with the channel's RF `:OUTP OFF`** — CH1 can carry the trigger
  marker without radiating into the chain. No need to move the cable to CH2 M1.
- `:MARK OFF` makes the output vanish entirely (the scope then cannot trigger),
  which is the clean way to confirm which physical connector you are looking at.
- `duc_cw_setup()` loads **segment 1**, not segment-equals-channel. Wrong segnum
  gives `211, assertion failed`.
- **Too short a marker array fails silently and badly** rather than erroring:
  1024 bytes where 2048 were needed produced a single 1.85 µs pulse.

### 6. Open / next session

1. **Coherent averaging test, triggered on the marker**, with `fresh=True`. This is
   the measurement the whole session was working toward; both things that would
   have invalidated it (stale frames, no marker) are now fixed. Nothing about
   averaging has actually been measured yet — the one attempt was invalid.
2. **NCO grid snapping.** The DUC NCO free-runs and is not reset at segment start,
   so carrier phase only repeats if `f_NCO × T_buffer` is an integer. Expected to
   matter for averaging; untested.
3. Scope-side reference guard exists (`Scope.require_external_reference()`) but is
   not called from the bench scripts.
4. `run_ref_drift.py` left in the repo, unused — see §2 before spending time on it.

**New/changed files:** `reflock.py`, `check_reference.py`, `run_ref_drift.py`,
`run_phase_coherence.py` (new); `awg.py`, `scope.py`, `gui.py` (reference control,
markers, stale-frame guard).

---

## 2026-10-20 — Coherent averaging measured and working · DUC carrier grid · NCO quantum

**Instruments:** Tabor Proteus P9482D (fw 1.237.0), Tektronix MSO64B (FV 2.10.5.1825),
Wenzel 501-27501-11 OCXO via ZX10R-2-183-S+, scope locked (`ROSc:STATe? → LOCKED`).

**Wiring:** AWG CH1 M1 → scope CH4 (trigger, CH1 RF output OFF); AWG CH2 → scope CH3
(signal). Both AWG channels configured inside one reset so their segments start
together — a marker is rigid only to its own channel's segment. Scope 12.5 GS/s,
1 µs record, NORMal trigger.

### 1. The headline: averaging works

This is the measurement §6 of the 19 Oct entry was working toward, and the first
time anything about averaging has actually been measured rather than assumed.

DUC CW at **808.593750 MHz** (inside the 730–830 MHz IF band), 25 shots spread
**irregularly over 60 s**, two independent runs:

| amplitude surviving an average | run 1 | run 2 |
|---|---|---|
| referenced to the interpolated trigger | 0.998 | 0.997 |
| referenced to the first sample | 0.993 | 0.992 |
| scope's own 64-average | 0.990 | 0.990 |

0.997 means a 1000-shot average buys the full √1000 = 31.6× in SNR to within 0.3%.
On-grid circular sd is 4.0–4.3°, i.e. ≈14 ps of shot-to-shot timing jitter at
808.6 MHz — marker, trigger and noise combined.

### 2. The control: the DUC carrier must close the buffer

The NCO free-runs and is **not** reset at segment start. The baseband buffer repeats
every `T = 4096 / (9 GS/s / 8) = 3.6409 µs`, so the output repeats only if the
carrier fits a whole number of cycles into it:

```
coherent grid = 1/T = 274658.203125 Hz,   f = k * grid
```

Same cables, same trigger, same code path, carrier moved 100 kHz off the grid:

| | run 1 | run 2 |
|---|---|---|
| trigger-referenced (floor 0.20 for 25 shots) | 0.208 | 0.370 |
| scope 64-average | 0.028 | 0.129 |

Complete collapse. **Caveat on which number to quote:** |⟨e^{iφ}⟩| over 25 shots is
a biased, high-variance statistic — the same physical condition gave 0.21 and 0.37.
The 64-average is the tighter estimate and is the one to cite for the collapsed case.

`k = 2944` was chosen as a multiple of 64, which puts the grid point on a whole
number of Hz.

### 3. NCO tuning quantum measured: 2.095 Hz = 9 GS/s / 2³²

Sat on an exact grid point, asked for grid + δ, 12 shots over 20 s per point. Phase
walk is the discriminator: 50 mHz of error is a full cycle in 20 s, far finer than
any spectrum estimate on a 1 µs record.

| δ (Hz) | 0 | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 4.0 | 1000 |
|---|---|---|---|---|---|---|---|---|
| coherence | 0.998 | 0.998 | 0.997 | 0.998 | 0.418 | 0.425 | 0.373 | 0.151 |

A clean step. The changeover is at δ = q/2, so **2 Hz < q < 3 Hz**; 9e9/2³² = 2.095 Hz
is the only power-of-two word that fits (2³¹ → 4.19, 2³³ → 1.05).

**Consequence: `:NCO:CFR1 %.0f` (whole Hz) is safe and needs no change.** It rounds by
at most 0.5 Hz, below q/2, so the request lands back on the grid point. This was about
to be "fixed" to `%.6f` on reasoning alone; the measurement showed there was nothing to
fix. What is *not* safe is asking for a carrier that is not a grid point at all.

### 4. Correction: residuals cannot go into the baseband for CW

The 14 Oct note said to snap the NCO to the grid and absorb the remainder into the
baseband. That holds for a **chirp** — the baseband chirp is a transient, so its
start/stop can be shifted by the residual while the buffer still repeats exactly. It
does **not** hold for **CW**: a baseband tone must itself repeat in the buffer, so it
can only sit on the same grid. The achievable set of phase-coherent CW carriers *is*
the grid; finer spacing needs a longer buffer (4096 → 274.7 kHz, 65536 → 17.2 kHz).

### 5. Scope-side vs PC-side averaging — first real number

The trigger lands between two samples. `WFMOutpre:XZEro`, read per shot, says where.

- trigger-referenced **0.997** — the ceiling for PC-side averaging with re-alignment
- first-sample **0.992** — what any record-stacking averager gets
- scope 64-average **0.990** — matches the sample-grid figure, as predicted

So the scope stacks on its sample grid and cannot recover the sub-sample offset. The
cost here is 0.5%, i.e. nothing. Predicted from an 80 ps sample period at 808.6 MHz:
sinc(πfΔt) = 0.993 against 0.992 measured. **It scales with frequency** — the same
80 ps at 350 GHz would be total cancellation, so this must be revisited if anything is
ever digitised at a higher IF.

### 6. Two method traps, both hit today

1. **A coherence number is meaningless without its time span.** 25 back-to-back shots
   occupy ~0.2 s; a 0.2 Hz error walks 15° in that time and reports 0.998, while the
   same error over 60 s walks 12 cycles and collapses. The first version of the NCO
   test had a negative control that refused to fail for exactly this reason. Every
   coherence number now carries the wall-clock span it was measured over.
2. **Evenly spaced shots can alias.** A linear phase ramp sampled at equal intervals
   can advance a near-integer number of cycles per shot, so every shot sees nearly the
   same phase and a badly off-grid carrier reports coherence ≈1. Shots are now placed
   at random times within the span, fixed seed so states stay comparable.

### 7. Code

- `scope.py` — `get_waveform(..., absolute_time=True)` sets t = 0 at the interpolated
  trigger via `WFMOutpre:XZEro`, queried per shot and never cached.
- `reflock.py` — `fit_tone(t, v, f, t_ref)` returns amplitude *and* phase from one
  projection; amplitude by coherent detection, so noise does not inflate it.
  `estimate_phase()` is now a wrapper and keeps its old signature.
- `awg.py` — `duc_grid_hz()`, `snap_duc_carrier()`, and a printed warning from
  `duc_cw_setup()` / `duc_cw_setup_dual()` when a carrier is >1 Hz off the grid.
- New: `run_marker_coherence.py`, `run_nco_resolution.py`; JSON + PNG under
  `marker_coherence/`.

### 8. Pulse averaging — works, and only the *carrier* must be on the grid

§1–2 used a continuous carrier, which is the easy case and not how the
spectrometer runs. `run_pulse_averaging.py` plays what the acquisition will
see: a gated burst at the start of the buffer, a long quiet stretch after it,
one marker per buffer, scope triggered on the marker, CH2 averaged. Flat-top
with raised-cosine edges — deliberately **not** a decaying FID, so the
amplitude comparison has no envelope mixed into it.

```
buffer    16384 complex @ 1.125 GS/s = 14.5636 us,  grid 68664.551 Hz
pulse     1.99 us flat top (13.7% duty), then 12.6 us of nothing
carrier   808.593750 MHz, k = 11776, exactly ON grid
baseband  +1.234567 MHz = 17.98 grid steps, deliberately OFF grid
          -> line at 809.8283 MHz, measured 809.8282 (DUC is upper-sideband)
```

Two runs, 25 shots over 30 s each, plus a hardware 64-average:

| | run 1 | run 2 | control (carrier +25 kHz off grid) |
|---|---|---|---|
| pulse coherence | 0.998 | 0.998 | 0.018 / 0.180 |
| 64-avg amplitude | 0.990 | 0.993 | 0.081 / 0.121 |
| noise RMS reduction | 7.5× | 8.3× | 7.4× / 7.7× |
| **SNR gain** | **7.4×** | **8.3×** | **0.6× / 0.9×** |

Averaging a pulse is lossless and the noise falls by √64 = 8.0 as it should
(7.5 and 8.3 bracket it), so the full SNR is bought.

**The baseband does not have to be on the grid — only the carrier does.** A
transient has decayed to zero before the buffer ends, so it never has to join
up across the boundary and nothing constrains it. This is what makes the
coarse 68.7 kHz carrier grid harmless in practice: put the carrier on a grid
point and let the baseband place the line anywhere. It confirms the transient
half of §4, which was asserted there and untested.

With the **carrier** off grid the 64-average SNR gain is 0.6–0.9× — averaging
leaves you *worse off than a single shot*. That is the cost of getting it wrong.

Marker-to-RF skew measured **+548 ns**, constant run to run.

### 9. Two more silent traps

1. **`DATa:WIDth` defaults to 1 byte**, so an averaged record truncates back to
   8 bits. The first pass measured the quiet stretch of a 64-average at RMS
   **exactly 0.000 mV** with infinite noise reduction — every sample had
   collapsed onto one code. With `width=2` the same measurement reads
   1.9 → 0.25 mV, the real 8×. This scope's ADC is 12-bit, so 8-bit readback
   discards resolution on *single shots* too. `get_waveform(..., width=2)` now
   exists; use it for anything involving noise, SNR or averaging.
2. **Ranging must be done on the trigger** once the duty cycle is low. The
   marker is 100 ns in a 14.56 µs buffer (0.7%); a short free-running look
   lands in the quiet stretch and reports 0 mVpp, indistinguishable from a
   missing cable.

Also added: `AWG.duc_iq_setup_dual(I, Q, carrier)` — arbitrary complex baseband
to CH1 and CH2 in one reset, CH2 the signal and CH1 the aligned copy that clocks
the marker.

### 10. Live view on a scope MATH channel (`demo_averaging.py`)

Set up so the effect can be *watched* rather than re-measured. Unlike the
measurement scripts this one deliberately does **not** restore the scope — it
leaves everything triggered and free-running.

```
MATH1 = AVG(CH3), weight 64, ACQuire:MODe SAMple (not AVErage, or it averages twice)
trigger CH4 NORMal at the marker midpoint, 8 us window, 10% pretrigger
```

| carrier | MATH1 / CH3 |
|---|---|
| ON grid (settled) | 0.989 / 0.992 / 0.993 / 0.990 |
| OFF grid | **0.138** |

Matches the 64-average amplitude of §8 (0.990–0.993). CH3 itself does not
change at all between the two — the whole effect lives in MATH1.

- The math average needs ~64 acquisitions to flush after a change: 3 s after
  switching back on-grid it still read **0.803** on its way to 0.99. Wait ~6 s
  before believing any number after a change.
- The SCPI name is **`MATH:MATH1:AVG:WEIGHT`**, not `AVG:WEIGHTing` — the
  latter is a command error (ESR 32) and silently leaves the depth alone.
  From scratch: `MATH:ADDNew 'MATH1'` → `MATH:MATH1:TYPe ADVanced` →
  `MATH:MATH1:DEFine "AVG(CH3)"` → `MATH:MATH1:AVG:WEIGHT 64`.

### 11. Three bugs found wiring up the live view

1. **`restore_state()` puts the trigger source back.** After the measurement
   runs the scope was triggering on CH1 again rather than the marker on CH4,
   so it was not triggering at all — and a math average of that is
   meaningless. An observation session must set the trigger itself.
2. **The preamble cache was not per-source.** `get_waveform()` cached
   `xincr/ymult/yzero` and only cleared on a timebase change, so reading
   `MATH1` straight after `CH3` scaled the math data with CH3's volts-per-code
   and returned **0.0 mVpp**. It would equally have corrupted a CH3-then-CH4
   read at different V/div. Masked until now only because `fresh=True` clears
   the cache and every measurement so far used it. Now keyed on the source.
3. **MATH sources return floats**, channels return integer codes; the ASCII
   parser did `int(x)` and threw. Now parses as float, which also drops the
   per-sample Python `int()` and speeds up the large transfers.

Separately: the pulse had been overwritten by a CW upload between sessions
(the GUI), which is why CH3 looked continuous. The loaded waveform is not
persistent state — check it rather than assume it.

### 13. Grid snapping in the production code — and the rule was wrong first time

The buffer-grid condition from §2/§8 (`f = k * rate_cx / n_total`) is **necessary but not
sufficient**. The NCO also quantises its tuning word to 2.095 Hz (§3), so a carrier must
satisfy both. With `n_total = 2**a * d` (d odd), only every *d*-th buffer-grid point is
representable and the usable carrier step is

```
rate_cx / 2**a        NOT        rate_cx / n_total
```

equal only for a power-of-two buffer. The default CP-FTMW timing is the bad case:
`n_total = 12512 = 2**5 * 391`, buffer grid **89.914 kHz**, usable step **35.156 MHz**.

Measured (`run_grid_achievable.py`), 20 shots over 30 s, gated 2 µs pulse, same cables:

| carrier | cycles per buffer | coherence |
|---|---|---|
| 808593750.000 Hz (achievable) | 8993.000000 | **0.999** |
| 808683663.683 Hz (buffer grid only) | 8994.000000 | **0.114** |

The second is an *exact* whole number of cycles per buffer and still collapses — the naive
rule would have passed it. One hertz of unrepresentable residual walks a full cycle of phase
per second, so it dies over any run longer than about a second.

**API** (`awg.py`): `nco_step_hz()`, `buffer_grid_hz(n)`, `coherent_step_hz(n)`,
`snap_duc_carrier(f, n)`. `send_waveforms`, `send_chirp_duc_sync` and `send_cw_duc_sync` all
take `coherent=True` by default.

Per channel the carrier snaps and the same residual comes back out of the baseband:

- **Gated** (chirp, burst) — windowed to zero at both ends, never wraps, baseband
  unconstrained → absolute frequency held **exactly**, whatever the carrier did.
- **Continuous** — the baseband is snapped to whole cycles anyway, which is the *fine*
  buffer grid, so the output lands within half of 89.9 kHz rather than half of 35.2 MHz.
  Pushing the residual into the baseband instead of leaving it in the carrier is the point.
- **DSB** — cannot hold carrier±f with one shift; snapped, not compensated.

Bench-verified: `send_chirp_duc_sync(1.0 GHz, 100→200 MHz)` snaps the carrier to
984.375 MHz, shifts the baseband to 115.625→215.625 MHz, and the output span is
**1100.0→1200.0 MHz exactly as asked**; `:NCO:CFR1?` reads 984375000.000.

Two consequences worth remembering:

- The **NCO-only fast path** (`duc_cw_step` / `retune`) cannot do the compensating half —
  the baseband is already in segment memory — so a carrier handed to it is used as given.
  It now warns (`retune()` always; `duc_cw_step` only with `warn_grid=True`, so sweeps stay
  quiet). **Re-send rather than retune when the result will be averaged.**
- `_warn_off_grid` now lives in `_duc_upload_ch`, so every DUC path is covered rather than
  the three that remembered to call it. Direct mode needs none of it.

### 14. Pre-existing bug: the amplitude ceiling is 0.55 Vpp

`:VOLT? MAX` reads **0.55** in both direct and DUC mode on this unit. `send_chirp_duc_sync`,
`send_cw_duc_sync`, `send_chirp_with_lo` and `send_chirp_with_lo_duc` all defaulted to
`sync_amp_vpp=0.8`. The instrument answers `204, data out of range in scpi` and **silently
keeps the previous level**, so the sync burst was never at the requested amplitude. Defaults
are now 0.5, and `_duc_upload_ch` raises rather than letting the write fail quietly.

### 15. Chirp averaging — the last gap closed

`run_chirp_averaging.py`. The A/B is the production flag itself, through
`send_waveforms()` — the same path the GUI uses — with the two states
differing *only* in `coherent`:

```
chirp 700 -> 900 MHz in 1 us, period 11.1218 us, marker trigger,
20 shots over 30 s; asked carrier 800 MHz, NOT on the 35.156 MHz grid
```

| | coherent=True | coherent=False |
|---|---|---|
| carrier | 808.593750 MHz | 800.000000 MHz |
| cycles per buffer | 8993.000 | 8897.422 |
| baseband | −108.594 / +91.406 MHz | −100.000 / +100.000 MHz |
| chirp coherence | **0.990 / 0.991** | 0.073 / 0.192 |
| 64-average amplitude | **0.988 / 0.990** | 0.017 / 0.031 |
| measured sweep | 700.0 → 900.2 MHz | 700.0 → 900.2 MHz |

A chirp averages essentially losslessly once its carrier is on the grid, and
cancels almost completely when it is not — two runs each.

**Both states swept the identical band**, so the baseband compensation really
did hold the sweep while the carrier moved 8.594 MHz. That is the claim grid
snapping rests on, now measured rather than argued.

A chirp has no single frequency to fit, so each shot is matched against the
first shot of its own run (complex inner product of the analytic signals over
the chirp window). Shot 0 is the reference and is excluded from the
statistic, else it contributes a guaranteed zero-phase term worth 1/N.

**How not to measure a chirp's sweep.** A spectral dB contour badly
understates it — a linear chirp's spectrum is not flat-topped (Fresnel ripple
plus the windowed edges). The same genuine 700–900 MHz chirp measured:

| contour | edges | width |
|---|---|---|
| −3 dB | 766.3 → 833.3 MHz | 67.0 MHz |
| −6 dB | 753.5 → 846.1 MHz | 92.6 MHz |
| −20 dB | 725.8 → 873.8 MHz | 147.9 MHz |

all wrong, the −6 dB one by better than a factor of two. The instantaneous
frequency (derivative of the unwrapped analytic phase) is unambiguous and
gives 200.2 MHz/µs directly. The first version of this script reported the
−6 dB number and it looked like the chirp generator was broken; it was not.

*Caveat:* single-shot noise in this script comes from one reference capture
rather than an average over shots, so the 5.9–7.7× spread against √64 = 8 is
mostly that. §8's pulse measurement is the better noise number.

### 16. Marker trigger in the GUI

Backlog item 1, finally done. The Timing panel gained a **Marker out** tickbox
with CH (1/2), M (1–4) and a width in ns. It is re-applied after every Send,
because the `*RST` inside a waveform upload deletes the segment the marker
data lives in; it survives a Retune untouched, which is correct.

Checked headlessly through the real code path (chirp on CH1, marker CH1 M1):

- marker **1039 mVpp** on scope CH4, width **99.6 ns** against 100 asked
- chirp carrier 800 → 808.59375 MHz snapped, span held at 700.000 → 900.000 MHz
- unticking → the scope can no longer trigger on CH4, i.e. genuinely off
- marker requested on a switched-off channel → refused with a message, no crash

The array length follows the segment as `:TRAC:DEF` saw it: `2*n_total` uint16
in DUC (interleaved I/Q), `n_total` real samples in direct mode. A first version
switched off a hardcoded CH1 M1 when unticked, which would have left a marker on
CH2 M3 running; it now switches off whichever CH/M is selected.

**To use:** Mode DUC, CH1 chirp with gate `chirp`, Marker out ticked on CH1 M1,
Send. Then trigger the Scope block on the channel the marker is wired to, in
**NORMal** mode (AUTO free-runs and washes out any average), level 0 V — the
marker swings about ±0.5 V into 50 Ω.

### 17. The ~450 kHz dip comb on a wideband chirp — not degradation

Reported: a 2–2.5 GHz chirp, averaged, with sharp dips in its FT from about
2.172 GHz repeating every ~450 kHz.

A comb of *nulls* (rather than lines) means two copies of the signal separated
by 1/450 kHz = **2.22 µs** — far too long for a reflection (~440 m of cable),
so it had to be timing, and 2.22 µs is a DUC buffer period.

Measured with `run_chirp_comb.py` (flatten the magnitude spectrum over the
chirp band, transform again, read the ripple period off as a delay):

| detect µs | buffer µs | 1/buffer kHz | ripple τ µs |
|---|---|---|---|
| 0.1 | 2.2187 | 450.7 | 2.2189 |
| 2.0 | 4.1244 | 242.5 | 4.1242 |
| 6.0 | 8.1067 | 123.4 | 8.1073 |

τ **is** the buffer period, over a 4× range. Chirp 2 µs + dead 0.1 + detect 0.1
gives 2.2187 µs = 450.7 kHz, the reported spacing; 2.172 GHz / 450.7 kHz =
4819.0, an integer, so the dips sit on that comb.

Decisive test with the chirp isolated (detect 8 µs, 20% duty):

| transform window | ripple contrast |
|---|---|
| whole 30 µs record (3 periods) | **2720** |
| one chirp only | **3.4** |

The chirp's own spectrum is flat. The comb appears only when the transform
spans more than one repetition — it is simply the line spectrum of a periodic
waveform. A chirp repeating every T has power only at multiples of 1/T, and
the gaps between those lines are the "dips".

- **FFT exactly one chirp**, not the whole capture.
- At chirp 2 µs + dead 0.1 + detect 0.1 the duty cycle is **90%** — the chirps
  are nearly back to back and there is no detection window at all. That timing
  could not have worked for spectroscopy regardless; lengthening `detect` fixes
  the duty cycle and the comb together.
- Open thought: a free-running buffer excites at a comb spaced 1/T. Once the
  Quantum Composers sets a low rep rate this becomes fine enough not to matter,
  but worth remembering if anything is run free-running.

### 18. "Chirp not flat" and "FT jumping" — two separate causes

Setup: 1 µs chirp 2.09→2.44 GHz at 500 MHz/µs, 1 µs record at 6.25 GS/s,
scope AVERAGE ×8, triggered on the marker.

**(a) Envelope structure is the chain's frequency response.** A chirp maps
frequency onto time, so its envelope *is* the response of everything between
DAC and scope.

| | |
|---|---|
| peak-to-peak | 2.50 dB |
| tilt | −6.54 dB/GHz (−2.28 dB across the sweep) |
| flatness after removing tilt | 0.87 dB pp, sd 0.27 dB |
| strongest ripple | 116.3 MHz period → 8.60 ns round trip, ≈86 cm |

The "features at the beginning" are the low-frequency end sitting ~1 dB high
on the tilt. DAC sinc accounts for only ~0.3 dB of it; the rest is cable and
chain. The 116 MHz ripple is a real mismatch about 86 cm away — worth chasing,
and a candidate for pre-emphasis.

**(b) The FT jumping is sub-sample trigger quantisation killing the average.**

| | |
|---|---|
| single-shot ptp | 391.3 ± 1.0 mVpp (**0.2%**) |
| 8-average ptp | 238 mVpp (ratio 0.61) |
| XZEro spread across shots | 156 ps (one sample = 160 ps) |
| coherence, sample grid | **0.520** |
| coherence, re-aligned on XZEro | **0.996** |

The chirp is coherent. The scope stacks acquisitions on its sample grid and
cannot recover where inside a sample the trigger fell. At 6.25 GS/s one sample
is 160 ps = **130° of phase at 2.25 GHz**, so every shot lands at a random
phase, the average keeps about half the amplitude — and a different half each
time, which is the jumping.

This scales with frequency and sample rate: the same effect at 808 MHz and
12.5 GS/s (§5) cost only 0.5%. Digitising a 2.25 GHz chirp directly is the
worst case; the real receiver downconverts to 730–830 MHz first, where it is
negligible. To average a high-frequency signal: raise the sample rate
(12.5 GS/s halves the quantisation), or average on the PC with XZEro
re-alignment, which recovers the full 0.996. **Scope-side averaging cannot.**

### 19. 50 GS/s fixes it — and corrects two numbers from §18

Scope set to 50 GS/s. (I claimed here that four active channels cost sample
rate — **wrong**, see §20.)

| coherence, 2.25 GHz chirp | 6.25 GS/s | 50 GS/s |
|---|---|---|
| sample grid (scope's own average) | 0.520 | **0.9904** |
| re-aligned on XZEro | 0.996 | 0.9883 |

Scope-side averaging keeps 99%, so it *is* usable at 2.25 GHz. And re-alignment
no longer helps — the residual ~1% is real jitter (sd 7.9° ≈ 10 ps) or noise,
not sample quantisation, so PC-side averaging would buy nothing. **That settles
the phase half of the scope-vs-PC question: scope-side is fine provided the
sample rate is high enough.**

**It was not aliasing of the fundamental.** 2.25 GHz is below the 3.125 GHz
Nyquist at 6.25 GS/s; what looked wrong on screen was 2.8 samples per cycle,
which cannot draw a sine. But the chirp's **2nd harmonic at 4–5 GHz does fold**
at 6.25 GS/s onto 1.25–2.25 GHz, right on top of the band — real aliasing, and
it contaminated the flatness measurement.

Corrected envelope numbers:

| | 6.25 GS/s (contaminated) | 50 GS/s (good) |
|---|---|---|
| peak-to-peak | 2.50 dB | 1.65 dB |
| tilt | −6.54 dB/GHz | −3.34 dB/GHz |
| ripple after removing tilt | 0.87 dB | 0.85 dB |

About half the apparent tilt was the folded harmonic, not the chain; the ripple
is real and unchanged. I also over-stated the reflection distance in §18: the
delay read 8.60 ns at 6.25 GS/s and 7.39 ns at 50 GS/s, but with only ~3 ripple
periods across a 0.4 GHz sweep the resolution is ±2.5 ns. Both are the same
number within error — a reflection at roughly **7–9 ns round trip (0.7–0.9 m)**.
To localise it, sweep wider: a 2–4 GHz chirp would give ±0.5 ns.

### 20. Correction, and the scope's real limits

**`SAMPLERate MAX` is not accepted on this firmware.** It raises a command error
(ESR 32) and is silently ignored, leaving the rate where it was — so
`scope.set_max_sample_rate()` never maximised anything, it returned whatever the
previous session had set. That is where the "12.5 GS/s" came from, and my claim
that four active channels cost sample rate was wrong. Measured directly:

| channels on | max rate |
|---|---|
| CH3 | 50.00 GS/s |
| CH3+4 | 50.00 GS/s |
| CH1+3 | 50.00 GS/s |
| CH3+4+1 | 50.00 GS/s |
| **CH1+2+3+4** | **50.00 GS/s** |

No channel-count penalty at all. Fixed: `set_max_sample_rate()` now reads
`ACQuire:MAXSamplerate?` and writes it explicitly. Do **not** "fix" it by writing
a huge number — the scope accepts 2500 GS/s of *interpolated* samples.

Real limits of this unit: `ACQuire:MAXSamplerate?` = **25 GS/s** (real-time ADC),
`CONFIGuration:ANALOg:BANDWidth?` = **4 GHz**, max record **62,500,000 pts** =
1250 µs at 50 GS/s.

**50 vs 25 GS/s is not only record length.** 50 GS/s is 2× interpolated — no new
information — but it still measurably helps, because the averager stacks on the
returned grid and interpolation puts that grid finer relative to the trigger:

| rate | sample-grid coherence, 2.25 GHz chirp |
|---|---|
| 50.0 GS/s | **0.9860** |
| 25.0 GS/s | 0.9612 |
| 12.5 GS/s | 0.8660 |

Keep 50 GS/s when averaging at GHz; at the real 730–830 MHz IF the difference is
under a percent and 25 GS/s is fine. There is a floor of about **1% (sd ≈ 8° ≈
10 ps)** that no sample rate removes — real trigger jitter. Scope-side averaging
cannot beat ~0.99 here.

**What a long record costs** — time for a 64-shot average:

| record pts | µs captured | 64-avg time | shots/s |
|---|---|---|---|
| 50,000 | 1.0 | 1.64 s | 39.0 |
| 500,000 | 10.0 | 4.07 s | 15.7 |
| 5,000,000 | 100.0 | 34.36 s | 1.9 |
| 50,000,000 | 1000.0 | >60 s | <1.1 |

1 ms of record is possible (max 1250 µs) but costs ~35× in averaging speed: a
1000-shot average is ~25 s at a 1 µs record and ~15 min at 1 ms. **Even at the
shortest record the scope manages only 39 shots/s while the AWG offers 90,000/s**
— the scope's rearm is the throughput bottleneck by ~2300×, which is the real
limit on scope-side averaging, not the data rate.

### 21. Multi-chirp per gas pulse + phase cycling — works

`run_phase_cycling.py`. The scope manages only ~32 acquisitions/s (78 with
FastFrame, §20), but a gas pulse lasts hundreds of µs while a chirp/FID cycle
is ~11 µs. So put K cycles in **one** AWG buffer: one marker, **one** scope
acquisition holding K FIDs, and the ~21 ms rearm is paid once per gas pulse
instead of once per FID. At a tens-of-Hz valve that is the whole game.

Each cycle carries a programmed chirp phase. The molecular response follows it;
chirp breakthrough, detector ringing, DC offset and digital pickup do not — so
combining with the conjugate phases adds signal and cancels the rest.

Test: 4 cycles × 3.1004 µs, chirp 1.99–2.49 GHz in 1 µs, carrier 2241.210938 MHz
(snapped), phases 0/180/0/180, scope 12.5 GS/s, 16-average.

| cycle | programmed | measured | error | amplitude | corr w/ cycle 0 |
|---|---|---|---|---|---|
| 0 | 0° | 0.0° | 0.0 | 161.6 mV | +1.000 |
| 1 | 180° | −180.0° | 0.0 | 160.5 mV | **−1.000** |
| 2 | 0° | 0.0° | 0.0 | 160.1 mV | +1.000 |
| 3 | 180° | −180.0° | 0.0 | 160.3 mV | **−1.000** |

```
phase-cycled sum   642.5 mV = 4.00x a single cycle (ideal 4.00)
straight sum         1.6 mV = 0.25%  -> non-chirp-following 52.1 dB down
```

Phases land exactly where programmed and the waveform correlations are ±1.000 —
the 180° cycles are exact inversions. Plot: `phase_cycling.png`.

**The first run was at 12.5 GS/s and the numbers were misleading:** phase errors
0.5–1.3°, 15% amplitude spread, straight sum 6.4% (−23.9 dB). I read the
amplitude spread as an artifact beating against the signal, B ≈ 12 mV. Wrong —
nearly all of it was my own `np.interp` **linear interpolation error**. At
2.24 GHz, 12.5 GS/s is only 5.6 samples per carrier cycle, where linear
interpolation injects percent-level amplitude and phase error; at 50 GS/s
(22 samples/cycle) it falls to 0.25%. **When re-sampling onto an exact grid,
either sample ≥20× the carrier or use sinc/FFT interpolation** — linear is not
adequate near Nyquist.

Two things that had to be right:

- **The NCO must be de-rotated per cycle.** It free-runs across the buffer, so
  cycle *k* starts with carrier phase 2π·f_c·t_k on top of the baseband;
  `build_cycles()` subtracts it. The carrier still has to sit on the coherent
  grid for the **full** buffer, not per cycle.
- **Cut cycles at exact times, not rounded samples.** The first pass stepped by
  `round(t_cycle/dt)` = 38756 samples where the true value is 38755.56. That
  0.44-sample slip is 35 ps = **28° per cycle at 2.24 GHz**, and showed up as a
  dead-straight linear error 0/28.5/57/85.5 that dragged the cycled sum to
  3.40×. Interpolating onto the exact grid fixed it. Same class of trap as the
  sub-sample trigger offset in §18.

**This is the architecture for the FID panel:** K phased cycles in one buffer,
one marker, hardware-average the whole record across gas pulses, then split into
K and combine in software with the conjugate phases.

### 22. Open / next session

1. ~~Chirp coherence on the bench.~~ **DONE (§15)** — 0.990/0.991 on grid against
   0.073/0.192 off it, with the sweep held identical across both.
2. ~~Grid snapping is warned about, not applied.~~ **DONE (§13)**, including the
   achievable-vs-buffer-grid correction.
3. **Noise reduction spread.** 7.5× vs 8.3× is wider than it should be; more
   repeats before quoting √N as exact.
4. **Longer averages.** 30–60 s confirmed; a real run may go minutes.
5. **Rep rate.** The buffer free-runs at 68.7 kHz. Once the Quantum Composers
   sets global timing the rep rate drops to the experiment's, and the
   marker-to-shot relationship needs re-checking.
6. **Throughput is still unmeasured.** Only the *phase* half of the
   scope-vs-PC averaging question is settled (§5). Transfer rates, ASCII vs
   binary and FastFrame all remain open.

### 24. The envelope ripple is a reflection (pad test), and the delay was wrong

2026-10-23. `run_pad_test.py`, chirp 2.0-2.5 GHz, 3.4 us, 0.5 Vpp, 50 GS/s,
16 hardware averages, **5 independent repeats per condition**, AWG CH2 ->
scope CH3 direct, marker CH1 M1 -> CH4. Vertical scale pinned at 65.4 mV/div
across both runs: the front-end attenuator setting changes the scope's input
match, which is one of the two reflection coefficients being measured, so it
must not move between conditions.

| | no pad | 7 dB pad |
|---|---|---|
| ripple depth | 0.7563 +- 0.0013 dB pp | 0.1890 +- 0.0010 dB pp |
| G1.G2 | 0.04351 | 0.01088 |
| round-trip delay | 5.9484 +- 0.0072 ns | 5.9606 +- 0.0042 ns |
| direct signal | 376.7 +- 2.7 mVpp | 170.9 +- 0.6 mVpp |
| tilt | -3.42 dB/GHz | -3.54 dB/GHz |

**It is a reflection.** The pad measures 6.87 dB in band (from the direct
signal, not from its label), so a pure through-path reflection predicts a
13.73 dB round-trip change; measured 12.04 dB.

**The 1.69 dB shortfall is expected, and a pad test cannot avoid it.**
Inserting a pad does not only attenuate the existing standing wave, it adds
its own port reflection, and the resonance between that port and the far end
never crosses the pad, so it is not suppressed. At a typical 20 dB pad return
loss that term is of order 0.02, the same scale as the 0.01 discrepancy. A
shortfall of this size is not evidence that part of the ripple is internal.

The delay is the strongest evidence: 5.9484 vs 5.9606 ns, a 0.012 ns
difference against 0.007 and 0.004 ns of scatter. Same physical path,
attenuated.

**Correction to 18-19: the round trip is 5.95 ns, not 7.39 ns.** The old
number picked the peak FFT bin of the detrended envelope, and I quoted the bin
width (2.5 ns) as the uncertainty. `ripple_fit()` in run_chirp_flatness.py now
uses the FFT only to locate the ripple and takes the delay and depth from a
least-squares sinusoid fit refined across +-1 bin. Repeat scatter is
0.004-0.007 ns, about 400x tighter than the bin. Depth is now the fitted
sinusoid amplitude rather than the detrended peak-to-peak, which had included
everything else in the envelope.

With the corrected delay the 50 cm cable nearly accounts for it:

| | |
|---|---|
| 50 cm cable at VF 0.66 | 5.05 ns |
| measured round trip | 5.95 ns |
| left for instrument internals | 0.89 ns, about 9 cm each in the AWG output stage and the scope front end |

Ordinary, where 7.39 ns demanded 2.6 ns and was not. If the two ports are
comparable, each is about 13.6 dB return loss, VSWR 1.53 - normal for a bare
instrument pair with no isolator.

Cross-checks: the tilt is unchanged within 0.12 dB/GHz, so the pad is flat and
the tilt is the chain's (DAC sinc + cable loss), not a mismatch. The no-pad
detrended pp of 0.819 dB reproduces 19's 0.85 dB, so this replicates the
earlier measurement - what the repeat bought was the pairing and the error
bar, and that is what moved the delay.

**Method note.** The first attempt sent the chirp with `send_chirp_duc_sync()`,
which puts the chirp on AWG CH1 and a 40 ns sync burst on CH2. With CH2 cabled
to scope CH3 the analysis fitted a "chirp" 0.035 us long sweeping
2.2786-2.5846 GHz - entirely an artifact of a 40 ns constant tone at the
carrier. Use `send_waveforms()` with the same chirp on both channels and CH1's
RF off, as run_chirp_averaging.py does. The tell is a measured sweep that
disagrees with what was programmed.

### 25. Open / next

1. ~~Cable swap (item 23c).~~ **DROPPED, and item 23 is closed.** It would
   have split the 5.95 ns into cable and instrument internals, but that
   standing wave exists only between an AWG and a scope with 50 cm of coax
   between them - a bench-test configuration. In the instrument the AWG
   drives the multiplier chain and the scope sees the receiver IF, so neither
   port survives. Step (a), widening the chirp, was not needed either: the
   fit, not the span, was what limited the delay.

   What carries over is the **tilt, -3.42 dB/GHz**, which is the AWG's own
   output response over 2.0-2.5 GHz (DAC sinc + cable loss) and not a
   mismatch - the pad did not change it. That is about 1.4 dB of slope across
   the chirp band presented to the first multiplier stage; how much it matters
   depends on how hard that stage is driven. And the method: `run_pad_test.py`
   plus `ripple_fit()` is chain-agnostic, so the same ~3 minute measurement
   localises any later mismatch (between multiplier stages, at the filters, at
   the mixer IF port) to better than 0.01 ns and says whether it is a
   reflection.
2. Re-run `run_phase_cycling.py` once the receiver is in place, so the 52 dB
   residual becomes a real measure of chirp breakthrough and detector ringing.
3. Rep rate: the buffer free-runs. Once the Quantum Composers sets global
   timing the marker-to-shot relationship needs re-checking.

### 26. Plan for the week of 2026-10-26

Three facts established 2026-10-26 that set this plan:

- **Some** of the 2026-10-07 Mini-Circuits basket has arrived; which parts is
  not recorded, so an inventory comes before any Phase 1 bench work.
- The Quantum Composers pulse generator exists but is committed to the
  ablation experiment.
- That same experiment has a vacuum chamber and pulsed nozzle the
  spectrometer could share.

**The QC and the chamber are the same setup, so the QC is a scheduling
constraint and not an availability one** - when the spectrometer sits on that
chamber the QC is there too. In the meantime the AWG's own `:TRIG:TIMer`
(200 ns - 2 s internal timed trigger) covers the rep-rate work. Item 25.3 was
parked waiting on hardware that does not have to be waited for.

**Mon-Tue: the acquisition architecture into production code.** Section 21
established it - K phased chirp/FID cycles in one buffer, one marker,
hardware-average the whole record across gas pulses, split into K, combine
with conjugate phases - but it exists only inside `run_phase_cycling.py`.
Promote `build_cycles()` into `awg.py` with its NCO de-rotation and its
exact-time segment cutting (both were traps that cost a measurement each,
section 21), add the scope-side counterpart, add split-and-combine.

**Tue-Wed: rep rate.** `:INIT:CONT OFF` + `:TRIG:TIMer` to run at a real tens
of Hz instead of free-running at 249 kHz, then re-check the marker-to-shot
relationship there. This also rewrites section 22's throughput conclusion: at
20 Hz the scope's 32 shots/s rearm is no longer the bottleneck, the gas pulse
is, which changes whether FastFrame is worth its complexity. Re-derive with
numbers rather than carrying the old framing forward.

**Wed-Thu: the GUI FID panel.** The item deferred by the user 2026-10-14.
Every precondition is now measured rather than assumed - marker trigger, grid
snapping, and the scope-vs-PC averaging question. Acquire-FID -> N shots ->
averaged FID -> spectrum, into the existing Analysis block.

**Thu-Fri: parts.** Inventory against the `Parts list` sheet of
assembly_parts_and_order_2026-10-07.xlsx first; it determines how far Phase 1
runs. Then as far up steps 1.1-1.6 as the parts allow, thru-then-amp so pads
and cables cancel. **Hard stop before 1.7** - the -6 dBm LO trim is a gate
with a damage limit 4 dB above it, not a casual measurement.

**Three things with lead times, any day, near-zero bench time:**

1. **The ACST questions.** The internal IF amp's P1dB and overload recovery
   time is the sole remaining determinant of dead time and has been open since
   August. Also: is -2 dBm LO an operating max or a damage limit; is there DC
   on the IF port; out-of-band input tolerance as a written dBc number.
2. **The mm-wave detector for 310-370 GHz.** Hard gate before Phase 3 (step
   3.2 is a standing safety rule - measure the power at the receive-horn
   position before the MixAMC is ever in the beam, which is damaged at 0 dBm).
   Nothing on hand, long lead time. Start the quote now.
3. **Survey the chamber.** Optical access for a >=0.5 m free-space path,
   window and flange options at 310-370 GHz, and the nozzle's trigger
   interface.

**Dropped:** further AWG-to-scope characterisation. Item 23 is closed, the
bare bench is understood, and the next flatness measurement worth taking is
through real hardware.
