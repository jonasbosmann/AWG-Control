# x8_chain — bench measurements for the multiplier-only transmit chain

Measurements for the chain **as physically built**:

```
AWG 1.614–1.927 GHz → ×2 → ×2 → ×2 → 12.91–15.42 GHz → AMC (×24) → 310–370 GHz
```

The design model lives in `../multiplier_chain.py`. This folder holds the bench
work that replaces its datasheet numbers with measured ones.

## The one rule

**A 10 dB pad sits in front of the measuring instrument for every measurement in
this campaign.** It is the largest pad on hand and it never moves, which is what
makes it safe to treat as a constant of the setup.

It lives in `config.PAD_AT_INSTRUMENT_DB` and nowhere else. Every run script
imports it and writes it into the saved file's metadata, so no measurement can
be orphaned from the attenuation that was in line when it was taken.

This exists because of what happened on 2026-07-27: two of three CW level scans
are unusable because a pad was changed mid-session and never recorded. The
offset varies ~20 dB across the band, so it cannot even be backed out. The
measurements were fine; the bookkeeping killed them.

**If a pad changes at the bench, change it in `config.py` in the same motion.**

Saved levels are always **raw as the instrument reported them**. The pad is
recorded, not applied — reporting adds it back. A later correction to the pad
value then does not require re-taking data.

## Files

| file | what it is |
|---|---|
| `config.py` | single source of truth — pads, bands, per-stage parts, instrument limits, safety checks |
| `chainlog.py` | JSON logging; writes the setup block itself, so a script cannot omit it |
| `run_drive_level.py` | closed-loop ladder of output levels, ending at the experiment's |
| `run_stage_gain.py` | gain vs frequency and gain vs drive (P1dB) for one amplifier stage |
| `verify_chain.py` | propagates a stage-A output level through the whole chain, both band edges |
| `measurements/` | saved data, one folder per session (gitignored) |

## Stage A — first up

```
AWG CH1 → 3 dB pad → ZX60-2522MA-S+ → 10 dB pad → scope
```

### 1. Measure the drive ladder (start here)

```
python x8_chain/run_drive_level.py
```

Measures a **full band sweep at each of +5, +9, +12 and +15 dBm** at the amp
output — the rungs of `config.OUT_LEVEL_BACKOFF_DB`, ending at the level the
experiment runs at. At each rung it converges the AWG amplitude closed-loop on
three spot frequencies, then sweeps the band.

Two reasons for the ladder rather than a single setting. It **ramps up**: the
first rung is 10 dB down where nothing is near a limit, so a wrong pad, an
unpowered amp or a cable on the wrong port shows up as an obviously wrong
number rather than as damage, and each rung's level is predicted from the one
below and checked against the scope's limit before being commanded. And **one
level is not a characterisation** — gain and flatness both move with drive, and
the question worth answering is whether the experiment's level is still linear.
One curve can't answer that; the family can, and the lowest rung is the
small-signal reference.

At +15 dBm out the scope sees +5 dBm — 15 dB below its 2.3 V<sub>RMS</sub>
(+20.2 dBm) limit. Expect roughly **0.34 Vpp** at the top rung if the gain is
near its 24.8 dB typical. A number far from that means something is wrong with
the setup, not with the amp — find out what before going further.

### 1a. Check what that level does to the rest of the chain

```
python x8_chain/verify_chain.py [stage_A_band_max_dBm ...]
```

Carries **both band edges** from the stage-A amp output to the AMC and checks
every part's input window at each. A part with a 5 dB window can be inside it
at one band edge and outside at the other, which a single-level walk down the
chain will not show.

Mind the convention: `run_drive_level.py` sets the band **maximum**, while
`multiplier_chain.full_chain()` quotes the band **minimum**. They differ by the
~1.2 dB tilt, so the design's +16.13 dBm is a band max of about **+17.3 dBm** —
asking for +15 dBm is ~2.3 dB below the design point, not at it.

### 2. Characterise the stage

```
python x8_chain/run_stage_gain.py thru     # amp replaced by a barrel
python x8_chain/run_stage_gain.py amp      # amp fitted
```

Gain is `amp − thru`, point by point. Every pad, cable, connector and scope
calibration error appears in both runs and cancels exactly, so gain does not
depend on knowing the AWG's absolute output, the pads' true values, or the
scope's absolute accuracy. The only residual is the barrel's own loss
(~0.05–0.1 dB), which biases gain high by that much.

The thru also gives the input level for free: it measures at exactly the plane
where the amp's input connector sits, so `Pin = thru + 10 dB`.

Two sweeps per run, because the cross product wastes bench time and neither
question needs it:

- **flatness** — full frequency sweep at every rung of the drive ladder. It
  reuses the amplitudes `run_drive_level.py` found, so thru and amp are taken
  at identical settings and each curve corresponds to a known output power.
  Gain flatness adds to the AWG's own 1.2 dB droop, and the sum has to fit the
  AMC's 7 dB window.
- **compression** — 10-step, 18 dB drive ladder at 3 spot frequencies, running
  right up to the AWG's ceiling. Gives P1dB in situ. This one *does* go to full
  drive; reaching compression is its whole point.

Everything is repeated and the scatter is reported. The AWG level scan showed
0.19 dB rms of per-point scatter, which is the same size as several of the
effects being looked for — a single pass is not a result.

## Watch out

**+15 dBm is a good level to measure at, not to operate at.** It is 5.3 dB
below the amp's +20.3 dBm P1dB, so well clear of the AM-to-PM the chain would
multiply by 192. But `verify_chain.py` shows it leaves the ZX90-2-24-S+ under
its +11 dBm input floor at the weak band edge, and the chain wants *more* drive
from stage A, not less. Target the design point (**band max +17.3 dBm**) once
the stage is characterised.

**The design table grants the ZX60-123LN-S+ 1.9 dB it may not have.**
`multiplier_chain.full_chain()` back-solves whatever gain lands the MMD-0415HS
at its +14 dBm nominal, prints "16.9 dB needed", and never compares it against
`AMP_123LN_GAIN_DB = 15.0` in the same file. At 15.0 dB the chain does not
close at either level; at ~17 dB (plausible for this part at 3.5 GHz, but
unverified) the design point does. **This makes stage B the highest-value
measurement, not stage A** — stage A has 5 dB of margin everywhere, stage B is
where the chain is ambiguous.

**The 3 dB drive pad stays in for the thru reference.** The thru replaces only
the amp. That is what makes the gain subtraction exact.

**Pad bandwidth.** A DC–6 GHz pad (VAT series) covers stages A and B but *not*
stage D's 12.9–15.4 GHz output — that needs DC–18 GHz (BW-S series). Check
before trusting a high-band number.

**Stage D is above the scope's bandwidth** and has to go to the EXA. The EXA
cannot be damaged by this chain (+30 dBm limit) but **set its internal
attenuation from `config.required_exa_atten_db()`, not habit** — the usual
10 dB would put +7 dBm on the first mixer and the analyzer would invent
harmonics that are not in the signal.

## Open questions this campaign exists to answer

1. **Do the doublers actually saturate?** The whole flatness case assumes they
   compress input tilt. Measure Pout vs Pin per stage — this is the
   load-bearing assumption of the entire design.
2. **Real gain vs the 22.6/24.8 dB datasheet spread** — 2.2 dB of guesswork
   about where the HMC188's +10 dBm input floor lands.
3. **ZX90 F1 suppression drops to 21.9 dBc at 7.0 GHz input** (vs 31–33 dB
   either side), inside the band stage C sweeps.
4. **The 1.500 GHz ambient line at −95 dBm** was measured as ambient pickup,
   probably not through the same cabling as the CW scan. Re-measure through an
   identical path before fixing the input filter's rejection spec.
