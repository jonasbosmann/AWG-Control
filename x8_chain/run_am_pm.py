"""Measure AM-to-PM conversion through the chain as built, on the EXA.

    AWG -> ZX60-2522MA-S+ -> PE87FL1012 -> HMC188 -> 10 dB pad -> EXA

No hardware change: this is exactly the run_doubler.py setup, measured a
different way.

WHY THIS EXISTS
---------------
Every level decision in this campaign has been argued with a backoff rule --
"keep 2 dB below P1dB" -- and that rule exists to keep AM-to-PM small. AM-to-PM
has never been measured here. Until it is, "2 dB of backoff is fine" is an
assumption wearing the clothes of a result.

It matters more than the usual amount because of how this chain is built.
Amplitude and phase errors are NOT symmetric on the way to 310-370 GHz:

  amplitude  an error at the AWG reaches the AMC INPUT scaled by the doublers'
             tilt transfer (measured r^3 = 4.2), and lands in a 7 dB window
             that has room to absorb it. Pre-emphasis can also correct it.

  phase      an error at the AWG is multiplied by 192 -- x2, x2, x2, then the
             AMC's x24 -- and arrives at the OUTPUT, where nothing absorbs it
             and no filter reaches it. Pre-emphasis cannot touch it either,
             because it is not a function of frequency: it is a function of
             the amplitude that pre-emphasis is busy changing.

So AM-to-PM is the mechanism that converts the cheap error into the expensive
one. A stage with 1 deg/dB of AM-to-PM turns the chain's own 2.2 dB of drive
tilt into 2.2 deg here and 211 deg at the AMC output. That is the whole
argument for measuring it.

HOW
---
The AWG plays a sinusoidally amplitude-modulated carrier (awg.duc_am_setup)
whose PEAK equals the CW operating amplitude, so the modulation explores
DOWNWARD from the operating point and no new worst case is created. The EXA
captures raw I/Q (specan.iq_setup / iq_capture) at the doubler's 2f output.
Envelope and phase come out of the same samples, so they are perfectly
co-registered -- no triggering, no alignment, no assumption about delay.

The slope is extracted as the ratio of the two signals' Fourier components AT
THE MODULATION FREQUENCY. That is the point of modulating rather than just
stepping the level: the analyzer's LO is not locked to the AWG, so the
measured phase carries a large ramp and a slow drift that have nothing to do
with the modulation. Both live near DC. Reading the answer out at a few
hundred kHz rejects them for free, which a level-stepping measurement cannot
do -- it would have to distinguish the phase step it wants from the drift
that happened while it was stepping.

WHAT IS AND IS NOT SEPARATED
----------------------------
This is the COMBINED AM-to-PM of the AWG, the stage-A amp, the filter and the
doubler. The AWG's own contribution cannot be separated without a thru
reference. MODE=awg takes that reference (AWG -> pad -> EXA, a real cable
change); the report subtracts it when one exists, doubling it first, since the
doubler doubles carrier phase and therefore doubles anything upstream of it.

Run:  python x8_chain/run_am_pm.py [n_freqs] [drive_offset_db]
      MODE=chain|awg     what is physically in the path (default chain)
      DEPTH_DB=6         peak-to-trough AM swing
      NO_SHOW=1          no windows,  LIVE=0  no live window
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import chainlog
import multiplier_chain as mc
from liveplot import LivePlot
from awg import AWG
from specan import SpecAn
from run_doubler import known_drive, measure_tone

STAGE = "A"

# What is in the path. 'chain' is the zero-hardware-change case: the doubler
# setup, measured for phase instead of level. 'awg' is the thru reference and
# DOES need the cable moved.
MODE = os.environ.get("MODE", "chain")
_MODES = {
    "chain": dict(tone=2,
                  element="AWG +ZX60-2522MA-S+ +PE87FL1012 +HMC188 "
                          "+VHF-2100+ +VLF-3800+, combined"),
    "predoubler": dict(tone=1,
                       element="AWG +ZX60-2522MA-S+ +PE87FL1012, combined"),
    "awg":   dict(tone=1,
                  element="AWG alone (thru reference)"),
}
# Three modes because there are three suspects and one measurement cannot
# separate them. Run all three and subtract:
#     awg                        -> the AWG (and the analyzer) alone
#     predoubler - awg           -> the stage-A amp + filter
#     chain - 2*predoubler       -> the HMC188 itself
# The factor 2 is the doubler doubling carrier phase, so everything ahead of
# it arrives at the 2f plane doubled. 'predoubler' needs the doubler
# unbolted; 'awg' needs the cable moved to the analyzer directly.
if MODE not in _MODES:
    raise SystemExit(f"MODE must be one of {sorted(_MODES)}, got {MODE!r}")
TONE = _MODES[MODE]["tone"]
ELEMENT = os.environ.get("ELEMENT", _MODES[MODE]["element"])
# The printed path is DERIVED from ELEMENT rather than being a second string
# that has to be kept in step with it. A run whose header says one thing and
# whose saved metadata says another is how two files ended up labelled with
# the wrong filter on 2026-07-31.
PATH = ELEMENT.split(",")[0].replace(" +", " -> ")

# x8 to the AMC input, then the AMC's x24 = 192 from the AWG plane. Measuring
# at TONE x the AWG frequency means TONE of those doublings already happened,
# so what is left to multiply the measured phase by is 192/TONE.
MULT_TO_OUTPUT = 192.0 / TONE

# ── modulation ────────────────────────────────────────────────────────────
MOD_HZ = 200e3
# High enough to sit clear of the LO drift and the analyzer's close-in phase
# noise, low enough that every stage responds quasi-statically -- at 200 kHz
# the envelope changes over ~5 us, which is slow against any electrical time
# constant in an amplifier or a diode doubler. If a bias or thermal effect
# with a longer time constant exists, it shows up as a phase lag away from
# 0/180 deg in the report, which is why that number is printed.
DEPTH_DB = float(os.environ.get("DEPTH_DB", "6.0"))
# Peak-to-trough. 6 dB brackets the region the backoff argument is about. The
# extracted slope is an average over the swing, so a smaller depth gives a
# more local slope and a worse signal-to-noise ratio.
N_TOTAL = 16384          # complex buffer samples at 1.125 GS/s -> 14.56 us

# ── capture ───────────────────────────────────────────────────────────────
IF_BW_HZ = 5e6
# The analyzer's digital IF bandwidth, and the ONLY acquisition knob it has --
# the sample rate is not settable on this unit and follows from this (see
# specan.iq_setup). Set 25x wider than the modulation on purpose: the IF
# filter is Gaussian, and any asymmetry in its response about the tone
# converts AM into PM, which is precisely the quantity being attributed to
# the chain. 25x out, its own contribution is ~0.02 dB of tilt across the
# sidebands. The price is noise floor, and there is 70 dB of signal to spend.
CAPTURE_S = 1e-3         # ~200 modulation cycles
N_FREQS = 3
BACKOFF_DB = (0.0, -2.0, -4.0, -6.0)
# Relative to the operating drive. Downward only, same reason as run_doubler:
# going above pushes the stage-A amp toward P1dB, and the interesting part of
# an AM-to-PM curve is watching it FALL as the drive backs off -- that is what
# distinguishes a compression effect from a constant instrumental offset.

EXA_ATTEN_DB = 20
REF_LEVEL_DBM = 10
SPAN_HZ = 50e6           # for the swept-mode sanity check only
RBW_HZ = 100e3
N_POINTS = 801
SETTLE_S = 0.25

MIN_TONE_SNR_DB = 20.0   # swept-mode check: is there a tone here at all
MIN_PM_SNR_DB = 10.0     # per-point: below this the slope is noise


# ── analysis ──────────────────────────────────────────────────────────────
def _bandpass(x, fs, lo_hz, hi_hz):
    """Keep only [lo_hz, hi_hz] of a real signal, by rfft masking."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[(f < lo_hz) | (f > hi_hz)] = 0.0
    return np.fft.irfft(X, n=len(x))


def am_pm_from_iq(z, fs, f_mod_hint, trim=0.02):
    """AM-to-PM slope in deg/dB from one I/Q record.

    Returns a dict. The headline is `slope_deg_per_db`, but `pm_snr_db` is
    what says whether to believe it: a phase measurement on a weak or noisy
    tone produces a perfectly confident number out of nothing.

    The slope is SIGNED: positive means the phase rises as the envelope
    rises. Its magnitude is |P|/|A| exactly as before, so values compare
    directly with runs saved before 2026-08-03 -- those stored the magnitude
    with the sign discarded.
    """
    n0 = len(z)
    k = int(n0 * trim)
    z = z[k:n0 - k] if k > 0 else z
    n = len(z)
    if n < 256:
        raise ValueError(f"only {n} samples after trimming")
    t = np.arange(n) / fs

    env_db = 20 * np.log10(np.abs(z))
    ph_deg = np.degrees(np.unwrap(np.angle(z)))

    # Remove the free-running LO: a frequency offset is a phase RAMP and the
    # slow drift on top of it is close to quadratic over a 2 ms record.
    # Neither is signal. A degree-2 fit takes both out without touching
    # anything at the modulation frequency, which is ~400 cycles away.
    ph_ac = ph_deg - np.polyval(np.polyfit(t, ph_deg, 2), t)
    env_ac = env_db - env_db.mean()

    # Find the modulation frequency in the ENVELOPE, where it is unambiguous,
    # and then use that same frequency for the phase. Measuring it rather than
    # trusting the commanded value costs nothing and covers any clock offset
    # between the AWG and the analyzer.
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(env_ac * win))
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    band = (fr > f_mod_hint * 0.5) & (fr < f_mod_hint * 1.5)
    if not band.any() or spec[band].max() <= 0:
        raise ValueError(f"no modulation found near {f_mod_hint/1e3:.1f} kHz")
    idx = int(np.arange(len(fr))[band][int(np.argmax(spec[band]))])
    f_mod = float(fr[idx])
    if 0 < idx < len(spec) - 1:                      # parabolic refine
        a_, b_, c_ = spec[idx - 1], spec[idx], spec[idx + 1]
        den = a_ - 2 * b_ + c_
        if den != 0:
            f_mod = float((idx + 0.5 * (a_ - c_) / den) * fs / n)

    # Complex Fourier component of each signal at f_mod. For x = X*cos(wt+p),
    # 2*sum(x*w)/sum(win) = X*exp(jp), so |.| is the peak amplitude in that
    # signal's own units: dB for the envelope, degrees for the phase.
    w = np.exp(-2j * np.pi * f_mod * t) * win
    norm = win.sum() / 2.0
    A = np.dot(env_ac, w) / norm
    P = np.dot(ph_ac, w) / norm
    am_db = float(abs(A))
    pm_deg = float(abs(P))

    # Per-bin phase noise, from bins well away from the harmonics. This is the
    # amplitude a pure-noise record would have produced at f_mod, so the ratio
    # to pm_deg is the honest confidence in the slope.
    pspec = np.abs(np.fft.rfft(ph_ac * win)) / norm
    mask = (fr > 0.2 * f_mod) & (fr < 6.0 * f_mod)
    for h in range(1, 7):
        mask &= np.abs(fr - h * f_mod) > 4.0 * fs / n
    noise = float(np.median(pspec[mask])) if mask.sum() > 16 else float("nan")
    snr_db = 20 * np.log10(pm_deg / noise) if noise and noise > 0 else float("nan")

    # Phase of the PM relative to the AM. A memoryless AM-to-PM gives 0 or
    # 180 deg. Anything else is a time constant -- a bias network, a thermal
    # effect -- and means the slope depends on modulation rate, so the number
    # would not extrapolate to a chirp.
    lag_raw = float(np.degrees(np.angle(P / A))) if am_db > 0 else float("nan")
    # 0 and 180 are not two flavours of memoryless: 180 means the phase FALLS
    # as the envelope rises. That is a sign, and it belongs on the slope --
    # the thru subtraction and the drive trend both depend on it. The folded
    # lag keeps only the distance from the nearest memoryless axis, which is
    # what the time-constant warning is about.
    sign = 1.0 if abs(lag_raw) <= 90.0 else -1.0
    lag_deg = (lag_raw + 90) % 180 - 90            # fold to +-90 about 0/180

    # Independent check, and the only view that shows curvature: both signals
    # bandpassed to the modulation and its first few harmonics, then phase
    # binned against envelope. A straight line means the slope is constant
    # over the swing; a bend means it steepens toward the top, which is what
    # a stage entering compression does.
    e_bp = _bandpass(env_ac, fs, 0.5 * f_mod, 4.5 * f_mod)
    p_bp = _bandpass(ph_ac, fs, 0.5 * f_mod, 4.5 * f_mod)
    nb = 25
    lo, hi = np.percentile(e_bp, [1, 99])
    edges = np.linspace(lo, hi, nb + 1)
    which = np.clip(np.digitize(e_bp, edges) - 1, 0, nb - 1)
    bx, by = [], []
    for b in range(nb):
        sel = which == b
        if sel.sum() >= 20:
            bx.append(float(np.mean(e_bp[sel])))
            by.append(float(np.mean(p_bp[sel])))
    bx, by = np.array(bx), np.array(by)
    if len(bx) >= 4:
        fit_lin = float(np.polyfit(bx, by, 1)[0])
        q = np.polyfit(bx, by, 2)
        fit_quad = float(q[0])
    else:
        fit_lin = fit_quad = float("nan")

    return dict(
        slope_deg_per_db=(sign * pm_deg / am_db if am_db > 0 else float("nan")),
        am_db=am_db, pm_deg=pm_deg, f_mod_hz=f_mod, lag_deg=lag_deg,
        lag_raw_deg=lag_raw,
        pm_noise_deg=noise, pm_snr_db=snr_db,
        slope_binned=fit_lin, curvature=fit_quad,
        bin_env_db=bx, bin_ph_deg=by,
        env_db=env_db, ph_ac_deg=ph_ac, t=t, n_used=n)


def am_fundamental_db(depth_db):
    """The AM the DUT is DRIVEN with: the fundamental component, in dB, of the
    envelope-in-dB waveform that duc_am_setup(depth_db) produces.

    The envelope is a sinusoid in linear amplitude, so in dB it is not one --
    20*log10(1 + m*cos t) carries harmonics. Its fundamental follows from
        int_0^2pi ln(1 + m cos t) cos t dt = 2*pi*(1 - sqrt(1-m^2))/m
    giving (40/ln10)*(1 - sqrt(1-m^2))/m. Checked against the synthetic
    dry-run signal, which returns 2.9705 dB for depth_db = 6.

    Comparing this to the AM measured at the output gives the TILT TRANSFER
    RATIO r for free at every single point -- an independent check on the
    value run_doubler.py gets from its drive sweep, by a completely different
    route (one modulated record versus a ladder of static levels).
    """
    ratio = 10 ** (depth_db / 20.0)
    m = (ratio - 1.0) / (ratio + 1.0)
    return (40.0 / np.log(10)) * (1.0 - np.sqrt(1.0 - m * m)) / m


def drive_ripple(freqs, drive):
    """(tilt, scatter_rms) of the measured drive across the band, in dB.

    Both come from the saved PE87FL1012 sweep rather than being retyped, so
    the consequence numbers in the report track the data. Tilt is the smooth
    part -- a slowly varying phase across a chirp. Scatter is what is left
    after a cubic, i.e. the ripple, which is what makes paired echoes.

    The residual is normalised by the remaining degrees of freedom, not by the
    point count. On an 11-point sweep a cubic absorbs 4 of the 11, and a plain
    np.std then reports ~0.11 dB where 0.19 was really there -- a 40% understatement
    of exactly the quantity that sets the spurious floor, in the flattering
    direction. Verified against injected noise in the dry run.
    """
    tilt = float(drive.max() - drive.min())
    deg = 3
    n = len(freqs)
    smooth = np.polyval(np.polyfit(freqs, drive, deg), freqs)
    dof = max(n - (deg + 1), 1)
    return tilt, float(np.sqrt(np.sum((drive - smooth) ** 2) / dof))


# ── the run ───────────────────────────────────────────────────────────────
def run(n_freqs=N_FREQS, drive_offset_db=0.0):
    pad = config.PAD_AT_INSTRUMENT_DB
    st = config.STAGES[STAGE]
    lo, hi = st["band_hz"]

    vpp, dfreqs, drive, dpath = known_drive()
    if vpp is None:
        print("No PE87FL1012 session found -- run run_exa_harmonics.py first "
              "so the drive is a measured quantity, not a guess.")
        return
    freqs = np.linspace(lo, hi, int(n_freqs))
    drive_at = np.interp(freqs, dfreqs, drive)
    tilt_db, scatter_db = drive_ripple(dfreqs, drive)

    if drive_offset_db:
        vpp = float(np.clip(vpp * 10 ** (drive_offset_db / 20), 0.001,
                            config.AWG_MAX_VPP))
        drive_at = drive_at + drive_offset_db

    print("=" * 78)
    print(f"AM-to-PM -- {ELEMENT}")
    print(f"  {PATH} -> {pad:g} dB pad -> EXA")
    print("=" * 78)
    print(f"  measuring at {TONE}x the AWG frequency "
          f"({TONE*lo/1e9:.3f}-{TONE*hi/1e9:.3f} GHz)")
    print(f"  phase here is multiplied by {MULT_TO_OUTPUT:.0f} to the AMC "
          f"output at 310-370 GHz")
    print(f"  drive at the doubler input {drive_at.max():+.2f} to "
          f"{drive_at.min():+.2f} dBm at {vpp:.4f} Vpp (peak of the AM)")
    print(f"    from {dpath}")
    if drive_offset_db:
        print(f"    includes a {drive_offset_db:+.2f} dB offset")
    am_in_db = am_fundamental_db(DEPTH_DB)
    print(f"  AM {MOD_HZ/1e3:.0f} kHz, {DEPTH_DB:.1f} dB peak-to-trough "
          f"({am_in_db:.3f} dB fundamental), peak = the CW operating level")
    print(f"  {len(freqs)} frequencies x {len(BACKOFF_DB)} drive levels "
          f"({', '.join(f'{b:+.0f}' for b in BACKOFF_DB)} dB)")

    # ---- safety: identical to run_doubler, because the setup is identical --
    # The AM's PEAK equals the CW amplitude that was already run, so nothing
    # here is driven harder than it has been. The checks are repeated anyway;
    # a script that assumes a previous script's checks still hold is one
    # cable-swap away from being wrong.
    if MODE in ("chain", "predoubler"):
        amp_out = drive_at.max() + 0.64          # + the filter's measured IL
        back = mc.ZX602522MA_P1DB_DBM - amp_out
        print(f"\n  stage-A amp at {amp_out:+.2f} dBm, {back:.2f} dB below its "
              f"P1dB {mc.ZX602522MA_P1DB_DBM:+.1f}")
        if back < 1.0:
            print("  ** under 1 dB of backoff -- ABORT **")
            return
    if MODE == "chain":
        if drive_at.max() > mc.HMC188_PIN_MAX_DBM:
            print(f"  ** {drive_at.max():+.2f} dBm exceeds the HMC188's "
                  f"{mc.HMC188_PIN_MAX_DBM:+.0f} dBm maximum. ABORT **")
            return

    # Worst case at the analyzer. With the amp in the path that is the full
    # drive feeding through; with the AWG alone it is just the AWG.
    at_exa = (config.dbm_from_vpp(vpp) if MODE == "awg"
              else drive_at.max()) - pad
    need = config.required_exa_atten_db(at_exa)
    print(f"  worst case at the EXA {at_exa:+.1f} dBm -> needs >= {need:.0f} dB "
          f"internal attenuation; using {EXA_ATTEN_DB:.0f} dB")
    if EXA_ATTEN_DB < need:
        print("  ** not enough -- ABORT **")
        return
    print(f"  EXA damage {config.EXA_DAMAGE_DBM:+.0f} dBm -> "
          f"{config.EXA_DAMAGE_DBM-at_exa:.0f} dB of margin\n")

    awg = AWG()
    sa = SpecAn()
    res = {}          # (freq, backoff) -> analysis dict
    tone_dbm = {}     # freq -> swept-mode level at TONE*f
    iq_info = {}
    am_info = {}
    try:
        # ---- 1. swept-mode sanity check --------------------------------
        # Before capturing anything, confirm in the mode that is known to
        # work that there is a real tone where the IQ capture is about to be
        # pointed, and that it stands clear of the floor. An I/Q record of
        # noise still yields a phase and still yields a slope.
        sa.set_trace_mode("NORM")
        sa.set_attenuation(EXA_ATTEN_DB)
        sa.set_ref_level(REF_LEVEL_DBM)
        sa.set_rbw(RBW_HZ)
        sa.set_vbw(None)
        sa.set_points(N_POINTS)
        print("  swept-mode check at the CW operating level:")
        for f in freqs:
            awg.duc_cw_setup(float(f), amplitude_vpp=float(vpp), channel=1)
            time.sleep(SETTLE_S)
            pk, floor, _tf, _ta = measure_tone(sa, float(f) * TONE)
            tone_dbm[float(f)] = pk + pad
            ok = (pk - floor) >= MIN_TONE_SNR_DB
            print(f"    {TONE*f/1e9:.3f} GHz  {pk+pad:+7.2f} dBm  "
                  f"{pk-floor:5.1f} dB above the floor  {'' if ok else '** LOW **'}")
            if not ok:
                print(f"  ** only {pk-floor:.1f} dB of signal-to-noise at "
                      f"{TONE*f/1e9:.3f} GHz. A phase measurement on that is "
                      f"noise with a number attached. ABORT **")
                return
        print()

        # ---- 2. IQ captures --------------------------------------------
        iq_info = sa.iq_setup(float(freqs[0]) * TONE, if_bw_hz=IF_BW_HZ,
                              duration_s=CAPTURE_S, atten_db=EXA_ATTEN_DB)
        print(f"  EXA IQ: DIF BW {iq_info['dif_bw_hz']/1e6:.3f} MHz "
              f"({iq_info['if_filter']}), {iq_info['duration_s']*1e3:.3f} ms, "
              f"{iq_info['n_samples']} samples -> "
              f"{iq_info['sample_rate_hz']/1e6:.4f} MS/s DERIVED "
              f"(the instrument claims "
              f"{iq_info['srat_claimed_hz']/1e6:.4f}; it is echoing the IF BW, "
              f"not the rate)")
        print(f"    atten {iq_info['atten_db']:.0f} dB, "
              f"{'big' if iq_info['big_endian'] else 'little'}-endian")
        if iq_info["soft_errors"]:
            print(f"    (ignored: {iq_info['soft_errors']})")
        cycles = iq_info["duration_s"] * MOD_HZ
        print(f"    {cycles:.0f} modulation cycles per record, "
              f"{iq_info['sample_rate_hz']/MOD_HZ:.1f} samples per cycle\n")

        lp = LivePlot(f"AM-to-PM: {ELEMENT}", "drive backoff from the "
                      "operating point (dB)", "AM-to-PM (deg/dB)")

        for f in freqs:
            print(f"  {TONE*f/1e9:.3f} GHz")
            for b in BACKOFF_DB:
                v = float(np.clip(vpp * 10 ** (b / 20), 0.001,
                                  config.AWG_MAX_VPP))
                am_info = awg.duc_am_setup(float(f), mod_hz=MOD_HZ,
                                           depth_db=DEPTH_DB,
                                           amplitude_vpp=v, channel=1,
                                           n_total=N_TOTAL)
                time.sleep(SETTLE_S)
                z, fs = sa.iq_capture(float(f) * TONE)
                try:
                    r = am_pm_from_iq(z, fs, am_info["mod_hz"])
                except ValueError as e:
                    print(f"    {b:+5.1f} dB   ** {e} -- skipped **")
                    continue
                res[(float(f), float(b))] = r
                # Independent check on the DERIVED sample rate: the AWG's
                # modulation frequency is known to ppm, so if the rate is
                # wrong the recovered frequency is wrong by the same factor.
                # The slope itself is a ratio taken within one record and does
                # not depend on the rate at all -- but a wrong frequency axis
                # would quietly mislabel everything else.
                dev = r["f_mod_hz"] / am_info["mod_hz"] - 1.0
                if abs(dev) > 0.02:
                    print(f"      ** recovered {r['f_mod_hz']/1e3:.2f} kHz vs "
                          f"{am_info['mod_hz']/1e3:.2f} commanded "
                          f"({dev*100:+.1f}%): the derived sample rate is off "
                          f"by that factor. The slope is unaffected; the "
                          f"frequency axis is not. **")
                weak = r["pm_snr_db"] < MIN_PM_SNR_DB
                r["am_gain"] = r["am_db"] / am_in_db
                print(f"    {b:+5.1f} dB   AM {r['am_db']:5.2f} dB "
                      f"(r={r['am_gain']:4.2f})   "
                      f"PM {r['pm_deg']:6.3f} deg   -> "
                      f"{r['slope_deg_per_db']:+6.3f} deg/dB   "
                      f"lag {r['lag_deg']:+6.1f} deg   "
                      f"SNR {r['pm_snr_db']:4.1f} dB"
                      + ("  ** noise-limited **" if weak else ""))
                if not weak:
                    lp.add(f"{TONE*f/1e9:.3f} GHz", b, r["slope_deg_per_db"])
                lp.note(f"AM-to-PM  {TONE*f/1e9:.3f} GHz  {b:+.1f} dB")
        lp.done()

        settings = {
            "mode": MODE, "element": ELEMENT, "tone": TONE,
            "mult_to_amc_output": MULT_TO_OUTPUT,
            "mod_hz": float(am_info.get("mod_hz", MOD_HZ)),
            "depth_db": DEPTH_DB, "n_total": N_TOTAL,
            "operating_vpp": float(vpp),
            "drive_offset_db": float(drive_offset_db),
            "drive_reference": dpath,
            "drive_tilt_db": tilt_db, "drive_scatter_db": scatter_db,
            "backoff_db": list(BACKOFF_DB),
            "atten_db": EXA_ATTEN_DB, "settle_s": SETTLE_S,
            "iq": {k: v for k, v in iq_info.items() if k != "soft_errors"},
            "tone_dbm": {f"{k/1e9:.4f}ghz": v for k, v in tone_dbm.items()},
        }
    finally:
        for name, fn in (("specan.iq_teardown", sa.iq_teardown),
                         ("awg.stop", awg.stop),
                         ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    if not res:
        print("\n  nothing measured.")
        return
    report(freqs, res, settings)


# ── reporting ─────────────────────────────────────────────────────────────
def thru_reference():
    """A previously saved MODE=awg run, if there is one. (slope, path).

    Files saved before 2026-08-03 stored the slope as a magnitude with the
    sign discarded, so an old reference subtracts as if it were in phase with
    the chain. At the measured 0.037 deg/dB (2% of the chain's slope) the
    worst case is 4% -- retake MODE=awg if the correction ever matters.
    """
    for p in chainlog.find(STAGE, label_contains="am_pm", thru=False):
        d = chainlog.load(p)
        s = d.get("instrument_settings", {})
        if s.get("mode") != "awg":
            continue
        key = chainlog.series_key("slope_mean")
        if key not in d or not len(d[key]):
            continue
        y = d[key]
        # The rung nearest 0 dB backoff, not blindly column 0 -- the saved
        # x axis holds only the rungs that actually produced data.
        x = d.get(d.get("x_name", ""), None)
        col = int(np.argmin(np.abs(x))) if x is not None and len(x) == len(y) \
            else 0
        return float(y[col]), p
    return None, None


def report(freqs, res, settings):
    backoffs = [b for b in BACKOFF_DB
                if any((f, b) in res for f in map(float, freqs))]
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)

    print(f"  AM-to-PM (deg/dB), by frequency and drive backoff:")
    print("    " + "freq".ljust(12)
          + "".join(f"{b:+.0f} dB".rjust(11) for b in backoffs))
    slope_tab = {}
    for f in map(float, freqs):
        row = []
        for b in backoffs:
            r = res.get((f, b))
            row.append(r["slope_deg_per_db"] if r else float("nan"))
        slope_tab[f] = np.array(row)
        print(f"    {settings['tone']*f/1e9:.3f} GHz".ljust(16)
              + "".join(f"{v:11.3f}" for v in row))

    mean_by_b = np.array([np.nanmean([slope_tab[f][i] for f in slope_tab])
                          for i in range(len(backoffs))])
    print("    " + "mean".ljust(12) + "".join(f"{v:11.3f}" for v in mean_by_b))

    # Free by-product: the AM measured at the output divided by the AM the AWG
    # drove with IS the tilt transfer ratio. run_doubler.py gets the same
    # quantity from a ladder of static levels; this gets it from a single
    # modulated record. Two independent routes to r, and they should agree.
    gain_tab = {}
    print(f"\n  tilt transfer ratio r (measured AM / driven AM), same axes:")
    print("    " + "freq".ljust(12)
          + "".join(f"{b:+.0f} dB".rjust(11) for b in backoffs))
    for f in map(float, freqs):
        row = np.array([res[(f, b)].get("am_gain", np.nan) if (f, b) in res
                        else np.nan for b in backoffs])
        gain_tab[f] = row
        print(f"    {settings['tone']*f/1e9:.3f} GHz".ljust(16)
              + "".join(f"{v:11.2f}" for v in row))
    if 0.0 in backoffs:
        r_op = np.array([gain_tab[f][backoffs.index(0.0)] for f in gain_tab])
        if np.isfinite(r_op).any():
            print(f"    at the operating point r = {np.nanmin(r_op):.2f} to "
                  f"{np.nanmax(r_op):.2f}, mean {np.nanmean(r_op):.2f} "
                  f"(run_doubler's drive sweep gave 1.62 mean, 1.10 at the "
                  f"strong edge and 1.92 at the weak)")

    b_op = 0.0 if 0.0 in backoffs else max(backoffs)
    i_op = backoffs.index(b_op)
    s_op = float(mean_by_b[i_op])
    # Signed slopes at different frequencies could cancel in the mean, but a
    # chirp accrues each frequency's phase error at that frequency's own
    # |slope| -- so the consequence numbers below use the mean magnitude.
    s_mag = float(np.nanmean([abs(slope_tab[f][i_op]) for f in slope_tab]))
    if b_op == 0.0:
        print(f"\n  at the operating point: {s_op:+.3f} deg/dB")
    else:
        print(f"\n  ** every 0 dB point was skipped: {s_op:+.3f} deg/dB is "
              f"the {b_op:+.0f} dB rung, the closest measured to the "
              f"operating point, NOT the operating point itself **")
    if s_mag > 0 and abs(s_mag - abs(s_op)) > 0.05 * s_mag:
        print(f"    frequencies disagree in sign, so the signed mean partly "
              f"cancels; the consequence below uses the mean magnitude "
              f"{s_mag:.3f} deg/dB")

    # Does it behave like compression? A compression-driven AM-to-PM falls as
    # the drive backs off. One that does not is either instrumental or comes
    # from something other than the stage's own nonlinearity.
    #
    # Fitted PER FREQUENCY and only reported as a verdict if the frequencies
    # agree. A single fit to the mean will always return some sign and reads
    # as a physical finding, even when the underlying curves point in opposite
    # directions -- which is exactly what happened on the first run of this
    # script, where a mean dominated by one noisy point announced
    # "it IS compression" over data that showed the reverse at 3.228 GHz.
    trends = {}
    sign_cross = []
    for f in slope_tab:
        y = slope_tab[f]
        fin = np.isfinite(y)
        if fin.sum() < 3:
            continue
        yy = y[fin]
        if (yy > 0).any() and (yy < 0).any():
            # The slope crosses zero inside the ladder: |slope| is V-shaped
            # and no single line describes the penalty. Say so instead of
            # fitting one anyway.
            sign_cross.append(f)
            continue
        # The verdict below is about the PENALTY |slope|. With every point on
        # one side of zero, d|slope|/d(backoff) is the signed fit times the
        # common sign -- and fitting the signed values stays linear.
        s = 1.0 if yy.mean() >= 0 else -1.0
        trends[f] = float(s * np.polyfit(np.array(backoffs)[fin], yy, 1)[0])
    if sign_cross:
        print("\n  slope crosses zero over the drive ladder at "
              + ", ".join(f"{settings['tone']*f/1e9:.3f} GHz"
                          for f in sign_cross)
              + " -- no single penalty trend exists there")
    if trends:
        print("\n  trend with drive (|slope| in deg/dB per dB of backoff; "
              "positive = falls as the drive backs off = compression):")
        for f, tr in trends.items():
            print(f"    {settings['tone']*f/1e9:.3f} GHz   {tr:+.4f}")
        v = np.array(list(trends.values()))
        if sign_cross:
            print("    -> verdict limited to the frequencies above; the "
                  "zero-crossing ones do not fit either story.")
        if np.all(v > 0.01):
            print("    -> all positive: amplifier-like. Backing the drive off "
                  "reduces the phase penalty, which is what the 2 dB backoff "
                  "rule assumes.")
        elif np.all(v < -0.01):
            print("    -> all NEGATIVE: the phase penalty per dB gets WORSE "
                  "as the drive backs off.")
            print("       For a DIODE MULTIPLIER that is the expected sign -- "
                  "it is least phase-sensitive when hardest driven, so here "
                  "more drive is better, not less. That inverts the backoff "
                  "rule for this stage.")
            print("       It can equally be an instrument contribution, since "
                  "backing off the drive also moves the AWG's own operating "
                  "point. MODE=awg settles which.")
        else:
            print("    -> THE FREQUENCIES DISAGREE IN SIGN. No compression "
                  "verdict is supportable from this run.")

    lags = [res[k]["lag_deg"] for k in res if np.isfinite(res[k]["lag_deg"])]
    if lags:
        print(f"  phase of the PM relative to the AM: "
              f"{np.mean(lags):+.1f} deg (0 = memoryless)")
        if abs(np.mean(lags)) > 20:
            print(f"    ** that is far from 0: there is a time constant in "
                  f"play, so this slope is specific to {MOD_HZ/1e3:.0f} kHz "
                  f"and does not extrapolate to a chirp without checking "
                  f"another modulation rate **")

    curv = [res[k]["curvature"] for k in res
            if np.isfinite(res[k].get("curvature", np.nan))]
    if curv:
        print(f"  curvature over the {settings['depth_db']:.0f} dB swing: "
              f"{np.mean(curv):+.4f} deg/dB^2 "
              f"(0 = the slope is constant across the swing)")

    snrs = [res[k]["pm_snr_db"] for k in res if np.isfinite(res[k]["pm_snr_db"])]
    if snrs:
        print(f"  phase SNR {min(snrs):.1f} to {max(snrs):.1f} dB; noise floor "
              f"{np.mean([res[k]['pm_noise_deg'] for k in res]):.4f} deg")

    # ---- what it costs at 310-370 GHz --------------------------------
    mult = settings["mult_to_amc_output"]
    tilt = settings["drive_tilt_db"]
    scat = settings["drive_scatter_db"]
    print(f"\n  CONSEQUENCE at the AMC output (x{mult:.0f} from here)")
    print(f"    measured drive across the band: {tilt:.2f} dB of tilt, "
          f"{scat:.3f} dB rms of ripple on top")
    print(f"    tilt   -> {tilt*s_mag:.2f} deg here -> "
          f"{tilt*s_mag*mult:.0f} deg at 310-370 GHz")
    print(f"      slowly varying across the sweep, so for a chirp this is a "
          f"phase distortion the matched filter partly absorbs; "
          f"pre-emphasis removes the cause")
    beta = np.radians(scat * s_mag * mult * np.sqrt(2))    # rms -> peak
    print(f"    ripple -> {scat*s_mag:.3f} deg here -> "
          f"{scat*s_mag*mult:.1f} deg rms at 310-370 GHz")
    if beta > 0:
        print(f"      as paired echoes: {20*np.log10(min(beta,2.0)/2):.1f} dBc")
        print(f"      (the filter choice was made at -36 dBc; anything worse "
              f"than that here means AM-to-PM, not the filter, sets the "
              f"spurious floor)")

    thru_s, thru_p = thru_reference()
    if thru_s is not None and settings["mode"] != "awg":
        # The doubler doubles carrier phase, so anything ahead of it arrives
        # at the measurement plane doubled.
        corrected = s_op - 2.0 * thru_s
        print(f"\n  thru reference (MODE=awg): {thru_s:+.3f} deg/dB")
        print(f"    x2 through the doubler = {2*thru_s:+.3f} deg/dB of the "
              f"{s_op:+.3f} measured -> {corrected:+.3f} deg/dB is the "
              f"amp+filter+doubler's own")
        print(f"    {thru_p}")
    elif settings["mode"] != "awg":
        print(f"\n  NO THRU REFERENCE: this number includes the AWG's own "
              f"AM-to-PM and cannot be attributed to the chain alone.")
        print(f"    MODE=awg python x8_chain/run_am_pm.py   "
              f"(AWG -> 10 dB pad -> EXA; needs the cable moved)")

    # ---- save ---------------------------------------------------------
    sess = chainlog.new_session(STAGE, "am_pm")
    x = np.array(backoffs, dtype=float)
    series = {"slope_mean": mean_by_b}
    for f in map(float, freqs):
        tag = f"{settings['tone']*f/1e9:.3f}ghz"
        series[f"slope_{tag}"] = slope_tab[f]
        series[f"pm_{tag}"] = np.array(
            [res[(f, b)]["pm_deg"] if (f, b) in res else np.nan
             for b in backoffs])
        series[f"am_{tag}"] = np.array(
            [res[(f, b)]["am_db"] if (f, b) in res else np.nan
             for b in backoffs])
        series[f"snr_{tag}"] = np.array(
            [res[(f, b)]["pm_snr_db"] if (f, b) in res else np.nan
             for b in backoffs])
        series[f"r_{tag}"] = gain_tab[f]
    p = chainlog.save(sess, "am_pm", STAGE, x, series, settings,
                      x_name="drive_backoff_db", thru=False,
                      notes=f"AM-to-PM, {ELEMENT}. slope_* are SIGNED deg/dB "
                            f"(positive = phase rises with the envelope; "
                            f"files before 2026-08-03 stored the magnitude) "
                            f"at {settings['tone']}x the AWG frequency; pm_* "
                            f"are degrees peak, am_* dB peak, snr_* dB. "
                            f"Multiply slope by {mult:.0f} for the AMC output. "
                            f"Drive reference: {settings['drive_reference']}")
    print(f"\n  saved {p}")

    # ---- figures ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for f in map(float, freqs):
        axes[0].plot(backoffs, slope_tab[f], "o-", ms=5,
                     label=f"{settings['tone']*f/1e9:.3f} GHz")
    axes[0].plot(backoffs, mean_by_b, "k--", lw=2, label="mean")
    axes[0].set_xlabel("drive backoff from the operating point (dB)")
    axes[0].set_ylabel("AM-to-PM (deg/dB)")
    axes[0].set_title("the decision plot")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    ax2 = axes[0].twinx()
    ax2.set_ylim(np.array(axes[0].get_ylim()) * tilt * mult)
    ax2.set_ylabel(f"deg at 310-370 GHz from {tilt:.2f} dB of tilt")

    k_op = None
    for f in map(float, freqs):
        if (f, 0.0) in res:
            k_op = (f, 0.0)
            break
    if k_op is None:
        k_op = list(res)[0]
    r = res[k_op]
    axes[1].plot(r["bin_env_db"], r["bin_ph_deg"], "o", ms=5)
    if len(r["bin_env_db"]) >= 2:
        xx = np.linspace(r["bin_env_db"].min(), r["bin_env_db"].max(), 50)
        axes[1].plot(xx, np.polyval(np.polyfit(r["bin_env_db"],
                                               r["bin_ph_deg"], 2), xx),
                     "-", lw=1.5,
                     label=f"{r['slope_binned']:.3f} deg/dB")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("envelope, relative (dB)")
    axes[1].set_ylabel("phase, relative (deg)")
    axes[1].set_title(f"{settings['tone']*k_op[0]/1e9:.3f} GHz at "
                      f"{k_op[1]:+.0f} dB: phase vs envelope")
    axes[1].grid(alpha=0.3)

    n_show = min(len(r["t"]), int(r["t"].size * 0.05))
    t_us = r["t"][:n_show] * 1e6
    axes[2].plot(t_us, r["env_db"][:n_show] - r["env_db"].mean(),
                 lw=0.8, label="envelope (dB)")
    axes[2].plot(t_us, r["ph_ac_deg"][:n_show], lw=0.8, label="phase (deg)")
    axes[2].set_xlabel("time (us)")
    axes[2].set_ylabel("dB  /  deg")
    axes[2].set_title(f"raw modulation at {r['f_mod_hz']/1e3:.1f} kHz")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "am_pm.png")
    fig.savefig(png, dpi=130)
    print(f"        {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_FREQS,
        float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
