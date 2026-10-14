"""Measure the AWG's ABSOLUTE output level across the x8 chain's input band,
on the SCOPE rather than the spectrum analyzer.

Why the scope: the three cw_level_check_* runs from 2026-07-27 disagree by
19-27 dB because an external pad was changed mid-session and never recorded
(see multiplier_chain.AWG_SCAN_RUN). The scope removes that entire class of
error -- it is a direct voltage measurement into its own 50 ohm input, with no
pad in the path at all, so there is nothing unrecorded to get wrong. At the
~0.55 Vpp the AWG produces, no attenuation is needed or wanted.

What this establishes:
  1. absolute output level vs frequency over 1.55-2.00 GHz (the x8 chain's
     AWG band is 1.614-1.927 GHz), at fine steps rather than the 100 MHz of
     the full-band scan;
  2. the band SPREAD, which is what has to fit the AMC's 7 dB input window;
  3. that commanding 0.55 Vpp really does give +0.83 dB over 0.50 Vpp --
     this exercises the silent-:VOLT-rejection path (awg._cmd prints ERROR but
     does not raise, and :VOLT is never read back), which has caused a wrong
     amplitude before.

Wiring:
    AWG CH1 -> short cable -> scope CH1 (50 ohm).  NO pad.

Run:  python run_awg_level_scope.py
"""
import os
import sys
import time
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from awg import AWG
from scope import Scope

# ── configuration ─────────────────────────────────────────────────────────
SIG_CH = 1                     # scope channel carrying the AWG output
FREQ_START_HZ = 1.55e9         # x8 chain needs 1.614-1.927 GHz; scan wider
FREQ_STOP_HZ = 2.00e9
FREQ_STEP_HZ = 10e6
AMPS_VPP = (0.50, 0.55)        # 0.50 = what the EXA scans used (comparable),
                               # 0.55 = the AC-coupled module ceiling
CAPTURE_S = 2e-6               # ~3000 cycles at 1.6 GHz: far more than RMS needs
SETTLE_S = 0.20

# Vertical scale. measure_vpp_auto() RESCALES whenever the level drifts outside
# 0.3-0.95 of full screen, and a rescale can shift the reading slightly -- which
# would appear as a step in the curve that has nothing to do with the AWG. The
# level only moves ~1.8 dB across this whole scan, so ONE fixed scale covers
# every point at both amplitudes and removes autoscale as a variable entirely.
# 80 mV/div -> 800 mV full screen; the 390-525 mVpp seen here fills 49-66%.
# Set to None to fall back to autoscaling.
VSCALE_V_PER_DIV = 0.08

RUN_LABEL = os.environ.get("RUN_LABEL", "awg_level_scope")

# x8 chain band, for reporting the number that actually matters
X8_LO_HZ, X8_HI_HZ = 12.91e9 / 8, 15.42e9 / 8

OUT_DIR = "awg_level_scope"


def theory_dbm(vpp):
    """A vpp sine into 50 ohm, in dBm."""
    return 10 * np.log10(((vpp / 2) ** 2 / (2 * 50)) / 1e-3)


def dbm_from_vpp(vpp):
    return 10 * np.log10(((vpp / 2) ** 2 / (2 * 50)) / 1e-3)


def phase_coverage_warning(freq_hz, fs_hz, min_phases=64):
    """Vpp here is derived from RMS, which is only unbiased if the samples
    cover the sine's phase uniformly.

    If freq/fs is a ratio with a small denominator q, the record only ever
    visits q distinct phases no matter how long it is -- e.g. 1.5625 GHz at
    12.5 GS/s is exactly fs/8, giving 8 phases and a biased RMS. Flag it
    rather than silently reporting a wrong level.
    """
    if not fs_hz:
        return None
    q = Fraction(freq_hz / fs_hz).limit_denominator(100000).denominator
    if q < min_phases:
        return (f"only ~{q} distinct sample phases (f/fs = 1/{q:g}) -- "
                f"RMS may be biased")
    return None


def crest_warning(v, lo=1.35, hi=1.55):
    """Crest factor sanity check on the captured record.

    A pure sine has peak/rms = sqrt(2) = 1.414 exactly. Clipping flattens the
    peaks and pushes it BELOW that; added noise, spurs or a non-sinusoidal
    output push it above. Either way the RMS-derived Vpp stops meaning what it
    is supposed to mean, so it is worth catching per-point rather than
    discovering a bent curve at the end. Needs no knowledge of the vertical
    scale, so it works regardless of what autoscale settled on.
    """
    if v is None or len(v) == 0:
        return None
    ac = v - np.mean(v)
    rms = np.sqrt(np.mean(ac ** 2))
    if rms <= 0:
        return "no signal"
    cf = np.max(np.abs(ac)) / rms
    if cf < lo:
        return f"crest {cf:.2f} < {lo} -- CLIPPED?"
    if cf > hi:
        return f"crest {cf:.2f} > {hi} -- noisy/not a clean sine?"
    return None


def run():
    awg = AWG()
    scope = Scope()

    freqs = np.arange(FREQ_START_HZ, FREQ_STOP_HZ + 1, FREQ_STEP_HZ)
    results = {}

    try:
        # Trigger on the signal itself: a CW sine crosses 0 V reliably, and
        # because the level comes from RMS over thousands of cycles the trigger
        # position is irrelevant -- it only has to yield a complete record.
        scope.setup(channel=SIG_CH, n_averages=1,
                    trigger_channel=SIG_CH, trigger_level=0.0)
        scope.set_active_channels([SIG_CH])
        if VSCALE_V_PER_DIV is not None:
            applied = scope.set_vertical(SIG_CH, VSCALE_V_PER_DIV)
            print(f"scope: vertical FIXED at {applied*1e3:.4g} mV/div "
                  f"({applied*10*1e3:.4g} mV full screen) -- no autoscale, so a "
                  f"rescale cannot masquerade as a step in the curve")
        fs = scope.set_max_sample_rate(CAPTURE_S)
        print(f"scope: {fs/1e9:.2f} GS/s, {CAPTURE_S*1e6:.1f} us record "
              f"({fs*CAPTURE_S:.0f} points)")
        print(f"       {fs/FREQ_STOP_HZ:.2f} samples/cycle at the top frequency "
              f"-- fine for an RMS-based level, but far too few for peak "
              f"detection (which is why measure_vpp uses RMS)\n")

        for amp in AMPS_VPP:
            print(f"=== {amp} Vpp  (theory {theory_dbm(amp):+.2f} dBm) ===")
            levels = []
            for i, f in enumerate(freqs):
                if i == 0:
                    awg.duc_cw_setup(float(f), amplitude_vpp=amp, channel=1)
                else:
                    awg.duc_cw_step(float(f), channel=1)
                time.sleep(SETTLE_S)

                if VSCALE_V_PER_DIV is None:
                    vpp, t, v = scope.measure_vpp_auto(channel=SIG_CH, settle=0.05)
                else:
                    vpp, t, v = scope.measure_vpp(channel=SIG_CH, settle=0.05)
                dbm = dbm_from_vpp(vpp)
                levels.append(dbm)

                notes = [n for n in (phase_coverage_warning(float(f), fs),
                                     crest_warning(v)) if n]
                print(f"  {f/1e9:.3f} GHz  {vpp*1000:7.1f} mVpp  {dbm:+7.2f} dBm"
                      + ("   " + "; ".join(notes) if notes else ""))
            results[amp] = np.array(levels)

    finally:
        for name, fn in (("awg.stop", awg.stop), ("scope.restore",
                         lambda: scope.restore(channel=SIG_CH))):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(freqs, results)
    return results


def report(freqs, results):
    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    band = (freqs >= X8_LO_HZ) & (freqs <= X8_HI_HZ)
    for amp, lv in results.items():
        th = theory_dbm(amp)
        print(f"\n  {amp} Vpp  (theory {th:+.2f} dBm)")
        print(f"    full scan {freqs[0]/1e9:.2f}-{freqs[-1]/1e9:.2f} GHz : "
              f"{lv.max():+.2f} to {lv.min():+.2f} dBm, spread {lv.max()-lv.min():.2f} dB")
        print(f"    peak vs theory                : {lv.max()-th:+.2f} dB")
        if band.any():
            b = lv[band]
            print(f"    x8 band {X8_LO_HZ/1e9:.3f}-{X8_HI_HZ/1e9:.3f} GHz    : "
                  f"{b.max():+.2f} to {b.min():+.2f} dBm, "
                  f"SPREAD {b.max()-b.min():.2f} dB")

    if len(results) == 2:
        a1, a2 = sorted(results)
        step = np.mean(results[a2] - results[a1])
        expect = theory_dbm(a2) - theory_dbm(a1)
        print(f"\n  AMPLITUDE STEP CHECK  {a1} -> {a2} Vpp")
        print(f"    measured {step:+.2f} dB (sd {np.std(results[a2]-results[a1]):.2f}), "
              f"expected {expect:+.2f} dB")
        if abs(step - expect) > 0.3:
            print(f"    ** MISMATCH {step-expect:+.2f} dB -- the commanded amplitude is")
            print(f"       NOT being applied as expected. awg._cmd() prints ERROR but does")
            print(f"       not raise, and :VOLT is never read back, so a rejected :VOLT")
            print(f"       leaves the previous amplitude silently in place. Check with")
            print(f"       :VOLT? before trusting either curve. **")
        else:
            print(f"    OK -- the commanded amplitude is being applied.")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for amp, lv in results.items():
        ax.plot(freqs / 1e9, lv, "o-", ms=4, label=f"{amp} Vpp measured")
        ax.axhline(theory_dbm(amp), ls="--", lw=1, alpha=0.6,
                   label=f"{amp} Vpp theory ({theory_dbm(amp):+.2f} dBm)")
    ax.axvspan(X8_LO_HZ / 1e9, X8_HI_HZ / 1e9, alpha=0.12, color="tab:green",
               label="x8 chain band")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("AWG output (dBm into 50 $\\Omega$)")
    ax.set_title("AWG absolute output, measured on the scope (no pad in path)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{RUN_LABEL}.png")
    fig.savefig(png, dpi=130)
    print(f"\n  saved {png}")

    npz = os.path.join(OUT_DIR, f"{RUN_LABEL}.npz")
    np.savez(npz, freqs_hz=freqs,
             **{f"dbm_{str(a).replace('.','p')}vpp": lv for a, lv in results.items()})
    print(f"  saved {npz}")
    # NO_SHOW=1 keeps the run non-blocking (the PNG is already written above),
    # which is what an automated/unattended invocation wants.
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run()
