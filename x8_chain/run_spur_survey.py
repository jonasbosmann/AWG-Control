"""Broad-band spur survey: sweep the WHOLE analyzer range and find everything.

WHY THIS EXISTS, SEPARATELY FROM run_doubler.py
-----------------------------------------------
run_doubler.py measures narrow spans centred on tones we PREDICTED -- integer
multiples of the AWG frequency. That makes it precise and fast, and completely
blind to anything we did not think of. Every number it has produced is
conditional on the assumption that all energy in this chain lands on k x f_AWG.

That assumption is reasonable (multipliers multiply) but it is an assumption,
and the failure modes it hides are exactly the ones that would matter most:

  * an amplifier oscillating -- parametric or out-of-band, at no harmonic
    relationship to anything, and invisible to a harmonic-indexed measurement;
  * subharmonics or fractional products (k/2) from a doubler driven hard;
  * external pickup -- the 1.500 GHz line already noted in the README, or
    anything else the lab radiates into an imperfectly screened path;
  * mixing with a spur ON the AWG's own output rather than with its carrier.

So this sweeps continuously, finds every peak by prominence, and then asks of
each one "is this k x f_AWG for an integer k?". Anything that is not gets
flagged. The point is not to measure known spurs more accurately -- it is to
find out whether the harmonic model is complete.

READ THE RESULT CAREFULLY
-------------------------
A survey trades resolution for coverage. With a wide span per segment the bin
width is MHz-scale, so reported peak FREQUENCIES are coarse and reported peak
LEVELS are lower bounds -- a spur narrower than a bin is under-measured. Use
this to decide WHERE to look; use run_doubler.py's narrow spans to measure
what is found. A peak here is a lead, not a number.

Run:  python x8_chain/run_spur_survey.py [n_awg_freqs]
      START_GHZ / STOP_GHZ / SEG_GHZ / RBW_KHZ  override the sweep plan
      EXPECT_MAX_DBM   strongest level expected AT THE DUT OUTPUT (safety)
      CHAIN="..."      the physical path, recorded with the data
      NO_SHOW=1        skip the plot window
"""
import glob
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
from awg import AWG
from specan import SpecAn, find_peaks

STAGE = "A"
N_AWG_FREQS = int(os.environ.get("N_AWG_FREQS", "3"))

START_HZ = float(os.environ.get("START_GHZ", "0.3")) * 1e9
STOP_HZ = float(os.environ.get("STOP_GHZ", "20.0")) * 1e9
SEG_HZ = float(os.environ.get("SEG_GHZ", "2.0")) * 1e9
RBW_HZ = float(os.environ.get("RBW_KHZ", "300")) * 1e3
N_POINTS = 1001
SETTLE_S = 0.15

# A peak must stand this far above the local noise floor to be reported. Set
# from experience with these traces: below ~8 dB the finder starts returning
# noise ripple, above ~15 dB it misses real low-level spurs.
MIN_PROMINENCE_DB = 10.0

# How close to k x f_AWG a peak must fall to be called harmonic. Generous on
# purpose: the bin width at these spans is MHz-scale, so a genuine harmonic can
# land a bin or two off. Too tight a tolerance would report real harmonics as
# "unexpected", which is the failure mode that would waste the most bench time.
HARMONIC_TOL_HZ = float(os.environ.get("HARMONIC_TOL_MHZ", "12")) * 1e6
MAX_K = 16

PREEMPH = os.environ.get("PREEMPH", "")
CHAIN = os.environ.get("CHAIN", "")
EXPECT_MAX_DBM = float(os.environ.get("EXPECT_MAX_DBM", "5.0"))


def classify(f_peak, f_awg):
    """(k, err_hz) if the peak sits on an integer multiple, else (None, None)."""
    k = f_peak / f_awg
    kr = int(round(k))
    if kr < 1 or kr > MAX_K:
        return None, None
    err = abs(f_peak - kr * f_awg)
    return (kr, err) if err <= HARMONIC_TOL_HZ else (None, None)


def survey_once(sa, f_awg, segments):
    """Full sweep at one AWG frequency. Returns (freqs, amps) concatenated."""
    fs, amps = [], []
    for i, (c, span) in enumerate(segments):
        sa.set_freq(c, span)
        time.sleep(0.05)
        f, a = sa.sweep_once()
        fs.append(f)
        amps.append(a)
        print(f"      segment {i+1}/{len(segments)}  "
              f"{(c-span/2)/1e9:5.2f}-{(c+span/2)/1e9:5.2f} GHz  "
              f"peak {np.max(a):+7.2f} dBm", flush=True)
    return np.concatenate(fs), np.concatenate(amps)


def run(n_awg=N_AWG_FREQS):
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = config.STAGES[STAGE]["band_hz"]
    awg_freqs = np.linspace(lo, hi, int(n_awg))

    at_exa = EXPECT_MAX_DBM - pad
    atten = config.required_exa_atten_db(at_exa)
    n_seg = int(np.ceil((STOP_HZ - START_HZ) / SEG_HZ))
    segments = []
    for i in range(n_seg):
        a = START_HZ + i * SEG_HZ
        b = min(a + SEG_HZ, STOP_HZ)
        segments.append(((a + b) / 2, b - a))

    print("=" * 78)
    print("BROAD SPUR SURVEY")
    print("=" * 78)
    print("  CHAIN AS CABLED:")
    for i, part in enumerate((CHAIN or "(not stated -- pass CHAIN=...)").split("->")):
        print(f"    {i+1:2d}. {part.strip()}")
    print(f"\n  sweep {START_HZ/1e9:.2f}-{STOP_HZ/1e9:.2f} GHz in {n_seg} segments "
          f"of {SEG_HZ/1e9:g} GHz, RBW {RBW_HZ/1e3:.0f} kHz")
    print(f"  bin width {SEG_HZ/N_POINTS/1e6:.2f} MHz -- peak frequencies are coarse "
          f"and levels are LOWER BOUNDS")
    print(f"  at {n_awg} AWG frequencies: "
          + ", ".join(f"{f/1e9:.3f}" for f in awg_freqs) + " GHz")
    print(f"\n  expecting at most {EXPECT_MAX_DBM:+.1f} dBm at the DUT output "
          f"-> {at_exa:+.1f} dBm at the EXA")
    print(f"  internal attenuation {atten:.0f} dB (keeps the first mixer at or below "
          f"{config.EXA_MAX_MIXER_DBM:+.0f} dBm)")
    if at_exa > config.EXA_DAMAGE_DBM - 10:
        print("  ** too close to the EXA damage level -- ABORT **")
        return

    awg = AWG()
    sa = SpecAn()
    results = {}
    try:
        sa.set_trace_mode("NORM")
        sa.set_attenuation(atten)
        sa.set_ref_level(max(10.0, at_exa + 10))
        sa.set_rbw(RBW_HZ)
        sa.set_vbw(None)
        sa.set_points(N_POINTS)
        print(f"  EXA: RBW {sa.get_rbw()/1e3:.0f} kHz, {N_POINTS} points/segment\n")

        for j, fa in enumerate(awg_freqs):
            vpp = config.AWG_DESIGN_VPP_NO_PAD
            print(f"  --- AWG {fa/1e9:.4f} GHz ---")
            awg.duc_cw_setup(float(fa), amplitude_vpp=float(vpp), channel=1)
            time.sleep(SETTLE_S)
            f, a = survey_once(sa, fa, segments)
            results[float(fa)] = (f, a)
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(results, pad, atten)
    return results


def report(results, pad, atten):
    print("\n" + "=" * 78)
    print("PEAKS FOUND")
    print("=" * 78)
    unexpected_all = []
    for fa, (f, a) in sorted(results.items()):
        pk = find_peaks(f, a, min_prominence_db=MIN_PROMINENCE_DB,
                        min_spacing_hz=SEG_HZ / N_POINTS * 3)
        # Reference for dBc: the strongest harmonic present, which is whatever
        # the chain is meant to be delivering at this build point.
        ref = max((amp for fp, amp in pk if classify(fp, fa)[0] is not None),
                  default=max(a))
        print(f"\n  AWG {fa/1e9:.4f} GHz -- {len(pk)} peaks above "
              f"{MIN_PROMINENCE_DB:.0f} dB prominence   (ref {ref+pad:+.2f} dBm)")
        print(f"    {'GHz':>9s} {'dBm':>9s} {'dBc':>8s}  identity")
        for fp, amp in pk:
            k, err = classify(fp, fa)
            if k is None:
                ident = "**  NOT a harmonic of f_AWG  **"
                unexpected_all.append((fa, fp, amp))
            else:
                ident = f"{k}f  (off by {err/1e6:+.1f} MHz)"
            print(f"    {fp/1e9:9.4f} {amp+pad:+9.2f} {amp-ref:8.1f}  {ident}")

    print("\n" + "=" * 78)
    if unexpected_all:
        print(f"** {len(unexpected_all)} PEAK(S) NOT EXPLAINED BY THE HARMONIC MODEL **")
        print("=" * 78)
        print("  These are the reason this script exists. For each, check whether it")
        print("  moves when the AWG frequency moves (chain-borne) or stays put")
        print("  (external pickup / analyzer artifact) -- the table below is")
        print("  grouped so that is visible at a glance.")
        for fa, fp, amp in unexpected_all:
            print(f"    AWG {fa/1e9:.4f} GHz -> peak {fp/1e9:9.4f} GHz  {amp+pad:+7.2f} dBm"
                  f"   (f_peak/f_AWG = {fp/fa:.4f})")
    else:
        print("NO UNEXPLAINED PEAKS -- every detection sits on an integer multiple")
        print("of the AWG frequency, so the harmonic model used everywhere else in")
        print("this campaign is complete at this prominence threshold.")
    print("=" * 78)

    sess = chainlog.new_session(STAGE, "spur_survey")
    for fa, (f, a) in sorted(results.items()):
        chainlog.save(sess, f"survey_{fa/1e9:.4f}ghz".replace(".", "p"), STAGE,
                      f, {"trace": a},
                      {"awg_hz": float(fa), "rbw_hz": RBW_HZ,
                       "n_points": N_POINTS, "atten_db": atten,
                       "start_hz": START_HZ, "stop_hz": STOP_HZ,
                       "seg_hz": SEG_HZ, "preemph": PREEMPH,
                       "min_prominence_db": MIN_PROMINENCE_DB,
                       "element": "broad spur survey"},
                      x_name="freqs_hz", thru=False,
                      arrangement=CHAIN or None,
                      notes=f"Broad survey at AWG {fa/1e9:.4f} GHz. RAW analyzer "
                            f"levels, add {pad:g} dB. Levels are LOWER BOUNDS: "
                            f"bin width {SEG_HZ/N_POINTS/1e6:.2f} MHz.")
    print(f"\n  saved {sess}")

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), squeeze=False, sharex=True)
    for ax, (fa, (f, a)) in zip(axes[:, 0], sorted(results.items())):
        ax.plot(f / 1e9, a + pad, lw=0.8, color="#2a78d6")
        for k in range(1, MAX_K + 1):
            fk = k * fa
            if START_HZ <= fk <= STOP_HZ:
                ax.axvline(fk / 1e9, ls=":", lw=0.9, color="#1baf7a", alpha=0.7)
        pk = find_peaks(f, a, min_prominence_db=MIN_PROMINENCE_DB,
                        min_spacing_hz=SEG_HZ / N_POINTS * 3)
        for fp, amp in pk:
            if classify(fp, fa)[0] is None:
                ax.plot(fp / 1e9, amp + pad, "v", ms=9, color="#e34948")
                ax.annotate(f"{fp/1e9:.3f}", (fp / 1e9, amp + pad), fontsize=7,
                            color="#e34948", xytext=(0, 9),
                            textcoords="offset points", ha="center")
        ax.set_ylabel("dBm")
        ax.set_title(f"AWG {fa/1e9:.4f} GHz   (green dotted = k x f_AWG, "
                     f"red = unexplained)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("frequency (GHz)")
    fig.suptitle("Broad spur survey -- is the harmonic model complete?", fontsize=12)
    fig.tight_layout()
    png = os.path.join(sess, "spur_survey.png")
    fig.savefig(png, dpi=130)
    print(f"  saved {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_AWG_FREQS)
