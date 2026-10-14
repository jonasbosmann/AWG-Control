"""Flatten the band by pre-emphasis, closed-loop, and verify that it worked.

WHY
---
The AM-to-PM measurement (run_am_pm.py, 2026-07-31) found 3.2 deg/dB in the
chain, with the AWG and analyzer contributing 1% of it. That converts leftover
amplitude ripple into phase and multiplies it by 96 to 310-370 GHz, where
nothing removes it. Working back from -36 dBc paired echoes gives a ripple
budget of roughly 0.003 dB rms. Measured today: 0.050 dB rms, reproducible at
a correlation of 0.996 between repeats.

Reproducible is the whole point. A deterministic function of frequency can be
inverted and commanded away; noise cannot. And the random part is 0.0033 dB
(0.0011 dB after averaging 8 sweeps), so the correction can be MEASURED some
30x finer than the error being corrected -- which is what makes this worth
attempting at all.

HOW -- and why it iterates
--------------------------
A dB commanded at the AWG does NOT arrive as a dB at the measurement plane.
Through a doubler it arrives as r dB, and r was measured today between 0.92
and 2.4 depending on drive and frequency. Applying the raw error as a
correction would overshoot by that factor and, at r > 2, diverge.

So the loop measures r for itself. Iteration 1 applies a damped correction;
from the change it actually produced, r is computed PER POINT as
(change at the output) / (change commanded), and iteration 2 applies the exact
correction. No prior knowledge of the chain is needed, and it converges
wherever the measurement plane happens to be.

MECHANISM
---------
Amplitude per frequency via :VOLT (awg.duc_cw_step). Measured 2026-07-31: no
staircase down to 20 milli-dB, but step increments quantised near ~6 milli-dB
and a 0.0046 dB rms deviation from a commanded ladder -- against 0.0011 dB of
read noise, so that floor is the AWG's, not the measurement's. That puts the
achievable flatness around 0.002-0.005 dB rms, right at the budget.

If the loop stalls there rather than at the target, the next step is to put
the correction in the DIGITAL baseband instead (15-bit scaling, ~0.0003 dB) at
the cost of a waveform re-upload per point. That is deliberately NOT built
yet: the simple mechanism buys 10x on its own, and the loop below reports
which floor it actually hit.

For the real experiment the question is moot -- a chirp is one waveform, so
its pre-emphasis is a digital envelope with full resolution and none of this
quantisation applies. This script flattens the CW-stepped case, which is what
every measurement in this campaign uses.

Run:  python x8_chain/run_preemph.py [n_freqs] [n_iters]
      MODE=chain|predoubler|awg    (default awg)
      NO_SHOW=1 no windows, LIVE=0 no live window.
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

os.environ.setdefault("MODE", "awg")

import config
import chainlog
from liveplot import LivePlot
from awg import AWG
from specan import SpecAn
import run_am_pm as R
from run_repeatability import decompose

STAGE = "A"
TONE, ELEMENT, PATH = R.TONE, R.ELEMENT, R.PATH

N_FREQS = 21
N_ITERS = 3
N_AVG = 8                # sweeps averaged per point -> 0.0011 dB read noise
SPAN_HZ = 1e6            # fine grid: 80 samples across the RBW, no scalloping
RBW_HZ = 100e3
N_POINTS = 801
SETTLE_S = 0.20
EXA_ATTEN_DB = 20
REF_LEVEL_DBM = 10

BASE_VPP = 0.3482
FIRST_DAMPING = 0.7
# Iteration 1 applies only this fraction of the measured error, because r is
# not yet known. 0.7 is safe for anything up to r = 1.4 and merely slow below
# that; iteration 2 onward uses the r the loop measured and needs no damping.
TOL_DB = 0.005           # stop when the reproducible residual is under this
SMOOTH_DEG = 3


def measure_band(sa, awg, freqs, vpps, n_avg=N_AVG, lp=None, tag=""):
    """One averaged pass across the band at a per-point amplitude.

    Returns (levels_dbm_raw, per_point_read_noise_db).
    """
    lv = np.empty(len(freqs))
    sd = np.empty(len(freqs))
    for i, (f, v) in enumerate(zip(freqs, vpps)):
        awg.duc_cw_step(float(f), channel=1, amplitude_vpp=float(v))
        sa.set_freq(float(f) * TONE, SPAN_HZ)
        time.sleep(SETTLE_S)
        reads = []
        for _ in range(n_avg):
            _fr, a = sa.sweep_once()
            reads.append(float(np.max(a)))
        lv[i] = float(np.mean(reads))
        sd[i] = float(np.std(reads, ddof=1)) / np.sqrt(n_avg)
        if lp is not None:
            lp.add(tag, f / 1e9, lv[i])
    return lv, sd


def run(n_freqs=N_FREQS, n_iters=N_ITERS):
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = config.STAGES[STAGE]["band_hz"]
    freqs = np.linspace(lo, hi, int(n_freqs))

    print("=" * 78)
    print(f"PRE-EMPHASIS -- {ELEMENT}")
    print(f"  {PATH} -> {pad:g} dB pad -> EXA")
    print("=" * 78)
    print(f"  {n_freqs} points {TONE*lo/1e9:.3f}-{TONE*hi/1e9:.3f} GHz, "
          f"{n_iters} iterations, {N_AVG} sweeps averaged per point")
    print(f"  base {BASE_VPP:.4f} Vpp, AWG ceiling {config.AWG_MAX_VPP} Vpp "
          f"-> {20*np.log10(config.AWG_MAX_VPP/BASE_VPP):.2f} dB of headroom")
    print(f"  target: flat at the band MAXIMUM, i.e. correct by RAISING the "
          f"weak points (never by throwing power away)")

    awg = AWG()
    sa = SpecAn()
    hist = []
    try:
        sa.set_trace_mode("NORM")
        sa.set_attenuation(EXA_ATTEN_DB)
        sa.set_ref_level(REF_LEVEL_DBM)
        sa.set_rbw(RBW_HZ)
        sa.set_vbw(None)
        sa.set_points(N_POINTS)
        awg.duc_cw_setup(float(freqs[0]), amplitude_vpp=float(BASE_VPP),
                         channel=1)
        time.sleep(0.4)

        lp = LivePlot(f"pre-emphasis: {ELEMENT}", "frequency (GHz)",
                      "level (dBm, raw)")

        vpps = np.full(len(freqs), float(BASE_VPP))
        lp.note("iteration 0: baseline")
        lv, sd = measure_band(sa, awg, freqs, vpps, lp=lp, tag="baseline")
        target = float(lv.max())
        print(f"\n  baseline: {lv.max()+pad:+.4f} to {lv.min()+pad:+.4f} dBm, "
              f"spread {lv.max()-lv.min():.4f} dB")
        print(f"    read noise {sd.mean():.4f} dB per point after averaging")

        need_db = target - lv.min()
        need_vpp = BASE_VPP * 10 ** (need_db / 20)
        print(f"    flattening to the maximum needs {need_db:.3f} dB at the "
              f"weakest point -> {need_vpp:.4f} Vpp")
        if need_vpp > config.AWG_MAX_VPP:
            print(f"  ** that exceeds the {config.AWG_MAX_VPP} Vpp ceiling. "
                  f"ABORT -- lower BASE_VPP or accept a lower target. **")
            return
        hist.append(dict(iter=0, vpps=vpps.copy(), levels=lv.copy(),
                         noise=sd.copy()))

        prev_cmd_db = np.zeros(len(freqs))
        r_est = np.ones(len(freqs))
        for it in range(1, int(n_iters) + 1):
            err = target - lv                     # dB short at each point
            if it == 1:
                step = err * FIRST_DAMPING
                how = f"damped x{FIRST_DAMPING} (r not yet known)"
            else:
                step = err / np.clip(r_est, 0.3, 4.0)
                how = (f"divided by the measured r "
                       f"({r_est.min():.2f}-{r_est.max():.2f})")
            cmd_db = prev_cmd_db + step
            new_vpp = np.clip(BASE_VPP * 10 ** (cmd_db / 20), 1e-3,
                              config.AWG_MAX_VPP)
            hit = int(np.sum(new_vpp >= config.AWG_MAX_VPP - 1e-9))
            print(f"\n  iteration {it}: correcting {how}")
            if hit:
                print(f"    ** {hit} point(s) clipped at the AWG ceiling **")

            lp.note(f"iteration {it}")
            lv_new, sd_new = measure_band(sa, awg, freqs, new_vpp, lp=lp,
                                          tag=f"iter {it}")

            # r, measured from what the correction actually did.
            d_cmd = 20 * np.log10(new_vpp / np.maximum(vpps, 1e-9))
            d_out = lv_new - lv
            ok = np.abs(d_cmd) > 0.02
            r_new = np.where(ok, d_out / np.where(ok, d_cmd, 1.0), r_est)
            r_est = np.clip(r_new, 0.3, 4.0)

            vpps, lv, sd = new_vpp, lv_new, sd_new
            prev_cmd_db = cmd_db
            resid = lv - np.polyval(np.polyfit(freqs, lv, SMOOTH_DEG), freqs)
            print(f"    spread {lv.max()-lv.min():.4f} dB, "
                  f"rms about the target {np.std(lv-target):.4f} dB, "
                  f"ripple about a smooth fit {np.std(resid):.4f} dB")
            print(f"    measured r: {r_est.min():.2f} to {r_est.max():.2f}, "
                  f"mean {r_est.mean():.2f}")
            hist.append(dict(iter=it, vpps=vpps.copy(), levels=lv.copy(),
                             noise=sd.copy(), r=r_est.copy()))
            if np.std(lv - target) < TOL_DB:
                print(f"    -> under the {TOL_DB} dB tolerance, stopping")
                break

        # Repeatability of the FINAL state, so the residual can be split into
        # what is really left and what is just measurement noise.
        print(f"\n  verifying the corrected state ({N_AVG}-averaged sweeps x3)")
        reps = [lv]
        for k in range(2):
            l2, _ = measure_band(sa, awg, freqs, vpps)
            reps.append(l2)
            print(f"    repeat {k+2}: spread {l2.max()-l2.min():.4f} dB")
        lp.done()
        dec = decompose(np.array(reps), freqs)

        settings = {"n_freqs": int(n_freqs), "n_iters": int(n_iters),
                    "n_avg": N_AVG, "span_hz": SPAN_HZ,
                    "rbw_hz": float(sa.get_rbw()), "n_points": N_POINTS,
                    "atten_db": EXA_ATTEN_DB, "base_vpp": BASE_VPP,
                    "mode": R.MODE, "element": ELEMENT, "tone": TONE,
                    "target_dbm_raw": target, "mechanism": "VOLT per point",
                    "first_damping": FIRST_DAMPING,
                    "r_final": [float(v) for v in r_est]}
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(freqs, hist, dec, settings)


def report(freqs, hist, dec, settings):
    pad = config.PAD_AT_INSTRUMENT_DB
    base, final = hist[0], hist[-1]
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  {'iter':>5}  {'spread':>9}  {'rms vs target':>14}  {'ripple':>9}")
    tgt = settings["target_dbm_raw"]
    for h in hist:
        lv = h["levels"]
        rp = np.std(lv - np.polyval(np.polyfit(freqs, lv, SMOOTH_DEG), freqs))
        print(f"  {h['iter']:5d}  {lv.max()-lv.min():8.4f} dB "
              f"{np.std(lv-tgt):13.4f} dB {rp:8.4f} dB")

    b = np.std(base["levels"] - np.polyval(
        np.polyfit(freqs, base["levels"], SMOOTH_DEG), freqs))
    print(f"\n  reproducible ripple in the CORRECTED state: "
          f"{dec['sigma_real']:.4f} dB rms "
          f"(detection floor {dec['floor']:.4f})")
    print(f"  random measurement noise:                    "
          f"{dec['sigma_noise']:.4f} dB rms")
    if not dec["detected"]:
        print(f"    -> no ripple detectable above the noise; the correction "
              f"has reached the measurement floor")

    slope, mult = 3.2, 192.0 / settings["tone"]
    def echo(sig):
        beta = np.radians(sig * slope * mult * np.sqrt(2))
        return 20 * np.log10(min(max(beta, 1e-12), 2.0) / 2)
    print(f"\n  WHAT IT BOUGHT (at {slope:.1f} deg/dB, x{mult:.0f} to the AMC "
          f"output)")
    print(f"    before  {b:.4f} dB rms -> {b*slope*mult:7.2f} deg rms -> "
          f"paired echoes {echo(b):+6.1f} dBc")
    print(f"    after   {dec['sigma_real']:.4f} dB rms -> "
          f"{dec['sigma_real']*slope*mult:7.2f} deg rms -> "
          f"paired echoes {echo(dec['sigma_real']):+6.1f} dBc")
    print(f"    improvement {20*np.log10(b/max(dec['sigma_real'],1e-9)):.1f} dB")
    need = np.degrees(10 ** (-36 / 20) * 2 / np.sqrt(2)) / (slope * mult)
    print(f"    -36 dBc needs under {need:.4f} dB rms -> "
          f"{'MET' if dec['sigma_real'] <= need else 'NOT met'}")
    if dec["sigma_real"] > need:
        print(f"       the :VOLT mechanism floors out near 0.002-0.005 dB "
              f"(measured); if this stalled there rather than at the target, "
              f"digital baseband scaling is the next lever")

    sess = chainlog.new_session(STAGE, "preemph")
    series = {"vpp_final": final["vpps"], "levels_baseline": base["levels"],
              "levels_final": final["levels"],
              "correction_db": 20 * np.log10(final["vpps"] / base["vpps"]),
              "r_measured": final.get("r", np.ones(len(freqs)))}
    p = chainlog.save(sess, "preemph", STAGE, freqs, series, settings,
                      x_name="freqs_hz", thru=False,
                      notes=f"{ELEMENT}. Closed-loop pre-emphasis. "
                            f"vpp_final is the per-frequency AWG amplitude to "
                            f"command; correction_db is it in dB relative to "
                            f"base_vpp. Levels RAW, add {pad:g} dB. Valid ONLY "
                            f"for this path and this base level.")
    print(f"\n  saved {p}")
    print(f"  -> that file IS the correction table: command vpp_final at each "
          f"frequency")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for h in hist:
        axes[0].plot(freqs / 1e9, h["levels"] + pad, "-o", ms=3,
                     label=f"iter {h['iter']}")
    axes[0].axhline(tgt + pad, ls="--", c="k", lw=1, alpha=0.6, label="target")
    axes[0].set_xlabel("frequency (GHz)"); axes[0].set_ylabel("dBm")
    axes[0].set_title("band levels through the loop")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].plot(freqs / 1e9, 20 * np.log10(final["vpps"] / BASE_VPP), "-o",
                 ms=4, c="tab:red")
    axes[1].set_xlabel("frequency (GHz)")
    axes[1].set_ylabel("correction (dB)")
    axes[1].set_title("the correction table")
    axes[1].grid(alpha=0.3)

    for h in (base, final):
        lv = h["levels"]
        axes[2].plot(freqs / 1e9,
                     lv - np.polyval(np.polyfit(freqs, lv, SMOOTH_DEG), freqs),
                     "-o", ms=3, label=f"iter {h['iter']}")
    axes[2].set_xlabel("frequency (GHz)")
    axes[2].set_ylabel("residual (dB)")
    axes[2].set_title(f"ripple: {b:.4f} -> {dec['sigma_real']:.4f} dB rms")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "preemph.png")
    fig.savefig(png, dpi=130)
    print(f"        {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_FREQS,
        int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERS)
