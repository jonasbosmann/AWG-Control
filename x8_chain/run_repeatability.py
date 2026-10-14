"""How much of the measured band ripple is real, and how much is the measurement?

The AM-to-PM result made this urgent. Amplitude ripple surviving pre-emphasis
is converted to phase at ~3.2 deg/dB and then multiplied by 96 to 310-370 GHz,
where nothing removes it. Working back from -36 dBc paired echoes gives a
ripple budget of a few thousandths of a dB. Before anyone designs to that, the
0.177 dB rms currently on the books has to be shown to be HARDWARE.

It probably is not. Three things are mixed into it:

  1. RANDOM measurement noise -- uncorrelated between repeats. Averaging kills
     it. Easy to spot: it does not repeat.

  2. SCALLOPING -- the trap. The earlier sweeps used a 50 MHz span with 801
     points: 62.5 kHz between bins against a 100 kHz RBW, i.e. ~1.6 samples
     across the resolution filter. Picking the peak off a grid that coarse
     costs a few tenths of a dB depending where the tone falls between bins.
     Because a given frequency always lands at the same offset from a bin,
     THAT ERROR IS IDENTICAL EVERY SWEEP. It correlates perfectly between
     repeats and is indistinguishable from hardware ripple by repetition
     alone. It is an artefact of the frequency grid, so it is separated by
     changing the grid: measure the same band at a fine span and a coarse one.

  3. REAL hardware ripple -- reproducible AND grid-independent. Only this
     makes paired echoes.

Method: sweep the band N times at each of two spans. Fit and remove a smooth
curve per repeat (so drift and tilt do not contaminate), then decompose the
residual variance:

    sigma_real^2  = mean over pairs i!=j of cov(residual_i, residual_j)
    sigma_noise^2 = mean_i var(residual_i) - sigma_real^2

Covariance between INDEPENDENT repeats sees only what reproduces; whatever is
left over is noise. Run it at both spans and the difference in sigma_real is
the scalloping.

Works in whatever configuration is cabled up -- it takes MODE from the same
table run_am_pm.py uses, so the path label cannot drift between them.

Run:  python x8_chain/run_repeatability.py [n_freqs] [n_repeats]
      MODE=chain|predoubler|awg     (default awg)
      NO_SHOW=1 to skip the window.
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
import run_am_pm as R          # single definition of the modes and the path

STAGE = "A"
TONE, ELEMENT, PATH = R.TONE, R.ELEMENT, R.PATH

N_FREQS = 21
N_REPEATS = 5
SPANS_HZ = (1e6, 50e6)
# 1 MHz: 1.25 kHz between bins, 80 samples across the 100 kHz RBW -- scalloping
#        is nothing.
# 50 MHz: what every earlier sweep in this campaign used. Kept so the two can
#        be compared directly on the same hardware in the same session.
RBW_HZ = 100e3
N_POINTS = 801
SETTLE_S = 0.25
EXA_ATTEN_DB = 20
REF_LEVEL_DBM = 10
SMOOTH_DEG = 3           # what counts as "the band shape" rather than ripple

OPERATING_VPP = 0.3482   # the level everything else today was taken at


def sweep_band(sa, awg, freqs, span_hz, vpp, first=False):
    """One pass across the band. Returns peak level per point, RAW."""
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        if first and i == 0:
            awg.duc_cw_setup(float(f), amplitude_vpp=float(vpp), channel=1)
        else:
            awg.duc_cw_step(float(f), channel=1)
        time.sleep(SETTLE_S)
        sa.set_freq(float(f) * TONE, span_hz)
        time.sleep(0.05)
        _fr, a = sa.sweep_once()
        out[i] = float(np.max(a))
    return out


def decompose(reps, freqs):
    """Split residual ripple into what reproduces and what does not.

    reps: (n_repeats, n_freqs) of levels in dB.

    A smooth curve is removed from EACH repeat separately, so a slow level
    drift between repeats cannot masquerade as either term.
    """
    res = np.array([r - np.polyval(np.polyfit(freqs, r, SMOOTH_DEG), freqs)
                    for r in reps])
    n = len(res)
    # Both terms divided by the SAME residual degrees of freedom. Removing a
    # cubic costs 4 of the n points, and np.cov/np.var would apply different
    # divisors (n-1 vs n-4) to the two halves of the subtraction -- an 18%
    # error at n=21 that lands entirely in the noise term. The residuals are
    # already zero-mean by construction (the fit carries a constant), so a
    # plain dot product is the right estimator.
    dof = max(len(freqs) - (SMOOTH_DEG + 1), 1)
    var_tot = float(np.mean([float(np.dot(r, r)) / dof for r in res]))
    # Noise from the DIFFERENCES between repeats. Any component common to both
    # -- real ripple, scalloping, anything reproducible -- cancels exactly in
    # res_i - res_j, so this is a clean measurement of the random part that
    # assumes nothing about how much real ripple is present.
    diffs = [float(np.dot(res[i] - res[j], res[i] - res[j])) / (2 * dof)
             for i in range(n) for j in range(n) if i < j]
    var_noise = float(np.mean(diffs))
    var_real = max(var_tot - var_noise, 0.0)

    # DETECTION FLOOR. With finite repeats the estimator returns a non-zero
    # sigma_real even when there is none, and this measurement exists to
    # decide whether ripple is real -- so the threshold is part of the answer,
    # not a footnote. Calibrated by simulation under the null hypothesis
    # (2026-07-31, 400 trials per cell): the 95th percentile of the reported
    # sigma_real is k * sigma_noise / (dof * R(R-1)/2)^0.25 with k = 1.25-1.43
    # across 11-41 points and 3-10 repeats, exactly proportional to
    # sigma_noise. 1.45 is the conservative end of that.
    floor = 1.45 * np.sqrt(var_noise) / (dof * n * (n - 1) / 2) ** 0.25

    # Behaviour, over 300 trials per case (2026-07-31): false positives 4.3%
    # against the 95% bound, detection 100% for ripple above the floor, and
    # sigma_real running a consistent 1.10x HIGH -- from taking the square
    # root of a difference of two noisy variances and clipping it at zero.
    # Left uncorrected: it errs toward reporting more ripple than is there,
    # which is the safe direction for a spec.
    corr = [float(np.corrcoef(res[i], res[j])[0, 1])
            for i in range(n) for j in range(n) if i != j]
    sig_real = float(np.sqrt(var_real))
    return dict(
        residuals=res,
        sigma_real=sig_real,
        sigma_noise=float(np.sqrt(var_noise)),
        sigma_total=float(np.sqrt(var_tot)),
        floor=float(floor),
        detected=bool(sig_real > floor),
        corr=float(np.mean(corr)),
        drift=float(np.max(np.mean(reps, axis=1)) - np.min(np.mean(reps, axis=1))),
    )


def run(n_freqs=N_FREQS, n_repeats=N_REPEATS):
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = config.STAGES[STAGE]["band_hz"]
    freqs = np.linspace(lo, hi, int(n_freqs))

    print("=" * 78)
    print(f"BAND REPEATABILITY -- {ELEMENT}")
    print(f"  {PATH} -> {pad:g} dB pad -> EXA")
    print("=" * 78)
    print(f"  {n_freqs} points {TONE*lo/1e9:.3f}-{TONE*hi/1e9:.3f} GHz, "
          f"{n_repeats} repeats, at spans "
          f"{', '.join(f'{s/1e6:g} MHz' for s in SPANS_HZ)}")
    for s in SPANS_HZ:
        print(f"    {s/1e6:5g} MHz span -> {s/N_POINTS/1e3:7.2f} kHz per bin, "
              f"{RBW_HZ/(s/N_POINTS):6.1f} samples across the {RBW_HZ/1e3:.0f} "
              f"kHz RBW")
    print(f"  the wider span is what every earlier sweep used; if its "
          f"reproducible\n  residual is bigger, the difference is scalloping, "
          f"not hardware")

    awg = AWG()
    sa = SpecAn()
    data = {}
    try:
        sa.set_trace_mode("NORM")
        sa.set_attenuation(EXA_ATTEN_DB)
        sa.set_ref_level(REF_LEVEL_DBM)
        sa.set_rbw(RBW_HZ)
        sa.set_vbw(None)
        sa.set_points(N_POINTS)
        print(f"\n  EXA: RBW {sa.get_rbw()/1e3:.0f} kHz, {N_POINTS} points, "
              f"atten {EXA_ATTEN_DB} dB\n")

        lp = LivePlot(f"repeatability: {ELEMENT}",
                      "frequency (GHz)", "residual after a smooth fit (dB)")

        first = True
        for span in SPANS_HZ:
            reps = []
            for k in range(int(n_repeats)):
                lp.note(f"{span/1e6:g} MHz span, repeat {k+1}/{n_repeats}")
                lv = sweep_band(sa, awg, freqs, span, OPERATING_VPP, first=first)
                first = False
                reps.append(lv)
                sm = np.polyval(np.polyfit(freqs, lv, SMOOTH_DEG), freqs)
                for f, v in zip(freqs, lv - sm):
                    lp.add(f"{span/1e6:g}MHz #{k+1}", f / 1e9, v)
                print(f"    {span/1e6:5g} MHz  repeat {k+1}: "
                      f"{lv.max()+pad:+7.3f} to {lv.min()+pad:+7.3f} dBm, "
                      f"residual rms {np.std(lv-sm):.4f} dB")
            data[span] = np.array(reps)
        lp.done()

        settings = {"n_freqs": int(n_freqs), "n_repeats": int(n_repeats),
                    "spans_hz": list(SPANS_HZ), "rbw_hz": float(sa.get_rbw()),
                    "n_points": N_POINTS, "atten_db": EXA_ATTEN_DB,
                    "smooth_deg": SMOOTH_DEG, "operating_vpp": OPERATING_VPP,
                    "element": ELEMENT, "mode": R.MODE, "tone": TONE,
                    "settle_s": SETTLE_S}
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(freqs, data, settings)


def report(freqs, data, settings):
    pad = config.PAD_AT_INSTRUMENT_DB
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)

    dec = {}
    print(f"  {'span':>10}  {'reproducible':>13} {'floor':>8}  {'random':>8}"
          f"  {'total':>8}  {'corr':>6}  {'drift':>7}")
    for span, reps in data.items():
        d = decompose(reps, freqs)
        dec[span] = d
        print(f"  {span/1e6:8g} MHz  {d['sigma_real']:11.4f} dB "
              f"{d['floor']:8.4f}  {d['sigma_noise']:8.4f}  "
              f"{d['sigma_total']:8.4f}  {d['corr']:6.3f}  {d['drift']:7.4f}"
              + ("" if d["detected"] else "   <- at/below the floor"))
    print("    reproducible = correlated between repeats (hardware OR grid "
          "artefact)")
    print("    floor        = what this estimator returns from PURE NOISE at "
          "this")
    print("                   repeat count; below it, no ripple is detected")
    print("    random       = uncorrelated; averaging N sweeps divides it by "
          "sqrt(N)")
    print("    drift        = spread of the per-repeat mean level")

    spans = sorted(data)
    if len(spans) >= 2:
        fine, coarse = spans[0], spans[-1]
        rf, rc = dec[fine]['sigma_real'], dec[coarse]['sigma_real']
        print(f"\n  SCALLOPING TEST")
        print(f"    reproducible residual: {rf:.4f} dB at {fine/1e6:g} MHz, "
              f"{rc:.4f} dB at {coarse/1e6:g} MHz")
        if rc > rf:
            sc = np.sqrt(max(rc ** 2 - rf ** 2, 0.0))
            print(f"    -> {sc:.4f} dB rms of the wide-span 'ripple' is "
                  f"GRID ARTEFACT, not hardware.")
            print(f"       It is reproducible, so repeating the sweep would "
                  f"never have caught it; only changing the grid does.")
            print(f"    -> real hardware ripple is {rf:.4f} dB rms")
        else:
            print(f"    -> no scalloping penalty; the wide span was fine")
        real = rf
    else:
        real = list(dec.values())[0]['sigma_real']

    # What it costs, given today's AM-to-PM.
    slope, mult = 3.2, 192.0 / settings["tone"]
    print(f"\n  CONSEQUENCE, at the {slope:.1f} deg/dB measured today "
          f"(x{mult:.0f} from this plane)")
    for label, sig in (("as measured", real),
                       ("averaging 10 sweeps", real)):
        beta = np.radians(sig * slope * mult * np.sqrt(2))
        print(f"    {label:22s} {sig:.4f} dB rms -> "
              f"{sig*slope*mult:7.2f} deg rms at 310-370 GHz -> "
              f"paired echoes {20*np.log10(min(beta,2.0)/2):+7.1f} dBc")
        break
    tgt = 10 ** (-36 / 20) * 2 / np.sqrt(2)
    need = np.degrees(tgt) / (slope * mult)
    print(f"    to reach -36 dBc (the level the filter was chosen at) the "
          f"ripple must be under {need:.4f} dB rms")
    print(f"    -> {'MET' if real <= need else 'NOT met'}: measured "
          f"{real:.4f} dB rms")
    print(f"    NOTE the random part does not count against this the same way "
          f"-- pre-emphasis is calibrated from an AVERAGED sweep, so it is the "
          f"REPRODUCIBLE ripple that survives into the correction and the "
          f"random part that limits how well the correction can be measured.")

    sess = chainlog.new_session(STAGE, "repeatability")
    series = {}
    for span, reps in data.items():
        for k, r in enumerate(reps):
            series[f"{span/1e6:g}mhz_rep{k+1}"] = r
        series[f"{span/1e6:g}mhz_mean"] = reps.mean(axis=0)
    p = chainlog.save(sess, "repeatability", STAGE, freqs, series, settings,
                      x_name="freqs_hz", thru=False,
                      notes=f"{ELEMENT}. RAW analyzer levels, add {pad:g} dB. "
                            f"Repeated band sweeps at two spans to separate "
                            f"reproducible ripple from random noise and from "
                            f"peak-pick scalloping. x is the AWG frequency; "
                            f"measured at {settings['tone']}x it.")
    print(f"\n  saved {p}")

    fig, axes = plt.subplots(1, len(data) + 1, figsize=(6 * (len(data) + 1), 5))
    axes = np.atleast_1d(axes)
    for ax, (span, reps) in zip(axes, data.items()):
        d = dec[span]
        for k, r in enumerate(d["residuals"]):
            ax.plot(freqs / 1e9, r, "-o", ms=3, lw=1, alpha=0.75,
                    label=f"repeat {k+1}")
        ax.plot(freqs / 1e9, d["residuals"].mean(axis=0), "k-", lw=2.2,
                label="mean")
        ax.set_title(f"{span/1e6:g} MHz span\nreproducible "
                     f"{d['sigma_real']:.4f} dB, random {d['sigma_noise']:.4f} dB",
                     fontsize=10)
        ax.set_xlabel("AWG frequency (GHz)")
        ax.set_ylabel("residual (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    ax = axes[-1]
    for span in data:
        ax.plot(freqs / 1e9, dec[span]["residuals"].mean(axis=0), "-o", ms=4,
                label=f"{span/1e6:g} MHz mean")
    ax.set_title("the two grids compared\n(shared structure = hardware)",
                 fontsize=10)
    ax.set_xlabel("AWG frequency (GHz)")
    ax.set_ylabel("residual (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(sess, "repeatability.png")
    fig.savefig(png, dpi=130)
    print(f"        {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_FREQS,
        int(sys.argv[2]) if len(sys.argv) > 2 else N_REPEATS)
