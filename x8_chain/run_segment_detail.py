"""What is actually inside one chirp segment?

The segment calibration samples each segment at 3 points and corrects it with a
single scalar. That is only right if nothing interesting lives between those
points. Three points across 5.2 MHz would completely miss a narrow suckout, a
spur, or a resonance -- and would report a smooth 22 milli-dB spread either way.

So: sweep a few segments at ~130 kHz resolution instead of ~2.6 MHz, repeat
each so real structure can be told from noise, and split what is there into

    TILT   a straight line across the segment. Benign: it is a slow amplitude
           gradient, it makes no paired echoes, and a per-segment scalar
           already handles its mean.
    WIGGLE periodic structure. This is what makes paired echoes, and what a
           3-point sample cannot see.
    SPIKE  a single narrow feature. The one that would silently corrupt a
           segment's calibration, because a 3-point sample either lands on it
           (and mis-sets that segment) or misses it (and leaves it in).

Reuses run_repeatability.decompose(), so "real" here means correlated between
independent repeats, with the same calibrated detection floor.

Run:  python x8_chain/run_segment_detail.py [n_segments] [points_per_segment]
      MODE=chain|predoubler|awg   (default awg)
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
from run_segments import segment_plan, CHAIN_MULT, BASE_VPP

STAGE = "A"
TONE, ELEMENT, PATH = R.TONE, R.ELEMENT, R.PATH

N_SEGMENTS = 3           # spread across the band: low, middle, high
N_PTS = 41               # across each segment -> 130 kHz at 1 GHz segments
N_REPEATS = 3
N_AVG = 4
SEGMENT_GHZ = 1.0
SPAN_HZ = 1e6
RBW_HZ = 100e3
N_POINTS = 801
SETTLE_S = 0.12
EXA_ATTEN_DB = 20
REF_LEVEL_DBM = 10
SPIKE_SIGMA = 4.0        # a point this far off the local trend is a feature


def sweep_fine(sa, awg, freqs, vpp, lp=None, tag=""):
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        awg.duc_cw_step(float(f), channel=1, amplitude_vpp=float(vpp))
        sa.set_freq(float(f) * TONE, SPAN_HZ)
        time.sleep(SETTLE_S)
        out[i] = float(np.mean([np.max(sa.sweep_once()[1])
                                for _ in range(N_AVG)]))
        if lp is not None:
            lp.add(tag, f / 1e6, out[i])
    return out


def pick_segments(n, n_total):
    """Which segments to look inside: the ones the calibration found WORST.

    Evenly spaced picking is the obvious choice and it is wrong. Run
    2026-07-31: linspace(0, 60, 3) selected segments 0/30/60, which happened
    to have internal spreads of 11/5/9 milli-dB against a band mean of 22 --
    three of the flattest in the band. The fine sweep then reported "nothing
    periodic here", which was true of those three and told us nothing about
    the 15/20/50 sitting at 31/26/36.

    So the previous calibration's own in-segment spread chooses the targets.
    If a per-segment scalar is going to fail anywhere, it fails there.
    """
    for p in chainlog.find(STAGE, label_contains="segments", thru=False):
        d = chainlog.load(p)
        k = chainlog.series_key("in_segment_spread_db")
        if k not in d or len(d[k]) != n_total:
            continue
        sp = np.asarray(d[k], dtype=float)
        order = np.argsort(sp)[::-1]
        chosen = list(order[:max(n - 1, 1)])
        med = int(order[len(order) // 2])       # one typical one, for contrast
        if n > 1 and med not in chosen:
            chosen.append(med)
        return np.array(sorted(chosen)), p, sp
    return (np.linspace(0, n_total - 1, n).round().astype(int), None, None)


def run(n_segments=N_SEGMENTS, n_pts=N_PTS):
    pad = config.PAD_AT_INSTRUMENT_DB
    cen, edges, cen_out = segment_plan(SEGMENT_GHZ)
    pick, src, sp_prior = pick_segments(int(n_segments), len(cen))

    print("=" * 78)
    print(f"INSIDE A SEGMENT -- {ELEMENT}")
    print(f"  {PATH} -> {pad:g} dB pad -> EXA")
    print("=" * 78)
    seg_bw = edges[1, 0] - edges[0, 0]
    print(f"  {len(pick)} segments of {seg_bw/1e6:.3f} MHz, {n_pts} points "
          f"each -> {seg_bw/(n_pts-1)/1e3:.1f} kHz resolution")
    print(f"  (the calibration samples the same span with 3 points, "
          f"{seg_bw/2/1e6:.2f} MHz apart)")
    print(f"  {N_REPEATS} repeats per segment, {N_AVG} sweeps averaged")
    if src is not None:
        print(f"  targets chosen by WORST in-segment spread from the "
              f"calibration, not evenly spaced:")
        for s in pick:
            print(f"    segment {s:2d} ({cen_out[s]/1e9:7.2f} GHz out) -- "
                  f"calibration measured {sp_prior[s]*1000:5.1f} milli-dB "
                  f"across it (band mean {sp_prior.mean()*1000:.1f})")
        print(f"    from {os.path.basename(src)}")
    else:
        print(f"  ** no prior calibration found -- falling back to evenly "
              f"spaced targets, which may miss the bad segments **")

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
        awg.duc_cw_setup(float(cen[pick[0]]), amplitude_vpp=float(BASE_VPP),
                         channel=1)
        time.sleep(0.4)

        lp = LivePlot(f"inside a segment: {ELEMENT}",
                      "AWG frequency (MHz)", "level (dBm, raw)")
        for s in pick:
            fs = np.linspace(edges[0, s], edges[1, s], int(n_pts))
            reps = []
            for k in range(N_REPEATS):
                lp.note(f"segment {s} ({cen_out[s]/1e9:.2f} GHz out), "
                        f"repeat {k+1}/{N_REPEATS}")
                reps.append(sweep_fine(sa, awg, fs, BASE_VPP, lp=lp,
                                       tag=f"seg{s} #{k+1}"))
            data[int(s)] = (fs, np.array(reps))
            print(f"    segment {s:2d} ({cen_out[s]/1e9:7.2f} GHz out) done")
        lp.done()

        settings = {"n_segments": int(n_segments), "n_points_per_seg": int(n_pts),
                    "n_repeats": N_REPEATS, "n_avg": N_AVG,
                    "segment_ghz": SEGMENT_GHZ, "span_hz": SPAN_HZ,
                    "rbw_hz": float(sa.get_rbw()), "atten_db": EXA_ATTEN_DB,
                    "base_vpp": BASE_VPP, "mode": R.MODE, "element": ELEMENT,
                    "tone": TONE, "segments_probed": [int(v) for v in pick],
                    "resolution_hz": float(seg_bw / (n_pts - 1))}
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(data, cen_out, settings)


def report(data, cen_out, settings):
    pad = config.PAD_AT_INSTRUMENT_DB
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  {'seg':>4} {'out GHz':>9} {'spread':>9} {'TILT':>9} "
          f"{'WIGGLE':>9} {'floor':>8} {'noise':>8} {'spikes':>7}")
    summary = {}
    for s, (fs, reps) in data.items():
        mean = reps.mean(axis=0)
        # tilt = the straight line; wiggle = everything left over
        lin = np.polyval(np.polyfit(fs, mean, 1), fs)
        wig = mean - lin
        d = decompose(reps, fs)                 # real vs random, degree-3 fit
        # spikes: points far off a local (degree-3) trend
        sm = np.polyval(np.polyfit(fs, mean, 3), fs)
        res = mean - sm
        rob = 1.4826 * np.median(np.abs(res - np.median(res)))
        spikes = int(np.sum(np.abs(res - np.median(res)) >
                            SPIKE_SIGMA * max(rob, 1e-6)))
        summary[s] = dict(tilt=float(lin[-1] - lin[0]), wiggle=float(np.std(wig)),
                          real=d["sigma_real"], floor=d["floor"],
                          noise=d["sigma_noise"], spikes=spikes,
                          spread=float(mean.max() - mean.min()),
                          detected=d["detected"], mean=mean, fs=fs, res=res)
        print(f"  {s:4d} {cen_out[s]/1e9:9.2f} {summary[s]['spread']*1000:8.1f}m "
              f"{summary[s]['tilt']*1000:+8.1f}m {summary[s]['wiggle']*1000:8.1f}m "
              f"{d['floor']*1000:7.1f}m {d['sigma_noise']*1000:7.1f}m "
              f"{spikes:7d}")
    print("    all figures milli-dB. TILT is signed (across the segment); "
          "WIGGLE is rms about the straight line.")

    tilts = np.array([v["tilt"] for v in summary.values()])
    wigs = np.array([v["wiggle"] for v in summary.values()])
    reals = np.array([v["real"] for v in summary.values()])
    nspk = sum(v["spikes"] for v in summary.values())

    print(f"\n  VERDICT")
    print(f"    tilt   {np.abs(tilts).mean()*1000:.1f} milli-dB mean across a "
          f"segment")
    print(f"    wiggle {wigs.mean()*1000:.1f} milli-dB rms, and the "
          f"reproducible part is {reals.mean()*1000:.1f} "
          f"(floor {np.mean([v['floor'] for v in summary.values()])*1000:.1f})")
    frac = wigs.mean() / max(np.abs(tilts).mean() / np.sqrt(12), 1e-9)
    if np.abs(tilts).mean() / np.sqrt(12) > 2 * wigs.mean():
        print(f"    -> DOMINATED BY TILT. A per-segment scalar is the right "
              f"correction and there is nothing periodic to make echoes.")
    elif wigs.mean() > 2 * np.abs(tilts).mean() / np.sqrt(12):
        print(f"    -> DOMINATED BY WIGGLE ({frac:.1f}x the tilt's rms). This "
              f"IS periodic structure, so it can make paired echoes and a "
              f"scalar per segment does not remove it.")
    else:
        print(f"    -> comparable tilt and wiggle ({frac:.1f}x).")
    if nspk:
        print(f"    ** {nspk} narrow feature(s) beyond {SPIKE_SIGMA} sigma -- "
              f"a 3-point sample would either land on one and mis-set that "
              f"segment, or miss it entirely **")
        for s, v in summary.items():
            if v["spikes"]:
                i = int(np.argmax(np.abs(v["res"])))
                print(f"       segment {s}: worst at {v['fs'][i]/1e6:.3f} MHz, "
                      f"{v['res'][i]*1000:+.1f} milli-dB off trend")
    else:
        print(f"    no narrow features: 3 points per segment is not missing "
              f"anything sharp")

    # what it costs, either way
    amp_pct = 100 * (10 ** (wigs.mean() / 20) - 1)
    beta = np.radians(wigs.mean() * np.sqrt(2) * 3.2 * 192.0 / settings["tone"])
    print(f"\n  CONSEQUENCE of the wiggle")
    print(f"    {amp_pct:.3f}% amplitude variation within a segment -> line "
          f"intensities inside one segment differ by that much")
    print(f"    via AM-to-PM at 3.2 deg/dB: "
          f"{wigs.mean()*3.2*192.0/settings['tone']:.2f} deg rms at "
          f"310-370 GHz -> paired echoes "
          f"{20*np.log10(min(max(beta,1e-12),2.0)/2):+.1f} dBc")

    sess = chainlog.new_session(STAGE, "segment_detail")
    for s, (fs, reps) in data.items():
        series = {f"rep{k+1}": r for k, r in enumerate(reps)}
        series["mean"] = reps.mean(axis=0)
        chainlog.save(sess, f"segment{s:02d}", STAGE, fs, series, settings,
                      x_name="freqs_hz", thru=False,
                      notes=f"{ELEMENT}. Fine sweep INSIDE segment {s} "
                            f"(centre {cen_out[s]/1e9:.3f} GHz at the output). "
                            f"RAW levels, add {pad:g} dB.")
    print(f"\n  saved {sess}")

    n = len(summary)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8), squeeze=False)
    for j, (s, v) in enumerate(summary.items()):
        fs, mean = v["fs"], v["mean"]
        ax = axes[0][j]
        ax.plot(fs / 1e6, mean + pad, "-o", ms=2.5, lw=1)
        lin = np.polyval(np.polyfit(fs, mean, 1), fs)
        ax.plot(fs / 1e6, lin + pad, "--", lw=1.5, c="k",
                label=f"tilt {v['tilt']*1000:+.1f} mdB")
        f3 = np.linspace(fs[0], fs[-1], 3)
        ax.plot(f3 / 1e6, np.interp(f3, fs, mean) + pad, "rv", ms=9,
                label="the 3 calibration points")
        ax.set_title(f"segment {s} -- {cen_out[s]/1e9:.2f} GHz out", fontsize=9)
        ax.set_xlabel("AWG frequency (MHz)"); ax.set_ylabel("dBm")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax = axes[1][j]
        ax.plot(fs / 1e6, (mean - lin) * 1000, "-o", ms=2.5, lw=1)
        ax.axhline(0, c="k", lw=0.8, alpha=0.5)
        ax.set_title(f"about the straight line: {v['wiggle']*1000:.1f} mdB rms",
                     fontsize=9)
        ax.set_xlabel("AWG frequency (MHz)"); ax.set_ylabel("milli-dB")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(sess, "segment_detail.png")
    fig.savefig(png, dpi=130)
    print(f"        {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_SEGMENTS,
        int(sys.argv[2]) if len(sys.argv) > 2 else N_PTS)
