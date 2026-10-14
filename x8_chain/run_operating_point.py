"""Measure a stage at ONE AWG amplitude -- the one the experiment runs at.

    AWG CH1 -> 3 dB pad -> ZX60-2522MA-S+ -> 10 dB pad -> scope     (stage A)

This is the whole measurement for an operating point: the level across the
band, the flatness, and the implied gain. No ladder, no compression sweep.
Those answer "how does the amp behave as drive changes", which is a different
question from "what does it do at the setting I am going to use", and only
the second one gates the build.

FIRST POINT FIRST
-----------------
The very first frequency is measured and reported BEFORE the sweep starts,
with an explicit verdict against what the level should be. A dead path, an
unpowered amp or a loose connector shows up in about two seconds instead of
after a full sweep -- and if the level is wildly wrong the script stops rather
than recording a band of nonsense.

Run:  python x8_chain/run_operating_point.py [stage] [vpp]
      vpp defaults to the operating amplitude from the most recent
      run_drive_level.py session for that stage.
      NO_SHOW=1 to skip the plot window.
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
from scope import Scope
from run_awg_level_scope import phase_coverage_warning, crest_warning


SIG_CH = 1
SETTLE_S = 0.20
CAPTURE_S = 2e-6
N_REPEATS = 2
SANITY_TOL_DB = 6.0      # how far the first point may sit from prediction
                          # before the run is abandoned as a broken path


def operating_vpp(stage_key):
    """The amplitude the last drive-ladder run settled on, or a datasheet
    estimate for the target output if there hasn't been one."""
    for p in chainlog.find(stage_key, label_contains="drive_ladder", thru=False):
        d = chainlog.load(p)
        v = d.get("instrument_settings", {}).get("operating_vpp")
        if v:
            print(f"  operating amplitude {v:.4f} Vpp, measured in {p}")
            return float(v)
    st = config.STAGES[stage_key]
    want = config.TARGET_OUT_DBM - (st["gain_typ_db"] or 0.0) + st["drive_pad_db"]
    v = float(np.clip(config.vpp_from_dbm(want), 0.001, config.AWG_MAX_VPP))
    print(f"  ** no drive-ladder session for stage {stage_key} -- using a "
          f"datasheet estimate of {v:.4f} Vpp **")
    return v


def predicted_scope_dbm(stage_key, vpp, freq_hz):
    """Where the first point should land, from the measured AWG curve."""
    st = config.STAGES[stage_key]
    resp = mc.load_scope_band_response()
    if resp is None:
        awg = config.dbm_from_vpp(vpp)
    else:
        awg = (float(np.interp(freq_hz, resp[0], resp[1]))
               + 20 * np.log10(vpp / mc.AWG_OPERATING_AMP_VPP))
    return (awg - mc.CABLE_IL_DB - st["drive_pad_db"]
            + (st["gain_typ_db"] or 0.0) - config.PAD_AT_INSTRUMENT_DB)


def measure(scope, fs, freq_hz, repeats=N_REPEATS):
    vals, notes = [], []
    for _ in range(repeats):
        vpp, _t, v = scope.measure_vpp(channel=SIG_CH, settle=0.05)
        vals.append(config.dbm_from_vpp(vpp) if vpp > 0 else np.nan)
        w = crest_warning(v)
        if w and w not in notes:
            notes.append(w)
    w = phase_coverage_warning(float(freq_hz), fs)
    if w:
        notes.append(w)
    return float(np.mean(vals)), float(np.std(vals)), notes


def run(stage_key="A", vpp=None):
    st = config.STAGES[stage_key]
    pad = config.PAD_AT_INSTRUMENT_DB
    freqs = config.band_freqs(stage_key)
    lo, hi = st["band_hz"]

    print("=" * 74)
    print(f"OPERATING POINT -- stage {stage_key}: {st['role']}")
    print(f"  {config.arrangement(stage_key)}")
    print("=" * 74)
    if vpp is None:
        vpp = operating_vpp(stage_key)
    pred = predicted_scope_dbm(stage_key, vpp, float(freqs[0]))
    print(f"  AWG {vpp:.4f} Vpp, {len(freqs)} points "
          f"{freqs[0]/1e9:.3f}-{freqs[-1]/1e9:.3f} GHz")
    print(f"  first point should read about {pred:+.1f} dBm at the scope "
          f"({pred+pad:+.1f} dBm at the amp output)")

    ok, _ = config.safety_check(stage_key,
                                pred + pad + 3.0)   # 3 dB of slack on the guess
    if not ok:
        print("\nABORTED on the safety check above.")
        return
    if st["instrument"] != "scope":
        print("\n  this stage is configured for the EXA; this script is scope-only")
        return

    awg = AWG()
    scope = Scope()
    try:
        scope.setup(channel=SIG_CH, n_averages=1,
                    trigger_channel=SIG_CH, trigger_level=0.0)
        scope.set_active_channels([SIG_CH])
        fs = scope.set_max_sample_rate(CAPTURE_S)
        vscale = min(max(config.volts_per_div_for(pred), 1e-3), 1.0)
        scope.set_vertical(SIG_CH, vscale)
        print(f"scope: {fs/1e9:.2f} GS/s, {CAPTURE_S*1e6:.1f} us record, "
              f"vertical FIXED at {vscale*1e3:.4g} mV/div")

        # ---- first point, as a live check on the path -------------------
        awg.duc_cw_setup(float(freqs[0]), amplitude_vpp=float(vpp), channel=1)
        time.sleep(SETTLE_S)
        m, s, notes = measure(scope, fs, float(freqs[0]))
        err = m - pred
        print(f"\n  FIRST POINT  {freqs[0]/1e9:.3f} GHz: {m:+.2f} dBm "
              f"({m+pad:+.2f} dBm out), {err:+.2f} dB vs prediction"
              + ("   " + "; ".join(notes) if notes else ""))
        if not np.isfinite(m) or abs(err) > SANITY_TOL_DB:
            print(f"\n  ** {abs(err):.1f} dB off (tolerance {SANITY_TOL_DB:g} dB) "
                  f"-- the signal path is not what this script assumes. **")
            print("     Check, in this order:")
            print("       - amp bias present (ZX60-2522MA-S+ wants 5 V, 94 mA)")
            print("       - the 3 dB pad is at the amp INPUT, the 10 dB at the "
                  "scope")
            print("       - every SMA joint actually tightened")
            print("       - AWG CH1 output on and cabled to the amp input")
            print("     Nothing recorded. Fix and re-run.")
            return
        print("     path looks right -- sweeping the band\n")

        lp = LivePlot(f"stage {stage_key}: {st['part']} at {vpp:.4f} Vpp",
                      "frequency (GHz)", "amp output (dBm)",
                      spans=[(lo / 1e9, hi / 1e9, "chain band")])
        if st["p1db_dbm"]:
            lp.hline(st["p1db_dbm"], color="tab:red",
                     label=f"P1dB {st['p1db_dbm']:+.1f} dBm")
        lp.add("output", freqs[0] / 1e9, m + pad)

        lv, sd, flagged = [m], [s], []
        for i, f in enumerate(freqs[1:], start=1):
            awg.duc_cw_step(float(f), channel=1)
            time.sleep(SETTLE_S)
            mi, si, ni = measure(scope, fs, float(f))
            lv.append(mi); sd.append(si)
            lp.add("output", f / 1e9, mi + pad)
            if ni:
                flagged.append((f, ni))
            print(f"    {f/1e9:.3f} GHz  {mi:+7.2f} dBm  ({mi+pad:+7.2f} out)"
                  + ("   " + "; ".join(ni) if ni else ""))
        lp.done()
        lv, sd = np.array(lv), np.array(sd)

        settings = {"sample_rate_hz": float(fs), "capture_s": CAPTURE_S,
                    "n_repeats": N_REPEATS, "settle_s": SETTLE_S,
                    "operating_vpp": float(vpp),
                    "fixed_vscale_v_per_div": float(vscale)}
    finally:
        for name, fn in (("awg.stop", awg.stop),
                         ("scope.restore", lambda: scope.restore(channel=SIG_CH))):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(stage_key, freqs, lv, sd, vpp, settings, flagged)
    return lv


def report(stage_key, freqs, lv, sd, vpp, settings, flagged):
    st = config.STAGES[stage_key]
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = st["band_hz"]
    ib = (freqs >= lo) & (freqs <= hi)
    out = lv + pad
    b = out[ib]

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  AWG {vpp:.4f} Vpp  ({20*np.log10(vpp/config.AWG_MAX_VPP):+.2f} dB "
          f"from the {config.AWG_MAX_VPP} Vpp ceiling)")
    print(f"  amp output over {lo/1e9:.3f}-{hi/1e9:.3f} GHz: "
          f"{b.max():+.2f} to {b.min():+.2f} dBm, FLATNESS {b.max()-b.min():.2f} dB")
    if st["p1db_dbm"]:
        print(f"  backoff from the datasheet P1dB ({st['p1db_dbm']:+.1f} dBm): "
              f"{st['p1db_dbm']-b.max():.2f} dB")

    resp = mc.load_scope_band_response()
    if resp is not None:
        awg = (np.interp(freqs, resp[0], resp[1])
               + 20 * np.log10(vpp / mc.AWG_OPERATING_AMP_VPP))
        g = (out - (awg - 0.5 - st["drive_pad_db"]))[ib]
        print(f"  implied gain {g.max():.2f} to {g.min():.2f} dB "
              f"(datasheet typ {st['gain_typ_db']:.1f}"
              + (f", min {st['gain_min_db']:.1f}" if st["gain_min_db"] else "")
              + f"), tilt {g.max()-g.min():.2f} dB")
        print("    INDIRECT -- assumes the stored AWG curve and a nominal "
              f"{st['drive_pad_db']:g} dB pad")

    print(f"\n  repeat sd {np.median(sd):.3f} dB median. This is back-to-back "
          f"acquisition noise only,")
    print(f"  NOT the measurement's uncertainty -- run-to-run scatter on this "
          f"bench is ~0.19 dB rms.")
    if flagged:
        print(f"\n  {len(flagged)} point(s) flagged:")
        for f, ni in flagged[:8]:
            print(f"    {f/1e9:.3f} GHz: {'; '.join(ni)}")

    print(f"\n  downstream, on datasheet typicals: feeds the HMC188 at "
          f"{b.max()-1.5-0.5:+.2f} dBm (needs {mc.HMC188_PIN_MIN_DBM:+.0f} to "
          f"{mc.HMC188_PIN_MAX_DBM:+.0f} dBm)")

    sess = chainlog.new_session(stage_key, "operating_point")
    p = chainlog.save(sess, "operating_point", stage_key, freqs,
                      {f"{vpp:.4f}vpp": lv}, settings, x_name="freqs_hz",
                      thru=False,
                      notes=f"single operating amplitude; levels are RAW scope "
                            f"readings, add {pad:g} dB for the amp output")
    print(f"\n  saved {p}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freqs / 1e9, out, "o-", ms=4, label=f"{vpp:.4f} Vpp")
    ax.axvspan(lo / 1e9, hi / 1e9, alpha=0.12, color="tab:green",
               label="chain band")
    if st["p1db_dbm"]:
        ax.axhline(st["p1db_dbm"], ls=":", color="tab:red",
                   label=f"datasheet P1dB {st['p1db_dbm']:+.1f} dBm")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("amp output (dBm)")
    ax.set_title(f"stage {stage_key}: {st['part']} at the operating point")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(sess, "operating_point.png")
    fig.savefig(png, dpi=130)
    print(f"  saved {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    v = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run(stage, v)
