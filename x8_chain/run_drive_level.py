"""Measure a stage's amplifier across a ladder of output levels, ending at the
level the experiment actually runs at.

    AWG CH1 -> 3 dB pad -> ZX60-2522MA-S+ -> 10 dB pad -> scope     (stage A)

The target is defined AT THE AMP OUTPUT, and the 10 dB pad is the only thing
between there and the scope, so it is measured directly:

    P_out = scope_reading + PAD_AT_INSTRUMENT_DB

Nothing here depends on knowing the AWG's absolute output, the amp's gain, or
the 3 dB pad's true value. Those all get measured as a by-product; none of
them are inputs.

WHAT IT DOES, PER LEVEL
-----------------------
For each rung of config.OUT_LEVEL_BACKOFF_DB (10, 6, 3, 0 dB below target):

  1. Converge the AWG amplitude onto that output level, closed-loop, using
     three spot frequencies. The correction applied is the measured error, so
     an error at one rung does not propagate to the next.
  2. Sweep the full band at the converged amplitude and save it.
  3. Predict the next rung's level from the one just measured and check it
     against the scope's input limit BEFORE commanding it. The instrument is
     never asked to take a level that has not already been shown to be safe by
     measurement.

The result is a family of curves, not a single point. That is what makes it a
characterisation rather than a setting: gain and flatness both move with drive,
and the question worth answering is whether the experiment's level is still in
the linear region. The lowest rung doubles as the small-signal reference, and
the per-level gain change against it is compression.

WHAT TO EXPECT AT STAGE A
-------------------------
Target +15 dBm out -> +5 dBm at the scope (limit +20.2 dBm, 15 dB of margin).
Datasheet typical gain 24.8 dB puts the AWG near 0.34 Vpp -- about 4 dB below
its 0.55 Vpp ceiling. If the measured amplitude comes out far from that,
believe the measurement and find out why before going further.

A NOTE ON +15 dBm AS AN OPERATING POINT
---------------------------------------
It is a good level to MEASURE at -- 5.3 dB below the amp's +20.3 dBm P1dB, so
well clear of AM-to-PM that the chain would multiply by 192. But carried
through the rest of the chain on datasheet typicals it lands +10.9 dBm at the
ZX90-2-24-S+, whose input floor is +11.0 dBm. The design point of +16.13 dBm
leaves exactly 1.0 dB there and +15 dBm spends all of it. Measure here, but do
not adopt +15 dBm as the chain's operating level until the stages in between
have been measured and the real numbers say it fits.

Run:  python x8_chain/run_drive_level.py [stage] [target_dbm]
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
from awg import AWG
from scope import Scope
from run_awg_level_scope import phase_coverage_warning, crest_warning


SIG_CH = 1
SETTLE_S = 0.20
CAPTURE_S = 2e-6

N_REPEATS_SET = 3        # while converging -- the level is the deliverable
N_REPEATS_SWEEP = 2      # during the band sweep, where there are 40+ points

CONVERGE_TOL_DB = 0.15   # stop correcting once the band max is this close
MAX_TRIM_ITERS = 4

SCOPE_MARGIN_DB = 3.0    # never command a level predicted within this of the
                          # scope's 50 ohm input limit

FIXED_VSCALE = True
FIXED_VSCALE_FILL = 0.80
# Use ONE vertical scale for every rung of the ladder, sized so the TOP rung
# fills FIXED_VSCALE_FILL of the screen.
#
# Why this matters more than it looks. The compression figure this script
# reports is a comparison BETWEEN rungs, so anything that changes between
# rungs contaminates it. Letting the vertical scale follow the level does
# exactly that: the 2026-07-31 run stepped 51 -> 81 -> 114 -> 161 mV/div,
# four different internal attenuator paths, and a range-to-range gain
# difference of 0.1-0.3 dB is entirely ordinary. That is the same size as the
# 0.33 dB of "compression" it reported, so the two could not be separated.
#
# With one scale the error is common to every rung and cancels in the
# comparison outright, rather than having to be subtracted away. It also
# makes the scope's own quantisation of the V/div setting irrelevant --
# whatever value it actually snaps to, it is the same for all four rungs.
#
# The cost is screen fill at the bottom rung: a 10 dB ladder at 80% fill on
# top leaves ~25% at the bottom, which is the lower edge of where an
# RMS-derived level stays trustworthy. Widening the ladder past ~10 dB would
# need this turned off (and the range error handled some other way).
#
# This is the same technique run_awg_level_scope.py uses, and for the same
# reason: a rescale must never be able to masquerade as a step in the data.


def awg_dbm_at(freq_hz, vpp):
    """Best available estimate of the AWG's own output, for a STARTING guess.

    Uses the averaged scope-measured band curve (2 runs, 0.19 dB rms per-point
    scatter) scaled by the commanded amplitude ratio. The 0.50 -> 0.55 Vpp step
    measured +0.824 dB against +0.83 expected with 0.010 dB sd, so scaling by
    20*log10(vpp/0.55) is sound. Falls back to the ideal-sine value if no scan
    is on disk. Only ever used to pick the first amplitude and a vertical
    scale -- every reported number comes from the measurement itself.
    """
    resp = mc.load_scope_band_response()
    ratio_db = 20 * np.log10(vpp / mc.AWG_OPERATING_AMP_VPP)
    if resp is None:
        return config.dbm_from_vpp(vpp)
    return float(np.interp(freq_hz, resp[0], resp[1])) + ratio_db


def sweep(scope, awg, fs, freqs, vpp, repeats, label=""):
    """Measure the scope level at every frequency for one AWG amplitude.

    Returns (levels_dbm, sds_dbm, notes_per_point).
    """
    lv, sd, allnotes = [], [], []
    for i, f in enumerate(freqs):
        # duc_cw_setup carries a *RST and is the only path that applies a new
        # :VOLT under DUC, so a new amplitude means a fresh setup; within one
        # amplitude the NCO is just retuned.
        if i == 0:
            awg.duc_cw_setup(float(f), amplitude_vpp=float(vpp), channel=1)
        else:
            awg.duc_cw_step(float(f), channel=1)
        time.sleep(SETTLE_S)

        vals, notes = [], []
        for _ in range(repeats):
            vpp_meas, _t, v = scope.measure_vpp(channel=SIG_CH, settle=0.05)
            vals.append(config.dbm_from_vpp(vpp_meas) if vpp_meas > 0 else np.nan)
            w = crest_warning(v)
            if w and w not in notes:
                notes.append(w)
        w = phase_coverage_warning(float(f), fs)
        if w:
            notes.append(w)
        lv.append(float(np.mean(vals)))
        sd.append(float(np.std(vals)))
        allnotes.append(notes)
    return np.array(lv), np.array(sd), allnotes


def converge(scope, awg, fs, spots, vpp, want_read, pad):
    """Trim the AWG amplitude until the band maximum sits at want_read.

    Returns (vpp, reached_ceiling). Sets on the band MAXIMUM so that nothing
    in the band exceeds the requested level.
    """
    for trim in range(MAX_TRIM_ITERS):
        lv, sd, notes = sweep(scope, awg, fs, spots, vpp, N_REPEATS_SET)
        for f, m, s, n_ in zip(spots, lv, sd, notes):
            print(f"      {f/1e9:.3f} GHz  scope {m:+7.2f} dBm (sd {s:.2f})"
                  f"  -> out {m+pad:+7.2f} dBm"
                  + ("   " + "; ".join(n_) if n_ else ""))
        err = want_read - lv.max()
        if abs(err) <= CONVERGE_TOL_DB:
            print(f"      on level ({err:+.2f} dB) at {vpp:.4f} Vpp")
            return vpp, False
        new_vpp = float(np.clip(vpp * 10 ** (err / 20), 0.001, config.AWG_MAX_VPP))
        if new_vpp >= config.AWG_MAX_VPP - 1e-9 and err > 0:
            print(f"      ** {err:+.2f} dB short and already at the AWG's "
                  f"{config.AWG_MAX_VPP} Vpp ceiling -- this level is not "
                  f"reachable with the {config.STAGES['A']['drive_pad_db']:g} dB "
                  f"drive pad in place. **")
            return new_vpp, True
        if trim < MAX_TRIM_ITERS - 1:
            print(f"      off by {err:+.2f} dB -> retrimming to {new_vpp:.4f} Vpp")
        vpp = new_vpp
    print(f"      ** did not converge in {MAX_TRIM_ITERS} trims -- using "
          f"{vpp:.4f} Vpp; the level below is what was actually measured **")
    return vpp, False


def run(stage_key="A", target_out_dbm=None):
    st = config.STAGES[stage_key]
    pad = config.PAD_AT_INSTRUMENT_DB
    target = config.TARGET_OUT_DBM if target_out_dbm is None else target_out_dbm
    levels = config.out_levels_dbm(target)
    spots = config.spot_freqs(stage_key)
    band = config.band_freqs(stage_key)
    scope_lim = config.scope_max_dbm()

    print("=" * 74)
    print(f"DRIVE LADDER -- stage {stage_key}: {st['role']}")
    print(f"  {config.arrangement(stage_key)}")
    print("=" * 74)
    print(f"  levels at the amp output: "
          + ", ".join(f"{l:+.1f}" for l in levels) + " dBm")
    print(f"  the last is the experiment's level; the first is the "
          f"small-signal reference")
    print(f"  -> at the scope, after the {pad:g} dB pad: "
          f"{levels[0]-pad:+.1f} to {levels[-1]-pad:+.1f} dBm")
    print(f"  scope 50 ohm limit {scope_lim:+.2f} dBm  -> "
          f"{scope_lim - (levels[-1]-pad):.1f} dB of margin at the top level")
    print(f"  band sweep: {len(band)} points, "
          f"{band[0]/1e9:.3f}-{band[-1]/1e9:.3f} GHz")
    if levels[-1] - pad > scope_lim - SCOPE_MARGIN_DB:
        print("\n  ** top level is too close to the scope's limit -- ABORT **")
        return
    if st["instrument"] != "scope":
        print("\n  this stage is configured for the EXA; this script is scope-only")
        return

    # Starting amplitude for the lowest rung, from datasheet gain. Refined by
    # measurement at every rung, so an error here costs one extra trim.
    want_awg = levels[0] - (st["gain_typ_db"] or 0.0) + st["drive_pad_db"]
    vpp = config.AWG_MAX_VPP * 10 ** (
        (want_awg - awg_dbm_at(float(spots[0]), config.AWG_MAX_VPP)) / 20)
    vpp = float(np.clip(vpp, 0.001, config.AWG_MAX_VPP))
    print(f"\n  starting at {vpp:.4f} Vpp (datasheet-gain estimate)\n")

    series, sds, level_vpp, achieved = {}, {}, {}, {}
    awg = AWG()
    scope = Scope()
    try:
        scope.setup(channel=SIG_CH, n_averages=1,
                    trigger_channel=SIG_CH, trigger_level=0.0)
        scope.set_active_channels([SIG_CH])
        fs = scope.set_max_sample_rate(CAPTURE_S)
        print(f"scope: {fs/1e9:.2f} GS/s, {CAPTURE_S*1e6:.1f} us record")

        vscale = None
        if FIXED_VSCALE:
            vscale = min(max(config.volts_per_div_for(
                levels[-1] - pad, fill_frac=FIXED_VSCALE_FILL), 1e-3), 1.0)
            scope.set_vertical(SIG_CH, vscale)
            print(f"scope: vertical FIXED at {vscale*1e3:.4g} mV/div for every "
                  f"rung -- no rescale between levels, so a range change cannot "
                  f"show up as compression")
            for l_ in levels:
                print(f"       {l_:+5.1f} dBm out -> "
                      f"{config.vpp_from_dbm(l_-pad)/(vscale*10)*100:4.0f}% "
                      f"of full screen")

        for i, lvl in enumerate(levels):
            want_read = lvl - pad
            print(f"\n  === level {lvl:+.1f} dBm out "
                  f"({want_read:+.1f} dBm at the scope), "
                  f"{target-lvl:g} dB below target ===")
            if not FIXED_VSCALE:
                scope.set_vertical(SIG_CH,
                                   min(max(config.volts_per_div_for(want_read), 1e-3), 1.0))

            vpp, ceiling = converge(scope, awg, fs, spots, vpp, want_read, pad)

            print(f"    sweeping the band at {vpp:.4f} Vpp ...")
            lv, sd, notes = sweep(scope, awg, fs, band, vpp, N_REPEATS_SWEEP)
            flagged = [(f, n) for f, n in zip(band, notes) if n]
            for f, n in flagged[:6]:
                print(f"      {f/1e9:.3f} GHz: {'; '.join(n)}")
            if len(flagged) > 6:
                print(f"      ... and {len(flagged)-6} more flagged points")

            key = config.level_key(lvl)
            series[key], sds[key] = lv, sd
            level_vpp[key], achieved[key] = float(vpp), float(lv.max() + pad)
            print(f"    band {lv.max()+pad:+.2f} to {lv.min()+pad:+.2f} dBm out, "
                  f"spread {lv.max()-lv.min():.2f} dB, "
                  f"repeat sd {np.median(sd):.2f} dB")

            if ceiling:
                print("    stopping the ladder here -- the AWG cannot go higher.")
                break
            if i + 1 < len(levels):
                step = levels[i + 1] - lvl
                pred = lv.max() + step
                print(f"    next level {levels[i+1]:+.1f} dBm out, predicted "
                      f"{pred:+.2f} dBm at the scope (limit {scope_lim:+.2f})")
                if pred > scope_lim - SCOPE_MARGIN_DB:
                    print(f"    ** predicted level is within {SCOPE_MARGIN_DB:g} dB "
                          f"of the scope limit -- STOPPING the ladder here. **")
                    break
                vpp = float(np.clip(vpp * 10 ** (step / 20), 0.001,
                                    config.AWG_MAX_VPP))

        settings = {"sample_rate_hz": float(fs), "capture_s": CAPTURE_S,
                    "n_repeats_set": N_REPEATS_SET,
                    "n_repeats_sweep": N_REPEATS_SWEEP,
                    "settle_s": SETTLE_S,
                    "target_out_dbm": target,
                    "out_levels_dbm": list(levels),
                    "level_vpp": level_vpp,
                    "achieved_out_dbm": achieved,
                    "fixed_vscale_v_per_div": vscale,
                    "vscale_note": ("one scale for all rungs" if vscale else
                                    "RESCALED PER RUNG -- range error is "
                                    "confounded with compression")}
    finally:
        for name, fn in (("awg.stop", awg.stop),
                         ("scope.restore", lambda: scope.restore(channel=SIG_CH))):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    if not series:
        print("\nno levels completed.")
        return
    report(stage_key, band, series, sds, level_vpp, achieved, target, settings)
    return level_vpp


def report(stage_key, band, series, sds, level_vpp, achieved, target, settings):
    st = config.STAGES[stage_key]
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = st["band_hz"]
    in_band = (band >= lo) & (band <= hi)
    keys = list(series)                     # ascending, as measured
    base = keys[0]                          # lowest rung = small-signal ref

    print("\n" + "=" * 74)
    print("RESULT -- amplifier output vs drive level")
    print("=" * 74)
    print(f"  {'requested':>10} {'AWG':>9} {'achieved':>9} {'in-band':>9} "
          f"{'flatness':>9} {'gain vs':>9}")
    print(f"  {'dBm out':>10} {'Vpp':>9} {'max dBm':>9} {'min dBm':>9} "
          f"{'dB':>9} {base:>9}")
    comp = {}
    for k in keys:
        b = series[k][in_band] + pad
        # Compression relative to the lowest rung.
        #
        # With FIXED_VSCALE on, the scope's range-to-range gain error is
        # common to every rung and drops out here. What does NOT drop out is
        # the AWG's own amplitude linearity: this assumes its output is
        # exactly proportional to commanded Vpp across the ladder. That was
        # verified over the 0.50 -> 0.55 Vpp step (+0.824 measured vs +0.830
        # expected, 0.010 dB sd) -- but this ladder runs at 0.10-0.33 Vpp,
        # a different and unmeasured part of the range.
        #
        # An AWG-straight-into-scope sweep at these same amplitudes settles
        # it; until then the figure below is an UPPER BOUND on the
        # amplifier's own compression.
        d = ((series[k][in_band] - series[base][in_band])
             - 20 * np.log10(level_vpp[k] / level_vpp[base]))
        comp[k] = float(np.mean(d))
        req = float(k.replace("out", "").replace("dBm", ""))
        print(f"  {req:>+10.1f} {level_vpp[k]:>9.4f} {b.max():>+9.2f} "
              f"{b.min():>+9.2f} {b.max()-b.min():>9.2f} "
              f"{comp[k]:>+9.2f}")

    print(f"\n  AWG amplitude for the experiment's {target:+.1f} dBm: "
          f"{level_vpp[keys[-1]]:.4f} Vpp")
    print(f"    ({config.AWG_MAX_VPP} Vpp ceiling -> "
          f"{20*np.log10(level_vpp[keys[-1]]/config.AWG_MAX_VPP):+.2f} dB "
          f"of AWG headroom left)")
    if st["p1db_dbm"]:
        print(f"    backoff from the datasheet P1dB ({st['p1db_dbm']:+.1f} dBm): "
              f"{st['p1db_dbm']-achieved[keys[-1]]:.2f} dB")

    worst = min(comp.values())
    print(f"\n  compression at the top level: {comp[keys[-1]]:+.2f} dB "
          f"relative to {base}")
    print(f"    scope range error: {'removed (one fixed vertical scale)' if FIXED_VSCALE else '** CONFOUNDED -- rescaled per rung **'}")
    print(f"    AWG amplitude linearity over {config.vpp_from_dbm(0)*0+min(level_vpp.values()):.4f}"
          f"-{max(level_vpp.values()):.4f} Vpp: NOT yet measured, so this is an "
          f"UPPER BOUND on the amp's own compression")
    if worst < -0.3:
        k = min(comp, key=comp.get)
        print(f"    ** the amp is compressing by {-worst:.2f} dB at {k}. Gain "
              f"AND flatness are understated there -- compression flattens "
              f"tilt. Back the experiment's level off, or accept that this "
              f"stage runs nonlinear. **")
    else:
        print("    linear across the whole ladder -- the experiment's level "
              "is in the linear region, so its flatness is the real flatness")

    fspread = {k: float((series[k][in_band]).max() - (series[k][in_band]).min())
               for k in keys}
    print(f"\n  in-band flatness {min(fspread.values()):.2f} to "
          f"{max(fspread.values()):.2f} dB across the ladder")
    print(f"  (the AMC's input window is {config.AMC_WINDOW_DB:g} dB wide and "
          f"this stage's tilt adds to the AWG's own ~1.2 dB droop)")
    print(f"\n  downstream, on datasheet typicals: the top level feeds the "
          f"HMC188 at {achieved[keys[-1]]-1.5-0.5:+.2f} dBm "
          f"(needs {mc.HMC188_PIN_MIN_DBM:+.0f} to "
          f"{mc.HMC188_PIN_MAX_DBM:+.0f} dBm)")

    sess = chainlog.new_session(stage_key, "drive_ladder")
    settings = dict(settings, operating_vpp=level_vpp[keys[-1]],
                    compression_vs_base_db=comp)
    p = chainlog.save(sess, "drive_ladder", stage_key, band, series, settings,
                      x_name="freqs_hz", thru=False,
                      notes=f"closed-loop ladder to {target:+.1f} dBm at the amp "
                            f"output; one full band sweep per level; levels are "
                            f"RAW scope readings, add {pad:g} dB")
    print(f"\n  saved {p}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k in keys:
        axes[0].plot(band / 1e9, series[k] + pad, "o-", ms=3,
                     label=f"{k}  ({level_vpp[k]:.3f} Vpp)")
    axes[0].axvspan(lo / 1e9, hi / 1e9, alpha=0.12, color="tab:green",
                    label="chain band")
    if st["p1db_dbm"]:
        axes[0].axhline(st["p1db_dbm"], ls=":", color="tab:red",
                        label=f"datasheet P1dB {st['p1db_dbm']:+.1f} dBm")
    axes[0].set_xlabel("frequency (GHz)")
    axes[0].set_ylabel("amp output (dBm)")
    axes[0].set_title(f"stage {stage_key}: {st['part']} output vs drive")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)

    x = [achieved[k] for k in keys]
    axes[1].plot(x, [comp[k] for k in keys], "o-", ms=6)
    axes[1].axhline(0, ls="--", lw=1, color="k", alpha=0.5)
    axes[1].axhline(-1.0, ls=":", color="tab:red", label="1 dB compression")
    axes[1].axvline(target, ls="--", color="tab:green",
                    label=f"experiment level {target:+.1f} dBm")
    axes[1].set_xlabel("achieved output (dBm)")
    axes[1].set_ylabel(f"gain change vs {base} (dB)")
    axes[1].set_title("compression across the ladder")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "drive_ladder.png")
    fig.savefig(png, dpi=130)
    print(f"  saved {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    tgt = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run(stage, tgt)
