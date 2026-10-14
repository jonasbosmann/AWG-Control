"""Measure one amplifier stage of the x8 chain: gain vs frequency, and gain
vs drive level (compression).

    AWG CH1 -> 3 dB pad -> ZX60-2522MA-S+ -> 10 dB pad -> scope     (stage A)

WHY A THRU REFERENCE
--------------------
Run this script TWICE. First with the amp replaced by a barrel/adapter:

    python x8_chain/run_stage_gain.py thru

then with the amp in place:

    python x8_chain/run_stage_gain.py amp

Gain is then (amp - thru), point by point. Every pad, cable, connector and
scope-calibration error in the path appears identically in both runs and
cancels exactly, so the gain number does not depend on knowing the AWG's
absolute output, the pads' true values, or the scope's absolute accuracy --
none of which are known to better than a few tenths of a dB. The only
residual is the adapter's own loss (~0.05-0.1 dB for an SMA barrel), which
biases gain HIGH by that amount.

That also makes the drive level absolute for free: the thru trace measures
the level at exactly the plane where the amp's input connector sits, so
    Pin = thru_reading + PAD_AT_INSTRUMENT_DB
with no reliance on the awg_level_scope curves at all.

WHAT IT PRODUCES
----------------
Two sweeps, because the cross product of both is a waste of bench time and
neither question needs it:

  1. FLATNESS -- full frequency sweep at every rung of the drive ladder that
     run_drive_level.py measured, ending at the level the experiment runs at.
     Gain flatness across 1.614-1.928 GHz adds directly to the AWG's own
     1.2 dB droop, and the sum is what has to fit the AMC's 7 dB window.
     Run run_drive_level.py FIRST: it closes the loop on the amp's actual
     output power, and this script reuses its amplitudes verbatim so that
     thru and amp are measured at identical settings.

  2. COMPRESSION -- full amplitude ladder at 3 spot frequencies. Gives P1dB in
     situ. This matters twice over: the design runs the amp 3 dB backed off to
     avoid AM-to-PM that would be multiplied by 192 downstream, and the
     datasheet gain spread (22.6 min vs 24.8 typ) is 2.2 dB of pure guesswork
     about where the HMC188's 10 dBm input floor lands.

Both are repeated N_REPEATS times per point and the run-to-run scatter is
reported. A single pass is not a result -- the AWG level scan showed 0.19 dB
rms of per-point scatter, which is the same size as several of the effects
being looked for here.

Run:  python x8_chain/run_stage_gain.py {thru|amp} [stage]
      NO_SHOW=1 to skip the blocking plot window.
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
from awg import AWG
from scope import Scope
# The phase-coverage and crest-factor guards were written and validated for
# the AWG level scan; the failure modes they catch (RMS biased by too few
# sample phases, clipped or non-sinusoidal record) are identical here.
from run_awg_level_scope import phase_coverage_warning, crest_warning


SIG_CH = 1
SETTLE_S = 0.20
CAPTURE_S = 2e-6
N_REPEATS = 2          # per point; scatter is reported, not hidden

# ── sweep 2: compression ──────────────────────────────────────────────────
# ~2 dB per step, 10 steps = 18 dB of drive range. At the top the amp should
# be at or just past P1dB; at the bottom it is unambiguously small-signal.
# This one deliberately DOES run to the AWG's ceiling -- reaching compression
# is the whole point of it, unlike the flatness sweep below.
AMPS_DRIVE_VPP = tuple(round(config.AWG_MAX_VPP * 10 ** (-2.0 * k / 20), 4)
                       for k in range(10))


def flatness_amps(stage_key):
    """The AWG amplitudes for the flatness sweep: the drive ladder that
    run_drive_level.py measured, one per output level, ending at the
    experiment's level.

    Taken from that script's saved session rather than recomputed, for two
    reasons. It closed the loop on the amp's actual OUTPUT power, so its
    amplitudes correspond to known output levels instead of to a datasheet
    guess. And the thru run and the amp run then use byte-identical
    amplitudes, which is what makes the (amp - thru) subtraction exact.

    The sweep must NOT run at the AWG's ceiling. Verified in simulation: an
    amp with 0.9 dB of real gain tilt, swept at 0.44/0.55 Vpp, reported
    0.32 dB of flatness and a mean gain 1.5 dB below the truth -- because at
    those levels it was compressed, and compression flattens tilt. The lowest
    rung of the ladder is the small-signal reference that catches this.

    Returns {label: vpp}, ascending.
    """
    st = config.STAGES[stage_key]
    for p in chainlog.find(stage_key, label_contains="drive_ladder", thru=False):
        d = chainlog.load(p)
        lv = d.get("instrument_settings", {}).get("level_vpp")
        if lv:
            print(f"  drive ladder from {p}")
            for k, v in sorted(lv.items(), key=lambda kv: kv[1]):
                print(f"    {k}  ->  {v:.4f} Vpp")
            return {k: float(v) for k, v in sorted(lv.items(),
                                                   key=lambda kv: kv[1])}

    # Fallback: reconstruct the ladder from datasheet gain.
    print(f"  ** no run_drive_level session found for stage {stage_key} -- "
          f"falling back to DATASHEET-estimated amplitudes.")
    print(f"     Run `python x8_chain/run_drive_level.py {stage_key}` first; "
          f"a flatness sweep at the wrong level measures the wrong thing. **")
    out = {}
    for lvl in config.out_levels_dbm():
        want = lvl - (st["gain_typ_db"] or 0.0) + st["drive_pad_db"]
        out[config.level_key(lvl)] = float(
            np.clip(config.vpp_from_dbm(want), 0.001, config.AWG_MAX_VPP))
    return out


def expected_at_instrument_dbm(stage_key, amp_vpp, thru):
    """Rough level at the scope, used only to pick a vertical scale.

    Deliberately uses TYPICAL gain and ignores the AWG's droop -- it only has
    to be right to a couple of dB to put the trace on screen, and the fill
    fraction is checked against the real reading afterwards.
    """
    st = config.STAGES[stage_key]
    gain = 0.0 if thru else (st["gain_typ_db"] or 0.0)
    return (config.dbm_from_vpp(amp_vpp) - st["drive_pad_db"] + gain
            - config.PAD_AT_INSTRUMENT_DB)


def set_scale_for(scope, dbm):
    vdiv = min(max(config.volts_per_div_for(dbm), 1e-3), 1.0)
    return scope.set_vertical(SIG_CH, vdiv)


def measure_point(scope, fs, freq_hz, repeats=N_REPEATS):
    """Measure one point `repeats` times. Returns (mean_dbm, sd_dbm, notes)."""
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
    vals = np.array(vals, dtype=float)
    return float(np.mean(vals)), float(np.std(vals)), notes


def check_fill(scope, dbm_measured, vdiv, stage_key):
    """Keep the trace in the part of the screen where an RMS reading is
    trustworthy. Rescales at most once per amplitude step, and says so, so a
    rescale can never be mistaken for a step in the curve."""
    frac = config.vpp_from_dbm(dbm_measured) / (vdiv * 10)
    if 0.25 <= frac <= 0.90:
        return vdiv, False
    new = set_scale_for(scope, dbm_measured)
    print(f"    (screen fill {frac*100:.0f}% -- rescaled to "
          f"{new*1e3:.4g} mV/div and re-measuring this amplitude)")
    return new, True


def sweep_flatness(awg, scope, fs, stage_key, thru, amps):
    """amps: {label: vpp}, ascending. Labels come from the drive ladder, so
    the thru run and the amp run produce matching series names."""
    freqs = config.band_freqs(stage_key)
    print(f"\n--- FLATNESS: {len(freqs)} points x {len(amps)} "
          f"levels x {N_REPEATS} repeats ---")
    series, sds = {}, {}
    for key, amp in amps.items():
        exp = expected_at_instrument_dbm(stage_key, amp, thru)
        print(f"\n  {key}: {amp:.4f} Vpp  (expect ~{exp:+.1f} dBm at the scope)")
        awg.duc_cw_setup(float(freqs[0]), amplitude_vpp=float(amp), channel=1)
        time.sleep(SETTLE_S)
        vdiv = set_scale_for(scope, exp)

        lv, sd = [], []
        for i, f in enumerate(freqs):
            if i:
                awg.duc_cw_step(float(f), channel=1)
            time.sleep(SETTLE_S)
            m, s, notes = measure_point(scope, fs, f)
            if i == 0:
                vdiv, rescaled = check_fill(scope, m, vdiv, stage_key)
                if rescaled:
                    m, s, notes = measure_point(scope, fs, f)
            lv.append(m); sd.append(s)
            print(f"    {f/1e9:.3f} GHz  {m:+7.2f} dBm  (sd {s:.2f})"
                  + ("   " + "; ".join(notes) if notes else ""))
        series[key] = np.array(lv)
        sds[key] = np.array(sd)
    return freqs, series, sds


def sweep_compression(awg, scope, fs, stage_key, thru):
    fspots = config.spot_freqs(stage_key)
    amps = np.array(AMPS_DRIVE_VPP)
    print(f"\n--- COMPRESSION: {len(amps)} drive levels x {len(fspots)} "
          f"frequencies x {N_REPEATS} repeats ---")
    series, sds = {}, {}
    for f in fspots:
        print(f"\n  {f/1e9:.3f} GHz")
        lv, sd = [], []
        for j, amp in enumerate(amps):
            # duc_cw_setup is the only path that applies a new :VOLT here, and
            # it carries a *RST, so it is re-run per amplitude rather than
            # writing :VOLT behind the DUC's back.
            awg.duc_cw_setup(float(f), amplitude_vpp=float(amp), channel=1)
            time.sleep(SETTLE_S)
            exp = expected_at_instrument_dbm(stage_key, float(amp), thru)
            vdiv = set_scale_for(scope, exp)
            m, s, notes = measure_point(scope, fs, f)
            vdiv, rescaled = check_fill(scope, m, vdiv, stage_key)
            if rescaled:
                m, s, notes = measure_point(scope, fs, f)
            lv.append(m); sd.append(s)
            print(f"    {amp:.4f} Vpp  {m:+7.2f} dBm  (sd {s:.2f})"
                  + ("   " + "; ".join(notes) if notes else ""))
        key = f"{f/1e9:.3f}ghz"
        series[key] = np.array(lv)
        sds[key] = np.array(sd)
    return amps, series, sds


def run(mode, stage_key="A"):
    thru = (mode == "thru")
    st = config.STAGES[stage_key]

    print("=" * 74)
    print(f"STAGE {stage_key}: {st['role']}")
    print(f"  {config.arrangement(stage_key, thru=thru)}")
    print("=" * 74)
    amps_flat = flatness_amps(stage_key)

    # Worst case: strongest AWG amplitude, max gain the datasheet allows.
    worst = (config.dbm_from_vpp(config.AWG_MAX_VPP) - st["drive_pad_db"]
             + (0.0 if thru else (st["gain_typ_db"] or 0.0)))
    ok, _ = config.safety_check(stage_key, worst)
    if not ok:
        print("\nABORTED on the safety check above.")
        return
    if st["instrument"] != "scope":
        print("\nThis stage is configured for the EXA, not the scope -- "
              "this script is scope-only.")
        return
    print()

    awg = AWG()
    scope = Scope()
    try:
        scope.setup(channel=SIG_CH, n_averages=1,
                    trigger_channel=SIG_CH, trigger_level=0.0)
        scope.set_active_channels([SIG_CH])
        fs = scope.set_max_sample_rate(CAPTURE_S)
        print(f"scope: {fs/1e9:.2f} GS/s, {CAPTURE_S*1e6:.1f} us record")

        settings = {"sample_rate_hz": float(fs), "capture_s": CAPTURE_S,
                    "n_repeats": N_REPEATS, "settle_s": SETTLE_S,
                    "vpp_method": "AC-RMS * 2*sqrt(2)",
                    "flatness_amps_vpp": amps_flat,
                    "compression_amps_vpp": list(AMPS_DRIVE_VPP)}

        f_ax, f_ser, f_sd = sweep_flatness(awg, scope, fs, stage_key, thru,
                                           amps_flat)
        a_ax, a_ser, a_sd = sweep_compression(awg, scope, fs, stage_key, thru)
    finally:
        for name, fn in (("awg.stop", awg.stop),
                         ("scope.restore", lambda: scope.restore(channel=SIG_CH))):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    sess = chainlog.new_session(stage_key, f"{mode}_gain")
    p1 = chainlog.save(sess, f"{mode}_flatness", stage_key, f_ax, f_ser,
                       settings, x_name="freqs_hz", thru=thru,
                       notes="gain vs frequency; pair with the matching "
                             "thru/amp run of the same session")
    p2 = chainlog.save(sess, f"{mode}_drive", stage_key, a_ax, a_ser,
                       settings, x_name="amps_vpp", thru=thru,
                       notes="gain vs drive level for P1dB")
    print(f"\nsaved {p1}\n      {p2}")

    med = np.median(np.concatenate([v for v in f_sd.values()]))
    print(f"\nper-point repeat scatter (flatness sweep): median sd {med:.2f} dB "
          f"over {N_REPEATS} repeats")

    if thru:
        print("\nTHRU reference stored. Now fit the amplifier and run:")
        print(f"    python x8_chain/run_stage_gain.py amp {stage_key}")
    else:
        analyse(stage_key, sess, f_ax, f_ser, a_ax, a_ser)


# ── analysis ──────────────────────────────────────────────────────────────
def analyse(stage_key, sess, f_ax, f_ser, a_ax, a_ser):
    """Subtract the most recent thru reference and report gain / P1dB."""
    st = config.STAGES[stage_key]
    lo, hi = st["band_hz"]
    pad = config.PAD_AT_INSTRUMENT_DB

    tf = chainlog.find_thru(stage_key, label_contains="flatness")
    td = chainlog.find_thru(stage_key, label_contains="drive")
    if not tf or not td:
        print("\nNo thru reference found -- run `thru` mode first. "
              "Levels above are raw scope readings only.")
        return
    thru_f, thru_d = chainlog.load(tf[0]), chainlog.load(td[0])
    print("\n" + "=" * 74)
    print("GAIN  (amp - thru, every pad and cable cancelling)")
    print("=" * 74)
    print(f"  thru references used:\n    {tf[0]}\n    {td[0]}")
    print("  ** confirm nothing but the DUT changed between those runs and "
          "this one. **")

    # --- gain vs frequency -----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    in_band = (f_ax >= lo) & (f_ax <= hi)
    flat = {}
    for key, amp_lv in f_ser.items():
        tk = chainlog.series_key(key)
        if tk not in thru_f or not np.allclose(thru_f["freqs_hz"], f_ax):
            print(f"  {key}: thru has no matching series/axis -- skipped")
            continue
        gain = amp_lv - thru_f[tk]
        b = gain[in_band]
        flat[key] = b
        print(f"\n  {key}")
        print(f"    gain in band {lo/1e9:.3f}-{hi/1e9:.3f} GHz : "
              f"{b.max():.2f} to {b.min():.2f} dB, FLATNESS {b.max()-b.min():.2f} dB")
        ds = f"    datasheet typ {st['gain_typ_db']:.1f} dB"
        if st["gain_min_db"]:
            ds += f", min {st['gain_min_db']:.1f} dB"
        print(ds)
        print(f"    measured mean {b.mean():.2f} dB "
              f"({b.mean()-st['gain_typ_db']:+.2f} dB vs typ)")
        axes[0].plot(f_ax / 1e9, gain, "o-", ms=3, label=key)
    axes[0].axvspan(lo / 1e9, hi / 1e9, alpha=0.12, color="tab:green",
                    label="chain band")
    axes[0].axhline(st["gain_typ_db"], ls="--", lw=1, color="k", alpha=0.5,
                    label=f"datasheet typ {st['gain_typ_db']:.1f} dB")
    axes[0].set_xlabel("frequency (GHz)")
    axes[0].set_ylabel("gain (dB)")
    axes[0].set_title(f"stage {stage_key}: {st['part']} gain")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    # The ladder's rungs are a matched set: if mean gain falls off as the level
    # rises, the upper rungs are compressing, and BOTH the tilt and the gain
    # measured there are understated -- compression flattens tilt. That is the
    # single most misreadable result this script can produce, so it is called
    # out rather than left to be spotted among the numbers above.
    if len(flat) >= 2:
        order = list(f_ser)              # ascending, as swept
        base = order[0]
        g0 = flat[base].mean()
        print(f"\n  gain vs level, against the small-signal rung ({base}):")
        worst = 0.0
        for k in order:
            d = flat[k].mean() - g0
            fl = flat[k].max() - flat[k].min()
            worst = min(worst, d)
            print(f"    {k:>14}  mean gain {flat[k].mean():6.2f} dB "
                  f"({d:+.2f} vs base)   flatness {fl:.2f} dB")
        if worst < -0.3:
            print(f"    ** compressing by up to {-worst:.2f} dB on the upper "
                  f"rungs. Their tilt and gain are both understated. The "
                  f"experiment's level is NOT in the linear region -- back it "
                  f"off, or accept that this stage runs nonlinear. **")
        else:
            print("    flat across the ladder -- the experiment's level is in "
                  "the linear region, so its flatness is the real flatness")

    # --- compression ------------------------------------------------------
    print("\n" + "-" * 74)
    print("COMPRESSION")
    print("-" * 74)
    for key, amp_lv in a_ser.items():
        tk = chainlog.series_key(key)
        if tk not in thru_d or not np.allclose(thru_d["amps_vpp"], a_ax):
            print(f"  {key}: thru has no matching series/axis -- skipped")
            continue
        p_in = thru_d[tk] + pad          # level at the amp's input connector
        p_out = amp_lv + pad             # level at its output connector
        gain = p_out - p_in
        # Small-signal gain from the three weakest drives, where compression
        # is not in play; P1dB is where gain has fallen 1 dB below that.
        order = np.argsort(p_in)
        g0 = float(np.mean(gain[order[:3]]))
        comp = g0 - gain
        print(f"\n  {key}   small-signal gain {g0:.2f} dB "
              f"(mean of the 3 lowest drives)")
        for pi, po, g, c in sorted(zip(p_in, p_out, gain, comp)):
            print(f"    Pin {pi:+7.2f}  Pout {po:+7.2f}  G {g:6.2f}  "
                  f"compression {c:5.2f} dB")
        if comp.max() >= 1.0:
            i = np.argsort(p_in)
            p1 = float(np.interp(1.0, comp[i], p_out[i]))
            print(f"    -> P1dB (output) {p1:+.2f} dBm", end="")
            if st["p1db_dbm"]:
                print(f"   datasheet {st['p1db_dbm']:+.1f} dBm "
                      f"({p1-st['p1db_dbm']:+.2f} dB)")
            else:
                print()
        else:
            print(f"    -> never reached 1 dB compression "
                  f"(max {comp.max():.2f} dB at {p_out.max():+.2f} dBm out) -- "
                  f"P1dB is above the drive available, which is the safe side")
        axes[1].plot(p_out, gain, "o-", ms=4, label=key)
    axes[1].set_xlabel("output level (dBm)")
    axes[1].set_ylabel("gain (dB)")
    axes[1].set_title("compression")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "gain.png")
    fig.savefig(png, dpi=130)
    print(f"\nsaved {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("thru", "amp"):
        print(__doc__)
        print("\nusage: python x8_chain/run_stage_gain.py {thru|amp} [stage]")
        sys.exit(1)
    run(mode, sys.argv[2] if len(sys.argv) > 2 else "A")
