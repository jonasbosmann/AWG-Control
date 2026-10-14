"""Per-segment level calibration -- one amplitude per chirp segment.

WHY THIS REPLACED THE CORRECTION TABLE
--------------------------------------
The full AWG band multiplied by 192 is a 60 GHz chirp at 310-370 GHz, which
nothing in the chain is flat across. Segmenting is standard CP-FTMW practice
anyway: at fixed chirp duration a narrower sweep is slower through each
transition, and polarization goes as mu*E/sqrt(dnu/dt).

A 1 GHz output segment is 5.21 MHz at the AWG. Measured 2026-07-31, the band
ripple across a window that narrow is 13.3 milli-dB peak to peak AND
essentially a straight line -- 0.15% in field amplitude, no paired echoes
(those need periodic ripple, not a tilt), and below the 5.2 milli-dB floor the
AWG's own :VOLT granularity imposes. There is nothing worth correcting inside
a segment.

What does NOT go away is the variation BETWEEN segments. The band's 1.3 dB of
tilt stops being a wobble inside one chirp and becomes 60 chirps each excited
at a different level. That is the thing that makes line intensities
non-comparable across the spectrum, and intensity ratios are how populations
and assignments get checked. So: one scalar per segment, measured once.

WHAT IT MEASURES
----------------
Each segment is calibrated with CW tones rather than with its own chirp. That
is justified by the same 13 milli-dB result -- the response is flat across a
segment, so a tone at the centre stands in for the chirp. N_PER_SEG points are
taken across each segment anyway, which costs almost nothing and VERIFIES that
flatness segment by segment instead of assuming it. If any segment comes back
with a large internal spread, the assumption has failed there and the report
says so.

Closed loop, same structure as run_preemph.py: a dB commanded at the AWG does
not arrive as a dB at the measurement plane, so the loop measures that ratio
for itself and needs no prior knowledge of the chain.

LIMIT, STATED UP FRONT
----------------------
This flattens to whatever plane the analyzer is on -- at best the AMC INPUT
(12.9-15.4 GHz, inside the EXA's range). The AMC's own response across
310-370 GHz, and the horn and optics after it, are beyond the bench and are
almost certainly less flat than anything here. Final intensity normalisation
will need a power measurement at 310 GHz or the molecular signal itself.

Run:  python x8_chain/run_segments.py [segment_GHz] [n_iters]
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
from awg import AWG, widen_for_flat_plateau
from specan import SpecAn
import run_am_pm as R

STAGE = "A"
TONE, ELEMENT, PATH = R.TONE, R.ELEMENT, R.PATH

AMC_MULT = 24.0                     # AMC input -> 310-370 GHz
CHAIN_MULT = 8.0 * AMC_MULT         # AWG -> output
OUT_LO_HZ = config.AMC_IN_LO_HZ * AMC_MULT
OUT_HI_HZ = config.AMC_IN_HI_HZ * AMC_MULT

SEGMENT_GHZ = 1.0                   # at the OUTPUT, 310-370 GHz
N_PER_SEG = 3                       # points across each segment
N_ITERS = 3
N_AVG = 4                           # -> ~0.0016 dB read noise, ample for a
                                     # level calibration
SPAN_HZ = 1e6
RBW_HZ = 100e3
N_POINTS = 801
SETTLE_S = 0.15
EXA_ATTEN_DB = 20
REF_LEVEL_DBM = 10

BASE_VPP = 0.3482
PROBE_DB = 0.15
# A deliberate, uniform, known step taken BEFORE any correction, purely to
# measure r. Guessing r instead is what limited the first version: a damped
# first correction is only stable while damping*r < 2, so a guess of 1.0 with
# damping 0.7 diverged for anything past r = 2.86. That covers today's plane
# (r = 0.9-2.4 after one doubler) but NOT a plane after two, where r compounds
# toward 4-6. A probe has no such ceiling. 0.15 dB is 25x the :VOLT
# quantisation and 100x the read noise, so the ratio is well conditioned, and
# a uniform step does not change the spread being minimised.
DAMPING = 0.8
# err_new = err*(1 - damping*r/r_measured), so this is stable even if r is
# wrong by 2x in either direction.
R_CLAMP = (0.3, 8.0)
# Later iterations, where the step is err/r_measured. Stable even if r is
# underestimated by 2x; the clamps below keep it inside that.
R_MIN_STEP_DB = 0.05
# A commanded step smaller than this is NOT used to estimate r. :VOLT
# quantises at ~6 milli-dB (measured), so a small commanded step can be
# delivered as nothing at all, giving a ratio near zero -- and dividing the
# next correction by that ratio is exactly how the 2026-07-31 first run
# diverged (0.207 dB improving, then 0.656 dB after r hit its 0.30 clamp).
TOL_DB = 0.01
SEG_FLAT_WARN_DB = 0.05             # a segment whose internal spread exceeds
                                     # this breaks the "flat inside a segment"
                                     # assumption this whole approach rests on


MIN_AMP_BACKOFF_DB = 1.5
FILTER_IL_DB = (0.64, 1.50)     # PE87FL1012, measured at the band edges


def amp_headroom_db(cen_awg):
    """How much MORE the AWG may be raised before the stage-A amp runs out.

    Returns (max_correction_db per segment, amp_output_at_base) or (None, None)
    if there is no measured drive curve to work from.

    This matters the moment the amp is back in the path. Flattening a band by
    RAISING its weak points is the right choice when the AWG is alone -- it
    keeps all the power. With the amp in, the weak segments are exactly the
    ones that get pushed toward P1dB, and the chain was already sitting at
    2.0 dB of backoff. Equalising upward by the measured 3.6 dB of doubler-
    output spread would take that to ~1.6 dB, which is past where this
    campaign has been willing to run.
    """
    try:
        from run_doubler import known_drive
        vpp0, dfreqs, drive, _p = known_drive()
        if vpp0 is None:
            return None, None
    except Exception:
        return None, None
    lo, hi = config.STAGES[STAGE]["band_hz"]
    il = np.interp(cen_awg, [lo, hi], FILTER_IL_DB)
    # drive is the FILTER OUTPUT at vpp0; the amp sits one filter earlier
    amp_out = (np.interp(cen_awg, dfreqs, drive)
               + 20 * np.log10(BASE_VPP / vpp0) + il)
    import multiplier_chain as mc
    return (mc.ZX602522MA_P1DB_DBM - MIN_AMP_BACKOFF_DB) - amp_out, amp_out


def segment_plan(segment_ghz=SEGMENT_GHZ):
    """(centres_awg_hz, edges_awg_hz, centres_out_hz) for the output band."""
    w = segment_ghz * 1e9
    n = int(np.ceil((OUT_HI_HZ - OUT_LO_HZ) / w))
    edges_out = OUT_LO_HZ + w * np.arange(n + 1)
    edges_out[-1] = min(edges_out[-1], OUT_HI_HZ)
    lo_out, hi_out = edges_out[:-1], edges_out[1:]
    return (0.5 * (lo_out + hi_out) / CHAIN_MULT,
            np.stack([lo_out, hi_out]) / CHAIN_MULT,
            0.5 * (lo_out + hi_out))


def measure_segments(sa, awg, edges, vpps, n_avg=N_AVG, lp=None, tag=""):
    """Mean level per segment, plus the spread across each segment.

    Returns (level_dbm_raw, internal_spread_db, read_noise_db).
    """
    n = edges.shape[1]
    lv = np.empty(n); sp = np.empty(n); nz = np.empty(n)
    for i in range(n):
        fs = np.linspace(edges[0, i], edges[1, i], N_PER_SEG)
        pts = []
        for f in fs:
            awg.duc_cw_step(float(f), channel=1, amplitude_vpp=float(vpps[i]))
            sa.set_freq(float(f) * TONE, SPAN_HZ)
            time.sleep(SETTLE_S)
            reads = [float(np.max(sa.sweep_once()[1])) for _ in range(n_avg)]
            pts.append(np.mean(reads))
        pts = np.array(pts)
        lv[i] = float(pts.mean())
        sp[i] = float(pts.max() - pts.min())
        nz[i] = float(np.std(reads, ddof=1)) / np.sqrt(n_avg)
        if lp is not None:
            lp.add(tag, float(fs.mean()) * CHAIN_MULT / 1e9, lv[i])
    return lv, sp, nz


def run(segment_ghz=SEGMENT_GHZ, n_iters=N_ITERS):
    pad = config.PAD_AT_INSTRUMENT_DB
    cen_awg, edges, cen_out = segment_plan(segment_ghz)
    n = len(cen_awg)

    print("=" * 78)
    print(f"SEGMENT LEVEL CALIBRATION -- {ELEMENT}")
    print(f"  {PATH} -> {pad:g} dB pad -> EXA")
    print("=" * 78)
    print(f"  output band {OUT_LO_HZ/1e9:.2f}-{OUT_HI_HZ/1e9:.2f} GHz in "
          f"{n} segments of {segment_ghz:g} GHz")
    print(f"  = {segment_ghz*1e9/CHAIN_MULT/1e6:.3f} MHz per segment at the "
          f"AWG ({cen_awg[0]/1e9:.4f}-{cen_awg[-1]/1e9:.4f} GHz)")
    print(f"  measuring at {TONE}x the AWG frequency, {N_PER_SEG} points per "
          f"segment, {N_AVG} sweeps averaged")
    print(f"  base {BASE_VPP:.4f} Vpp, ceiling {config.AWG_MAX_VPP} Vpp -> "
          f"{20*np.log10(config.AWG_MAX_VPP/BASE_VPP):.2f} dB of headroom")

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
        awg.duc_cw_setup(float(cen_awg[0]), amplitude_vpp=float(BASE_VPP),
                         channel=1)
        time.sleep(0.4)

        lp = LivePlot(f"segment levels: {ELEMENT}",
                      "output frequency (GHz)", "level (dBm, raw)")

        vpps = np.full(n, float(BASE_VPP))
        lp.note("baseline")
        lv, sp, nz = measure_segments(sa, awg, edges, vpps, lp=lp,
                                      tag="baseline")
        target = float(lv.max())
        print(f"\n  baseline: {lv.max()+pad:+.4f} to {lv.min()+pad:+.4f} dBm "
              f"across segments, spread {lv.max()-lv.min():.4f} dB")
        print(f"    read noise {nz.mean():.4f} dB")
        # The assumption this whole approach rests on, checked rather than
        # asserted. The prediction was extrapolated from a 21-point band sweep
        # that could not resolve anything faster than 31 MHz, so it is a lower
        # bound on what a 5 MHz window can contain, not an estimate.
        pred = 13.3 * (segment_ghz / 1.0)
        print(f"    WITHIN-segment spread: mean {sp.mean()*1000:.2f}, "
              f"worst {sp.max()*1000:.2f} milli-dB "
              f"(extrapolation from coarse data said ~{pred:.0f})")
        if sp.mean() * 1000 > 1.5 * pred:
            print(f"    ** {sp.mean()*1000/pred:.1f}x the extrapolation -- "
                  f"there is structure finer than the 21-point sweep could "
                  f"see. The extrapolation was a lower bound, and this is the "
                  f"real number. **")
        print(f"    -> in field amplitude: mean "
              f"{100*(10**(sp.mean()/20)-1):.2f}%, worst "
              f"{100*(10**(sp.max()/20)-1):.2f}% across a segment")
        print(f"    -> via AM-to-PM at 3.2 deg/dB: "
              f"{sp.mean()*3.2*192/TONE:.1f} deg mean across a segment at "
              f"310-370 GHz")
        bad = int(np.sum(sp > SEG_FLAT_WARN_DB))
        if bad:
            print(f"    ** {bad} segment(s) over {SEG_FLAT_WARN_DB*1000:.0f} "
                  f"milli-dB -- a single scalar is not the right correction "
                  f"there **")
        print(f"    NOTE {N_PER_SEG} points per segment cannot tell a TILT "
              f"from a WIGGLE, and only a wiggle makes paired echoes. A fine "
              f"sweep across one segment is what settles that.")

        print(f"    spread to remove: {target - lv.min():.2f} dB AT THIS "
              f"PLANE -- the AWG correction needed is that divided by r, "
              f"which the probe measures next")
        hist.append(dict(iter=0, vpps=vpps.copy(), levels=lv.copy(),
                         spread=sp.copy()))

        prev = np.zeros(n)
        best = dict(rms=float(np.std(lv - target)), vpps=vpps.copy(),
                    levels=lv.copy(), spread=sp.copy(), iter=0, cmd=prev.copy())

        # ---- probe: measure r before correcting anything ----------------
        probe_cmd = prev + PROBE_DB
        probe_vpp = np.clip(BASE_VPP * 10 ** (probe_cmd / 20), 1e-3,
                            config.AWG_MAX_VPP)
        lp.note(f"probe step (+{PROBE_DB} dB, measuring r)")
        lv_p, sp_p, _ = measure_segments(sa, awg, edges, probe_vpp, lp=lp,
                                         tag="probe")
        d_cmd = 20 * np.log10(probe_vpp / np.maximum(vpps, 1e-9))
        ok = np.abs(d_cmd) > R_MIN_STEP_DB
        ratios = np.where(ok, (lv_p - lv) / np.where(ok, d_cmd, 1.0), np.nan)
        r_glob = (float(np.clip(np.nanmedian(ratios), *R_CLAMP))
                  if np.isfinite(ratios).any() else 1.0)
        keep = (np.isfinite(ratios) & (ratios > 0.5 * r_glob)
                & (ratios < 2.0 * r_glob))
        r_est = np.where(keep, ratios, r_glob)
        print(f"\n  probe +{PROBE_DB} dB: r = {r_glob:.2f} (median), "
              f"{int(keep.sum())}/{n} segments used their own")
        print(f"    a dB commanded at the AWG arrives as {r_glob:.2f} dB here; "
              f"corrections are divided by it")
        vpps, lv, sp, prev = probe_vpp, lv_p, sp_p, probe_cmd

        # With r now measured, check the target is REACHABLE without running
        # the stage-A amp out of backoff, and lower it if not. Equalising down
        # costs power at the strong edge; running the amp into compression
        # costs AM-to-PM at 3.2 deg/dB x 96, which is not recoverable.
        # Now that r is known, decide what target is REACHABLE. Two separate
        # ceilings, and the tighter one per segment wins:
        #   the AWG's 0.55 Vpp, and the stage-A amp's remaining backoff.
        ceil_awg = 20 * np.log10(config.AWG_MAX_VPP / BASE_VPP) - PROBE_DB
        max_corr, amp_base = amp_headroom_db(cen_awg)
        if max_corr is None and R.MODE != "awg":
            print(f"\n  ** no measured drive curve, so the stage-A amp's "
                  f"backoff cannot be checked and raising its drive would be "
                  f"blind. ABORT -- run run_exa_harmonics.py first. **")
            return
        room = (np.full(n, ceil_awg) if max_corr is None
                else np.minimum(ceil_awg, max_corr - PROBE_DB))
        achievable = lv + np.clip(r_est, *R_CLAMP) * room
        feasible = float(np.min(achievable))

        print(f"\n  headroom check (r now known):")
        print(f"    AWG ceiling allows {ceil_awg:+.2f} dB more")
        if max_corr is not None:
            import multiplier_chain as mc
            print(f"    stage-A amp {amp_base.min():+.2f} to "
                  f"{amp_base.max():+.2f} dBm, P1dB "
                  f"{mc.ZX602522MA_P1DB_DBM:+.1f} -> "
                  f"{mc.ZX602522MA_P1DB_DBM-amp_base.max():.2f} dB backoff at "
                  f"the strong edge; allows {max_corr.min()-PROBE_DB:+.2f} to "
                  f"{max_corr.max()-PROBE_DB:+.2f} dB more")
        print(f"    binding limit {room.min():+.2f} to {room.max():+.2f} dB of "
              f"AWG, i.e. {(np.clip(r_est,*R_CLAMP)*room).min():+.2f} to "
              f"{(np.clip(r_est,*R_CLAMP)*room).max():+.2f} dB here")
        if feasible < target - 0.01:
            print(f"    ** cannot flatten UP to {target+pad:+.3f} dBm. "
                  f"Lowering the target to {feasible+pad:+.3f} dBm "
                  f"({target-feasible:.2f} dB down) -- equalising by REDUCING "
                  f"the strong segments instead. That costs power at the "
                  f"strong edge; compressing the amp would cost AM-to-PM at "
                  f"3.2 deg/dB x 96, which is not recoverable. **")
            target = feasible
        else:
            print(f"    -> the target fits, {feasible-target:.2f} dB to spare")

        for it in range(1, int(n_iters) + 1):
            err = target - lv
            step = DAMPING * err / np.clip(r_est, *R_CLAMP)
            cmd = prev + step
            new_vpp = np.clip(BASE_VPP * 10 ** (cmd / 20), 1e-3,
                              config.AWG_MAX_VPP)
            lp.note(f"iteration {it}")
            lv_new, sp_new, _ = measure_segments(sa, awg, edges, new_vpp,
                                                 lp=lp, tag=f"iter {it}")

            # r, robustly. Only steps big enough to survive :VOLT quantisation
            # vote, the median of those sets a global value, and a per-segment
            # ratio is kept only if it is within 2x of that. r is a property of
            # the chain and varies smoothly with frequency, so a lone segment
            # returning something wild is a measurement artefact, not physics.
            d_cmd = 20 * np.log10(new_vpp / np.maximum(vpps, 1e-9))
            d_out = lv_new - lv
            ok = np.abs(d_cmd) > R_MIN_STEP_DB
            ratios = np.where(ok, d_out / np.where(ok, d_cmd, 1.0), np.nan)
            if np.isfinite(ratios).any():
                r_glob = float(np.clip(np.nanmedian(ratios), *R_CLAMP))
            # else: KEEP the previous r_glob. Once the loop has converged no
            # step is large enough to re-measure r, and resetting to 1.0 there
            # would throw away everything learned -- then a later correction
            # on a chain with r = 2.4 would be applied at 2.4x, undoing the
            # convergence that caused the fallback in the first place.
            keep = (np.isfinite(ratios) & (ratios > 0.5 * r_glob)
                    & (ratios < 2.0 * r_glob))
            r_est = np.where(keep, ratios, r_glob)

            rms = float(np.std(lv_new - target))
            print(f"\n  iteration {it}: segment spread "
                  f"{lv_new.max()-lv_new.min():.4f} dB, rms {rms:.4f} dB")
            print(f"    r: global median {r_glob:.2f}, "
                  f"{int(keep.sum())}/{n} segments used their own "
                  f"({int(ok.sum())} had a step above {R_MIN_STEP_DB} dB)")
            hist.append(dict(iter=it, vpps=new_vpp.copy(),
                             levels=lv_new.copy(), spread=sp_new.copy(),
                             r=r_est.copy(), rms=rms))
            vpps, lv, sp, prev = new_vpp, lv_new, sp_new, cmd

            if rms < best["rms"]:
                best = dict(rms=rms, vpps=vpps.copy(), levels=lv.copy(),
                            spread=sp.copy(), iter=it, cmd=cmd.copy())
            elif rms > 1.2 * best["rms"]:
                # Diverging. Put the hardware back in the best state found and
                # stop -- an uncorrected loop that is getting worse will keep
                # getting worse, and the saved table must never be the worst
                # iteration (which is what the first run saved).
                print(f"    ** worse than iteration {best['iter']} "
                      f"({best['rms']:.4f} dB). Diverging -- reverting and "
                      f"stopping. **")
                vpps, prev = best["vpps"].copy(), best["cmd"].copy()
                lv, sp, _ = measure_segments(sa, awg, edges, vpps)
                break
            if rms < TOL_DB:
                print(f"    -> under {TOL_DB} dB, stopping")
                break
        lp.done()

        # The table that gets saved is the BEST state, restored on the
        # hardware, not whatever the last iteration happened to leave behind.
        if best["iter"] != hist[-1]["iter"]:
            print(f"\n  keeping iteration {best['iter']} "
                  f"(rms {best['rms']:.4f} dB) as the result")
            hist.append(dict(iter=best["iter"], vpps=best["vpps"],
                             levels=lv, spread=sp, rms=best["rms"],
                             restored=True))

        settings = {"segment_ghz": float(segment_ghz), "n_segments": n,
                    "n_per_segment": N_PER_SEG, "n_avg": N_AVG,
                    "span_hz": SPAN_HZ, "rbw_hz": float(sa.get_rbw()),
                    "atten_db": EXA_ATTEN_DB, "base_vpp": BASE_VPP,
                    "mode": R.MODE, "element": ELEMENT, "tone": TONE,
                    "chain_mult": CHAIN_MULT, "amc_mult": AMC_MULT,
                    "out_lo_hz": OUT_LO_HZ, "out_hi_hz": OUT_HI_HZ,
                    "target_dbm_raw": target, "n_iters_run": len(hist) - 1}
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(cen_awg, cen_out, edges, hist, settings)


def report(cen_awg, cen_out, edges, hist, settings):
    pad = config.PAD_AT_INSTRUMENT_DB
    base, final = hist[0], hist[-1]
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    for h in hist:
        lv = h["levels"]
        print(f"  iter {h['iter']}: segment-to-segment spread "
              f"{lv.max()-lv.min():.4f} dB, rms "
              f"{np.std(lv-settings['target_dbm_raw']):.4f} dB")
    b = base["levels"].max() - base["levels"].min()
    a = final["levels"].max() - final["levels"].min()
    print(f"\n  {b:.4f} -> {a:.4f} dB across the band "
          f"({20*np.log10(max(b,1e-9)/max(a,1e-9)):.1f} dB better)")
    print(f"  in field amplitude that is {100*(10**(b/20)-1):.2f}% -> "
          f"{100*(10**(a/20)-1):.2f}% of excitation non-uniformity")

    sp = final["spread"]
    print(f"\n  within-segment spread after correction: mean "
          f"{sp.mean()*1000:.2f}, worst {sp.max()*1000:.2f} milli-dB")
    print(f"    (unchanged by a per-segment scalar, as intended -- this is the "
          f"part the analysis says is negligible)")

    print(f"\n  the table, every {max(1, len(cen_awg)//12)}th segment:")
    print(f"    {'#':>3} {'out GHz':>9} {'AWG MHz':>10} {'Vpp':>9} "
          f"{'corr dB':>8} {'level':>9} {'in-seg':>8}")
    for i in range(0, len(cen_awg), max(1, len(cen_awg) // 12)):
        print(f"    {i:3d} {cen_out[i]/1e9:9.3f} {cen_awg[i]/1e6:10.3f} "
              f"{final['vpps'][i]:9.5f} "
              f"{20*np.log10(final['vpps'][i]/settings['base_vpp']):8.3f} "
              f"{final['levels'][i]+pad:+9.3f} {sp[i]*1000:7.1f}m")

    sess = chainlog.new_session(STAGE, "segments")
    series = {"vpp": final["vpps"],
              "correction_db": 20 * np.log10(final["vpps"] /
                                             settings["base_vpp"]),
              "level_final": final["levels"],
              "level_baseline": base["levels"],
              "in_segment_spread_db": final["spread"],
              "seg_lo_awg_hz": edges[0], "seg_hi_awg_hz": edges[1],
              "centre_out_hz": cen_out}
    p = chainlog.save(sess, "segments", STAGE, cen_awg, series, settings,
                      x_name="freqs_hz", thru=False,
                      notes=f"{ELEMENT}. Per-segment excitation level "
                            f"calibration. x is the segment CENTRE at the AWG; "
                            f"seg_lo/hi_awg_hz are its edges; vpp is the AWG "
                            f"amplitude to command for that segment. Levels "
                            f"RAW, add {pad:g} dB. Valid only for this path "
                            f"and base level.")
    print(f"\n  saved {p}")
    print(f"  -> play segment i with play_segment(awg, load(p), i, chirp_us)")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(cen_out / 1e9, base["levels"] + pad, "-o", ms=3,
                 label="baseline")
    axes[0].plot(cen_out / 1e9, final["levels"] + pad, "-o", ms=3,
                 label="calibrated")
    axes[0].axhline(settings["target_dbm_raw"] + pad, ls="--", c="k", lw=1,
                    alpha=0.6)
    axes[0].set_xlabel("output frequency (GHz)")
    axes[0].set_ylabel("segment level (dBm)")
    axes[0].set_title(f"excitation level per segment\n{b:.3f} -> {a:.3f} dB")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].plot(cen_out / 1e9,
                 20 * np.log10(final["vpps"] / settings["base_vpp"]),
                 "-o", ms=3, c="tab:red")
    axes[1].set_xlabel("output frequency (GHz)")
    axes[1].set_ylabel("AWG correction (dB)")
    axes[1].set_title("the segment table")
    axes[1].grid(alpha=0.3)

    axes[2].plot(cen_out / 1e9, final["spread"] * 1000, "-o", ms=3)
    axes[2].axhline(13.3, ls="--", c="k", lw=1, alpha=0.6,
                    label="13.3 mdB predicted")
    axes[2].set_xlabel("output frequency (GHz)")
    axes[2].set_ylabel("spread within a segment (milli-dB)")
    axes[2].set_title("is a segment really flat inside?")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "segments.png")
    fig.savefig(png, dpi=130)
    print(f"        {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


def play_segment(awg, doc, i, chirp_us=1.0, dead_us=1.0, window_frac=0.05,
                 widen=True):
    """Transmit segment i of a saved table, at its calibrated amplitude.

    The commanded sweep is WIDENED by default: the chirp's Gaussian taper
    leaves only the middle (1 - 2*window_frac) of the commanded span at full
    amplitude, so asking for exactly the segment edges would under-cover the
    segment by that fraction on each side (awg.widen_for_flat_plateau, verified
    on bench data 2026-07-24).
    """
    lo = doc[chainlog.series_key("seg_lo_awg_hz")][i]
    hi = doc[chainlog.series_key("seg_hi_awg_hz")][i]
    vpp = float(doc[chainlog.series_key("vpp")][i])
    if widen:
        lo, hi = widen_for_flat_plateau(lo, hi, window_frac)
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    awg.send_chirp_duc_sync(centre, -half, +half, chirp_us, dead_us,
                            amplitude_vpp=vpp, window_frac=window_frac)
    return dict(segment=i, centre_awg_hz=centre, bw_awg_hz=2 * half,
                centre_out_hz=centre * CHAIN_MULT,
                bw_out_hz=2 * half * CHAIN_MULT, vpp=vpp)


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else SEGMENT_GHZ,
        int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERS)
