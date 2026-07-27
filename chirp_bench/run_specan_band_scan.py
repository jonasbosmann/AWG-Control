"""Bench script: cross-check the AWG's swept amplitude response using the
Keysight N9010A EXA signal analyzer instead of the scope -- same designated
frequency segments as run_chirp_quality.py, so the two independent
measurements (scope-FFT-derived swept_amplitude_response()/rolloff_frequency()
vs. this EXA-measured trace) cover identical bands and are directly
comparable.

The EXA gives amplitude only, no phase (see project_chirp_quality memory's
"Scope boundary" note) -- so this only ever targets the rolloff/flatness
question, not linearity/EVM/PSLR. Those stay scope-only.

Wiring: AWG CH1 -> EXA RF IN (swap the cable from the scope, or T off it if
simultaneous capture is ever wanted). CH2's sync pulse still gets programmed
(send_chirp_duc_sync() always creates it) but isn't used for anything here --
no trigger needed. Watch the EXA's input power rating if this ever gets
pointed downstream of an amplifier; the raw AWG output alone (<=0.5 Vpp) is
well within range.

Why no "loop" code is needed on the AWG side: send_chirp_duc_sync() leaves
the AWG in `:INIT:CONT ON` -- once commanded, a segment (chirp + dead-time
silence) repeats indefinitely until the next segment loads or awg.stop()
is called.

Why MAX HOLD and not a single Normal sweep (MEASURED 2026-07-27, the first
version of this script got it wrong): a swept analyzer's RBW filter only
dwells at each frequency for sweep_time/points, and the chirp only
illuminates any given frequency for about T_chirp*RBW/B (~3 ns here) once
per 1.5 us period. Those two rarely coincide, so a single Normal sweep
mostly records noise where it missed the chirp -- measured on the real
0.2-1.0 GHz segment: peak -29.3 dBm but MEDIAN -77.1 dBm, a 47.8 dB spread
across a band that should be nearly flat. The same segment in MAX HOLD
converges to a 0.6 dB peak-to-median spread (i.e. an actually flat
envelope, as expected) within ~2 s and is stable out to at least 20 s.
So: MAXH + start_continuous() + settle + get_trace(). That also means
single_sweep() is unusable here anyway -- it raises in MAXH/AVER mode on
this instrument (documented firmware hang, see project_exa_specan memory).

Run with `python chirp_bench/run_specan_band_scan.py` from the project
root (or `cd chirp_bench` first) -- same convention as the rest of
chirp_bench/.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# awg.py/specan.py/specanlog.py live one level up (project root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from awg import AWG, widen_for_flat_plateau
from specan import SpecAn
import specanlog

import chirp_quality as cq
import run_chirp_quality as rcq

# Segment tiling. Defaults to 0.1 GHz-wide segments across 0.2-4.2 GHz --
# the SAME tiling run_fine_band_scan.py uses on the scope, so this EXA scan
# is directly comparable to the scope's fine band scan segment-for-segment.
# Narrow segments are also better on the AWG side (0.8 GHz-wide segments
# measured 3-25x worse dechirp residual/EVM than 0.2 GHz ones), so there's
# no reason to sweep wide here. Widen SEGMENT_SPAN_HZ (or set SEGMENTS to
# rcq.SEGMENTS) only if you specifically want the old 0.8 GHz tiling.
SEGMENT_SPAN_HZ = 0.1e9
BAND_START_HZ   = 0.2e9
BAND_STOP_HZ    = 4.2e9
SEGMENTS = [(round(f, 9), SEGMENT_SPAN_HZ)
            for f in np.arange(BAND_START_HZ, BAND_STOP_HZ, SEGMENT_SPAN_HZ)]

ATTEN_DB = 10        # the EXA's INTERNAL input attenuator. The analyzer
                      # already refers its readings to its own input
                      # connector, so this does NOT shift measured dBm --
                      # it only trades noise floor against input headroom.
                      # Fixed across the whole run for level comparability
                      # (same convention as the earlier LO spur-hunt work).

EXTERNAL_ATTEN_DB = 10.0
# A PHYSICAL pad sitting between the AWG and the EXA input. Unlike ATTEN_DB
# above, the analyzer has no idea this exists, so every reading is low by
# exactly this much -- it must be added back to refer levels to the AWG's
# own output. Confirmed 2026-07-27: CW at the flat low end read -12.01 dBm,
# +10 dB = -2.01 dBm, against the -2.04 dBm a 0.5 Vpp sine into 50 ohm
# gives in theory -- agreement to 0.03 dB, which also confirms the AWG
# really delivers its commanded amplitude (no DUC digital backoff).
# SET THIS TO 0.0 IF THE PAD IS REMOVED, or the levels will read 10 dB high.
# Raw traces are always SAVED as the analyzer measured them; the correction
# is applied to reported/plotted levels, with the value recorded in each
# trace's metadata so the raw numbers stay recoverable.
REF_LEVEL_DBM = 0     # adjust at the bench if the trace clips or sits low
MEAS_SPAN_MULT = 1.1  # measured span vs. the AWG-commanded (widened) band --
                      # a bit wider so the tapered edges are visible too,
                      # even though only the DESIGNATED range feeds the
                      # rolloff finding / stitched plot.
N_POINTS = 801
SETTLE_S = 0.3        # after commanding the AWG, before touching the EXA

# Max Hold accumulation time per segment. MEASURED 2026-07-27 on the real
# 0.2-1.0 GHz segment: the trace had already converged to its final shape
# (0.6 dB peak-to-median spread) by 2 s and was unchanged at 5/10/20 s, so
# these give real margin over observed convergence rather than being a
# guess. Sized as sweeps-worth-of-time with a floor, so it adapts if a
# future segment/RBW combination sweeps much more slowly.
MAXH_MIN_SETTLE_S = 3.0
MAXH_SWEEPS = 10
LIVE_POLL_S = 0.4     # how often to push an intermediate trace to an
                       # on_trace callback while Max Hold accumulates, so a
                       # GUI can show the trace building up instead of
                       # freezing for the whole settle. Only polled when a
                       # callback is actually supplied -- the standalone
                       # script just sleeps, no extra instrument traffic.


def new_run_subdir():
    """A fresh specan_traces/ subdir name, timestamped at CALL time (not
    import time) -- run_segment_specan()/run() take this as an explicit
    argument rather than a module-level constant so a long-lived caller
    (specan_gui.py's Chirp Band Scan button) gets a new folder on every
    run instead of every run silently overwriting the first one's files."""
    return time.strftime("chirp_band_scan_%Y-%m-%d_%H%M%S")


MAX_RBW_HZ = 3e6      # this N9010A's real RBW ceiling, MEASURED 2026-07-27:
                       # requesting 3.333 MHz came back from :BAND:RES? as
                       # 3.000 MHz. The instrument clamps silently (no SCPI
                       # error), so asking for more than this doesn't fail
                       # loudly, it just quietly gives a narrower filter than
                       # the comb-smoothing below assumes.
COMB_SMOOTH_MIN = 3.0  # required RBW / PRF ratio -- below this the pulse
                       # comb is no longer smoothed into an envelope.


def configure_specan(specan, carrier_hz, span_hz):
    """Set up one segment's sweep. Returns the RBW the instrument ACTUALLY
    applied (not the one requested -- see MAX_RBW_HZ)."""
    specan.set_freq(carrier_hz, span_hz)
    specan.set_points(N_POINTS)
    # RBW wide enough to smooth the burst's pulse-repetition comb (line
    # spacing = PRF = 1/(chirp_us+dead_us)) into the smooth swept-amplitude
    # envelope actually wanted here, but still narrow enough to resolve
    # real rolloff structure across the segment.
    prf = 1.0 / ((rcq.CHIRP_US + rcq.DEAD_US) * 1e-6)
    rbw = min(max(5 * prf, span_hz / 300), MAX_RBW_HZ)
    specan.set_rbw(rbw)
    specan.set_vbw(rbw)
    # MAXH, not NORM -- a single Normal sweep mostly misses this pulsed,
    # fast-swept signal (see module docstring for the measured 47.8 dB vs.
    # 0.6 dB peak-to-median comparison that settled this).
    specan.set_trace_mode("MAXH")
    # Verify rather than assume: the clamp is silent, and if the achieved
    # RBW ends up narrower than the comb spacing the trace stops being a
    # swept-amplitude envelope at all and becomes a picket fence of PRF
    # lines -- which would still LOOK like a plausible trace while making
    # every level reading (and the rolloff derived from them) meaningless.
    actual = specan.get_rbw()
    # RBW is pinned near 3 MHz by the comb-smoothing requirement below, so
    # shrinking SEGMENT_SPAN_HZ eventually makes the filter a large
    # fraction of the span and visibly rounds the band edges. At the 0.1 GHz
    # default this is ~2.5% (measured flat tops of 0.7-1.1 dB, fine); it
    # only becomes a problem for much narrower segments. The lever is dead
    # time, not RBW: raising DEAD_US lowers the PRF, which relaxes the
    # minimum RBW without changing the chirp being measured at all.
    if actual > span_hz / 15:
        print(f"  NOTE: RBW {actual/1e6:.2f} MHz is {100*actual/span_hz:.0f}% of the "
              f"{span_hz/1e6:.0f} MHz span -- band edges will be visibly rounded. "
              f"Raise DEAD_US (lowers PRF -> allows narrower RBW) or widen "
              f"SEGMENT_SPAN_HZ if edge shape matters.")
    ratio = actual / prf
    if ratio < COMB_SMOOTH_MIN:
        print(f"  WARNING: applied RBW {actual/1e6:.3f} MHz is only {ratio:.1f}x the "
              f"{prf/1e6:.3f} MHz pulse-repetition rate (want >={COMB_SMOOTH_MIN:.0f}x). "
              f"This instrument caps RBW at {MAX_RBW_HZ/1e6:.0f} MHz, so the burst's "
              f"comb will NOT be smoothed into an envelope and levels/rolloff from "
              f"this run are not trustworthy -- lengthen CHIRP_US+DEAD_US (lower the "
              f"PRF) in run_chirp_quality.py to fix.")
    return actual


def _settle_with_updates(specan, settle_s, on_trace):
    """Wait out the Max Hold accumulation, optionally pushing intermediate
    traces to on_trace(freqs, amps) as they build up.

    Traces handed to on_trace are the FULL commanded span, deliberately NOT
    cropped to the designated band -- that way a live view matches what the
    analyzer's own screen is showing (skirts included), rather than the
    narrower flat-top crop that the final stitched plot uses."""
    if on_trace is None:
        time.sleep(settle_s)
        return
    deadline = time.time() + settle_s
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(LIVE_POLL_S, remaining))
        try:
            on_trace(*specan.get_trace())
        except Exception as e:
            # A live-view update failing must never abort the measurement
            # itself -- the real trace is read after this loop regardless.
            print(f"  (live update skipped: {e})")


def run_segment_specan(awg, specan, f_start, span_hz, run_subdir, on_trace=None):
    f_stop = f_start + span_hz
    f_start_cmd, f_stop_cmd = widen_for_flat_plateau(f_start, f_stop)
    carrier_hz = (f_start_cmd + f_stop_cmd) / 2
    half_span_hz = (f_stop_cmd - f_start_cmd) / 2
    label = f"{f_start/1e9:.2f}-{f_stop/1e9:.2f} GHz"

    print(f"\n--- [{label}] ---")
    awg.send_chirp_duc_sync(carrier_hz, -half_span_hz, half_span_hz,
                             rcq.CHIRP_US, rcq.DEAD_US, amplitude_vpp=rcq.AMP_VPP,
                             sync_pulse_ns=rcq.SYNC_PULSE_NS,
                             sync_amp_vpp=rcq.SYNC_AMP_VPP,
                             sync_carrier_hz=rcq.SYNC_CARRIER_HZ)
    time.sleep(SETTLE_S)

    meas_span = (f_stop_cmd - f_start_cmd) * MEAS_SPAN_MULT
    actual_rbw = configure_specan(specan, carrier_hz, meas_span)
    # Accumulate Max Hold rather than taking one sweep -- see module
    # docstring. clear_trace() first so this segment can't inherit the
    # PREVIOUS segment's held peaks (MAXH accumulates forever otherwise,
    # which across a stepped band scan would smear every segment's trace
    # into all the later ones).
    specan.clear_trace()
    specan.start_continuous()
    time.sleep(0.3)          # let the instrument recompute sweep time for the
                              # new span/RBW before asking -- querying straight
                              # away returns the previous setting's value
                              # (same stale-read trap hit in the spur hunt)
    sweep_s = specan.get_sweep_time()
    settle_s = max(MAXH_MIN_SETTLE_S, sweep_s * MAXH_SWEEPS + 0.5)
    print(f"  EXA: center {carrier_hz/1e9:.4f} GHz, span {meas_span/1e6:.1f} MHz, "
          f"RBW applied {actual_rbw/1e6:.3f} MHz, sweep {sweep_s*1e3:.1f} ms "
          f"-> MaxHold settle {settle_s:.1f} s")
    _settle_with_updates(specan, settle_s, on_trace)
    freqs, amps = specan.get_trace()

    mask = (freqs >= f_start) & (freqs <= f_stop)
    # Same external-pad correction as the CW path -- note this still isn't
    # absolute chirp power (the analyzer reads power DENSITY, scaling as
    # 1/B, plus swept-filter settling loss); it just removes the one
    # instrumentation offset that IS exactly known.
    freqs_crop, amps_crop = freqs[mask], amps[mask] + EXTERNAL_ATTEN_DB
    if len(amps_crop):
        peak_i = int(np.argmax(amps_crop))
        print(f"  [{label}] designated-band peak {amps_crop[peak_i]:.2f} dBm "
              f"@ {freqs_crop[peak_i]/1e9:.4f} GHz")
    else:
        print(f"  [{label}] WARNING: no trace points landed inside the "
              f"designated band -- check span/crop math")

    settings = {
        "atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM,
        "rbw_hz": actual_rbw, "vbw_hz": actual_rbw, "points": N_POINTS,
        "trace_mode": "MAXH", "maxh_settle_s": settle_s,
        "chirp": {
            "f_start_designated_hz": f_start, "f_stop_designated_hz": f_stop,
            "f_start_cmd_hz": f_start_cmd, "f_stop_cmd_hz": f_stop_cmd,
            "carrier_hz": carrier_hz, "chirp_us": rcq.CHIRP_US,
            "dead_us": rcq.DEAD_US, "amp_vpp": rcq.AMP_VPP,
        },
    }
    notes = ("EXA cross-check of the scope-based swept_amplitude_response() "
             "rolloff finding -- same designated band/AWG params as "
             "run_chirp_quality.py's scope scan, looped continuously "
             "(:INIT:CONT ON) rather than triggered single-shot.")
    path = specanlog.save_trace(label, notes, settings, freqs, amps,
                                 subdir=run_subdir)
    print(f"  saved {path}")
    return label, f_start, f_stop, freqs_crop, amps_crop


def stitch_and_plot(results, fig=None):
    """Build the stitched amplitude-vs-frequency view + rolloff finding.

    fig=None (default, standalone-script use): creates a fresh Figure.
    fig=<existing Figure> (specan_gui.py's Chirp Band Scan use): clears and
    redraws into that Figure instead, so the GUI's embedded canvas can show
    the exact same stitching/rolloff logic as the standalone script -- one
    implementation, not two that could drift apart."""
    kept = [(f, a) for _, _, _, f, a in results if len(f)]
    if not kept:
        # Bare np.concatenate([]) raises "need at least one array to
        # concatenate", which says nothing about what actually went wrong.
        # Reaching here means EVERY segment's trace missed its designated
        # crop window -- in practice that's the EXA not having applied the
        # commanded center/span (e.g. a rejected :FREQ write leaving it on a
        # previous setting, or the instrument left in zero span), not a
        # signal-level problem. Say so.
        raise RuntimeError(
            f"none of the {len(results)} segments returned any trace points inside "
            "their designated band -- the EXA's frequency axis doesn't match what was "
            "commanded. Check :FREQ:CENT/:FREQ:SPAN actually applied (query them "
            "after a segment) and that the analyzer isn't in zero span.")
    all_f = np.concatenate([f for f, _ in kept])
    all_a = np.concatenate([a for _, a in kept])
    order = np.argsort(all_f)
    all_f, all_a = all_f[order], all_a[order]

    # Reference = median of the lowest-frequency segment's designated-band
    # amplitude ("how far down from the known-good low end are we"), same
    # -3dB-from-plateau convention as chirp_quality.rolloff_frequency() --
    # anchored to the first segment (not a whole-band median) so a
    # genuinely rolled-off top segment can't drag its own reference down.
    _, _, _, first_f, first_a = results[0]
    ref_dbm = float(np.median(first_a)) if len(first_a) else float(np.median(all_a))
    amp_db_rel = all_a - ref_dbm
    rolloff_hz = cq.rolloff_frequency(all_f, amp_db_rel, threshold_db=-3.0)

    if fig is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig.clear()
        ax = fig.add_subplot(111)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for (label, f_start, f_stop, f, a), c in zip(results, colors):
        if len(f):
            ax.plot(f / 1e9, a, lw=1.4, color=c, label=label)
    ax.axhline(ref_dbm, color="gray", ls="--", lw=1,
               label=f"ref ({ref_dbm:.1f} dBm, seg 1 median)")
    ax.axhline(ref_dbm - 3, color="red", ls=":", lw=1, label="-3 dB")
    if rolloff_hz is not None:
        ax.axvline(rolloff_hz / 1e9, color="red", lw=1.2)
        title_roll = f"EXA rolloff (-3dB): {rolloff_hz/1e9:.3f} GHz"
    else:
        title_roll = "EXA: flat over whole scanned band (no -3dB crossing found)"
    ax.set_xlabel("GHz"); ax.set_ylabel("dBm")
    ax.set_title(f"EXA-measured swept amplitude response, {len(results)} segments "
                 f"({results[0][1]/1e9:.2f}-{results[-1][2]/1e9:.2f} GHz)\n{title_roll}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig, rolloff_hz


# ── CW level check (absolute-level calibration reference) ──────────
# The chirp scan's dBm numbers are NOT calibrated absolute power: the
# analyzer reads power density (so level scales as 1/B, verified
# 2026-07-27), and on top of that a 1 us chirp sweeps through the 3 MHz RBW
# filter faster than it can settle, costing a further fixed amount of
# level. Both effects are constant at fixed span/duration, so they don't
# distort the droop-vs-frequency SHAPE -- but they do mean you can't read
# absolute power off the chirp trace.
#
# A CW tone has neither problem: continuous (no pulse comb, no duty-cycle
# correction) and stationary in frequency (the filter fully settles), so
# the analyzer reports its true level. Stepping a CW tone across the band
# therefore gives both (a) genuine absolute dBm vs frequency and (b) an
# INDEPENDENT measurement of the amplitude droop to check the chirp-derived
# one against -- a different signal type through the same cable/analyzer.
CW_FREQS = [round(f, 9) for f in np.arange(0.2e9, 4.21e9, 0.1e9)]
CW_SPAN_HZ = 10e6     # narrow: the DUC NCO is frequency-exact, so the tone
                       # lands where commanded; this only needs to contain it
CW_RBW_HZ = 100e3     # narrow RBW is fine (and more accurate) for a
                       # stationary tone -- none of the chirp scan's
                       # comb-smoothing constraints apply here
CW_POINTS = 401
CW_SETTLE_S = 0.25
# Same amplitude the chirp scan commands, so the CW and chirp curves are
# driven identically and any level difference between them is attributable
# to the measurement (density + filter settling), not to a different drive.
AMP_VPP_CW = rcq.AMP_VPP


def run_cw_point(awg, specan, freq_hz, run_subdir, first=False):
    """Park a CW tone at freq_hz and read its true level off the analyzer.

    first=True does the one-time DUC CW upload; afterwards only the NCO is
    retuned (duc_cw_step, ~ms) with no waveform re-upload."""
    if first:
        awg.duc_cw_setup(freq_hz, amplitude_vpp=AMP_VPP_CW)
    else:
        awg.duc_cw_step(freq_hz)
    time.sleep(CW_SETTLE_S)

    specan.set_freq(freq_hz, CW_SPAN_HZ)
    specan.set_points(CW_POINTS)
    specan.set_rbw(CW_RBW_HZ)
    specan.set_vbw(CW_RBW_HZ)
    specan.set_trace_mode("NORM")   # a stationary tone needs no Max Hold;
                                     # NORM also keeps single_sweep() usable
    freqs, amps = specan.sweep_once()

    i = int(np.argmax(amps))
    # Refer to the AWG's output by adding back the external pad the
    # analyzer can't know about (see EXTERNAL_ATTEN_DB).
    peak_dbm, peak_hz = float(amps[i]) + EXTERNAL_ATTEN_DB, float(freqs[i])
    # The NCO is exact, so a peak far from the commanded frequency means
    # something is wrong (wrong segment playing, aliasing, analyzer not
    # retuned) rather than a real measurement -- worth surfacing.
    err_hz = peak_hz - freq_hz
    flag = "  <-- PEAK OFF-FREQUENCY" if abs(err_hz) > CW_SPAN_HZ / 4 else ""
    print(f"  CW {freq_hz/1e9:.3f} GHz -> {peak_dbm:7.2f} dBm "
          f"(peak at {peak_hz/1e9:.4f} GHz, {err_hz/1e6:+.3f} MHz){flag}")

    specanlog.save_trace(
        f"cw_{freq_hz/1e9:.3f}GHz".replace(".", "p"),
        "CW absolute-level reference for the chirp band scan",
        {"atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM,
         "rbw_hz": CW_RBW_HZ, "vbw_hz": CW_RBW_HZ, "points": CW_POINTS,
         "trace_mode": "NORM", "cw_freq_hz": freq_hz, "amp_vpp": AMP_VPP_CW,
         # amps_dbm below is RAW as measured; add this to refer to the AWG output
         "external_atten_db": EXTERNAL_ATTEN_DB},
        freqs, amps, subdir=run_subdir)
    return peak_hz, peak_dbm


def plot_cw_scan(cw_freqs, cw_dbm, chirp_results=None, fig=None):
    """Absolute CW level vs frequency, plus (if a chirp scan is supplied) a
    shape-only overlay: both curves referenced to their own value at the
    lowest frequency, so the two droops can be compared directly despite
    sitting at completely different absolute levels."""
    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    else:
        fig.clear()
        axes = fig.subplots(2, 1, sharex=True)
    ax_abs, ax_shape = axes

    cw_freqs = np.asarray(cw_freqs); cw_dbm = np.asarray(cw_dbm)
    ax_abs.plot(cw_freqs / 1e9, cw_dbm, "o-", ms=3, lw=1.3, color="C1", label="CW tone")
    # What the commanded amplitude should produce into 50 ohm, as a
    # reference line -- the low end sitting on it is the end-to-end
    # confirmation that the AWG delivers what it's told to.
    theo_dbm = 10 * np.log10(((AMP_VPP_CW / 2) ** 2 / (2 * 50)) / 1e-3)
    ax_abs.axhline(theo_dbm, color="gray", ls="--", lw=1,
                   label=f"theory: {AMP_VPP_CW} Vpp into 50$\\Omega$ = {theo_dbm:.2f} dBm")
    ax_abs.set_ylabel("dBm at AWG output")
    ax_abs.set_title(f"CW level check -- absolute output vs frequency "
                     f"({len(cw_freqs)} points, {AMP_VPP_CW} Vpp, "
                     f"{EXTERNAL_ATTEN_DB:.0f} dB external pad corrected out)")
    ax_abs.grid(True, alpha=0.3); ax_abs.legend(fontsize=8)

    ax_shape.plot(cw_freqs / 1e9, cw_dbm - cw_dbm[0], "o-", ms=3, lw=1.3,
                  color="C1", label="CW (shape)")
    if chirp_results:
        cf, ca = [], []
        for _, _, _, f, a in chirp_results:
            if len(f):
                cf.append(np.median(f)); ca.append(np.median(a))
        if cf:
            cf, ca = np.array(cf), np.array(ca)
            o = np.argsort(cf); cf, ca = cf[o], ca[o]
            ax_shape.plot(cf / 1e9, ca - ca[0], "s--", ms=3, lw=1.3,
                          color="C0", label="chirp scan (shape)")
    ax_shape.set_xlabel("GHz"); ax_shape.set_ylabel("dB rel. to lowest freq")
    ax_shape.set_title("Shape comparison -- if these agree, the droop is real "
                       "and not an artifact of how the chirp is measured", fontsize=9)
    ax_shape.grid(True, alpha=0.3); ax_shape.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run_cw_scan(awg, specan, run_subdir, freqs_hz=None, on_point=None):
    """Step a CW tone across the band, returning (freqs_hz, levels_dbm).

    on_point(freqs_so_far, levels_so_far) is called after each point, so a
    GUI can watch the response curve build up point by point."""
    freqs_hz = CW_FREQS if freqs_hz is None else freqs_hz
    print(f"\n=== CW level check: {len(freqs_hz)} points, "
          f"{freqs_hz[0]/1e9:.2f}-{freqs_hz[-1]/1e9:.2f} GHz ===")
    out_f, out_a = [], []
    for i, f in enumerate(freqs_hz):
        _, pa = run_cw_point(awg, specan, f, run_subdir, first=(i == 0))
        out_f.append(f); out_a.append(pa)
        if on_point is not None:
            try:
                on_point(np.array(out_f), np.array(out_a))
            except Exception as e:
                print(f"  (live update skipped: {e})")
    return np.array(out_f), np.array(out_a)


def run():
    awg = AWG()
    specan = SpecAn()
    specan.set_attenuation(ATTEN_DB)
    specan.set_ref_level(REF_LEVEL_DBM)

    run_subdir = new_run_subdir()
    results = []
    try:
        for f_start, span_hz in SEGMENTS:
            results.append(run_segment_specan(awg, specan, f_start, span_hz, run_subdir))
    finally:
        try:
            awg.stop()
        except Exception as e:
            print(f"cleanup (awg.stop): {e}\n")

    fig, rolloff_hz = stitch_and_plot(results)
    out_dir = os.path.join(specanlog.TRACE_DIR, run_subdir)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "_band_overview.png")
    fig.savefig(png_path, dpi=130)
    print(f"\nsaved stitched plot: {png_path}")

    if rolloff_hz is not None:
        print(f"\nEXA-measured -3dB rolloff: {rolloff_hz/1e9:.3f} GHz "
              f"(single N=1 run -- not yet repeated, see feedback_bench_rigor)")
    else:
        print("\nEXA found no -3dB rolloff within the scanned band "
              "(single N=1 run -- not yet repeated)")
    print("Cross-check against the scope-based swept_amplitude_response()/"
          "rolloff_frequency() finding from run_chirp_quality.py/"
          "band_overview.py for the same segments.")

    plt.show()
    return results, rolloff_hz


if __name__ == "__main__":
    run()
