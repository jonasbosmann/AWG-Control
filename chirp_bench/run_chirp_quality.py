"""Bench script: characterize the raw AWG chirp output across the band,
using chirp_quality.py's Hilbert-demod + matched-filter analysis.

Wiring (same convention as compare_duc_direct.py):
    AWG CH1 -> scope CH1   (chirp under test)
    AWG CH2 -> scope CH2   (start-of-buffer sync pulse -> scope trigger)

Each DUC segment can only cover ~1 GHz of absolute output before the
complex-baseband Nyquist (1.125 GS/s -> +-0.5625 GHz around its NCO carrier)
starts to alias, so SEGMENTS below tiles the 0.2-4.2 GHz band in 800 MHz
steps rather than one continuous sweep. Each entry is (f_start_hz, span_hz)
-- the segment covers f_start .. f_start+span; the carrier is placed at the
center internally. Edit SEGMENTS / the per-run constants at the bench.

Run with a short SEGMENTS list first (e.g. just the one you care about) to
confirm wiring + trigger before looping the whole band.
"""

import json
import os
import sys
import time

import numpy as np

# awg.py/scope.py live one level up (project root) -- this script moved
# into chirp_bench/ but the hardware drivers stayed put, shared with the
# other (non-chirp-quality) bench tools there.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from awg import AWG, widen_for_flat_plateau
from scope import Scope
import chirp_quality as cq

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirp_captures")

# ── config — edit these at the bench ───────────────────────────────
SIG_CH  = 1
TRIG_CH = 2

# (f_start_hz, span_hz) -> absolute band = f_start .. f_start+span.
# 800 MHz segments tiling 0.2-4.2 GHz; shrink/move this list to test one
# region at a time. Each segment's carrier is placed at its center
# internally (DUC needs a carrier + baseband span around it) -- you only
# ever specify start/span here.
SEGMENTS = [
    (0.2e9, 0.8e9),   # 0.2-1.0 GHz
    (1.0e9, 0.8e9),   # 1.0-1.8 GHz
    (1.8e9, 0.8e9),   # 1.8-2.6 GHz
    (2.6e9, 0.8e9),   # 2.6-3.4 GHz
    (3.4e9, 0.8e9),   # 3.4-4.2 GHz -- the interesting one near the DC
                      # module's 4.5 GHz analog bandwidth edge.
]

CHIRP_US = 1.0
DEAD_US  = 0.5
AMP_VPP       = 0.5
# NOTE 2026-07-23: 0.8 Vpp was silently REJECTED by this AWG in DUC mode
# ("204, data out of range in scpi") while 0.5 Vpp (AMP_VPP above) succeeded
# -- brackets this unit's real ceiling to ~0.5-0.55 Vpp, matching the
# Proteus manual's AC-coupled/direct-DAC module spec (550 mVpp, 9 GHz BW),
# not the DC-coupled one (1.2 Vpp, 4.5 GHz). Keep amplitudes <= 0.5 Vpp.
SYNC_AMP_VPP  = 0.5
SYNC_PULSE_NS = 40
TRIG_LEVEL_V  = 0.15     # comfortably below SYNC_AMP_VPP
# Fixed carrier for the CH2 sync burst, used for every segment regardless of
# that segment's own carrier -- 0 Hz (true DC/square) measured ~30 mV instead
# of 0.5 Vpp on this AC-coupled-module AWG, so this needs to be a genuine RF
# tone. Comfortably in the flat, already-confirmed-working low band so the
# trigger stays reliable even for segments near the top of the sweep.
SYNC_CARRIER_HZ = 0.6e9

N_AVG    = 1             # scope-HARDWARE averaging -- kept at 1 (not used).
                          # ACQuire:MODe AVErage needs the trigger to land at
                          # the same point in the RF cycle every shot; CH2's
                          # sync burst can't guarantee that (see N_SHOTS_AVG
                          # below for the software alternative that doesn't
                          # depend on trigger precision).
N_SHOTS_AVG = 1         # number of independent single-shot captures to
                          # align (via cross-correlation on CH1's own
                          # waveform) and average IN SOFTWARE per segment --
                          # see chirp_quality.align_and_average(). Set to 1
                          # to skip averaging and use a single raw shot.
SETTLE_S = 0.15
TRIG_MODE = "NORMal"     # only acquire on genuine CH2 triggers

INTERACTIVE = True        # pause between segments so you can check the live scope
SHOW_PLOTS  = True         # pop up chirp_quality.plot() for each segment


def pause(msg):
    if INTERACTIVE:
        try:
            input(msg)
        except EOFError:
            pass


CAPTURE_MARGIN_MULT = 2.0   # capture window = chirp duration * this multiplier,
                            # sized from the scope's ACTUAL achieved sample
                            # rate (queried in set_max_sample_rate, not
                            # assumed) -- e.g. 2x a 1 us chirp = 2 us, vs. the
                            # earlier fixed RECORD_LENGTH=100000 which, at
                            # this scope's real 12.5 GS/s, gave 8 us of data
                            # per shot for a 1 us chirp (8x more than needed,
                            # 20x over with N_SHOTS_AVG capturing that much
                            # ASCII data per shot).
DISPLAY_MARGIN_MULT = 1.0  # on-screen display window = chirp duration * this
                            # multiplier -- deliberately DIFFERENT from
                            # CAPTURE_MARGIN_MULT. The scope can acquire more
                            # into memory (CAPTURE_MARGIN_MULT's safety margin
                            # against trigger/cable skew) than it displays at
                            # once (RECOrdlength/SAMPLERate vs. HORizontal:
                            # SCAle are independent) -- showing the full 2x
                            # capture window on screen was needlessly wide for
                            # visually checking the pulse; this frames just
                            # the chirp itself.


def arm_scope(scope, chirp_us):
    # CH1's chirp and CH2's sync pulse both start at the same buffer sample
    # (0), so trigger on CH2's RISing edge -- the chirp begins essentially
    # AT the trigger point. (FALLing doesn't help here: the sync burst is an
    # RF tone at SYNC_CARRIER_HZ with many rising/falling edges per period,
    # not a single clean pulse, so "falling" would just fire on the very
    # next half-cycle rather than at any meaningful "end of pulse".)
    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
                trigger_level=TRIG_LEVEL_V, trigger_mode=TRIG_MODE,
                trigger_slope="RISe")
    # Only CH1+CH2 are needed here -- leaving CH3/CH4 on (e.g. from an
    # earlier session) silently caps the real-time sample rate at a slower
    # tier (observed: 12.5 GS/s instead of the scope's rated 50 GS/s).
    scope.set_active_channels([SIG_CH, TRIG_CH])

    # Always capture at the scope's fastest possible sample rate, with the
    # record length sized from that ACTUAL rate (not assumed) -- don't try
    # to fit a specific on-screen time window (that's what was silently
    # trading away sample rate for segments needing faster sampling), and
    # don't just pick a big fixed sample count either (that overshoots once
    # the real rate is known). Crop to the chirp afterward in software
    # (find_burst_window()) instead.
    capture_s = chirp_us * 1e-6 * CAPTURE_MARGIN_MULT
    rate = scope.set_max_sample_rate(capture_s)
    scope.set_horizontal_position(0)
    # set_max_sample_rate() only sets the ACQUIRED record (rate + length) --
    # it never touches HORizontal:SCAle, so the on-screen display keeps
    # whatever timebase was left over from before (usually much wider than
    # the chirp), needing a manual scale turn to actually see the pulse.
    # Set the display window separately from the (wider) acquired one --
    # DISPLAY_MARGIN_MULT frames just the chirp, not the full safety margin.
    display_s = chirp_us * 1e-6 * DISPLAY_MARGIN_MULT
    actual_spd = scope.set_timebase_direct(display_s / 10)
    print(f"  scope: {rate/1e9:.2f} GS/s, {capture_s*1e6:.2f} us captured "
          f"({int(round(capture_s*rate))} samples) for a "
          f"{chirp_us*1e3:.0f} ns chirp, displaying {display_s*1e6:.2f} us "
          f"({actual_spd*1e6:.3f} us/div)")


def save_result(label, f_start, f_stop, t, v, result, fig,
                 f_start_designated=None, f_stop_designated=None):
    """Save one segment's raw capture + computed metrics as a JSON (mirrors
    sweeplog.py's compact-array convention) plus the diagnostic PNG, so
    results survive past the terminal and can be reloaded/re-analyzed later
    without re-running the bench.

    f_start/f_stop: the ACTUAL commanded range (post-widen_for_flat_plateau)
    -- stored as f_start_hz/f_stop_hz, same field names as always, so every
    existing tool that reads a capture and uses these directly for the
    reference phase model (pipeline_view.py, chirp_quality.analyze()) keeps
    working correctly without needing to know about widening at all.

    f_start_designated/f_stop_designated: the narrower ORIGINAL range that
    should actually be flat -- optional, stored separately for tools that
    want to crop out the tapered edges when aggregating many segments
    (band_overview.py). None for callers that don't widen (kept equal to
    f_start/f_stop in that case, so downstream code always has a sensible
    designated range to fall back on)."""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    safe_label = label.replace(".", "p").replace("-", "_").replace(" ", "")
    base = f"{ts}_{safe_label}"

    lin, mf = result["linearity"], result["matched_filter"]
    dc, evm = result["dechirp"], result["evm"]
    doc = {
        "label": label,
        "timestamp": ts,
        "f_start_hz": f_start,
        "f_stop_hz": f_stop,
        "f_start_designated_hz": f_start_designated if f_start_designated is not None else f_start,
        "f_stop_designated_hz": f_stop_designated if f_stop_designated is not None else f_stop,
        "params": {
            "chirp_us": CHIRP_US, "dead_us": DEAD_US, "amp_vpp": AMP_VPP,
            "sync_amp_vpp": SYNC_AMP_VPP, "sync_pulse_ns": SYNC_PULSE_NS,
            "sync_carrier_hz": SYNC_CARRIER_HZ, "n_avg": N_AVG,
            "n_shots_avg": N_SHOTS_AVG,
        },
        # np.median, not t[1]-t[0] -- matches what every chirp_quality.py
        # function internally uses (dt = np.median(np.diff(t))) to derive
        # its own dt, so a reloaded capture reconstructs the identical time
        # axis metrics were actually computed against, not just an
        # approximation of it (a real, if tiny, sub-ppm gap otherwise).
        "dt_s": float(np.median(np.diff(t))),
        "n_points": len(t),
        "metrics": {
            "linearity_rms_hz": lin["rms_hz"],
            "linearity_peak_hz": lin["peak_hz"],
            "rolloff_hz": result["rolloff_hz"],
            "mainlobe_width_s": mf["mainlobe_width_s"],
            "ideal_width_s": mf["ideal_width_s"],
            "psl_db": mf["psl_db"],
            "dechirp_freq_err_rms_hz": dc["freq_err_rms_hz"],
            "dechirp_freq_err_peak_hz": dc["freq_err_peak_hz"],
            "evm_rms_pct": evm["evm_rms_pct"],
            "evm_peak_pct": evm["evm_peak_pct"],
        },
        "voltage_v": "@@WF@@",   # placeholder -> compact one-line array below
    }
    text = json.dumps(doc, indent=2)
    arr = "[" + ",".join(f"{float(x):.6g}" for x in v) + "]"
    text = text.replace('"@@WF@@"', arr, 1)

    json_path = os.path.join(CAPTURE_DIR, base + ".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    png_path = os.path.join(CAPTURE_DIR, base + ".png")
    fig.savefig(png_path, dpi=130)
    print(f"  saved {json_path}\n  saved {png_path}")
    return json_path, png_path


def run_segment(awg, scope, f_start, span_hz):
    # f_start/f_stop here are the DESIGNATED range -- what should end up
    # flat/full-amplitude, used for the label and for grouping/cropping in
    # aggregate views (band_overview.py). The AWG actually gets commanded a
    # WIDER range (f_start_cmd/f_stop_cmd) so the flat plateau -- not the
    # tapered edges -- lands on the designated band; see
    # awg.widen_for_flat_plateau() for the reasoning + the real-data
    # confirmation that a chirp is NOT actually at full power at its own
    # nominal edge frequencies otherwise. All signal-processing downstream
    # (cq.analyze/plot, and this segment's saved f_start_hz/f_stop_hz used
    # by every other tool that re-loads this capture) uses f_start_cmd/
    # f_stop_cmd, since that's what the hardware actually produced -- the
    # reference phase model has to match reality, not the aspiration.
    f_stop = f_start + span_hz
    f_start_cmd, f_stop_cmd = widen_for_flat_plateau(f_start, f_stop)
    carrier_hz = (f_start_cmd + f_stop_cmd) / 2
    half_span_hz = (f_stop_cmd - f_start_cmd) / 2
    label = f"{f_start/1e9:.2f}-{f_stop/1e9:.2f} GHz"

    pause(f"\n[{label}] press Enter to generate + arm scope...")
    # sync_carrier_hz=0 was tried and measured ~30 mV / a few tens of ns on
    # the scope instead of a clean 0.5 Vpp / 40 ns pulse -- this AWG has the
    # AC-coupled output module (see project_awg_proteus memory), which
    # blocks DC/low frequencies by design, so a near-DC "square" pulse gets
    # differentiated into a weak edge-spike rather than passing through
    # flat. Use a genuine RF burst instead (which this module handles fine),
    # just FIXED across all segments rather than tracking the DUT carrier --
    # that decouples trigger reliability from the rolloff we're testing for,
    # without fighting the AC coupling.
    awg.send_chirp_duc_sync(carrier_hz, -half_span_hz, half_span_hz,
                             CHIRP_US, DEAD_US, amplitude_vpp=AMP_VPP,
                             sync_pulse_ns=SYNC_PULSE_NS,
                             sync_amp_vpp=SYNC_AMP_VPP,
                             sync_carrier_hz=SYNC_CARRIER_HZ)
    arm_scope(scope, CHIRP_US)
    pause(f"[{label}] live -- verify CH1 chirp + CH2 trigger on the scope. "
          f"Enter to capture {N_SHOTS_AVG} shot(s)...")

    shots = []
    for shot_i in range(N_SHOTS_AVG):
        vpp, t, v = scope.measure_vpp(channel=SIG_CH, settle=SETTLE_S)
        t, v = np.asarray(t), np.asarray(v)
        if shot_i == 0:
            captured_us = (t[-1] - t[0]) * 1e6
            sample_rate_gsps = 1e-9 / (t[1] - t[0])
            print(f"  captured {len(t)} samples, {captured_us:.3f} us @ "
                  f"{sample_rate_gsps:.2f} GS/s (requested {CHIRP_US:.3f} us chirp)")
            if captured_us < CHIRP_US:
                print(f"  WARNING: captured window is SHORTER than the chirp "
                      f"itself -- raise CAPTURE_MARGIN_MULT (currently "
                      f"{CAPTURE_MARGIN_MULT}).")
        shots.append((t, v))

    if len(shots) > 1:
        t, v = cq.align_and_average(shots)
        print(f"  software-aligned + averaged {len(shots)} shots")
    else:
        t, v = shots[0]

    result = cq.analyze(t, v, f_start_cmd, f_stop_cmd, CHIRP_US * 1e-6)
    lin, mf = result["linearity"], result["matched_filter"]
    dc, evm = result["dechirp"], result["evm"]
    roll = f"{result['rolloff_hz']/1e9:.2f} GHz" if result["rolloff_hz"] else "none in segment"
    print(f"  [{label}] (commanded {f_start_cmd/1e9:.4f}-{f_stop_cmd/1e9:.4f} GHz for a "
          f"flat plateau there) lin rms={lin['rms_hz']/1e6:.2f} MHz peak={lin['peak_hz']/1e6:.2f} MHz  "
          f"rolloff(-3dB)={roll}  IRW={mf['mainlobe_width_s']*1e9:.2f} ns "
          f"(ideal {mf['ideal_width_s']*1e9:.2f} ns)  PSLR={mf['psl_db']:.1f} dB")
    print(f"  [{label}] dechirp freq err rms={dc['freq_err_rms_hz']/1e6:.3f} MHz "
          f"peak={dc['freq_err_peak_hz']/1e6:.3f} MHz  "
          f"EVM rms={evm['evm_rms_pct']:.2f}% peak={evm['evm_peak_pct']:.2f}%")

    import matplotlib.pyplot as plt
    fig = cq.plot(t, v, result, f_start_cmd, f_stop_cmd, title=f"chirp {label}")
    save_result(label, f_start_cmd, f_stop_cmd, t, v, result, fig,
                f_start_designated=f_start, f_stop_designated=f_stop)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return label, result


def run():
    awg = AWG()
    # Lock the AWG to the scope's 10 MHz reference -- needs a physical BNC
    # cable from the scope's REF OUT to the AWG's REF IN. Independent
    # clocks drift the whole waveform's timebase slightly differently on
    # every shot (not just a fixed offset), which align_and_average()'s
    # simple time-shift alignment cannot correct -- this removes that
    # drift at the source instead, on top of the software alignment.
    awg.set_reference_external(freq_mhz=10)
    scope = None
    summary = []
    try:
        scope = Scope()
        for f_start, span_hz in SEGMENTS:
            label, result = run_segment(awg, scope, f_start, span_hz)
            summary.append((label, result))
    finally:
        cleanups = [("awg.stop", awg.stop)]
        if scope is not None:
            cleanups.append(("scope.restore", lambda: scope.restore(channel=SIG_CH)))
        for name, fn in cleanups:
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}\n")

    print("\n=== summary ===")
    for label, result in summary:
        lin, mf = result["linearity"], result["matched_filter"]
        evm = result["evm"]
        roll = f"{result['rolloff_hz']/1e9:.2f} GHz" if result["rolloff_hz"] else "flat"
        print(f"{label:16s}  rms={lin['rms_hz']/1e6:6.2f} MHz  "
              f"peak={lin['peak_hz']/1e6:7.2f} MHz  rolloff={roll:10s}  "
              f"PSLR={mf['psl_db']:6.1f} dB  EVM={evm['evm_rms_pct']:5.2f}%")
    return summary


if __name__ == "__main__":
    run()
