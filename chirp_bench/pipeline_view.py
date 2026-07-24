"""Interactive viewer for chirp_quality.py's per-metric internals --
recomputes every intermediate array (calling the SAME production functions
chirp_quality.py itself uses, not reimplementations) on a saved bench
capture and plots the whole chain in one zoomable matplotlib window,
instead of only the final rms/peak numbers.

This is the generalized/renamed successor to dechirp_pipeline_view.py --
same tool, extended with --metric so every metric can be walked through
the same rigorous way dechirp was (real capture -> exact formulas from
chirp_quality.py -> intermediate arrays plotted), one at a time.

Usage:
    python pipeline_view.py                              interactive picker, dechirp
    python pipeline_view.py 1p00_1p80                     match by substring, dechirp
    python pipeline_view.py 1p00_1p80 --metric linearity  trace a different metric
    python pipeline_view.py chirp_captures\\foo.json --metric linearity
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, butter, filtfilt

import chirp_quality as cq
from chirp_quality import sinc_reconstruct as _sinc_reconstruct

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirp_captures")


def _capture_summary(path):
    """One-line label + key metrics for a capture, so a file listing says
    something about what's actually IN it (freq range, how clean it was)
    instead of just a timestamped filename -- picking blind by index/name
    alone doesn't tell you what you're about to open."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception as e:
        return f"(couldn't read: {e})"
    m = doc.get("metrics", {})
    dc_rms = m.get("dechirp_freq_err_rms_hz")
    evm = m.get("evm_rms_pct")
    psl = m.get("psl_db")
    parts = [doc.get("label", "?")]
    if dc_rms is not None:
        parts.append(f"dechirp={dc_rms/1e6:.3f}MHz")
    if evm is not None:
        parts.append(f"EVM={evm:.1f}%")
    if psl is not None:
        parts.append(f"PSLR={psl:.1f}dB")
    return "  ".join(parts)


def pick_capture(query):
    files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.json")))
    if not files:
        raise SystemExit(f"no captures found in {CAPTURE_DIR}")

    if query:
        if os.path.isfile(query):
            return query
        matches = [f for f in files if query in os.path.basename(f)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"'{query}' matches {len(matches)} captures -- showing the most recent:")
            for m in matches:
                print(f"  {os.path.basename(m)}  --  {_capture_summary(m)}")
            return matches[-1]
        raise SystemExit(f"no capture matches '{query}'")

    print("captures in", CAPTURE_DIR)
    for i, f in enumerate(files):
        ts = os.path.basename(f).split("_")[0] + " " + os.path.basename(f).split("_")[1]
        print(f"  [{i:3d}] {ts}  {_capture_summary(f)}")
    default = len(files) - 1
    idx = input(f"pick one [0-{default}, default={default} (most recent)]: ").strip()
    return files[int(idx) if idx else default]


def load_capture(capture_path):
    with open(capture_path) as f:
        doc = json.load(f)
    v = np.array(doc["voltage_v"])
    dt = doc["dt_s"]
    t = np.arange(len(v)) * dt
    f_start, f_stop = doc["f_start_hz"], doc["f_stop_hz"]
    duration = doc["params"]["chirp_us"] * 1e-6
    return doc, t, v, f_start, f_stop, duration


# ---------------------------------------------------------------------------
# dechirp_residual() pipeline
# ---------------------------------------------------------------------------

def trace_dechirp(doc, t, v, f_start, f_stop, duration, lowpass_hz=20e6,
                   edge_frac=0.05, window_frac=0.05):
    """Same math as chirp_quality.dechirp_residual(), but keeping every
    intermediate array instead of only the final rms/peak -- so the whole
    chain can be plotted, not just the answer."""
    dt = float(np.median(np.diff(t)))
    t0 = cq.refine_t0(t, v, f_start, f_stop, duration, window_frac=window_frac)

    # "perfect digital data": the exact reference waveform, placed at t0
    ref_wave = cq.ideal_chirp_waveform(f_start, f_stop, duration, dt, window_frac)
    ref_full = np.zeros_like(v)
    i0 = int(round((t0 - t[0]) / dt))
    ref_full[i0:i0 + len(ref_wave)] = ref_wave[:len(ref_full) - i0]

    tau = t - t0
    tau_c = np.clip(tau, 0, duration)
    phase_ideal = 2 * np.pi * (f_start * tau_c + (f_stop - f_start) / (2 * duration) * tau_c ** 2)

    analytic = hilbert(v - np.mean(v))
    mixed = analytic * np.exp(-1j * phase_ideal)   # THE PRODUCT

    nyq = 0.5 / dt
    wn = min(lowpass_hz / nyq, 0.99)
    b, a = butter(4, wn, btype="low")
    residual = filtfilt(b, a, mixed)

    lo, hi = duration * edge_frac, duration * (1 - edge_frac)
    mask = (tau >= lo) & (tau <= hi)
    phase_err = np.unwrap(np.angle(residual[mask]))
    phase_err = phase_err - phase_err[0]
    t_mask = tau[mask]
    freq_err = np.gradient(phase_err, t_mask) / (2 * np.pi)

    angle_raw = np.unwrap(np.angle(mixed[mask]))
    angle_raw = angle_raw - angle_raw[0]
    angle_res = np.unwrap(np.angle(residual[mask]))
    angle_res = angle_res - angle_res[0]
    phase_measured = np.unwrap(np.angle(analytic))

    # Envelope (Hilbert magnitude), NOT the raw oscillation, for panel 2.
    # Reconstructing amplitude from a discrete sinusoid's local min/max
    # breaks down whenever samples/cycle drifts near a low integer ratio
    # (confirmed 2026-07-24: a 4.15 GHz carrier at 12.5 GS/s -> ~3.0
    # samples/cycle produced a spurious ~10% "dip" in the RAW SAMPLES'
    # local peak values, present identically in both v and the pure-math
    # ref_full, purely from (sample rate, instantaneous frequency)
    # geometry -- nothing to do with either signal's real content, and NOT
    # fixable by decimating harder since the artifact is in the samples
    # themselves, not the rendering). The Hilbert envelope is immune to
    # this -- it uses the whole analytic signal's phase, not local extrema.
    env_meas = np.abs(analytic)
    env_ref = np.abs(hilbert(ref_full))
    spc = (1.0 / dt) / np.maximum(np.abs(cq.ideal_instantaneous_freq(tau_c, f_start, f_stop, duration)), 1.0)
    min_spc = float(np.min(spc[(tau >= 0) & (tau <= duration)])) if np.any((tau >= 0) & (tau <= duration)) else float("nan")

    # Constant absolute-phase offset (phi0): the arbitrary, per-capture
    # reference phase confirmed 2026-07-24 to sit at a large, essentially
    # CONSTANT value across the whole pulse (varies capture-to-capture,
    # -176..+155 deg observed, no shared phase reference between the AWG's
    # NCO and the scope's clock across independent trigger events) -- this
    # is exactly what phase_err's "-= phase_err[0]" anchoring removes, so
    # dechirp's own trusted numbers never see it. Estimated here as the mean
    # of the (already lowpassed) residual's angle over the trimmed window --
    # averaging the complex value first, then taking angle, is robust to the
    # residual's own tiny wobble without needing unwrap.
    phi0 = float(np.angle(np.mean(residual[mask])))

    # "Matched" reference: the SAME ideal chirp, rotated by phi0 -- lets the
    # real-valued overlay actually visually align (when the chirp itself is
    # good), instead of every capture looking rotated by an arbitrary,
    # meaningless amount. This does NOT hide a real defect: phi0 is a single
    # constant for the whole capture, so any genuine time-varying phase
    # error (the actual freq_err signal) is untouched and still visible as
    # a growing/shrinking mismatch, not rotated away.
    n_active = len(ref_wave)
    tau_local = np.arange(n_active) * dt
    phase_local = 2 * np.pi * (f_start * tau_local + (f_stop - f_start) / (2 * duration) * tau_local ** 2)
    # cos(), not sin() -- the mixing math (analytic * exp(-i*phase_ideal))
    # implicitly assumes a COSINE convention (hilbert(cos(th))=exp(i*th),
    # but hilbert(sin(th))=-i*exp(i*th), a built-in 90deg offset between the
    # two) while ideal_chirp_waveform()/ref_full use sin() -- harmless for
    # ref_full (only ever compared via envelope or matched-filter magnitude,
    # never a signed real overlay) but would silently rotate this panel by
    # 90 deg if left as sin(). Verified empirically against real capture 80.
    ref_wave_matched = np.cos(phase_local + phi0) * cq._gauss_win(n_active, window_frac)
    ref_matched_full = np.zeros_like(v)
    ref_matched_full[i0:i0 + len(ref_wave_matched)] = ref_wave_matched[:len(ref_matched_full) - i0]

    return dict(doc=doc, t=t, v=v, t0=t0, ref_full=ref_full, phase_ideal=phase_ideal,
                phase_measured=phase_measured, t_mask=t_mask, angle_raw=angle_raw,
                angle_res=angle_res, phase_err=phase_err, freq_err=freq_err,
                env_meas=env_meas, env_ref=env_ref, min_samples_per_cycle=min_spc,
                phi0=phi0, ref_matched_full=ref_matched_full)


def plot_dechirp(r):
    doc, t, v, t0 = r["doc"], r["t"], r["v"], r["t0"]
    t_mask, freq_err = r["t_mask"], r["freq_err"]
    rms_mhz = np.sqrt(np.mean(freq_err ** 2)) / 1e6
    peak_mhz = np.max(np.abs(freq_err)) / 1e6

    fig, axes = plt.subplots(7, 1, figsize=(10, 17))

    ax = axes[0]
    xd, yd = _sinc_reconstruct(t * 1e6, v)
    ax.plot(xd, yd, lw=0.6, color="0.2", zorder=1, label="reconstructed")
    ax.plot(t * 1e6, v, 'o', ms=2, color="C3", zorder=2, alpha=0.35, label="raw samples")
    ax.axvline(t0 * 1e6, color="C3", ls=":", lw=1, label=f"t0={t0*1e6:.3f} us")
    ax.set_title("1. MEASURED DATA  —  v(t)  (raw samples over band-limited reconstruction, same idea as scope Sin(x)/x)", fontsize=9)
    ax.set_ylabel("V"); ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    v_n = v / np.max(np.abs(v))
    ref_n = r["ref_full"] / (np.max(np.abs(r["ref_full"])) + 1e-30)
    xd1, yd1 = _sinc_reconstruct(t * 1e6, v_n)
    xd2, yd2 = _sinc_reconstruct(t * 1e6, ref_n)
    ax.plot(xd1, yd1, lw=0.6, color="C1", zorder=1, label="measured (norm.)")
    ax.plot(t * 1e6, v_n, 'o', ms=2, color="C1", zorder=2, alpha=0.3)
    ax.plot(xd2, yd2, lw=0.6, color="C0", ls="--", zorder=1, label="perfect digital data (norm.)")
    ax.plot(t * 1e6, ref_n, 'o', ms=2, color="C0", zorder=2, alpha=0.3)
    spc = r["min_samples_per_cycle"]
    ax.set_title(f"2. PERFECT DIGITAL DATA vs MEASURED (raw samples over reconstruction, "
                 f"min samples/cycle={spc:.2f})", fontsize=9)
    ax.set_ylabel("norm."); ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    vm_n = v / np.max(np.abs(v))
    refm_n = r["ref_matched_full"] / (np.max(np.abs(r["ref_matched_full"])) + 1e-30)
    xd3, yd3 = _sinc_reconstruct(t * 1e6, vm_n)
    xd4, yd4 = _sinc_reconstruct(t * 1e6, refm_n)
    ax.plot(xd3, yd3, lw=0.6, color="C1", zorder=1, label="measured (norm.)")
    ax.plot(t * 1e6, vm_n, 'o', ms=2, color="C1", zorder=2, alpha=0.3)
    ax.plot(xd4, yd4, lw=0.6, color="C0", ls="--", zorder=1, label="ideal, phase-corrected (norm.)")
    ax.plot(t * 1e6, refm_n, 'o', ms=2, color="C0", zorder=2, alpha=0.3)
    ax.set_title(f"3. MATCHED: same reference rotated by the constant offset phi0={np.degrees(r['phi0']):.1f}deg "
                 f"found by the mixing step below -- should now visually align if the chirp is actually clean", fontsize=9)
    ax.set_ylabel("norm."); ax.legend(fontsize=8, loc="upper right")

    ax = axes[3]
    ax.plot(t * 1e6, r["phase_measured"], lw=1.2, color="C1", label="measured")
    ax.plot(t * 1e6, r["phase_ideal"], lw=1.2, color="C0", ls="--", label="ideal")
    ax.set_title("4. UNWRAPPED PHASE: measured vs ideal")
    ax.set_ylabel("radians"); ax.legend(fontsize=8, loc="upper left")

    ax = axes[4]
    ax.plot(t_mask * 1e6, np.degrees(r["angle_raw"]), lw=0.4, color="0.75",
            label="angle(mixed) — raw, pre-lowpass")
    ax.plot(t_mask * 1e6, np.degrees(r["angle_res"]), lw=1.2, color="C3",
            label="angle(residual) — after lowpass")
    ax.set_title("5. THE PRODUCT: before/after lowpass")
    ax.set_ylabel("deg (anchored)"); ax.legend(fontsize=8, loc="upper right")

    ax = axes[5]
    ax.plot(t_mask * 1e6, r["phase_err"], lw=1.2, color="C3")
    ax.set_title("6. phase_err(t)")
    ax.set_ylabel("rad")

    ax = axes[6]
    ax.plot(t_mask * 1e6, freq_err / 1e6, lw=1.2, color="C3")
    ax.set_title(f"7. freq_err(t)  —  rms={rms_mhz:.3f} MHz  peak={peak_mhz:.3f} MHz")
    ax.set_ylabel("MHz"); ax.set_xlabel("time (us)")

    fig.suptitle(f"dechirp_residual() pipeline: {doc['label']} ({doc['timestamp']})", y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# linearity_error() pipeline
# ---------------------------------------------------------------------------

def trace_linearity(doc, t, v, f_start, f_stop, duration, edge_frac=0.05):
    """Traces linearity_error() -- calls the actual production functions
    (refine_t0, demodulate, demodulate_raw, ideal_instantaneous_freq,
    linearity_error itself) rather than reimplementing them, so the
    numbers shown here are guaranteed identical to what run_chirp_quality.py
    reports. The only thing duplicated is the raw Hilbert phase BEFORE
    smoothing/differentiation (demodulate()'s first two lines) -- that
    intermediate isn't returned by demodulate() itself, but is the most
    useful new panel here: it shows whether the phase itself looks clean
    BEFORE the derivative step amplifies noise, same idea as the dechirp
    tracer's phase-comparison panel.
    """
    t0 = cq.refine_t0(t, v, f_start, f_stop, duration)

    analytic = hilbert(v - np.mean(v))
    phase_measured_raw = np.unwrap(np.angle(analytic))

    tau_full = t - t0
    tau_full_c = np.clip(tau_full, 0, duration)
    phase_ideal_full = 2 * np.pi * (f_start * tau_full_c +
                                     (f_stop - f_start) / (2 * duration) * tau_full_c ** 2)

    lin = cq.linearity_error(t, v, f_start, f_stop, duration, t0=t0, edge_frac=edge_frac)

    # Sanity check: these must match doc["metrics"] exactly (same t0, same
    # capture, same function) -- printed, not just trusted, per the standing
    # "show real numbers" preference.
    saved_rms = doc.get("metrics", {}).get("linearity_rms_hz")
    saved_peak = doc.get("metrics", {}).get("linearity_peak_hz")
    if saved_rms is not None:
        # Relative tolerance: captures saved before the dt_s fix (see
        # run_chirp_quality.py save_result()) reconstruct a t-axis that's
        # only an approximation (sub-ppm) of the one metrics were actually
        # computed against -- not a real discrepancy, just old-format noise.
        rel_err = abs(lin["rms_hz"] - saved_rms) / max(abs(saved_rms), 1.0)
        print(f"  sanity check vs saved JSON: rms {lin['rms_hz']/1e6:.6f} MHz "
              f"(saved {saved_rms/1e6:.6f})  peak {lin['peak_hz']/1e6:.6f} MHz "
              f"(saved {saved_peak/1e6:.6f})  "
              f"{'MATCH' if rel_err < 1e-4 else f'MISMATCH! (rel err {rel_err:.2e})'}")

    return dict(doc=doc, t=t, v=v, t0=t0, phase_measured_raw=phase_measured_raw,
                phase_ideal_full=phase_ideal_full, tau_full=tau_full, lin=lin)


def plot_linearity(r):
    doc, t, v, t0, lin = r["doc"], r["t"], r["v"], r["t0"], r["lin"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 13))

    ax = axes[0]
    xd, yd = _sinc_reconstruct(t * 1e6, v)
    ax.plot(xd, yd, lw=0.6, color="0.2", zorder=1, label="reconstructed")
    ax.plot(t * 1e6, v, 'o', ms=2, color="C3", zorder=2, alpha=0.35, label="raw samples")
    ax.axvline(t0 * 1e6, color="C3", ls=":", lw=1, label=f"t0={t0*1e6:.3f} us")
    ax.set_title("1. MEASURED DATA  —  v(t)  (raw samples over band-limited reconstruction, same idea as scope Sin(x)/x)", fontsize=9)
    ax.set_ylabel("V"); ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(r["tau_full"] * 1e6, r["phase_measured_raw"], lw=1.0, color="C1", label="measured (Hilbert, unwrapped)")
    ax.plot(r["tau_full"] * 1e6, r["phase_ideal_full"], lw=1.0, color="C0", ls="--", label="ideal (commanded)")
    ax.set_title("2. UNWRAPPED PHASE, before any smoothing/differentiation")
    ax.set_ylabel("radians"); ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    ax.plot(lin["t_raw"] * 1e6, lin["freq_meas_raw"] / 1e9, color="0.75", lw=0.3,
            label="measured, raw two-point diff", zorder=1)
    ax.plot(lin["t"] * 1e6, lin["freq_ideal"] / 1e9, "k--", lw=1, label="commanded", zorder=3)
    ax.plot(lin["t"] * 1e6, lin["freq_meas"] / 1e9, lw=1.0, color="C1",
            label="measured, Savitzky-Golay smoothed", zorder=2)
    ax.set_title("3. INSTANTANEOUS FREQUENCY  —  d(phase)/dt, raw vs smoothed, vs commanded ramp")
    ax.set_ylabel("GHz"); ax.legend(fontsize=8)

    ax = axes[3]
    ax.plot(lin["t_raw"] * 1e6, lin["err_hz_raw"] / 1e6, color="0.75", lw=0.3,
            label="raw", zorder=1)
    ax.plot(lin["t"] * 1e6, lin["err_hz"] / 1e6, lw=1.2, color="C3",
            label="smoothed", zorder=2)
    ax.set_title("4. freq_err(t) = measured − commanded")
    ax.set_ylabel("MHz"); ax.legend(fontsize=8)

    ax = axes[4]
    ax.plot(lin["t"] * 1e6, lin["amp"], lw=0.8, color="C4")
    ax.set_title("5. envelope amplitude |analytic(t)| (carried through, used by swept_amplitude_response)")
    ax.set_ylabel("V"); ax.set_xlabel("time (us)")

    fig.suptitle(f"linearity_error() pipeline: {doc['label']} ({doc['timestamp']})  "
                 f"rms={lin['rms_hz']/1e6:.3f} MHz  peak={lin['peak_hz']/1e6:.3f} MHz "
                 f"(smoothed)  |  raw rms={lin['rms_hz_raw']/1e6:.3f} peak={lin['peak_hz_raw']/1e6:.3f} MHz",
                 y=1.0, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# matched_filter_score() pipeline
# ---------------------------------------------------------------------------

def trace_matched_filter(doc, t, v, f_start, f_stop, duration, window_frac=0.05,
                          guard_factor=3.0):
    """Traces matched_filter_score() -- calls the actual production function
    for every returned number (lag_s, corr_db, mainlobe_width_s,
    ideal_width_s, psl_db), so those are guaranteed identical to what
    run_chirp_quality.py reports. The only thing duplicated is the segment
    extraction (seg = the mean-subtracted measured window actually fed into
    the correlation) -- matched_filter_score() computes this internally but
    doesn't return it, and seeing exactly what got correlated against what
    is the point of this panel."""
    t0 = cq.refine_t0(t, v, f_start, f_stop, duration, window_frac=window_frac)
    dt = float(np.median(np.diff(t)))
    ref = cq.ideal_chirp_waveform(f_start, f_stop, duration, dt, window_frac)
    n = len(ref)
    i0 = int(round((t0 - t[0]) / dt))
    seg = v[max(i0, 0):i0 + n] - np.mean(v[max(i0, 0):i0 + n])
    t_seg = np.arange(len(seg)) * dt

    mf = cq.matched_filter_score(t, v, f_start, f_stop, duration, t0=t0,
                                  window_frac=window_frac, guard_factor=guard_factor)

    saved = doc.get("metrics", {})
    if saved.get("mainlobe_width_s") is not None:
        print(f"  IRW {mf['mainlobe_width_s']*1e9:.4f} ns  PSLR {mf['psl_db']:.4f} dB  "
              f"(saved JSON has IRW {saved['mainlobe_width_s']*1e9:.4f} ns / PSLR {saved['psl_db']:.4f} dB "
              f"from BEFORE the 2026-07-24 envelope-based mainlobe fix -- a difference here is "
              f"expected, not a bug; old captures' matched_filter numbers are stale)")

    # re-walk the ENVELOPE-based mainlobe search the same way
    # matched_filter_score() does internally, purely so the same
    # left/right/guard region (used for the plot) is available here --
    # matched_filter_score() returns corr_env_db itself now, so this no
    # longer needs to recompute the envelope from scratch.
    peak_idx = mf["peak_idx"]
    corr_db = mf["corr_db"]
    corr_env_db = mf["corr_env_db"]
    above = corr_env_db > -3.0
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(above) - 1 and above[right + 1]:
        right += 1
    guard = int(round(guard_factor * max(right - left, 1) / 2)) + 1
    guard_lo = max(peak_idx - guard, 0)
    guard_hi = min(peak_idx + guard, len(corr_db) - 1)

    return dict(doc=doc, t=t, v=v, t0=t0, seg=seg, t_seg=t_seg, ref=ref, mf=mf,
                guard_lo=guard_lo, guard_hi=guard_hi, left=left, right=right,
                corr_env_db=corr_env_db)


def plot_matched_filter(r):
    doc, mf = r["doc"], r["mf"]
    lag_ns = mf["lag_s"] * 1e9
    corr_db = mf["corr_db"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    ax = axes[0]
    xd, yd = _sinc_reconstruct(r["t_seg"] * 1e6, r["seg"] / np.max(np.abs(r["seg"])))
    xd2, yd2 = _sinc_reconstruct(r["t_seg"] * 1e6, r["ref"] / np.max(np.abs(r["ref"])))
    ax.plot(xd, yd, lw=0.6, color="C1", zorder=1, label="measured segment (norm.)")
    ax.plot(r["t_seg"] * 1e6, r["seg"] / np.max(np.abs(r["seg"])), 'o', ms=2,
            color="C1", alpha=0.3, zorder=2)
    ax.plot(xd2, yd2, lw=0.6, color="C0", ls="--", zorder=1, label="ideal reference (norm.)")
    ax.set_title("1. WHAT GETS CORRELATED: measured segment vs ideal reference chirp")
    ax.set_xlabel("time (us)"); ax.set_ylabel("norm."); ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(lag_ns, corr_db, lw=0.8, color="C1")
    ax.axhline(-3.0, color="k", ls=":", lw=0.8)
    ax.axvspan(mf["lag_s"][r["guard_lo"]] * 1e9, mf["lag_s"][r["guard_hi"]] * 1e9,
               color="0.85", alpha=0.5, label="PSLR search excludes this guard region")
    ax.set_ylim(-40, 2)
    ax.set_title("2. FULL CORRELATION (dB) -- whole record")
    ax.set_xlabel("lag (ns)"); ax.set_ylabel("dB"); ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    zoom_ns = max(mf["mainlobe_width_s"], mf["ideal_width_s"]) * 15e9
    zoom_ns = max(zoom_ns, 5.0)  # keep a sane minimum span even if IRW is tiny
    ax.plot(lag_ns, corr_db, lw=0.5, color="0.75", label="raw |corr| (rides carrier ripple)")
    ax.plot(lag_ns, r["corr_env_db"], lw=1.4, color="C1", label="envelope (used for IRW)")
    ax.axhline(-3.0, color="k", ls=":", lw=0.8, label="-3dB (IRW threshold)")
    ax.axvline(lag_ns[r["left"]], color="C3", ls="--", lw=0.8)
    ax.axvline(lag_ns[r["right"]], color="C3", ls="--", lw=0.8)
    ax.set_xlim(-zoom_ns, zoom_ns)
    ax.set_ylim(-40, 2)
    ax.set_title(f"3. ZOOM ON MAINLOBE  —  IRW={mf['mainlobe_width_s']*1e9:.3f} ns "
                 f"(ideal {mf['ideal_width_s']*1e9:.3f} ns)  PSLR={mf['psl_db']:.2f} dB")
    ax.set_xlabel("lag (ns)"); ax.set_ylabel("dB"); ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"matched_filter_score() pipeline: {doc['label']} ({doc['timestamp']})", y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# overview -- the original all-metrics-in-one-window dashboard
# (chirp_quality.analyze() + chirp_quality.plot(), the same 4x2 grid
# run_chirp_quality.py saves a PNG of after every bench segment), reachable
# through the same picker/CLI workflow as the per-metric deep-dive views
# instead of only as a static post-bench PNG.
# ---------------------------------------------------------------------------

def trace_overview(doc, t, v, f_start, f_stop, duration):
    result = cq.analyze(t, v, f_start, f_stop, duration)
    return dict(doc=doc, t=t, v=v, f_start=f_start, f_stop=f_stop, result=result)


def plot_overview(r):
    title = f"{r['doc']['label']} ({r['doc']['timestamp']})"
    return cq.plot(r["t"], r["v"], r["result"], r["f_start"], r["f_stop"], title=title)


METRICS = {
    "dechirp": (trace_dechirp, plot_dechirp),
    "linearity": (trace_linearity, plot_linearity),
    "matched_filter": (trace_matched_filter, plot_matched_filter),
    "overview": (trace_overview, plot_overview),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--metric", choices=list(METRICS.keys()), default="dechirp")
    args = parser.parse_args()

    capture_path = pick_capture(args.query)
    print("tracing", os.path.basename(capture_path), "--metric", args.metric)
    doc, t, v, f_start, f_stop, duration = load_capture(capture_path)
    trace_fn, plot_fn = METRICS[args.metric]
    result = trace_fn(doc, t, v, f_start, f_stop, duration)
    plot_fn(result)
    plt.show()
