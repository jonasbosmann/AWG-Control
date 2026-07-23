"""Quantify AWG chirp quality from a scope capture.

A linear-FM (LFM) chirp maps time <-> frequency, so a single wideband scope
capture is enough to recover, from the SAME data, both:

  1. Frequency-vs-time linearity -- is the sweep actually landing where it
     was commanded, or does it compress/expand/hop near the module's
     bandwidth edge?
  2. The chain's swept amplitude response -- plotting the envelope against
     the *measured* instantaneous frequency (not time) gives you an
     AWG+cable+amp frequency response for free, no VNA/spec-an sweep needed.

Both come from a Hilbert-transform (analytic-signal) demodulation of the
capture. A third, independent check -- matched-filter / pulse-compression
against the ideal commanded chirp, the standard LFM-radar QC trick -- gives
one aggregate number (mainlobe width, peak sidelobe level) that's robust to
spurs strong enough to confuse the instantaneous-frequency estimate.

This module has no hardware dependency -- it operates on plain (t, v) numpy
arrays, whether from scope.get_waveform() or a synthetic test signal (see
the __main__ self-test at the bottom, runnable with no instruments attached).
"""

import os
import tempfile

import numpy as np
from scipy.signal import (correlate as _correlate, hilbert, savgol_filter,
                           spectrogram as _spectrogram)


# ---------------------------------------------------------------------------
# Ideal (commanded) chirp model -- matches awg.py's _chirp_windowed_u16:
# quadratic phase (linear FM) inside a Gaussian-edged flat-top window.
# ---------------------------------------------------------------------------

def _gauss_win(n, frac):
    """Same window as awg.py's _gauss_win, duplicated here so this module
    stays hardware-free. Gaussian-edged flat-top; edges span frac*n samples,
    capped at n//2 so head and tail can never overlap."""
    w = np.ones(n)
    if n < 2:
        return w
    n_w = min(max(int(frac * n), 1), n // 2)
    sigma = n_w / 3.0
    idx = np.arange(n_w, dtype=np.float64)
    w[:n_w] = np.exp(-0.5 * ((idx - n_w) / sigma) ** 2)
    w[-n_w:] = np.exp(-0.5 * (idx / sigma) ** 2)
    return w


def ideal_instantaneous_freq(tau, f_start, f_stop, duration):
    """Commanded linear ramp f(tau) = f_start + (f_stop-f_start)*tau/duration,
    tau = time since the chirp's active window started. Matches the
    derivative of the quadratic phase term used to generate the chirp."""
    return f_start + (f_stop - f_start) * np.clip(tau, 0, duration) / duration


def ideal_chirp_waveform(f_start, f_stop, duration, dt, window_frac=0.05):
    """Sampled ideal chirp (unit amplitude, same phase/window model as
    awg.py), used as the matched-filter reference."""
    n = int(round(duration / dt))
    t = np.arange(n, dtype=np.float64) * dt
    phase = 2 * np.pi * (f_start * t + (f_stop - f_start) / (2 * duration) * t ** 2)
    return np.sin(phase) * _gauss_win(n, window_frac)


# ---------------------------------------------------------------------------
# Analytic-signal demodulation
# ---------------------------------------------------------------------------

def find_burst_window(t, v, thresh_frac=0.1):
    """Locate a windowed burst's start/stop via the Hilbert envelope --
    useful when the capture's timing relative to the chirp isn't known
    precisely (e.g. a raw scope grab with some pre/post dead time either
    side). Returns (t_start, t_stop).

    This is only a COARSE estimate -- the threshold crossing sits inside
    the taper region, not at the true start, biased by a sizeable fraction
    of the taper width (confirmed: ~14 ns on a 1 us, 5%-tapered chirp).
    Since the ideal ramp is exactly linear in time, that constant t0 bias
    shows up as a constant frequency-error BIAS in linearity_error(), not
    noise -- use refine_t0() for anything where absolute timing precision
    matters, which is any actual linearity measurement.
    """
    env = np.abs(hilbert(v - np.mean(v)))
    thresh = thresh_frac * np.max(env)
    idx = np.flatnonzero(env > thresh)
    if len(idx) == 0:
        raise ValueError("no signal above thresh_frac * peak -- check the capture")
    return float(t[idx[0]]), float(t[idx[-1]])


def refine_t0(t, v, f_start, f_stop, duration, t0_guess=None, window_frac=0.05,
              search_frac=0.2):
    """Precisely locate the chirp's start via matched filtering (cross-
    correlate against the ideal commanded chirp), with sub-sample parabolic
    interpolation of the correlation peak.

    Far more precise than find_burst_window()'s envelope-threshold crossing
    (see its docstring) -- matched filtering is the optimal estimator for a
    known waveform's arrival time. Always use this (not find_burst_window()
    directly) before computing linearity_error(); find_burst_window() is
    still fine as a coarse starting guess.
    """
    if t0_guess is None:
        t0_guess, _ = find_burst_window(t, v)

    dt = float(np.median(np.diff(t)))
    ref = ideal_chirp_waveform(f_start, f_stop, duration, dt, window_frac)
    n = len(ref)

    # Search a window around t0_guess wide enough to comfortably cover the
    # taper-crossing bias, so the true start is guaranteed to fall inside it.
    search_n = max(int(search_frac * duration / dt), 1)
    i_guess = int(round((t0_guess - t[0]) / dt))
    lo = max(i_guess - search_n, 0)
    hi = min(i_guess + search_n + n, len(v))
    seg = v[lo:hi] - np.mean(v[lo:hi])
    if len(seg) < n:
        return t0_guess   # not enough room to refine -- fall back to the guess

    corr = _correlate(seg, ref, mode="valid", method="fft")
    peak = int(np.argmax(np.abs(corr)))
    if 0 < peak < len(corr) - 1:
        y0, y1, y2 = np.abs(corr[peak - 1]), np.abs(corr[peak]), np.abs(corr[peak + 1])
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        frac = 0.0
    return float(t[lo] + (peak + frac) * dt)


def demodulate(t, v, smooth_window=21, poly_order=3):
    """Analytic-signal demod -> instantaneous frequency (Hz) and envelope
    amplitude, at the same sample times as the input.

    Differentiation is a high-pass operation (multiplies by iw in the
    frequency domain), so a raw two-point phase difference amplifies
    whatever noise is on the capture by ~1/dt -- noise far too small to see
    on a time-domain plot turns into large frequency-estimate jitter, and
    it gets WORSE at a higher sample rate (smaller dt), not better. Fit a
    local degree-`poly_order` polynomial to the unwrapped phase over
    `smooth_window` samples and differentiate THAT (Savitzky-Golay)
    instead of differencing adjacent samples -- suppresses sample-to-
    sample noise while preserving the chirp's genuine sweep rate, which
    varies far more slowly than `smooth_window` samples at any oversampling
    ratio worth having.
    """
    dt = float(np.median(np.diff(t)))
    analytic = hilbert(v - np.mean(v))
    phase = np.unwrap(np.angle(analytic))
    w = min(smooth_window, len(phase) - 1 + len(phase) % 2)
    if w % 2 == 0:
        w -= 1
    w = max(w, poly_order + 1 + (poly_order + 1) % 2 + 2)   # smallest valid odd window
    inst_freq = savgol_filter(phase, w, poly_order, deriv=1, delta=dt) / (2 * np.pi)
    inst_amp = np.abs(analytic)
    return t, inst_freq, inst_amp


def demodulate_raw(t, v):
    """Unsmoothed two-point-difference instantaneous frequency.

    Noisier than demodulate() (see its docstring) but full-bandwidth --
    smoothing is a low-pass filter and cannot distinguish random noise from
    a genuine fast defect (a mode-hop, interpolation ripple, a glitch), so
    a clean SMOOTHED trace is not proof the chirp is actually clean. Always
    inspect this raw trace alongside the smoothed one, not instead of it.
    The only way to reduce noise without also risking hiding a real defect
    is averaging more shots (more SNR from repeated, coherent captures),
    not filtering a single capture harder.
    """
    analytic = hilbert(v - np.mean(v))
    phase = np.unwrap(np.angle(analytic))
    dt = np.diff(t)
    inst_freq = np.diff(phase) / (2 * np.pi * dt)
    t_mid = t[:-1] + dt / 2
    return t_mid, inst_freq


# ---------------------------------------------------------------------------
# Software-aligned multi-shot averaging -- for when the hardware trigger
# isn't precise enough for scope-native coherent averaging (e.g. triggering
# on a multi-cycle RF burst rather than a single clean edge, where the
# scope's ACQuire:MODe AVErage would smear at high frequency). Each shot is
# aligned using its OWN captured waveform via cross-correlation, so this
# doesn't depend on trigger precision at all beyond "captured a complete,
# non-clipped shot" -- random noise averages down across shots; anything
# repeatable (including genuine hardware defects) survives untouched,
# unlike smoothing a single shot (see demodulate_raw()'s docstring).
# ---------------------------------------------------------------------------

def align_and_average(shots):
    """Align single-shot (t, v) captures to sub-sample precision via
    cross-correlation against the first shot, then average.

    shots: list of (t, v) tuples, all at the same dt (from the same scope
    setup). Returns (t, v_avg) on the first shot's time base.
    """
    if len(shots) == 1:
        return shots[0]

    t_ref, v_ref = shots[0]
    v_ref = np.asarray(v_ref, dtype=np.float64) - np.mean(v_ref)
    n = len(v_ref)
    aligned = [v_ref]

    for t_i, v_i in shots[1:]:
        v_i = np.asarray(v_i, dtype=np.float64) - np.mean(v_i)
        v_i = v_i[:n] if len(v_i) >= n else np.pad(v_i, (0, n - len(v_i)))

        corr = _correlate(v_i, v_ref, mode="full", method="fft")
        peak = int(np.argmax(np.abs(corr)))
        # Parabolic interpolation of the 3 samples around the peak for a
        # sub-sample-precision estimate of the true correlation maximum.
        if 0 < peak < len(corr) - 1:
            y0, y1, y2 = np.abs(corr[peak - 1]), np.abs(corr[peak]), np.abs(corr[peak + 1])
            denom = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        else:
            frac = 0.0
        # np.correlate(v_i, v_ref, 'full')'s peak sits at index (n-1) + shift,
        # where shift is how many samples v_i leads v_ref by.
        shift = (peak - (n - 1)) + frac

        # Resample v_i onto v_ref's sample grid, undoing that shift.
        src_idx = np.arange(len(v_i))
        want_idx = np.arange(n) + shift
        v_aligned = np.interp(want_idx, src_idx, v_i, left=0.0, right=0.0)
        aligned.append(v_aligned)

    return t_ref, np.mean(aligned, axis=0)


# ---------------------------------------------------------------------------
# Metric 1: frequency-vs-time linearity
# ---------------------------------------------------------------------------

def linearity_error(t, v, f_start, f_stop, duration, t0=None, edge_frac=0.05):
    """RMS/peak deviation of the measured instantaneous frequency from the
    commanded linear ramp, over the chirp's active window (with the taper
    edges trimmed -- both the Gaussian window and the Hilbert transform are
    unreliable right at the burst edges).

    Returns both the smoothed (demodulate()) and raw (demodulate_raw())
    frequency traces/errors -- see demodulate_raw()'s docstring for why the
    raw one always needs to be checked too, not just the smoothed summary.

    t0: chirp start time in the capture's time base. Auto-detected via
    refine_t0() (matched-filter, sub-sample precision) if not given -- do
    NOT substitute find_burst_window() here, its envelope-threshold crossing
    is biased by a sizeable fraction of the taper width, which (since the
    ideal ramp is linear) shows up as a constant bias in every number this
    function returns, not as noise.
    """
    if t0 is None:
        t0 = refine_t0(t, v, f_start, f_stop, duration)

    t_mid, inst_freq, inst_amp = demodulate(t, v)
    tau = t_mid - t0
    ideal = ideal_instantaneous_freq(tau, f_start, f_stop, duration)

    lo, hi = duration * edge_frac, duration * (1 - edge_frac)
    mask = (tau >= lo) & (tau <= hi)
    if not np.any(mask):
        raise ValueError("no samples fall inside the trimmed chirp window -- "
                          "check t0/duration against the capture's time axis")
    err = inst_freq[mask] - ideal[mask]

    t_raw, freq_raw = demodulate_raw(t, v)
    tau_raw = t_raw - t0
    mask_raw = (tau_raw >= lo) & (tau_raw <= hi)
    ideal_raw = ideal_instantaneous_freq(tau_raw, f_start, f_stop, duration)
    err_raw = freq_raw[mask_raw] - ideal_raw[mask_raw]

    return {
        "t": tau[mask], "freq_meas": inst_freq[mask], "freq_ideal": ideal[mask],
        "amp": inst_amp[mask], "err_hz": err,
        "rms_hz": float(np.sqrt(np.mean(err ** 2))),
        "peak_hz": float(np.max(np.abs(err))),
        "t_raw": tau_raw[mask_raw], "freq_meas_raw": freq_raw[mask_raw],
        "err_hz_raw": err_raw,
        "rms_hz_raw": float(np.sqrt(np.mean(err_raw ** 2))),
        "peak_hz_raw": float(np.max(np.abs(err_raw))),
        "t0": t0,
    }


# ---------------------------------------------------------------------------
# Metric 2: swept amplitude response (envelope vs. measured frequency)
# ---------------------------------------------------------------------------

def swept_amplitude_response(lin_result, n_bins=200):
    """Bin envelope amplitude by measured instantaneous frequency -- the
    chain's swept frequency response, extracted from the same capture used
    for the linearity check. Returns (freq_hz, amp_linear, amp_db) with
    amp_db normalized to the passband median (nan where a bin has no
    samples, e.g. if the sweep rate is very nonuniform)."""
    freq, amp = lin_result["freq_meas"], lin_result["amp"]
    order = np.argsort(freq)
    freq_s, amp_s = freq[order], amp[order]

    edges = np.linspace(freq_s[0], freq_s[-1], n_bins + 1)
    idx = np.clip(np.digitize(freq_s, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    amp_binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = amp_s[idx == i]
        if len(sel):
            amp_binned[i] = sel.mean()

    ref = np.nanmedian(amp_binned)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp_db = 20 * np.log10(amp_binned / ref)
    return centers, amp_binned, amp_db


def rolloff_frequency(centers, amp_db, threshold_db=-3.0):
    """First frequency (scanning low->high) beyond which the swept response
    stays below threshold_db all the way to the top of the band. Returns
    None if it never drops below threshold (i.e. flat over the whole
    capture -- 'good up to at least the top of this capture')."""
    valid = ~np.isnan(amp_db)
    if not np.any(valid):
        return None
    c, a = centers[valid], amp_db[valid]
    below = a < threshold_db
    stays_below = np.array([np.all(below[i:]) for i in range(len(below))])
    if not stays_below.any():
        return None
    return float(c[np.argmax(stays_below)])


# ---------------------------------------------------------------------------
# Metric 3: matched filter / pulse compression (LFM radar QC trick)
# ---------------------------------------------------------------------------

def matched_filter_score(t, v, f_start, f_stop, duration, t0=None,
                          window_frac=0.05, guard_factor=3.0):
    """Cross-correlate the capture against the ideal commanded chirp.
    Returns the compressed pulse's mainlobe width (compare to the
    diffraction-limited 1/bandwidth) and peak sidelobe level (dB relative to
    the mainlobe peak) -- a single aggregate quality number, standard
    practice for characterizing LFM/radar chirps, robust to spurs that
    would confuse instantaneous-frequency demod."""
    if t0 is None:
        t0 = refine_t0(t, v, f_start, f_stop, duration, window_frac=window_frac)

    dt = float(np.median(np.diff(t)))
    ref = ideal_chirp_waveform(f_start, f_stop, duration, dt, window_frac)
    n = len(ref)

    i0 = int(round((t0 - t[0]) / dt))
    seg = v[max(i0, 0):i0 + n]
    if len(seg) < n:
        raise ValueError(
            f"captured segment ({len(seg)} samples, {len(seg)*dt*1e6:.3f} us) is "
            f"shorter than the ideal reference chirp ({n} samples, "
            f"{n*dt*1e6:.3f} us) -- t0={t0*1e6:.3f} us was detected "
            f"{(t[-1]-t0)*1e6:.3f} us before the end of a {(t[-1]-t[0])*1e6:.3f} us "
            f"capture. Likely a bad/untriggered capture (e.g. the scope never "
            f"saw a valid trigger) rather than a real too-short record -- check "
            f"the trigger signal amplitude/level before re-running.")
    seg = seg - np.mean(seg)

    corr = np.correlate(seg, ref, mode="full")
    peak_idx = int(np.argmax(np.abs(corr)))
    peak = np.abs(corr[peak_idx])
    lag = (np.arange(len(corr)) - (n - 1)) * dt
    with np.errstate(divide="ignore"):
        corr_db = 20 * np.log10(np.abs(corr) / peak + 1e-300)

    # Mainlobe = contiguous region around the peak above -6 dB.
    above = corr_db > -6.0
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(above) - 1 and above[right + 1]:
        right += 1
    mainlobe_width_s = (right - left) * dt

    guard = int(round(guard_factor * max(right - left, 1) / 2)) + 1
    outside = np.ones(len(corr), dtype=bool)
    outside[max(peak_idx - guard, 0):peak_idx + guard + 1] = False
    psl_db = float(np.max(corr_db[outside])) if np.any(outside) else float("nan")

    return {
        "lag_s": lag, "corr_db": corr_db, "peak_idx": peak_idx,
        "mainlobe_width_s": mainlobe_width_s,
        "ideal_width_s": 1.0 / abs(f_stop - f_start),
        "psl_db": psl_db,
    }


# ---------------------------------------------------------------------------
# Spectrogram (visual-only sanity check for glitches/mode-hops the RMS
# metrics above would average away)
# ---------------------------------------------------------------------------

def spectrogram(t, v, nperseg=256):
    fs = 1.0 / float(np.median(np.diff(t)))
    f, tt, sxx = _spectrogram(v, fs=fs, nperseg=nperseg,
                               noverlap=nperseg - max(nperseg // 8, 1))
    return f, tt + t[0], sxx


# ---------------------------------------------------------------------------
# Top-level bundle + plot
# ---------------------------------------------------------------------------

def analyze(t, v, f_start, f_stop, duration, t0=None, edge_frac=0.05, n_bins=200):
    lin = linearity_error(t, v, f_start, f_stop, duration, t0=t0, edge_frac=edge_frac)
    centers, amp_binned, amp_db = swept_amplitude_response(lin, n_bins=n_bins)
    rolloff = rolloff_frequency(centers, amp_db)
    mf = matched_filter_score(t, v, f_start, f_stop, duration, t0=lin["t0"])
    return {
        "linearity": lin,
        "response_freq_hz": centers, "response_amp": amp_binned, "response_db": amp_db,
        "rolloff_hz": rolloff,
        "matched_filter": mf,
    }


def plot(t, v, result, f_start, f_stop, title=None):
    import matplotlib.pyplot as plt

    lin = result["linearity"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    if title:
        fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(t * 1e6, v, lw=0.5)
    ax.set_xlabel("time (us)"); ax.set_ylabel("V"); ax.set_title("raw capture")

    ax = axes[0, 1]
    ax.plot(lin["t_raw"] * 1e6, lin["freq_meas_raw"] / 1e9, color="0.75", lw=0.3,
            label="measured (raw)", zorder=1)
    ax.plot(lin["t"] * 1e6, lin["freq_ideal"] / 1e9, "k--", lw=1, label="commanded", zorder=3)
    ax.plot(lin["t"] * 1e6, lin["freq_meas"] / 1e9, lw=0.8, label="measured (smoothed)", zorder=2)
    ax.set_xlabel("time (us)"); ax.set_ylabel("GHz"); ax.legend(fontsize=7)
    ax.set_title("instantaneous frequency")

    ax = axes[1, 0]
    ax.plot(lin["t_raw"] * 1e6, lin["err_hz_raw"] / 1e6, color="0.75", lw=0.3, zorder=1)
    ax.plot(lin["t"] * 1e6, lin["err_hz"] / 1e6, lw=0.8, zorder=2)
    ax.set_xlabel("time (us)"); ax.set_ylabel("error (MHz)")
    ax.set_title(f"freq error  smoothed rms={lin['rms_hz']/1e6:.2f} peak={lin['peak_hz']/1e6:.2f}  "
                 f"|  raw rms={lin['rms_hz_raw']/1e6:.2f} peak={lin['peak_hz_raw']/1e6:.2f} (MHz)",
                 fontsize=9)

    ax = axes[1, 1]
    ax.plot(result["response_freq_hz"] / 1e9, result["response_db"], lw=1)
    ax.axhline(-3.0, color="r", ls=":", lw=1)
    if result["rolloff_hz"]:
        ax.axvline(result["rolloff_hz"] / 1e9, color="r", ls=":", lw=1)
    ax.set_xlabel("instantaneous frequency (GHz)"); ax.set_ylabel("dB (norm. to median)")
    roll_txt = f"{result['rolloff_hz']/1e9:.2f} GHz" if result["rolloff_hz"] else "none in band"
    ax.set_title(f"swept amplitude response  -3dB @ {roll_txt}")

    ax = axes[2, 0]
    f, tt, sxx = spectrogram(t, v)
    ax.pcolormesh(tt * 1e6, f / 1e9, 10 * np.log10(sxx + 1e-20), shading="auto")
    ax.set_ylim(0, max(f_start, f_stop) * 1.2 / 1e9)
    ax.set_xlabel("time (us)"); ax.set_ylabel("GHz"); ax.set_title("spectrogram")

    ax = axes[2, 1]
    mf = result["matched_filter"]
    ax.plot(mf["lag_s"] * 1e9, mf["corr_db"], lw=0.8)
    zoom_ns = max(mf["mainlobe_width_s"], mf["ideal_width_s"]) * 15e9
    ax.set_xlim(-zoom_ns, zoom_ns)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("lag (ns)"); ax.set_ylabel("dB")
    ax.set_title(f"pulse compression  width={mf['mainlobe_width_s']*1e9:.2f} ns "
                 f"(ideal {mf['ideal_width_s']*1e9:.2f} ns)  PSL={mf['psl_db']:.1f} dB")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Self-test: synthetic capture with a known rolloff + known nonlinearity, so
# the metrics above can be sanity-checked without a scope attached.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    f_start, f_stop = 0.2e9, 4.0e9
    duration = 2e-6
    fs = 20e9
    t_pad = 0.3e-6
    t0 = t_pad

    n_active = int(round(duration * fs))
    tau = np.arange(n_active) / fs

    # Known nonlinearity: a small sinusoidal wobble on top of the ideal
    # linear-FM ramp (simulates e.g. DUC interpolation phase ripple), with a
    # 0.8 MHz PEAK INSTANTANEOUS-FREQUENCY deviation -- phase amplitude is
    # freq_dev/f_wobble (standard FM phase<->frequency relation), not
    # freq_dev itself.
    phase_ideal = 2 * np.pi * (f_start * tau + (f_stop - f_start) / (2 * duration) * tau ** 2)
    freq_dev_hz, f_wobble = 0.8e6, 3 / duration
    phase_wobble = (freq_dev_hz / f_wobble) * np.sin(2 * np.pi * f_wobble * tau)
    sig = np.sin(phase_ideal + phase_wobble) * _gauss_win(n_active, 0.05)

    # Known rolloff: -15 dB/GHz decline above 3.5 GHz (simulates a
    # DC-coupled module's analog bandwidth edge -- crosses -3dB near 3.7 GHz).
    inst_f = f_start + (f_stop - f_start) * tau / duration
    rolloff_db = np.clip(-15.0 * (inst_f - 3.5e9) / 1e9, -30, 0)
    sig *= 10 ** (rolloff_db / 20)

    n_pad = int(round(t_pad * fs))
    v = np.concatenate([rng.normal(0, 0.001, n_pad), sig,
                         rng.normal(0, 0.001, n_pad)])
    v += rng.normal(0, 0.001, len(v))
    t = np.arange(len(v)) / fs

    # Auto-detect t0 (t0=None), matching real usage in run_chirp_quality.py --
    # NOT passing the known true t0 here is what originally let the
    # find_burst_window() taper-crossing bias (~14 ns on this signal) slip
    # through undetected: it showed up as a constant bias in every metric,
    # invisible in a self-test that always supplied the exact answer.
    result = analyze(t, v, f_start, f_stop, duration)
    print(f"t0: true={t0*1e9:.2f} ns  refined={result['linearity']['t0']*1e9:.2f} ns")
    lin = result["linearity"]
    mf = result["matched_filter"]

    print(f"linearity: rms={lin['rms_hz']/1e6:.2f} MHz  peak={lin['peak_hz']/1e6:.2f} MHz")
    print(f"swept response -3dB rolloff: "
          f"{result['rolloff_hz']/1e9:.3f} GHz" if result["rolloff_hz"] else "no rolloff in band")
    print(f"matched filter: mainlobe={mf['mainlobe_width_s']*1e9:.2f} ns "
          f"(ideal {mf['ideal_width_s']*1e9:.2f} ns)  PSL={mf['psl_db']:.1f} dB")

    fig = plot(t, v, result, f_start, f_stop, title="chirp_quality.py self-test (synthetic)")
    out_path = os.path.join(tempfile.gettempdir(), "chirp_quality_selftest.png")
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
