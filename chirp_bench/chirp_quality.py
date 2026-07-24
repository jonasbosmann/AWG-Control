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
from scipy.signal import (butter, correlate as _correlate, filtfilt, hilbert,
                           resample as _resample, savgol_filter,
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


def _gauss_win_continuous(tau, duration, frac):
    """Continuous-time version of _gauss_win -- same taper shape, but
    evaluated at arbitrary tau (seconds since chirp start) instead of a
    fixed sample grid, so it can be aligned to a capture's own time axis
    (used by evm_score() to build the ideal complex reference on the
    capture's actual sample times). Zero outside [0, duration]."""
    tau = np.asarray(tau, dtype=np.float64)
    w = np.ones_like(tau)
    nw_t = min(max(frac * duration, 0.0), duration / 2)
    if nw_t > 0:
        sigma_t = nw_t / 3.0
        rising = tau < nw_t
        w[rising] = np.exp(-0.5 * ((tau[rising] - nw_t) / sigma_t) ** 2)
        falling = tau > duration - nw_t
        w[falling] = np.exp(-0.5 * ((tau[falling] - (duration - nw_t)) / sigma_t) ** 2)
    w = np.where((tau < 0) | (tau > duration), 0.0, w)
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
    Returns the compressed pulse's mainlobe width -- IRW, "impulse response
    width", the standard SAR/radar convention of measuring at -3dB from the
    peak (compare to the diffraction-limited 1/bandwidth) -- and peak
    sidelobe level (PSLR, dB relative to the mainlobe peak) -- a single
    aggregate quality number, standard practice for characterizing
    LFM/radar chirps, robust to spurs that would confuse instantaneous-
    frequency demod."""
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

    corr = _correlate(seg, ref, mode="full", method="fft")

    # Real-valued correlation of two bandpass signals rides a fast carrier-
    # frequency ripple on top of the true (slowly-varying) compression
    # envelope: corr(tau) ~ envelope(tau)*cos(2*pi*f_carrier*tau). For a
    # narrow-span (low relative bandwidth) chirp the envelope stays near its
    # peak for many carrier cycles, so that ripple swings tens of dB
    # sample-to-sample even while genuinely still "in the mainlobe" --
    # confirmed 2026-07-24 on real bench data (0.1 GHz-span 1.70-1.80 GHz
    # capture): raw |corr| swung from 0 dB to -19 dB one sample apart,
    # collapsing the naive -3dB contiguous-region search to a single sample
    # (IRW -> 0) even though the true envelope stayed within -3dB for ~9.4 ns,
    # matching the 10 ns diffraction-limited (1/bandwidth) expectation almost
    # exactly. Using the Hilbert ENVELOPE of corr for peak-finding and the
    # IRW search sidesteps this -- it's immune to the ripple by construction,
    # the same reason dechirp_residual() uses the analytic signal rather than
    # a raw real-valued comparison.
    corr_env = np.abs(hilbert(corr))
    peak_idx = int(np.argmax(corr_env))
    peak = corr_env[peak_idx]
    lag = (np.arange(len(corr)) - (n - 1)) * dt
    with np.errstate(divide="ignore"):
        corr_db = 20 * np.log10(np.abs(corr) / peak + 1e-300)
        corr_env_db = 20 * np.log10(corr_env / peak + 1e-300)

    # Mainlobe = contiguous region around the peak above -3 dB (IRW convention),
    # measured on the envelope -- see note above.
    above = corr_env_db > -3.0
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(above) - 1 and above[right + 1]:
        right += 1
    mainlobe_width_s = (right - left) * dt

    # PSLR guard region derives from mainlobe_width_s, so this is fixed
    # automatically now too: the old collapsed (~0-width) mainlobe gave a
    # near-zero guard, letting the PSLR search see ripple points right next
    # to the true peak (reporting a falsely-bad ~0 dB PSLR) instead of only
    # genuinely distant sidelobes.
    guard = int(round(guard_factor * max(right - left, 1) / 2)) + 1
    outside = np.ones(len(corr), dtype=bool)
    outside[max(peak_idx - guard, 0):peak_idx + guard + 1] = False
    psl_db = float(np.max(corr_db[outside])) if np.any(outside) else float("nan")

    return {
        "lag_s": lag, "corr_db": corr_db, "corr_env_db": corr_env_db, "peak_idx": peak_idx,
        "mainlobe_width_s": mainlobe_width_s,
        "ideal_width_s": 1.0 / abs(f_stop - f_start),
        "psl_db": psl_db,
    }


# ---------------------------------------------------------------------------
# Metric 4: coherent dechirp residual (LFM-radar "stretch processing")
# ---------------------------------------------------------------------------

def dechirp_residual(t, v, f_start, f_stop, duration, t0=None, window_frac=0.05,
                      edge_frac=0.05, lowpass_hz=20e6):
    """Coherent 'dechirp' (a.k.a. stretch processing) residual -- the
    standard LFM-radar technique for isolating chirp nonlinearity and phase
    noise, used in receivers/instrumentation specifically to determine
    chirp linearity: mix the capture DOWN against the exact, known,
    commanded reference chirp (multiply by its conjugate), then low-pass.
    A perfect chirp collapses to a constant after this; any ramp
    nonlinearity, phase noise, or mode-hop survives as a residual that's
    SLOW (near DC), not fast -- because the GHz-scale sweep itself has
    already been cancelled by the mix, not just measured.

    This sidesteps demodulate()'s core tension (differentiating a fast-
    swept phase is a high-pass operation that amplifies noise by ~1/dt,
    see its docstring) by using information linearity_error() throws away:
    since the reference is DUC/IQ-generated, the exact ideal complex
    baseband is already known exactly, not just its envelope -- mixing
    against it does the "remove the GHz sweep" step with one multiply
    instead of a derivative. The residual phase is consequently already
    well-conditioned; unlike demodulate(), no extra smoothing stage is
    needed before differentiating it for a frequency-error estimate.

    lowpass_hz: cutoff for isolating the near-DC residual from the (already
    single-sideband, via the analytic signal) mixer output and noise --
    default 20 MHz assumes genuine defects vary slower than that and
    f_start is well above it (so no real content sits near DC before
    mixing); tune down for a cleaner residual if the capture is very
    low-noise, or up if you need to resolve a suspected fast defect.
    """
    if t0 is None:
        t0 = refine_t0(t, v, f_start, f_stop, duration, window_frac=window_frac)

    dt = float(np.median(np.diff(t)))
    tau = t - t0
    tau_c = np.clip(tau, 0, duration)
    phase_ideal = 2 * np.pi * (f_start * tau_c + (f_stop - f_start) / (2 * duration) * tau_c ** 2)

    analytic = hilbert(v - np.mean(v))
    mixed = analytic * np.exp(-1j * phase_ideal)

    nyq = 0.5 / dt
    wn = min(lowpass_hz / nyq, 0.99)
    b, a = butter(4, wn, btype="low")
    residual = filtfilt(b, a, mixed)

    lo, hi = duration * edge_frac, duration * (1 - edge_frac)
    mask = (tau >= lo) & (tau <= hi)
    if not np.any(mask):
        raise ValueError("no samples fall inside the trimmed chirp window -- "
                          "check t0/duration against the capture's time axis")

    phase_err = np.unwrap(np.angle(residual[mask]))
    phase_err -= phase_err[0]   # anchor to 0; only the SHAPE (drift/wobble)
                                 # of the residual is meaningful -- the
                                 # absolute value is an arbitrary reference
                                 # phase set by t0/cable length.
    t_mask = tau[mask]
    freq_err = np.gradient(phase_err, t_mask) / (2 * np.pi)

    return {
        "t": t_mask, "residual": residual[mask],
        "phase_err_rad": phase_err,
        "phase_err_rms_rad": float(np.sqrt(np.mean(phase_err ** 2))),
        "phase_err_peak_rad": float(np.max(np.abs(phase_err))),
        "freq_err_hz": freq_err,
        "freq_err_rms_hz": float(np.sqrt(np.mean(freq_err ** 2))),
        "freq_err_peak_hz": float(np.max(np.abs(freq_err))),
        "t0": t0,
    }


# ---------------------------------------------------------------------------
# Metric 5: Error Vector Magnitude (EVM) -- standard RF/telecom AWG fidelity
# metric (used by Keysight/Tek/R&S to spec their own AWGs' output quality).
# ---------------------------------------------------------------------------

def evm_score(t, v, f_start, f_stop, duration, t0=None, window_frac=0.05,
              edge_frac=0.05):
    """Error Vector Magnitude: direct complex-baseband comparison against
    the known ideal reference, instead of comparing derived quantities
    (instantaneous frequency, envelope) separately as the other metrics do.
    Well-suited here because the reference is DUC/IQ-generated, so the
    exact ideal complex baseband is already known -- this is the metric a
    vendor datasheet or an RF engineer would reach for first to quantify
    "how faithfully does the hardware reproduce the commanded waveform".

    Aligns t0 (matched filter) and fits a single complex scalar (gain +
    constant phase offset) by least squares before computing the error --
    standard EVM practice, so an uncalibrated cable loss or an arbitrary
    absolute phase reference doesn't get counted as "error". Does NOT fit
    away a chirp-rate/span scaling error -- that would be a real defect,
    not an alignment nuisance.

    Returns EVM as % rms and % peak (both normalized to the aligned
    reference's rms amplitude), the conventional way EVM is quoted.
    """
    if t0 is None:
        t0 = refine_t0(t, v, f_start, f_stop, duration, window_frac=window_frac)

    tau = t - t0
    tau_c = np.clip(tau, 0, duration)
    phase_ideal = 2 * np.pi * (f_start * tau_c + (f_stop - f_start) / (2 * duration) * tau_c ** 2)
    ideal_complex = _gauss_win_continuous(tau, duration, window_frac) * np.exp(1j * phase_ideal)

    lo, hi = duration * edge_frac, duration * (1 - edge_frac)
    mask = (tau >= lo) & (tau <= hi)
    if not np.any(mask):
        raise ValueError("no samples fall inside the trimmed chirp window -- "
                          "check t0/duration against the capture's time axis")

    analytic = hilbert(v - np.mean(v))
    meas, ideal = analytic[mask], ideal_complex[mask]

    c = np.sum(meas * np.conj(ideal)) / np.sum(np.abs(ideal) ** 2)
    aligned_ideal = c * ideal
    err = meas - aligned_ideal
    ref_rms = float(np.sqrt(np.mean(np.abs(aligned_ideal) ** 2)))

    evm_t_pct = 100 * np.abs(err) / ref_rms
    return {
        "t": tau[mask], "err": err, "gain": complex(c),
        "evm_t_pct": evm_t_pct,
        "evm_rms_pct": float(np.sqrt(np.mean(np.abs(err) ** 2)) / ref_rms * 100),
        "evm_peak_pct": float(np.max(np.abs(err)) / ref_rms * 100),
        "t0": t0,
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


def spectrum_db(t, v):
    """Magnitude spectrum (periodogram) of the WHOLE capture -- unlike
    spectrogram() this doesn't trade frequency resolution for time
    resolution, so it's the better view for spotting narrow spurs, harmonics,
    or out-of-band content sitting below what a coarse spectrogram bin could
    resolve (e.g. the ~1.5 GHz environmental spur found 2026-07-24 was only
    ever confirmed via a spectrogram; a plain FFT would show it as a single
    sharp, unambiguous line rather than a faint horizontal streak). Real
    input -> only 0..Nyquist is meaningful, returned via rfft.

    Deliberately NOT windowed. An extra Hann taper applied across the whole
    record (dead time included) was tried first and rounded the swept-band
    peak into a smooth Gaussian-looking bump -- confirmed 2026-07-24 to be a
    self-inflicted artifact, not a real AWG characteristic: with no extra
    window the peak sits flat at ~0dB across nearly the entire commanded
    band with sharp edges, matching the textbook flat-top LFM spectrum shape.
    No windowing is actually needed here anyway -- the commanded envelope
    (chirp_quality._gauss_win()) already tapers smoothly to true zero at both
    ends of the record, so there's no abrupt edge for a window to protect
    against leakage from."""
    fs = 1.0 / float(np.median(np.diff(t)))
    n = len(v)
    spec = np.fft.rfft(v - np.mean(v))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag_db = 20 * np.log10(np.abs(spec) / (np.max(np.abs(spec)) + 1e-300) + 1e-300)
    return freqs, mag_db


def sinc_reconstruct(x, y, upsample=8):
    """Band-limited (sinc / Whittaker-Shannon) reconstruction of a real
    voltage trace for DISPLAY -- the same thing a scope's own Sin(x)/x
    display mode does, exact as long as the signal is genuinely sampled
    above Nyquist (comfortably true for any capture used with this module).
    scipy.signal.resample does this via FFT zero-padding; independently
    cross-checked 2026-07-24 against a hand-coded literal sinc sum on real
    capture data (agreement ~0.3% away from window edges).

    Returns the upsampled (x, y) directly -- deliberately NOT decimated down
    to a fixed point/bin count afterward. An earlier version did that
    (min/max per display bin) and it was wrong: collapsing bins to [min,max]
    pairs joined by straight lines looks like a filled envelope only when
    zoomed out; zoom in on the actual plot and you're looking at a handful
    of those bins, and alternating min/max points connected by lines is a
    sawtooth by construction, regardless of the true signal's shape. Handing
    matplotlib the real reconstructed samples and letting ITS zoom/pan do
    the work means every zoom level shows the genuine curve, not an artifact
    of a decimation scheme tuned for one zoom level. Used both by plot()
    here and by chirp_bench/pipeline_view.py's per-metric deep-dive views."""
    n = len(y)
    if n <= 1:
        return x, y
    y_up = _resample(y, n * upsample)
    x_up = np.interp(np.arange(n * upsample), np.arange(0, n * upsample, upsample), x)
    return x_up, y_up


# ---------------------------------------------------------------------------
# Top-level bundle + plot
# ---------------------------------------------------------------------------

def analyze(t, v, f_start, f_stop, duration, t0=None, edge_frac=0.05, n_bins=200):
    lin = linearity_error(t, v, f_start, f_stop, duration, t0=t0, edge_frac=edge_frac)
    centers, amp_binned, amp_db = swept_amplitude_response(lin, n_bins=n_bins)
    rolloff = rolloff_frequency(centers, amp_db)
    mf = matched_filter_score(t, v, f_start, f_stop, duration, t0=lin["t0"])
    dechirp = dechirp_residual(t, v, f_start, f_stop, duration, t0=lin["t0"], edge_frac=edge_frac)
    evm = evm_score(t, v, f_start, f_stop, duration, t0=lin["t0"], edge_frac=edge_frac)
    return {
        "linearity": lin,
        "response_freq_hz": centers, "response_amp": amp_binned, "response_db": amp_db,
        "rolloff_hz": rolloff,
        "matched_filter": mf,
        "dechirp": dechirp,
        "evm": evm,
    }


def plot(t, v, result, f_start, f_stop, title=None):
    """Dashboard layout (2026-07-24, reordered): PRIMARY metrics first
    (dechirp residual -- the one step-by-step-verified against real bench
    data -- and EVM, the industry-standard AWG fidelity number), then
    supporting/cross-check metrics (swept amplitude response, pulse
    compression), then raw capture + spectrum for sanity checking, and
    finally the smoothed-instantaneous-frequency view demoted to the bottom
    with an explicit caveat -- it's largely SUPERSEDED by dechirp_residual()
    for the same underlying question (frequency-vs-time linearity):
    differentiating phase is a high-pass operation that amplifies noise, so
    it needs heavy smoothing to be usable, and that smoothing can't tell a
    genuine fast defect (mode-hop, glitch) apart from noise -- a clean
    SMOOTHED trace is not proof of a clean chirp. dechirp_residual() avoids
    this by mixing down against the known reference first, never needing
    that smoothing in the first place.

    spectrogram() dropped from this routine dashboard entirely -- confirmed
    the plain FFT spectrum shows spectral content (e.g. spurs) MORE sharply
    since it doesn't trade frequency resolution for time resolution, and
    dechirp_residual() already gives quantitative time-resolved defect
    info a spectrogram could only show as a fuzzy color blob. The function
    itself is kept (not deleted) -- still worth reaching for by hand during
    genuinely exploratory investigation of an unfamiliar signal, which is
    how it earned its keep originally (finding the ~1.5 GHz environmental
    spur, 2026-07-24)."""
    import matplotlib.pyplot as plt

    lin = result["linearity"]
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    if title:
        fig.suptitle(title)

    ax = axes[0, 0]
    # Same rendering as pipeline_view.py's deep-dive panels: band-limited
    # (sinc) reconstruction -- what a scope's own Sin(x)/x display mode
    # does -- with the real raw samples overlaid as markers on top, so the
    # curve's fidelity to the actual data is directly checkable by eye
    # rather than trusting a naive line-connect (which looks like a fake
    # sawtooth at low samples/cycle) or losing the continuous shape
    # entirely (markers alone, tried and superseded).
    xd, yd = sinc_reconstruct(t * 1e6, v)
    ax.plot(xd, yd, lw=0.5, color="C0", zorder=1)
    ax.plot(t * 1e6, v, ".", ms=1.5, alpha=0.35, color="C3", zorder=2)
    ax.set_xlabel("time (us)"); ax.set_ylabel("V"); ax.set_title("raw capture")

    ax = axes[0, 1]
    f_spec, mag_db = spectrum_db(t, v)
    ax.plot(f_spec / 1e9, mag_db, lw=0.5)
    ax.set_xlim(0, max(f_start, f_stop) * 1.5 / 1e9)
    ax.set_ylim(-80, 2)
    ax.set_xlabel("GHz"); ax.set_ylabel("dB (norm. to peak)")
    ax.set_title("FFT magnitude spectrum (whole capture)")

    ax = axes[1, 0]
    dc = result["dechirp"]
    ax.plot(dc["t"] * 1e6, dc["freq_err_hz"] / 1e6, lw=0.8, color="C3")
    ax.set_xlabel("time (us)"); ax.set_ylabel("error (MHz)")
    ax.set_title(f"PRIMARY: dechirp residual freq error  rms={dc['freq_err_rms_hz']/1e6:.3f} "
                 f"peak={dc['freq_err_peak_hz']/1e6:.3f} MHz", fontsize=9)

    ax = axes[1, 1]
    evm = result["evm"]
    ax.plot(evm["t"] * 1e6, evm["evm_t_pct"], lw=0.8, color="C4")
    ax.set_xlabel("time (us)"); ax.set_ylabel("EVM (%)")
    ax.set_title(f"EVM (read w/ amplitude response, not alone)  "
                 f"rms={evm['evm_rms_pct']:.2f}%  peak={evm['evm_peak_pct']:.2f}%", fontsize=9)

    ax = axes[2, 0]
    ax.plot(result["response_freq_hz"] / 1e9, result["response_db"], lw=1)
    ax.axhline(-3.0, color="r", ls=":", lw=1)
    if result["rolloff_hz"]:
        ax.axvline(result["rolloff_hz"] / 1e9, color="r", ls=":", lw=1)
    ax.set_xlabel("instantaneous frequency (GHz)"); ax.set_ylabel("dB (norm. to median)")
    roll_txt = f"{result['rolloff_hz']/1e9:.2f} GHz" if result["rolloff_hz"] else "none in band"
    ax.set_title(f"swept amplitude response  -3dB @ {roll_txt}")

    ax = axes[2, 1]
    mf = result["matched_filter"]
    zoom_ns = max(mf["mainlobe_width_s"], mf["ideal_width_s"], 5e-9) * 15e9
    ax.plot(mf["lag_s"] * 1e9, mf["corr_db"], lw=0.4, color="0.75")
    ax.plot(mf["lag_s"] * 1e9, mf["corr_env_db"], lw=1.0, color="C0")
    ax.axhline(-3.0, color="r", ls=":", lw=1)
    ax.set_xlim(-zoom_ns, zoom_ns)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("lag (ns)"); ax.set_ylabel("dB")
    ax.set_title(f"pulse compression (cross-check)  IRW={mf['mainlobe_width_s']*1e9:.2f} ns "
                 f"(ideal {mf['ideal_width_s']*1e9:.2f} ns)  PSLR={mf['psl_db']:.1f} dB", fontsize=8)

    ax = axes[3, 0]
    ax.plot(lin["t_raw"] * 1e6, lin["freq_meas_raw"] / 1e9, color="0.75", lw=0.3,
            label="measured (raw)", zorder=1)
    ax.plot(lin["t"] * 1e6, lin["freq_ideal"] / 1e9, "k--", lw=1, label="commanded", zorder=3)
    ax.plot(lin["t"] * 1e6, lin["freq_meas"] / 1e9, lw=0.8, label="measured (smoothed)", zorder=2)
    ax.set_xlabel("time (us)"); ax.set_ylabel("GHz"); ax.legend(fontsize=7)
    ax.set_title("instantaneous frequency (SECONDARY -- see dechirp instead)", fontsize=8)

    ax = axes[3, 1]
    ax.plot(lin["t_raw"] * 1e6, lin["err_hz_raw"] / 1e6, color="0.75", lw=0.3, zorder=1)
    ax.plot(lin["t"] * 1e6, lin["err_hz"] / 1e6, lw=0.8, zorder=2)
    ax.set_xlabel("time (us)"); ax.set_ylabel("error (MHz)")
    ax.set_title(f"freq error (SECONDARY, MHz)  smoothed rms={lin['rms_hz']/1e6:.2f} peak={lin['peak_hz']/1e6:.2f}  "
                 f"|  raw rms={lin['rms_hz_raw']/1e6:.2f} peak={lin['peak_hz_raw']/1e6:.2f}",
                 fontsize=7)

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
    print(f"matched filter: IRW={mf['mainlobe_width_s']*1e9:.2f} ns "
          f"(ideal {mf['ideal_width_s']*1e9:.2f} ns)  PSLR={mf['psl_db']:.1f} dB")

    dc, evm = result["dechirp"], result["evm"]
    print(f"dechirp residual: freq err rms={dc['freq_err_rms_hz']/1e6:.3f} MHz "
          f"peak={dc['freq_err_peak_hz']/1e6:.3f} MHz  "
          f"(injected wobble peak was {freq_dev_hz/1e6:.2f} MHz -- should be in "
          f"the same ballpark, recovered without demodulate()'s smoothing)")
    print(f"EVM: rms={evm['evm_rms_pct']:.2f}%  peak={evm['evm_peak_pct']:.2f}%")

    fig = plot(t, v, result, f_start, f_stop, title="chirp_quality.py self-test (synthetic)")
    out_path = os.path.join(tempfile.gettempdir(), "chirp_quality_selftest.png")
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
