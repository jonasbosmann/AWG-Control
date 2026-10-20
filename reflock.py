"""Check whether the AWG is locked to the 10 MHz reference -- permanently,
using the signal path that is already cabled.

The idea
--------
The scope's timebase is locked to the Wenzel OCXO, so the scope IS a
calibrated frequency reference (ROSc:STATe? -> LOCKED proves it). Anything the
AWG emits into a scope channel can therefore be measured against that
reference. If the AWG is locked too, a tone it believes is f_nominal measures
as exactly f_nominal. If it is running on its internal TCXO, it measures
1-2.5 ppm off.

So no AUX-OUT cable, no spare splitter arm and no re-patching: the check runs
over whatever AWG->scope connection the measurement already uses.

Two things it depends on
------------------------
1. The nominal frequency must be what the AWG can ACTUALLY synthesise, not
   what was asked for. In Direct mode `send_sine()` rounds to an integer
   number of cycles in the 2048-sample buffer, so at 2.5 GS/s a requested
   10 MHz comes out at 9.7656 MHz -- a 234 kHz error that swamps the ~10 Hz
   per ppm we are looking for. Always use the value `send_sine()` RETURNS
   (or pass exact=True), never the value asked for.

2. The AWG must stay out of 9 GS/s DIRECT mode. That configuration
   (interleaved DACs, CH2 disabled) drops the external reference every time
   -- bench-measured 0/3, against 3/3 for both 2.5 GS/s Direct and 9 GS/s
   DUC. check_lock() therefore forces 2.5 GS/s for the test tone.

3. Resolution scales with how many cycles are captured:

       fractional precision  ~  phase_accuracy / (2*pi * f * T)

   so a high carrier and a long record. 1 GHz for 20 us is 2e4 cycles and
   resolves well under 0.1 ppm; 10 MHz for 1 us is 10 cycles and resolves
   nothing. estimate_frequency() reports cycles observed so the caller can
   tell whether the answer means anything.

Per the lab's standing rule on unreplicated bench numbers, check_lock()
repeats the capture and reports the spread, not a single value.
"""
import numpy as np
from scipy import signal

from awg import SAMPLE_RATE_DUAL

# A clock error this large or larger means the AWG is not on the reference.
# The internal TCXO is 1-2.5 ppm, so 0.3 ppm separates the two cases cleanly
# while staying well above the estimator's own noise on a decent capture.
UNLOCKED_PPM = 0.3


def estimate_frequency(t, v, edge_trim=0.05):
    """Frequency of a single clean tone, far below one FFT bin of precision.

    Coarse FFT peak with parabolic interpolation, then heterodyne to near DC
    and fit the slope of the unwrapped residual phase. Returns
    (frequency_hz, n_cycles).

    edge_trim discards a fraction from each end before the phase fit, because
    the Hilbert transform misbehaves at the edges of a finite record.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = v.size
    if n < 64:
        raise ValueError(f"need at least 64 samples, got {n}")
    v = v - v.mean()
    dt = (t[-1] - t[0]) / (n - 1)

    # ── coarse: interpolated FFT peak ────────────────────────────────
    mag = np.abs(np.fft.rfft(v * np.hanning(n)))
    k = int(np.argmax(mag[1:]) + 1)
    if 1 <= k < mag.size - 1:
        a, b, c = (np.log(mag[k - 1] + 1e-300), np.log(mag[k] + 1e-300),
                   np.log(mag[k + 1] + 1e-300))
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    f0 = (k + delta) / (n * dt)

    # ── fine: residual phase slope after heterodyning to near DC ─────
    tt = t - t[0]
    z = signal.hilbert(v) * np.exp(-2j * np.pi * f0 * tt)
    trim = max(int(edge_trim * n), 8)
    if n - 2 * trim < 32:
        trim = 0
    zc, tc = z[trim:n - trim], tt[trim:n - trim]
    phase = np.unwrap(np.angle(zc))
    slope = np.polyfit(tc, phase, 1)[0]

    f = f0 + slope / (2 * np.pi)
    return f, abs(f) * (t[-1] - t[0])


def fractional_error(t, v, f_nominal):
    """(measured - nominal) / nominal, as a fraction. Multiply by 1e6 for ppm.

    f_nominal must be the frequency the AWG actually synthesises -- the value
    returned by send_sine(), not the one requested.
    """
    f, cycles = estimate_frequency(t, v)
    return f / f_nominal - 1.0, f, cycles


def check_lock(awg, scope, f_test_hz=500e6, amp_vpp=0.5, awg_channel=1,
               scope_channel=1, capture_seconds=20e-6, n_reps=5,
               max_points=200000, verbose=True, leave_output_on=False):
    """Measure the AWG's clock error against the (reference-locked) scope.

    Drives a CW tone, captures it n_reps times and returns a dict with the
    mean fractional error, the spread across repeats, and a verdict. The
    spread is the honest error bar: trust the mean only when it exceeds it.

    The caller's waveform cannot be preserved -- send_sine()'s *RST destroys
    it -- so the test tone is switched OFF on the way out rather than left
    running on a channel that is supposed to carry something else. Pass
    leave_output_on=True to keep it (e.g. to probe it by hand afterwards).

    Returns {'ppm', 'ppm_sd', 'cycles', 'f_nominal', 'f_measured',
             'locked', 'conclusive', 'reps'}.
    """
    st = scope.reference_status()
    if not (st["external"] and st["locked"]):
        raise RuntimeError(
            f"the scope itself is not locked to the external reference "
            f"(SOUR={st['source']}, STATE={st['state']}) -- it cannot be used "
            "as the frequency standard until it is")

    # Two constraints decide how the tone is generated, and they conflict with
    # the obvious choice of exact=True:
    #
    #  * 9 GS/s DIRECT mode (the interleaved, CH2-disabled configuration)
    #    DROPS the external reference -- reproducibly, 0/3 on the bench
    #    2026-10-19, while 2.5 GS/s Direct and 9 GS/s DUC both keep it 3/3.
    #    exact=True re-derives the sample rate and lands there, so using it
    #    would un-lock the AWG and then "measure" a guaranteed false negative.
    #  * the nominal frequency must be exactly what is synthesised, or the
    #    synthesis error is read as clock error.
    #
    # Both are satisfied by staying at 2.5 GS/s and taking the frequency
    # send_sine() RETURNS: that value is cycles*rate/n_samples, exact by
    # construction, whatever was asked for.
    rate_was = awg.sample_rate
    awg.sample_rate = SAMPLE_RATE_DUAL
    try:
        f_nominal = awg.send_sine(f_test_hz, amp_vpp, channel=awg_channel,
                                  exact=False)
    finally:
        awg.sample_rate = rate_was
    if verbose:
        print(f"  test tone {f_nominal/1e6:.6f} MHz at 2.5 GS/s "
              f"(asked {f_test_hz/1e6:g} MHz; grid is "
              f"{SAMPLE_RATE_DUAL/awg.n_samples/1e6:.4f} MHz)")

    # Leave the other channels alone. Disabling them would buy a faster
    # sample-rate tier, but 500 MHz needs nothing like the top tier and
    # switching off channels somebody is watching is not worth it.
    saved = scope.save_state()
    try:
        scope.set_max_sample_rate(capture_seconds)
        scope.setup(channel=scope_channel, n_averages=1,
                    trigger_channel=scope_channel, trigger_mode="AUTO")

        errs, freqs, cycles = [], [], []
        for i in range(n_reps):
            t, v = scope.get_waveform(scope_channel, max_points=max_points)
            err, f, cyc = fractional_error(t, v, f_nominal)
            errs.append(err)
            freqs.append(f)
            cycles.append(cyc)
            if verbose:
                print(f"  rep {i+1}/{n_reps}: {f/1e6:.6f} MHz  "
                      f"{err*1e6:+8.3f} ppm  ({cyc:.0f} cycles)")
    finally:
        scope.restore_state(saved)
        # Don't abandon a 500 MHz test tone on a channel the user thinks is
        # carrying their waveform -- it is gone either way (the *RST inside
        # send_sine wiped it), so leaving it radiating only misleads.
        if not leave_output_on:
            with awg._lock:
                awg._dev.write(f":INST:CHAN {awg_channel}")
                awg._dev.write(":OUTP OFF")
            if verbose:
                print(f"  test tone switched off; CH{awg_channel} needs "
                      "re-sending from the GUI")

    ppm = float(np.mean(errs)) * 1e6
    ppm_sd = float(np.std(errs, ddof=1)) * 1e6 if len(errs) > 1 else float("nan")

    # Conclusive only if the shot-to-shot spread is small enough to separate
    # "locked" from the TCXO's 1-2.5 ppm in the first place.
    conclusive = bool(np.isfinite(ppm_sd) and ppm_sd < UNLOCKED_PPM)
    locked = bool(abs(ppm) < UNLOCKED_PPM)

    return {"ppm": ppm, "ppm_sd": ppm_sd, "cycles": float(np.mean(cycles)),
            "f_nominal": f_nominal, "f_measured": float(np.mean(freqs)),
            "locked": locked, "conclusive": conclusive, "reps": n_reps}


def verdict_text(res):
    """One line summarising a check_lock() result."""
    base = f"{res['ppm']:+.3f} ± {res['ppm_sd']:.3f} ppm"
    if not res["conclusive"]:
        return (f"AWG clock {base} — INCONCLUSIVE, scatter too large "
                f"(need < {UNLOCKED_PPM} ppm; try a higher test frequency "
                f"or a longer capture)")
    if res["locked"]:
        return f"AWG clock {base} — LOCKED to the reference"
    return (f"AWG clock {base} — NOT LOCKED "
            f"(internal TCXO is 1–2.5 ppm; at 350 GHz this is "
            f"{abs(res['ppm']) * 350e3:.0f} kHz of line position)")

def fit_tone(t, v, f_hz, t_ref=None):
    """Least-squares fit of a known-frequency tone: (amplitude, phase).

    Returns the zero-to-peak amplitude and the phase at `t_ref`, for the model
        v = amplitude * cos(2*pi*f_hz*(t - t_ref) + phase)

    t_ref=None means t[0] (the first sample). Pass t_ref=0.0 together with a
    time axis from get_waveform(absolute_time=True) to reference the phase to
    the TRIGGER instant instead -- which is the only version that means
    anything across shots, since the first sample sits at a random sub-sample
    offset from the trigger.

    Amplitude comes from the same projection, so it is coherent detection at
    one frequency: broadband noise averages out of it instead of inflating it
    the way peak-to-peak or RMS does. That matters when the point of the
    exercise is to compare an averaged trace against a single shot.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    origin = t[0] if t_ref is None else float(t_ref)
    ph = 2 * np.pi * f_hz * (t - origin)
    # least-squares projection onto cos/sin at the known frequency
    c, sn = np.cos(ph), np.sin(ph)
    a = 2 * np.mean(v * c)
    b = -2 * np.mean(v * sn)
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def estimate_phase(t, v, f_hz, t_ref=None):
    """Phase of a known-frequency tone at `t_ref` (default t[0]), in radians.

    Used for shot-to-shot coherence: capture the same tone repeatedly against
    an INDEPENDENT trigger and look at how much this value moves. Fitting at a
    known frequency (rather than estimating it per shot) keeps the answer
    referenced to the trigger instant.
    """
    return fit_tone(t, v, f_hz, t_ref)[1]


def coherence(phases):
    """How much amplitude survives coherent averaging of shots at `phases`.

    |<exp(i*phi)>| -- 1.0 means every shot adds in phase and averaging is
    lossless; 0 means the shots cancel. This is the number that decides
    whether N-shot averaging actually buys sqrt(N) of SNR.

    Returns (coherence, circular_sd_degrees).
    """
    z = np.exp(1j * np.asarray(phases))
    r = float(abs(z.mean()))
    sd = float(np.degrees(np.sqrt(-2 * np.log(r)))) if r > 0 else float("inf")
    return r, sd
