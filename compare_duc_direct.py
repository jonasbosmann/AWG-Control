"""Compare DUC (IQM-ONE) vs direct-DAC generation on the P9482D.

Wiring:  AWG CH1 -> scope CH1 (signal under test)
         AWG CH2 -> scope CH2 (start-of-buffer sync pulse -> scope trigger)

The CH2 pulse shares CH1's buffer period, so it marks the identical CH1 phase
every period -> coherent scope triggering / averaging without the hardware marker.

Two tests, both at a carrier BOTH modes can reach (~1 GHz):
  1. CW SFDR   — FFT of an averaged CW capture; signal-to-worst-spur (dBc).
                 A stationary tone, so this is robust even to trigger quality.
  2. Chirp f(t) — averaged chirp capture -> instantaneous frequency via the
                 analytic signal; compare ramp linearity (residual Hz RMS) and
                 phase residual direct vs DUC. Needs the CH2 coherent trigger.

Run at the bench and tune the numbers below. Start with RUN_CW=True only to
confirm the CH2 trigger + averaging before enabling the chirp test.
"""

import numpy as np
import matplotlib.pyplot as plt

from awg import AWG
from scope import Scope

# ── config — edit these at the bench ───────────────────────────────
SIG_CH   = 1          # scope channel measuring AWG CH1
TRIG_CH  = 2          # scope channel receiving AWG CH2 sync pulse

CW_FREQ_HZ    = 1.0e9     # CW tone (both modes reach 1 GHz)
CHIRP_CTR_HZ  = 1.0e9     # chirp center frequency

CHIRP_BW_HZ   = 200e6     # total chirp sweep width (center ± BW/2)
CHIRP_US      = 1.0
DEAD_US       = 0.5

AMP_VPP       = 0.5       # CH1 signal amplitude
SYNC_AMP_VPP  = 0.8       # CH2 sync pulse amplitude
SYNC_PULSE_NS = 40        # CH2 pulse width
TRIG_LEVEL_V  = 0.2       # scope edge-trigger level on CH2 (~half sync amp)

N_AVG    = 64             # hardware averages (coherent via CH2 trigger)
SETTLE_S = 0.15
TRIG_MODE = "NORMal"      # NORMal = only average genuine triggers (coherent).
                          # AUTO free-runs w/o a trigger -> averaging washes out.

RUN_CW    = True
RUN_CHIRP = False

INTERACTIVE = True        # pause between steps so you can inspect the live scope
SHOW_EACH   = True        # pop up each arm's capture+FFT before continuing

# CW capture window: wide enough for fine FFT resolution
CW_TDIV_S = 2e-7          # scope s/div for the CW test

# DAC first-Nyquist edge per mode = reconstruction-filter passband.
# SFDR is measured only below this; DAC sampling images at k*Fs ± f sit above it
# (direct: images at 2.5±f GHz etc.; DUC: images near 9 GHz) and are excluded.
DAC_NYQ_DIRECT = 1.25e9   # 2.5 GS/s direct-dual Nyquist
DAC_NYQ_DUC    = 4.5e9    # 9 GS/s DUC Nyquist


# ── analysis ───────────────────────────────────────────────────────
def analytic_signal(v):
    """Numpy-only Hilbert: real -> complex analytic signal."""
    n = len(v)
    V = np.fft.fft(v)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[1:n // 2] = 2
        h[n // 2] = 1
    else:
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(V * h)


def sfdr(t, v, guard_hz=20e6, fmax_hz=None):
    """Return (freqs, mag_db, sig_idx, spur_idx, sfdr_db) of an averaged CW capture.

    fmax_hz: only search for signal/spurs below this (the DAC first-Nyquist edge =
    what a reconstruction filter would pass). This excludes the DAC sampling images
    at k*Fs ± f, which are NOT spurious content and would otherwise dominate SFDR.
    """
    v = np.asarray(v) - np.mean(v)
    n = len(v)
    dt = t[1] - t[0]
    # 4-term Blackman-Harris (~-92 dB sidelobes) so the signal's leakage skirt
    # doesn't masquerade as a spur; guard_hz then skips its main lobe.
    k = np.arange(n)
    w = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / n)
         + 0.14128 * np.cos(4 * np.pi * k / n) - 0.01168 * np.cos(6 * np.pi * k / n))
    mag = np.abs(np.fft.rfft(v * w))
    freqs = np.fft.rfftfreq(n, dt)
    inband = freqs <= fmax_hz if fmax_hz else np.ones(len(freqs), bool)
    sig_idx = int(np.argmax(mag * inband))
    guard = (np.abs(freqs - freqs[sig_idx]) > guard_hz) & inband
    guard[0] = False                     # ignore DC
    spur_idx = int(np.argmax(mag * guard))
    sfdr_db = 20 * np.log10(mag[sig_idx] / max(mag[spur_idx], 1e-12))
    mag_db = 20 * np.log10(mag / max(mag.max(), 1e-12) + 1e-12)
    return freqs, mag_db, sig_idx, spur_idx, sfdr_db


def inst_freq(t, v):
    """Instantaneous frequency (Hz) vs time from the analytic signal."""
    a = analytic_signal(np.asarray(v) - np.mean(v))
    phase = np.unwrap(np.angle(a))
    dt = t[1] - t[0]
    f = np.diff(phase) / (2 * np.pi * dt)
    return t[:-1], f


def chirp_linearity(t, f):
    """Fit a line to f(t) over its central 80% and return (fit, residual_rms_hz)."""
    lo, hi = int(0.1 * len(t)), int(0.9 * len(t))
    tc, fc = t[lo:hi], f[lo:hi]
    coeff = np.polyfit(tc, fc, 1)
    fit = np.polyval(coeff, t)
    resid_rms = float(np.sqrt(np.mean((fc - np.polyval(coeff, tc)) ** 2)))
    return fit, resid_rms


# ── acquisition + interactive stepping ─────────────────────────────
def acquire_avg(scope):
    """One averaged capture of the signal channel (CH2-triggered)."""
    _vpp, t, v = scope.measure_vpp(channel=SIG_CH, settle=SETTLE_S)
    return np.asarray(t), np.asarray(v)


def pause(msg):
    if INTERACTIVE:
        try:
            input(msg)
        except EOFError:
            pass


def _arm_scope(scope, tdiv):
    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
                trigger_level=TRIG_LEVEL_V, trigger_mode=TRIG_MODE)
    scope.set_timebase_direct(tdiv)


def show_cw(label, t, v, f, md, si, sp, s, fmax):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
    a1.plot(np.asarray(t) * 1e9, np.asarray(v) * 1e3)
    a1.set_title(f"{label}: averaged CH1 (clean tone = coherent; flat = trigger not locking)")
    a1.set_xlabel("Time (ns)"); a1.set_ylabel("mV"); a1.grid(True)
    a2.plot(f / 1e6, md)
    a2.plot(f[si] / 1e6, md[si], 'g^', label=f"sig {f[si]/1e6:.2f} MHz")
    a2.plot(f[sp] / 1e6, md[sp], 'rv', label=f"worst in-band spur {f[sp]/1e6:.2f} MHz")
    a2.axvline(fmax / 1e6, color='orange', ls='--', lw=1,
               label=f"DAC Nyquist {fmax/1e6:.0f} MHz")
    a2.axvspan(fmax / 1e6, f.max() / 1e6, color='orange', alpha=0.08)
    a2.text(fmax / 1e6, 2, " images →", color='orange', fontsize=8, va='top')
    a2.set_title(f"SFDR = {s:.1f} dBc  (in-band only; images beyond Nyquist excluded)")
    a2.set_xlabel("Frequency (MHz)"); a2.set_ylabel("dB"); a2.set_ylim(-100, 5)
    a2.grid(True); a2.legend(fontsize=8)
    fig.tight_layout(); plt.show()


def cw_arm(scope, label, gen_fn, fmax):
    pause(f"\n[{label}] press Enter to generate + arm scope...")
    gen_fn()
    _arm_scope(scope, CW_TDIV_S)
    pause(f"[{label}] live — verify on scope: CH1 = clean tone, CH2 = stable trigger pulse. "
          f"Enter to capture {N_AVG}x avg...")
    t, v = acquire_avg(scope)
    f, md, si, sp, s = sfdr(t, v, fmax_hz=fmax)
    print(f"  [{label}] SFDR = {s:.1f} dBc in-band (<{fmax/1e9:.2f} GHz)  "
          f"(sig {f[si]/1e6:.2f} MHz, worst spur {f[sp]/1e6:.2f} MHz)")
    if SHOW_EACH:
        show_cw(label, t, v, f, md, si, sp, s, fmax)
    return f, md, s


def show_chirp(label, t, fi, fit, res):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.asarray(t) * 1e6, np.asarray(fi) / 1e6, '.', ms=1, label="f(t)")
    ax.plot(np.asarray(t) * 1e6, np.asarray(fit) / 1e6, 'k-', lw=1, label="linear fit")
    ax.set_title(f"{label}: chirp f(t), residual {res/1e3:.2f} kHz RMS")
    ax.set_xlabel("Time (µs)"); ax.set_ylabel("Inst. freq (MHz)"); ax.grid(True); ax.legend()
    fig.tight_layout(); plt.show()


def chirp_arm(scope, label, gen_fn):
    pause(f"\n[{label}] press Enter to generate + arm scope...")
    gen_fn()
    _arm_scope(scope, CHIRP_US * 1e-6 / 10 * 1.2)   # ~full chirp on screen
    pause(f"[{label}] live — verify CH1 chirp + CH2 trigger. Enter to capture {N_AVG}x avg...")
    t, v = acquire_avg(scope)
    tf, fi = inst_freq(t, v)
    fit, res = chirp_linearity(tf, fi)
    print(f"  [{label}] f(t) residual = {res/1e3:.2f} kHz RMS")
    if SHOW_EACH:
        show_chirp(label, tf, fi, fit, res)
    return tf, fi, fit, res


def run():
    awg = AWG()
    scope = Scope()
    results = {}

    try:
        if RUN_CW:
            print("\n=== CW SFDR: direct vs DUC ===")
            fd, md, sfdr_d = cw_arm(scope, "DIRECT CW",
                lambda: awg.send_cw_direct_sync(CW_FREQ_HZ, amplitude_vpp=AMP_VPP,
                            sync_pulse_ns=SYNC_PULSE_NS, sync_amp_vpp=SYNC_AMP_VPP),
                DAC_NYQ_DIRECT)
            fu, mu, sfdr_u = cw_arm(scope, "DUC CW",
                lambda: awg.send_cw_duc_sync(CW_FREQ_HZ, amplitude_vpp=AMP_VPP,
                            sync_pulse_ns=SYNC_PULSE_NS, sync_amp_vpp=SYNC_AMP_VPP),
                DAC_NYQ_DUC)
            print(f"  --> DUC penalty: {sfdr_d - sfdr_u:+.1f} dB")
            results['cw'] = (fd, md, fu, mu, sfdr_d, sfdr_u)

        if RUN_CHIRP:
            print("\n=== Chirp f(t) linearity: direct vs DUC ===")
            f_start = CHIRP_CTR_HZ - CHIRP_BW_HZ / 2
            f_stop  = CHIRP_CTR_HZ + CHIRP_BW_HZ / 2
            td, fd_i, fit_d, res_d = chirp_arm(scope, "DIRECT CHIRP",
                lambda: awg.send_chirp_direct_sync(f_start, f_stop, CHIRP_US, DEAD_US,
                            amplitude_vpp=AMP_VPP, sync_pulse_ns=SYNC_PULSE_NS,
                            sync_amp_vpp=SYNC_AMP_VPP))
            tu, fu_i, fit_u, res_u = chirp_arm(scope, "DUC CHIRP",
                lambda: awg.send_chirp_duc_sync(CHIRP_CTR_HZ, -CHIRP_BW_HZ / 2, CHIRP_BW_HZ / 2,
                            CHIRP_US, DEAD_US, amplitude_vpp=AMP_VPP,
                            sync_pulse_ns=SYNC_PULSE_NS, sync_amp_vpp=SYNC_AMP_VPP))
            print(f"  --> DUC penalty: {res_u - res_d:+.2f} kHz RMS")
            results['chirp'] = (td, fd_i, fit_d, tu, fu_i, fit_u)
    finally:
        # Always leave the instruments idle/normal — even on Ctrl-C or error.
        for name, fn in (("awg.stop", awg.stop),
                         ("scope.restore", lambda: scope.restore(channel=SIG_CH))):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}\n")

    pause("\nAll arms done — press Enter to show the direct-vs-DUC overlay...")
    _plot(results)
    return results


def _plot(results):
    nplots = ('cw' in results) + ('chirp' in results)
    if nplots == 0:
        return
    fig, axes = plt.subplots(nplots, 1, figsize=(9, 4 * nplots), squeeze=False)
    row = 0
    if 'cw' in results:
        fd, md, fu, mu, sfdr_d, sfdr_u = results['cw']
        ax = axes[row][0]
        ax.plot(fd / 1e6, md, label=f"direct (SFDR {sfdr_d:.1f} dBc)", alpha=0.8)
        ax.plot(fu / 1e6, mu, label=f"DUC (SFDR {sfdr_u:.1f} dBc)", alpha=0.8)
        ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Magnitude (dB)")
        ax.set_ylim(-100, 5); ax.grid(True); ax.legend(); ax.set_title("CW spectrum")
        row += 1
    if 'chirp' in results:
        td, fd_i, fit_d, tu, fu_i, fit_u = results['chirp']
        ax = axes[row][0]
        ax.plot(td * 1e6, fd_i / 1e6, '.', ms=1, label="direct f(t)")
        ax.plot(tu * 1e6, fu_i / 1e6, '.', ms=1, label="DUC f(t)")
        ax.set_xlabel("Time (µs)"); ax.set_ylabel("Inst. freq (MHz)")
        ax.grid(True); ax.legend(); ax.set_title("Chirp instantaneous frequency")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
