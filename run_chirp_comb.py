"""Where does the ~450 kHz dip comb on a wideband chirp come from?

Symptom (2026-10-20): a 2-2.5 GHz chirp, averaged, shows sharp dips in its
FT starting around 2.172 GHz and repeating every ~450 kHz.

A regular comb of NULLS (as opposed to a comb of lines) is the signature of
two copies of the same signal separated by a delay: |1 + exp(-j.2.pi.f.tau)|
nulls every 1/tau. So the dip spacing names the delay directly:

    1 / 450 kHz = 2.22 us

That is far too long to be an RF reflection -- 2.22 us of round trip is
about 440 m of cable -- so it has to be a timing artifact. 2.22 us is
suspiciously close to a DUC buffer period, which is the one thing in this
system that repeats on that scale.

Two candidates, and they are easy to tell apart:

  1. THE ANALYSIS WINDOW. The buffer repeats every T, so a capture longer
     than T contains the chirp more than once. Transforming two copies
     separated by T produces nulls every 1/T, and nothing is wrong with the
     chirp at all.
  2. SOMETHING REAL, e.g. the marker coupling into the output, or the chirp
     being contaminated by its neighbour in the buffer.

This measures the ripple delay directly -- take the magnitude spectrum over
the chirp band, flatten it, and transform again; a comb of period df shows
up as a peak at delay 1/df -- and then varies the things that would move it:

    * change the buffer period T   -> if tau follows T, it is the period
    * transform ONE chirp only     -> if the ripple vanishes, it was the window
    * switch the marker off        -> if the ripple vanishes, it was the marker

Run:  python run_chirp_comb.py
"""
import sys

import numpy as np
from scipy import signal as sig

from awg import AWG
from scope import Scope
from run_pulse_averaging import range_on_trigger

F_LO, F_HI = 2.0e9, 2.5e9
F_CARRIER = 0.5 * (F_LO + F_HI)
CHIRP_US = 2.0
AMP_VPP = 0.5
AWG_MARK_CH, MARKER = 1, 1
SIG_CH, TRIG_CH = 3, 4
SAMPLE_RATE = 6.25e9          # 2.5 GHz needs nothing faster; halves the bytes
CAPTURE_S = 30e-6
MAX_POINTS = 190000
N_AVG = 64


def specs():
    one = {"kind": "chirp", "gate": "chirp", "carrier_hz": F_CARRIER,
           "f_start_hz": F_LO - F_CARRIER, "f_stop_hz": F_HI - F_CARRIER,
           "amplitude_vpp": AMP_VPP, "sideband": "upper"}
    return {1: dict(one), 2: dict(one)}


def ripple_delay(t, v, win, f_lo=F_LO, f_hi=F_HI):
    """Delay implied by the ripple on the magnitude spectrum inside `win`.

    Returns (tau_s, contrast) -- contrast is the ripple peak relative to the
    mean of the delay spectrum, so a flat (ripple-free) band gives a small
    number however noisy it is.
    """
    m = (t >= win[0]) & (t < win[1])
    x = v[m] - v[m].mean()
    if x.size < 2048:
        return float("nan"), 0.0
    dt = float(t[1] - t[0])
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, dt)
    band = (f >= f_lo) & (f <= f_hi)
    if band.sum() < 64:
        return float("nan"), 0.0
    mb, fb = mag[band], f[band]
    k = max(mb.size // 40, 5) | 1
    smooth = np.convolve(mb, np.ones(k) / k, mode="same")
    r = mb / np.maximum(smooth, 1e-18) - 1.0
    r = r[k:-k]                                  # drop convolution edges
    if r.size < 64:
        return float("nan"), 0.0
    df = float(fb[1] - fb[0])
    d = np.abs(np.fft.rfft(r * np.hanning(r.size)))
    tau = np.fft.rfftfreq(r.size, df)
    lo = max(int(0.2e-6 / (tau[1] - tau[0])), 2)   # ignore the DC lobe
    i = int(np.argmax(d[lo:]) + lo)
    return float(tau[i]), float(d[i] / max(np.mean(d[lo:]), 1e-18))


def capture(awg, scope, trig_level, n_avg=N_AVG):
    scope.setup(channel=SIG_CH, n_averages=n_avg, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                              width=2, absolute_time=True, acq_timeout=30.0)
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    return t, v


def find_chirp(t, v):
    env = np.convolve(np.abs(sig.hilbert(v - v.mean())), np.ones(101) / 101,
                      mode="same")
    idx = np.flatnonzero(env > 0.5 * env.max())
    return float(t[idx[0]]), float(t[idx[-1]])


def send(awg, detect_us, marker=True):
    plan = awg.send_waveforms(specs(), mode="duc",
                              timing={"chirp_us": CHIRP_US, "dead_us": 0.1,
                                      "detect_us": detect_us}, verbose=False)
    if marker:
        awg.marker_pulse(AWG_MARK_CH, segnum=1,
                         n_wave_points=2 * plan["n_total"], marker=MARKER,
                         high_frac=100e-9 / plan["period_s"], level_vpp=1.0)
    return plan


def main():
    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)
    print(f"\nchirp {F_LO/1e9:.2f} -> {F_HI/1e9:.2f} GHz in {CHIRP_US} us\n")

    plan = send(awg, 0.1)
    with scope._lock:
        scope._dev.write("HORizontal:MODE MANual")
        scope._dev.write(f"HORizontal:MODE:SAMPLERate {SAMPLE_RATE:.0f}")
        scope._dev.write(
            f"HORizontal:MODE:RECOrdlength {int(CAPTURE_S*SAMPLE_RATE)}")
        scope._dev.write(f"CH{TRIG_CH}:SCAle 0.25")
    scope._pre.clear()
    scope.set_horizontal_position(5)
    m_ptp, m_mid = range_on_trigger(scope, TRIG_CH, 0.0)
    if m_ptp < 0.1:
        print("no marker on the trigger channel -- abort")
        return 1
    range_on_trigger(scope, SIG_CH, m_mid)
    with scope._lock:
        got = float(scope._dev.query("HORizontal:SAMPLERate?"))
    print(f"scope {got/1e9:.2f} GS/s, {CAPTURE_S*1e6:g} us record, "
          f"resolution {1/CAPTURE_S/1e3:.1f} kHz\n")

    print("=" * 72)
    print("1) does the ripple delay follow the BUFFER PERIOD?")
    print("=" * 72)
    print(f"  {'detect us':>10}{'period us':>11}{'1/period kHz':>14}"
          f"{'ripple tau us':>15}{'contrast':>10}")
    for detect_us in (0.1, 2.0, 6.0):
        plan = send(awg, detect_us)
        t, v = capture(awg, scope, m_mid)
        tau, c = ripple_delay(t, v, (t[0], t[-1]))
        print(f"  {detect_us:>10.1f}{plan['period_s']*1e6:>11.4f}"
              f"{1/plan['period_s']/1e3:>14.1f}{tau*1e6:>15.4f}{c:>10.1f}")

    print("\n" + "=" * 72)
    print("2) same capture, transformed over ONE chirp only")
    print("=" * 72)
    plan = send(awg, 0.1)
    t, v = capture(awg, scope, m_mid)
    c0, c1 = find_chirp(t, v)
    tau_all, con_all = ripple_delay(t, v, (t[0], t[-1]))
    tau_one, con_one = ripple_delay(t, v, (c0, c1))
    print(f"  whole {(t[-1]-t[0])*1e6:.1f} us record : tau "
          f"{tau_all*1e6:.4f} us, contrast {con_all:.1f}")
    print(f"  one chirp ({(c1-c0)*1e6:.3f} us)   : tau "
          f"{tau_one*1e6:.4f} us, contrast {con_one:.1f}")
    print(f"  (buffer period {plan['period_s']*1e6:.4f} us)")

    print("\n" + "=" * 72)
    print("3) marker off -- is it coupling into the output?")
    print("=" * 72)
    send(awg, 0.1, marker=True)
    t, v = capture(awg, scope, m_mid)
    tau_on, con_on = ripple_delay(t, v, (t[0], t[-1]))
    print(f"  marker ON  : tau {tau_on*1e6:.4f} us, contrast {con_on:.1f}")
    send(awg, 0.1, marker=True)
    awg.marker_off(AWG_MARK_CH, MARKER)
    # the scope can no longer trigger on the marker, so free-run for this one
    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=SIG_CH,
                trigger_level=0.0, trigger_mode="AUTO")
    t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                              width=2, absolute_time=True, acq_timeout=30.0)
    tau_off, con_off = ripple_delay(t, v, (t[0], t[-1]))
    print(f"  marker OFF : tau {tau_off*1e6:.4f} us, contrast {con_off:.1f}")
    print("  (untriggered, so the average is washed out -- only the ripple "
          "matters here)")

    send(awg, 0.1)
    print("\ndone; chirp and marker left running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
