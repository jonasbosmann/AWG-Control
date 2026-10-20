"""What is the structure on a chirp's envelope?

A linear chirp maps frequency onto time, so the amplitude envelope IS the
frequency response of everything between the DAC and the scope. Anything
that is not flat means something: DAC sinc rolloff and cable loss show up as
a smooth tilt, a mismatch between two components shows up as a periodic
ripple whose period in frequency gives the round-trip delay, and a settling
transient shows up only at the start.

This reads whatever the scope is currently set to -- it changes nothing
except arming one acquisition -- then reports:

    tilt        a straight-line fit of envelope against frequency, dB
    ripple      periodicity of the detrended envelope, as a delay and a
                depth; a mismatch at distance d shows up at 2d/v
    start       the first 10% of the sweep against the rest, which is where
                a filter or interpolator transient would sit

Run:  python run_chirp_flatness.py
"""
import sys

import numpy as np
from scipy import signal as sig

from scope import Scope

SIG_CH = 3
EDGE_TRIM = 0.06          # ignore the windowed ends before judging flatness


def main():
    s = Scope()
    with s._lock:
        rl = int(float(s._dev.query("HORizontal:MODE:RECOrdlength?")))
        sr = float(s._dev.query("HORizontal:SAMPLERate?"))
        acq = s._dev.query("ACQuire:MODe?").strip()
        navg = s._dev.query("ACQuire:NUMAVg?").strip()
    print(f"record {rl} pts at {sr/1e9:.3f} GS/s = {rl/sr*1e6:.4f} us, "
          f"{acq} x{navg}")

    t, v = s.get_waveform(SIG_CH, max_points=min(rl, 200000), fresh=True,
                          width=2, absolute_time=True, acq_timeout=20.0)
    dt = float(t[1] - t[0])
    z = sig.hilbert(v - v.mean())
    env = np.abs(z)
    k = max(env.size // 500, 21) | 1
    env_s = np.convolve(env, np.ones(k) / k, mode="same")
    on = env_s > 0.5 * env_s.max()
    idx = np.flatnonzero(on)
    if idx.size < 500:
        print("no chirp found in the record")
        return 1
    a, b = idx[0], idx[-1]
    trim = int(EDGE_TRIM * (b - a))
    a, b = a + trim, b - trim
    print(f"chirp {t[a]*1e9:+.0f} to {t[b]*1e9:+.0f} ns "
          f"({(t[b]-t[a])*1e6:.3f} us after trimming {EDGE_TRIM*100:.0f}% "
          f"off each windowed end)")

    seg = v[a:b] - np.mean(v[a:b])
    zz = sig.hilbert(seg)
    ph = np.unwrap(np.angle(zz))
    f_i = np.gradient(ph, dt) / (2 * np.pi)
    e_i = np.abs(zz)
    kk = max(e_i.size // 200, 21) | 1
    f_s = np.convolve(f_i, np.ones(kk) / kk, mode="valid")
    e_s = np.convolve(e_i, np.ones(kk) / kk, mode="valid")
    db = 20 * np.log10(np.maximum(e_s, 1e-12) / np.median(e_s))

    print(f"sweep {f_s[0]/1e9:.4f} -> {f_s[-1]/1e9:.4f} GHz, "
          f"{np.polyfit(np.arange(f_s.size)*dt, f_s, 1)[0]/1e12:.2f} MHz/us")
    print(f"\nenvelope: peak-to-peak {db.max()-db.min():.2f} dB, "
          f"sd {db.std():.2f} dB")

    slope, icept = np.polyfit(f_s, db, 1)
    print(f"  tilt      {slope*1e9:+.2f} dB/GHz "
          f"({slope*(f_s[-1]-f_s[0]):+.2f} dB across the sweep)")
    flat = db - (slope * f_s + icept)
    print(f"  after removing the tilt: pp {flat.max()-flat.min():.2f} dB, "
          f"sd {flat.std():.2f} dB")

    print("\n  envelope across the sweep:")
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        i = int(frac * (len(db) - 1))
        print(f"     {f_s[i]/1e9:.4f} GHz   {db[i]:+6.2f} dB")

    # periodic ripple -> a delay
    df = float(np.mean(np.diff(f_s)))
    if df > 0 and flat.size > 256:
        w = flat * np.hanning(flat.size)
        d = np.abs(np.fft.rfft(w))
        tau = np.fft.rfftfreq(flat.size, df)
        lo = max(int(1e-9 / (tau[1] - tau[0])), 2)
        i = int(np.argmax(d[lo:]) + lo)
        con = d[i] / max(np.mean(d[lo:]), 1e-18)
        print(f"\n  strongest ripple: period "
              f"{1/tau[i]/1e6:.1f} MHz -> delay {tau[i]*1e9:.2f} ns, "
              f"contrast {con:.1f}")
        if con > 8:
            print(f"     a round trip of {tau[i]*1e9:.2f} ns is about "
                  f"{tau[i]*2e8/2*100:.0f} cm of cable each way")
        else:
            print("     (weak -- no single dominant reflection)")

    n10 = max(len(db) // 10, 1)
    print(f"\n  first 10% of the sweep: mean {db[:n10].mean():+.2f} dB, "
          f"pp {db[:n10].max()-db[:n10].min():.2f} dB")
    print(f"  the rest:               mean {db[n10:].mean():+.2f} dB, "
          f"pp {db[n10:].max()-db[n10:].min():.2f} dB")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(t * 1e6, v * 1e3, lw=0.3)
        ax[0].set_xlabel("time [us]")
        ax[0].set_ylabel("mV")
        ax[0].set_title("chirp as captured")
        ax[1].plot(f_s / 1e9, db, lw=0.8)
        ax[1].plot(f_s / 1e9, slope * f_s + icept, 'r--', lw=1,
                   label=f"tilt {slope*1e9:+.2f} dB/GHz")
        ax[1].set_xlabel("instantaneous frequency [GHz]")
        ax[1].set_ylabel("envelope [dB]")
        ax[1].set_title("envelope against frequency = the chain's response")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("marker_coherence/chirp_flatness.png", dpi=130)
        plt.close(fig)
        print("\nplot: marker_coherence/chirp_flatness.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
