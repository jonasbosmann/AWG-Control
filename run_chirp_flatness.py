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

The ripple is not read off the FFT bin. The FFT only locates it; the delay
and depth then come from a least-squares sinusoid fit refined over +-1 bin,
so the delay resolution is NOT the 1/span bin width. See analyse().

`analyse(t, v)` is importable -- run_pad_test.py uses it so that the
pad-in/pad-out comparison runs through exactly one copy of this analysis.

Run:  python run_chirp_flatness.py
"""
import sys

import numpy as np
from scipy import signal as sig

from scope import Scope

SIG_CH = 3
EDGE_TRIM = 0.06          # ignore the windowed ends before judging flatness


def ripple_fit(f_s, flat, n_eff=None, n_refine=201):
    """Locate the dominant envelope ripple and fit it properly.

    The FFT of the detrended envelope against frequency has bins of width
    1/span, which is where the "+-2.5 ns" in LABBOOK 19 came from. That is a
    bound on where the PEAK BIN is, not on where the ripple is: a clean
    sinusoid at high contrast localises to a fraction of a bin. So the FFT
    only picks the starting point, and the delay comes from scanning tau
    across +-1 bin and keeping the least-squares best fit.

    Returns a dict, or None if the envelope is too short to judge.
    """
    df = float(np.mean(np.diff(f_s)))
    if not (df > 0 and flat.size > 256):
        return None
    w = flat * np.hanning(flat.size)
    d = np.abs(np.fft.rfft(w))
    tau = np.fft.rfftfreq(flat.size, df)
    lo = max(int(1e-9 / (tau[1] - tau[0])), 2)
    i = int(np.argmax(d[lo:]) + lo)
    contrast = float(d[i] / max(np.mean(d[lo:]), 1e-18))
    bin_w = float(tau[1] - tau[0])

    def amp_at(tt):
        """Least-squares amplitude of a sinusoid of delay tt in the envelope."""
        A = np.column_stack([np.cos(2 * np.pi * tt * f_s),
                             np.sin(2 * np.pi * tt * f_s)])
        c, *_ = np.linalg.lstsq(A, flat, rcond=None)
        return float(np.hypot(c[0], c[1])), c

    grid = np.linspace(max(tau[i] - bin_w, 0.1e-9), tau[i] + bin_w, n_refine)
    amps = np.array([amp_at(tt)[0] for tt in grid])
    j = int(np.argmax(amps))
    tau_fit = float(grid[j])
    amp, coef = amp_at(tau_fit)

    # residual scatter -> a real 1-sigma on the delay, rather than the bin
    model = coef[0] * np.cos(2 * np.pi * tau_fit * f_s) + \
        coef[1] * np.sin(2 * np.pi * tau_fit * f_s)
    resid = float(np.std(flat - model))
    span = float(f_s[-1] - f_s[0])
    # Cramer-Rao bound for the frequency of a sinusoid in WHITE noise, in
    # delay units: sigma_tau = sqrt(3) / (pi * SNR * sqrt(N) * span).
    #
    # Two reasons this is OPTIMISTIC and is printed as a floor, not an error
    # bar. The envelope is boxcar-smoothed, so neighbouring points are not
    # independent -- n_eff is the number of smoothing windows, not the number
    # of samples. And the residual is not noise: it is the rest of the
    # frequency response (other reflections, filter shape), which is
    # structured. The error bar that counts is the scatter over repeats,
    # which run_pad_test.py measures.
    n = n_eff if n_eff else flat.size
    snr = amp / max(resid, 1e-12)
    sigma_tau = float(np.sqrt(3.0) / (np.pi * max(snr, 1e-9) *
                                      np.sqrt(max(n, 2)) * max(span, 1.0)))

    pp_db = 2.0 * amp
    g = (10 ** (pp_db / 20) - 1) / (10 ** (pp_db / 20) + 1)
    return {"delay_s": tau_fit, "period_hz": 1.0 / tau_fit,
            "pp_db": pp_db, "contrast": contrast, "bin_s": bin_w,
            "sigma_s": sigma_tau, "resid_db": resid, "gamma_product": g,
            "span_hz": span, "n_periods": span * tau_fit}


def analyse(t, v, edge_trim=EDGE_TRIM, verbose=True):
    """Envelope-against-frequency analysis of one captured chirp."""
    dt = float(t[1] - t[0])
    z = sig.hilbert(v - v.mean())
    env = np.abs(z)
    k = max(env.size // 500, 21) | 1
    env_s = np.convolve(env, np.ones(k) / k, mode="same")
    on = env_s > 0.5 * env_s.max()
    idx = np.flatnonzero(on)
    if idx.size < 500:
        if verbose:
            print("no chirp found in the record")
        return None
    a, b = idx[0], idx[-1]
    trim = int(edge_trim * (b - a))
    a, b = a + trim, b - trim
    if verbose:
        print(f"chirp {t[a]*1e9:+.0f} to {t[b]*1e9:+.0f} ns "
              f"({(t[b]-t[a])*1e6:.3f} us after trimming {edge_trim*100:.0f}% "
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

    rate_hz_s = float(np.polyfit(np.arange(f_s.size) * dt, f_s, 1)[0])
    slope, icept = np.polyfit(f_s, db, 1)
    flat = db - (slope * f_s + icept)
    rip = ripple_fit(f_s, flat, n_eff=max(flat.size // kk, 8))
    n10 = max(len(db) // 10, 1)

    out = {"t_start_s": float(t[a]), "t_end_s": float(t[b]),
           "f_lo_hz": float(f_s[0]), "f_hi_hz": float(f_s[-1]),
           "chirp_rate_hz_per_s": rate_hz_s,
           "pp_db": float(db.max() - db.min()), "sd_db": float(db.std()),
           "tilt_db_per_hz": float(slope),
           "flat_pp_db": float(flat.max() - flat.min()),
           "flat_sd_db": float(flat.std()),
           "ptp_v": float(np.ptp(v)), "ripple": rip,
           "first10_mean_db": float(db[:n10].mean()),
           "rest_mean_db": float(db[n10:].mean()),
           "_f_s": f_s, "_db": db, "_flat": flat,
           "_slope": slope, "_icept": icept, "_t": t, "_v": v}

    if verbose:
        print(f"sweep {f_s[0]/1e9:.4f} -> {f_s[-1]/1e9:.4f} GHz, "
              f"{rate_hz_s/1e12:.2f} MHz/us")
        print(f"\nenvelope: peak-to-peak {out['pp_db']:.2f} dB, "
              f"sd {out['sd_db']:.2f} dB")
        print(f"  tilt      {slope*1e9:+.2f} dB/GHz "
              f"({slope*(f_s[-1]-f_s[0]):+.2f} dB across the sweep)")
        print(f"  after removing the tilt: pp {out['flat_pp_db']:.2f} dB, "
              f"sd {out['flat_sd_db']:.2f} dB")

        print("\n  envelope across the sweep:")
        for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            i = int(frac * (len(db) - 1))
            print(f"     {f_s[i]/1e9:.4f} GHz   {db[i]:+6.2f} dB")

        if rip:
            print(f"\n  strongest ripple: period {rip['period_hz']/1e6:.1f} MHz"
                  f" -> delay {rip['delay_s']*1e9:.3f} ns"
                  f" +- {rip['sigma_s']*1e9:.3f} ns (fit),"
                  f" contrast {rip['contrast']:.1f}")
            print(f"     fitted depth {rip['pp_db']:.3f} dB pp "
                  f"-> |G1.G2| = {rip['gamma_product']:.4f}, "
                  f"{rip['n_periods']:.1f} periods across the sweep")
            print(f"     (FFT bin {rip['bin_s']*1e9:.2f} ns; white-noise floor "
                  f"on the fit {rip['sigma_s']*1e9:.3f} ns -- a FLOOR, not an "
                  f"error bar; repeats give the real one)")
            if rip["contrast"] <= 8:
                print("     WEAK -- no single dominant reflection")

        print(f"\n  first 10% of the sweep: mean {out['first10_mean_db']:+.2f} "
              f"dB, pp {db[:n10].max()-db[:n10].min():.2f} dB")
        print(f"  the rest:               mean {out['rest_mean_db']:+.2f} dB, "
              f"pp {db[n10:].max()-db[n10:].min():.2f} dB")
    return out


def plot(out, path="marker_coherence/chirp_flatness.png", title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, v, f_s, db = out["_t"], out["_v"], out["_f_s"], out["_db"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(t * 1e6, v * 1e3, lw=0.3)
    ax[0].set_xlabel("time [us]")
    ax[0].set_ylabel("mV")
    ax[0].set_title(title or "chirp as captured")
    ax[1].plot(f_s / 1e9, db, lw=0.8)
    ax[1].plot(f_s / 1e9, out["_slope"] * f_s + out["_icept"], 'r--', lw=1,
               label=f"tilt {out['_slope']*1e9:+.2f} dB/GHz")
    rip = out.get("ripple")
    if rip:
        ax[1].plot([], [], ' ',
                   label=f"ripple {rip['pp_db']:.2f} dB pp @ "
                         f"{rip['delay_s']*1e9:.2f} ns")
    ax[1].set_xlabel("instantaneous frequency [GHz]")
    ax[1].set_ylabel("envelope [dB]")
    ax[1].set_title("envelope against frequency = the chain's response")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


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
    out = analyse(t, v)
    if out is None:
        return 1
    try:
        print(f"\nplot: {plot(out)}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
