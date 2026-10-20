"""Choosing a sample rate for a fixed FID window: the whole trade-off.

The window is set by the experiment (chirp + dead + trigger edge, then ~10 us
of FID). Sample rate then decides two things that pull against each other:

  COHERENCE. The scope's averager stacks acquisitions on its sample grid and
    cannot recover where inside a sample the trigger fell. A coarser grid
    means a bigger random phase per shot, so the average keeps less
    amplitude. Worse at higher signal frequency.

  SPEED. Record length = rate x window, and the scope's rearm time grows with
    record length. Fewer shots per second means fewer shots in a given run.

Averaging N shots gives SNR proportional to coherence * sqrt(N), and in a
fixed wall-clock time T you get N = shots_per_s * T, so the thing to maximise
is

        figure of merit = coherence * sqrt(shots per second)

Neither term alone answers it: the fastest rate has the best coherence and
the worst speed. This measures both on the actual instrument and prints the
product, so the choice is made on numbers rather than on which effect came
to mind first.

Coherence at the real IF is extrapolated from a measurement at whatever is
currently on the signal channel: phase error grows linearly with frequency,
so the coherence exponent scales as (f_target/f_measured)^2.

Run:  python run_rate_tradeoff.py [window_us] [f_target_MHz]
"""
import sys
import time

import numpy as np
from scipy import signal as sig

from scope import Scope
import reflock

SIG_CH, TRIG_CH = 3, 4
RATES = (3.125e9, 6.25e9, 12.5e9, 25e9, 50e9)
N_AVG = 64
N_SHOTS = 8               # for the coherence estimate at each rate


def coherence_at(scope, npts):
    ref, ph = None, []
    for _ in range(N_SHOTS):
        t, v = scope.get_waveform(SIG_CH, max_points=min(npts, 200000),
                                  fresh=True, width=2, acq_timeout=20.0)
        z = sig.hilbert(v - v.mean())
        if ref is None:
            ref = z
        else:
            n = min(ref.size, z.size)
            ph.append(float(np.angle(np.vdot(ref[:n], z[:n]))))
    return reflock.coherence(np.array(ph))[0]


def main():
    window_s = (float(sys.argv[1]) if len(sys.argv) > 1 else 10.0) * 1e-6
    f_target = (float(sys.argv[2]) if len(sys.argv) > 2 else 830.0) * 1e6
    s = Scope()
    with s._lock:
        was = {k: s._dev.query(q).strip() for k, q in (
            ("mode", "HORizontal:MODE?"), ("sr", "HORizontal:MODE:SAMPLERate?"),
            ("rl", "HORizontal:MODE:RECOrdlength?"),
            ("acq", "ACQuire:MODe?"), ("navg", "ACQuire:NUMAVg?"),
            ("stop", "ACQuire:STOPAfter?"))}
        rlmax = 62_500_000

    print(f"window {window_s*1e6:g} us, target signal {f_target/1e6:g} MHz, "
          f"{N_AVG}-shot averages\n")
    rows = []
    try:
        for rate in RATES:
            npts = int(round(rate * window_s))
            if npts > rlmax:
                continue
            with s._lock:
                s._dev.write("HORizontal:MODE MANual")
                s._dev.write(f"HORizontal:MODE:SAMPLERate {rate:.0f}")
                s._dev.write(f"HORizontal:MODE:RECOrdlength {npts}")
                s._dev.write("ACQuire:MODe SAMple")
                s._dev.query("*OPC?")
            s._pre.clear()
            c_meas = coherence_at(s, npts)

            with s._lock:
                s._dev.write("ACQuire:MODe AVErage")
                s._dev.write(f"ACQuire:NUMAVg {N_AVG}")
                s._dev.write("ACQuire:STATE STOP")
                s._dev.write("ACQuire:STOPAfter SEQuence")
                t0 = time.perf_counter()
                s._dev.write("ACQuire:STATE RUN")
                while time.perf_counter() - t0 < 90:
                    if s._dev.query("ACQuire:STATE?").strip() in ("0", "0.0"):
                        break
                    time.sleep(0.002)
                el = time.perf_counter() - t0
            rows.append({"rate": rate, "npts": npts, "c_meas": c_meas,
                         "secs": el, "shots_s": N_AVG / el})
            print(f"  {rate/1e9:5.2f} GS/s  {npts:>9,} pts   "
                  f"coherence {c_meas:.4f}   {N_AVG}-avg {el:6.2f}s "
                  f"-> {N_AVG/el:6.1f} shots/s")
    finally:
        with s._lock:
            s._dev.write("HORizontal:MODE MANual")
            s._dev.write(f"HORizontal:MODE:SAMPLERate {was['sr']}")
            s._dev.write(f"HORizontal:MODE:RECOrdlength {was['rl']}")
            s._dev.write(f"HORizontal:MODE {was['mode']}")
            s._dev.write(f"ACQuire:MODe {was['acq']}")
            s._dev.write(f"ACQuire:NUMAVg {was['navg']}")
            s._dev.write(f"ACQuire:STOPAfter {was['stop']}")
            s._dev.write("ACQuire:STATE RUN")
        s._pre.clear()
        print("\nrestored")

    if not rows:
        return 1
    # what is on the channel now, so the extrapolation is honest about it
    f_meas = 2.25e9
    print(f"\nextrapolating coherence from the {f_meas/1e9:g} GHz signal "
          f"currently on CH{SIG_CH} to {f_target/1e6:g} MHz")
    print(f"(phase error scales with frequency, so the coherence exponent "
          f"scales as (f2/f1)^2)\n")
    scale = (f_target / f_meas) ** 2
    print(f"  {'rate':>9}{'points':>11}{'shots/s':>10}"
          f"{'coh @IF':>10}{'FoM':>9}{'rel':>7}")
    best = 0.0
    for r in rows:
        c = float(np.exp(np.log(max(r["c_meas"], 1e-9)) * scale))
        r["c_if"] = c
        r["fom"] = c * np.sqrt(r["shots_s"])
        best = max(best, r["fom"])
    for r in rows:
        print(f"  {r['rate']/1e9:7.2f}G{r['npts']:>11,}{r['shots_s']:>10.1f}"
              f"{r['c_if']:>10.4f}{r['fom']:>9.2f}{r['fom']/best:>7.2f}")

    win = max(rows, key=lambda r: r["fom"])
    print(f"\n  best: {win['rate']/1e9:g} GS/s "
          f"({win['npts']:,} pts, {win['shots_s']:.1f} shots/s, "
          f"coherence {win['c_if']:.4f} at {f_target/1e6:g} MHz)")
    for r in rows:
        if r is win:
            continue
        t_ratio = (win["fom"] / r["fom"]) ** 2
        print(f"    vs {r['rate']/1e9:5.2f} GS/s: needs {t_ratio:4.2f}x the "
              f"wall-clock time for the same SNR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
