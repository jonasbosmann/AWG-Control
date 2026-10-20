"""Several phase-cycled chirp/FID cycles inside one AWG buffer.

Two things at once, both standard practice in modern CP-FTMW:

  MORE FIDS PER GAS PULSE. A gas pulse lasts hundreds of microseconds but a
    chirp/FID cycle only ~11 us, so K cycles fit inside one pulse. One marker
    still fires per buffer, so the scope takes ONE acquisition holding K
    FIDs and pays its ~21 ms rearm once per gas pulse instead of once per
    FID. That is the bottleneck measured 2026-10-20: the scope manages about
    32 acquisitions a second whatever you do.

  PHASE CYCLING. Each cycle's chirp carries a programmed phase. The
    molecular response follows the chirp phase; the things that spoil a
    measurement mostly do not -- chirp breakthrough into the receiver,
    detector ringing, DC offset, digital pickup. So combining the K sub-FIDs
    with the conjugate phases ADDS the signal and CANCELS everything that
    ignored the chirp.

      alternating sum (0, pi, 0, pi ... with signs +,-,+,-)  -> signal
      straight sum    (same data, all +)                     -> artifacts

    Both are computed here, which means the artifact level is measured
    rather than assumed. With no sample present the "signal" is the AWG's
    own chirp, so the straight sum is a direct measurement of everything in
    the chain that does not follow the chirp phase.

Phase bookkeeping
-----------------
The NCO free-runs across the whole buffer, so cycle k starts with a carrier
phase 2*pi*f_c*t_k on top of whatever the baseband carries. That is
subtracted when the waveform is built, so the phase actually emitted by
cycle k is the phase asked for and nothing else. The carrier still has to
sit on the coherent grid for the FULL buffer, which snap_duc_carrier handles.

Wiring: AWG CH1 M1 -> scope CH4 (trigger), AWG CH2 -> scope CH3 (signal).

Run:  python run_phase_cycling.py [n_cycles] [detect_us]
"""
import sys

import numpy as np
from scipy import signal as sig

from awg import AWG
from scope import Scope
from run_pulse_averaging import range_on_trigger

RATE_DAC, INTERP = 9e9, 8
RATE_CX = RATE_DAC / INTERP
F_LO, F_HI = 1.99e9, 2.49e9
CHIRP_US, DEAD_US = 1.0, 0.1
AMP_VPP = 0.5
EDGE_FRAC = 0.05
AWG_SIG_CH, AWG_MARK_CH, MARKER = 2, 1, 1
SIG_CH, TRIG_CH = 3, 4
SCOPE_RATE = 50e9      # 22 samples per carrier cycle at 2.24 GHz, so
                       # the zoomed waveform is a smooth sine rather
                       # than 5.6 jagged points; also the best
                       # sample-grid coherence for the average
N_AVG = 16
MARKER_NS = 100


def build_cycles(n_cycles, detect_us, phases):
    """K chirp/dead/detect cycles in one buffer. Returns (I, Q, meta)."""
    n_chirp = int(round(CHIRP_US * 1e-6 * RATE_CX))
    n_cycle = int(round((CHIRP_US + DEAD_US + detect_us) * 1e-6 * RATE_CX))
    n_cycle = int(np.ceil(n_cycle / 32)) * 32          # keep cycles aligned
    n_total = n_cycle * n_cycles
    t_cycle = n_cycle / RATE_CX

    carrier = AWG.snap_duc_carrier(0.5 * (F_LO + F_HI), n_total,
                                   RATE_DAC, INTERP)[0]
    bb_lo, bb_hi = F_LO - carrier, F_HI - carrier

    t = np.arange(n_chirp) / RATE_CX
    T = t[-1] if n_chirp > 1 else 1e-9
    base = 2 * np.pi * (bb_lo * t + (bb_hi - bb_lo) / (2 * T) * t ** 2)
    env = AWG._gauss_win(n_chirp, EDGE_FRAC)

    i_arr, q_arr = np.zeros(n_total), np.zeros(n_total)
    emitted = []
    for k in range(n_cycles):
        # undo the carrier's own advance so the emitted phase is the one asked
        carrier_adv = 2 * np.pi * carrier * (k * t_cycle)
        phi = phases[k] - carrier_adv
        z = env * np.exp(1j * (base + phi))
        s = k * n_cycle
        i_arr[s:s + n_chirp] = z.real
        q_arr[s:s + n_chirp] = z.imag
        emitted.append(phases[k])
    peak = max(np.abs(i_arr + 1j * q_arr).max(), 1e-12)
    i_arr, q_arr = i_arr / peak, q_arr / peak
    return i_arr, q_arr, {"n_chirp": n_chirp, "n_cycle": n_cycle,
                          "n_total": n_total, "t_cycle": t_cycle,
                          "carrier": carrier, "bb": (bb_lo, bb_hi),
                          "phases": emitted}


def main():
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    detect_us = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    phases = [(k % 2) * np.pi for k in range(n_cycles)]     # 0, pi, 0, pi ...

    i_arr, q_arr, m = build_cycles(n_cycles, detect_us, phases)
    print(f"\n{n_cycles} cycles x {m['t_cycle']*1e6:.4f} us = "
          f"{m['n_total']/RATE_CX*1e6:.4f} us buffer ({m['n_total']:,} "
          f"complex samples)")
    print(f"carrier {m['carrier']/1e6:.6f} MHz, baseband "
          f"{m['bb'][0]/1e6:+.3f} to {m['bb'][1]/1e6:+.3f} MHz "
          f"-> chirp {F_LO/1e9:.3f}-{F_HI/1e9:.3f} GHz")
    print(f"programmed phases: "
          f"{', '.join(f'{np.degrees(p):.0f}' for p in phases)} deg")

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)
    awg.duc_iq_setup_dual(i_arr, q_arr, m["carrier"], AMP_VPP)
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * m["n_total"],
                     marker=MARKER,
                     high_frac=MARKER_NS * 1e-9 / (m["n_total"] / RATE_CX),
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")

    npts = int(round(m["n_total"] / RATE_CX * SCOPE_RATE))
    with scope._lock:
        scope._dev.write("HORizontal:MODE MANual")
        scope._dev.write(f"HORizontal:MODE:SAMPLERate {SCOPE_RATE:.0f}")
        scope._dev.write(f"HORizontal:MODE:RECOrdlength {npts}")
        scope._dev.write(f"CH{TRIG_CH}:SCAle 0.25")
    scope._pre.clear()
    scope.set_horizontal_position(2)
    m_ptp, m_mid = range_on_trigger(scope, TRIG_CH, 0.0)
    if m_ptp < 0.1:
        print(f"no marker on CH{TRIG_CH} -- abort")
        return 1
    range_on_trigger(scope, SIG_CH, m_mid)
    print(f"scope {SCOPE_RATE/1e9:g} GS/s, {npts:,} pts, "
          f"{N_AVG}-average on the marker\n")

    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
                trigger_level=m_mid, trigger_mode="NORMal")
    t, v = scope.get_waveform(SIG_CH, max_points=npts, fresh=True, width=2,
                              absolute_time=True, acq_timeout=30.0)
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=m_mid, trigger_mode="NORMal")

    # locate the first chirp, then step by exactly one cycle
    env = np.convolve(np.abs(sig.hilbert(v - v.mean())),
                      np.ones(151) / 151, mode="same")
    first = int(np.flatnonzero(env > 0.5 * env.max())[0])
    dt = float(t[1] - t[0])
    n_win = int(round(CHIRP_US * 1e-6 / dt))

    # Cycles must be cut at their EXACT times, not at a rounded number of
    # samples. One cycle is 3.100444 us = 38755.56 scope samples at 80 ps, so
    # stepping by a whole 38756 leaves 0.44 samples = 35 ps of slip per
    # cycle, which at 2.24 GHz is 28 deg of phase -- it showed up as a
    # perfectly linear phase error across the cycles. Interpolating onto the
    # exact grid removes it. Same trap as the sub-sample trigger offset.
    t0 = float(t[first])
    segs, raw = [], []
    for k in range(n_cycles):
        grid = t0 + k * m["t_cycle"] + np.arange(n_win) * dt
        if grid[-1] > t[-1]:
            print(f"  cycle {k} runs past the record -- shorten detect_us")
            return 1
        seg = np.interp(grid, t, v)
        raw.append(seg - seg.mean())
        segs.append(sig.hilbert(seg - seg.mean()))
    segs = np.array(segs)
    raw = np.array(raw)

    print(f"  {'cycle':>6}{'programmed':>12}{'measured':>11}{'error':>9}"
          f"{'amplitude':>11}")
    ref = segs[0]
    for k, z in enumerate(segs):
        rel = float(np.degrees(np.angle(np.vdot(ref, z))))
        want = float(np.degrees(phases[k] - phases[0]))
        err = (rel - want + 180) % 360 - 180
        print(f"  {k:>6}{want:>11.1f}d{rel:>10.1f}d{err:>8.1f}d"
              f"{np.abs(z).mean()*1e3:>10.2f}mV")

    print(f"\n  waveform correlation with cycle 0 "
          f"(-1.00 = exactly inverted):")
    for k in range(n_cycles):
        c = float(np.dot(raw[0], raw[k]) /
                  np.sqrt(np.dot(raw[0], raw[0]) * np.dot(raw[k], raw[k])))
        print(f"    cycle {k} ({np.degrees(phases[k]):>5.0f} deg): {c:+.3f}")

    signs = np.array([np.exp(-1j * p) for p in phases])
    cycled = np.abs((segs * signs[:, None]).sum(0)).mean()
    straight = np.abs(segs.sum(0)).mean()
    single = np.abs(segs).mean()
    print(f"\n  phase-cycled sum : {cycled*1e3:8.2f} mV   "
          f"({cycled/single:.2f}x a single cycle, ideal {n_cycles:.2f})")
    print(f"  straight sum     : {straight*1e3:8.2f} mV   "
          f"({straight/cycled*100:.2f}% of the cycled sum)")
    print(f"\n  -> everything that does not follow the chirp phase is "
          f"{20*np.log10(max(straight/cycled, 1e-12)):.1f} dB down")
    print(f"     after {n_cycles}-step cycling. That is the residual phase "
          f"cycling removes.")

    try:
        _plot(t, v, raw, segs, signs, phases, m, dt, t0, n_win, n_cycles)
        print("\n  plot: marker_coherence/phase_cycling.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return 0


def _plot(t, v, raw, segs, signs, phases, m, dt, t0, n_win, n_cycles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = [int(round(np.degrees(p))) % 360 for p in phases]
    cols = plt.cm.viridis(np.linspace(0, 0.85, n_cycles))
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))

    # (a) the whole buffer, with each cycle shaded and labelled
    ax[0][0].plot(t * 1e6, v * 1e3, lw=0.25, color="0.25")
    for k in range(n_cycles):
        a = t0 + k * m["t_cycle"]
        ax[0][0].axvspan(a * 1e6, (a + n_win * dt) * 1e6, color=cols[k],
                         alpha=0.25)
        ax[0][0].text((a + n_win * dt / 2) * 1e6, v.max() * 1e3 * 1.08,
                      f"{deg[k]}\u00b0", ha="center", color=cols[k],
                      fontweight="bold")
    ax[0][0].set_xlabel("time from trigger [us]")
    ax[0][0].set_ylabel("mV")
    ax[0][0].set_title(f"{n_cycles} chirp/FID cycles in one buffer "
                       f"({m['t_cycle']*1e6:.3f} us each)", fontsize=10)

    # (b) the actual phase flip, a few cycles of carrier from each
    mid = n_win // 2
    span = int(round(1.6e-9 / dt))
    tt = np.arange(span) * dt * 1e9
    for k in range(n_cycles):
        c = float(np.dot(raw[0], raw[k]) /
                  np.sqrt(np.dot(raw[0], raw[0]) * np.dot(raw[k], raw[k])))
        ax[0][1].plot(tt, raw[k, mid:mid + span] * 1e3, lw=1.8,
                      ls="-" if deg[k] == 0 else "--", color=cols[k],
                      label=f"cycle {k}: {deg[k]}\u00b0  (corr {c:+.2f})")
    ax[0][1].plot(tt, (raw[0, mid:mid + span] + raw[1, mid:mid + span]) * 1e3,
                  lw=2.5, color="0.45", alpha=0.8,
                  label="cycle 0 + cycle 1  (cancels)")
    ax[0][1].axhline(0, color="0.7", lw=0.5)
    ax[0][1].set_xlabel("time within the cycle [ns, offset]")
    ax[0][1].set_ylabel("mV")
    ax[0][1].set_title("same instant of each chirp: 180\u00b0 cycles are "
                       "inverted", fontsize=10)
    ax[0][1].legend(fontsize=7.5, ncol=1, loc="upper right")

    # (c) measured against programmed
    ref = segs[0]
    # unwrap each measurement onto its own programmed value, otherwise -1 deg
    # is drawn at 359 and looks like a catastrophic error
    meas = []
    for k, z in enumerate(segs):
        raw_deg = float(np.degrees(np.angle(np.vdot(ref, z))))
        meas.append(deg[k] + ((raw_deg - deg[k] + 180) % 360 - 180))
    ax[1][0].plot(range(n_cycles), deg, "o--", ms=11, mfc="none",
                  color="0.4", label="programmed")
    ax[1][0].plot(range(n_cycles), meas, "x", ms=11, mew=2.5,
                  color="crimson", label="measured")
    for k in range(n_cycles):
        e = (meas[k] - deg[k] + 180) % 360 - 180
        ax[1][0].annotate(f"{e:+.1f}\u00b0", (k, meas[k]),
                          textcoords="offset points", xytext=(8, 6),
                          fontsize=8, color="crimson")
    ax[1][0].set_xticks(range(n_cycles))
    ax[1][0].set_xlabel("cycle")
    ax[1][0].set_ylabel("chirp phase [deg]")
    ax[1][0].set_ylim(min(deg) - 60, max(deg) + 60)
    ax[1][0].grid(alpha=0.3)
    ax[1][0].legend(fontsize=8)
    ax[1][0].set_title("emitted phase follows the programmed phase",
                       fontsize=10)

    # (d) what the two sums do
    cyc = np.real((segs * signs[:, None]).sum(0))
    stg = np.real(segs.sum(0))
    tw = np.arange(n_win) * dt * 1e6
    ax[1][1].plot(tw, cyc * 1e3, lw=0.5, color="tab:green",
                  label=f"phase-cycled sum  ({np.abs((segs*signs[:,None]).sum(0)).mean()*1e3:.0f} mV)")
    ax[1][1].plot(tw, stg * 1e3, lw=0.7, color="tab:red",
                  label=f"straight sum  ({np.abs(segs.sum(0)).mean()*1e3:.0f} mV)")
    ax[1][1].set_xlabel("time within the cycle [us]")
    ax[1][1].set_ylabel("mV")
    ax[1][1].legend(fontsize=8)
    ax[1][1].set_title("cycling adds the chirp and cancels what ignores it",
                       fontsize=10)

    fig.suptitle("Phase cycling across chirps in one AWG buffer", fontsize=12)
    fig.tight_layout()
    fig.savefig("marker_coherence/phase_cycling.png", dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
