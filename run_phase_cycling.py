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

    Both sums are computed, which means the artifact level is measured
    rather than assumed. With no sample present the "signal" is the AWG's
    own chirp, so the straight sum is a direct measurement of everything in
    the chain that does not follow the chirp phase.

This script drives the PRODUCTION path -- `AWG.send_waveforms()` with
`timing["n_cycles"]` and a per-channel `phases` list, plus `phasecycle` for
the analysis -- which is the same path the GUI uses. It is therefore the
regression test for both. Pass criteria, from 2026-10-20 at 50 GS/s: phases
within 0.0 deg of programmed, waveform correlations +-1.000, cycled sum
4.00x a single cycle, artifacts >= 50 dB down.

Phase bookkeeping lives in _build_channel: the NCO free-runs across the
buffer, so cycle k starts with a carrier phase 2*pi*f_c*t_k on top of
whatever the baseband carries, and that is subtracted when the samples are
built. The carrier still has to sit on the coherent grid for the FULL
buffer, which send_waveforms' grid snapping does.

Wiring: AWG CH1 M1 -> scope CH4 (trigger), AWG CH2 -> scope CH3 (signal).

Run:  python run_phase_cycling.py [n_cycles] [detect_us]
"""
import sys

import numpy as np

import phasecycle
from awg import AWG
from scope import Scope
from run_pulse_averaging import range_on_trigger

F_LO, F_HI = 1.99e9, 2.49e9
F_CARRIER = 0.5 * (F_LO + F_HI)
CHIRP_US, DEAD_US = 1.0, 0.1
AMP_VPP = 0.5
EDGE_FRAC = 0.05
AWG_SIG_CH, AWG_MARK_CH, MARKER = 2, 1, 1
MARKER_NS = 100
SIG_CH, TRIG_CH = 3, 4
SCOPE_RATE = 50e9      # 22 samples per carrier cycle at 2.24 GHz. Below ~20
                       # the linear interpolation in phasecycle.split_cycles
                       # injects its own phase and amplitude error -- at
                       # 12.5 GS/s it faked a -23.9 dB artifact floor where
                       # the truth was -52 dB.
N_AVG = 16


def specs_for(phases):
    """Same chirp on both channels: CH2 is measured, CH1 clocks the marker.

    Both channels carry the same spec so their segments are the same length
    and the marker stays rigid to the measured waveform. CH1's RF is shut
    off after the upload.
    """
    one = {"kind": "chirp", "gate": "chirp", "carrier_hz": F_CARRIER,
           "f_start_hz": F_LO - F_CARRIER, "f_stop_hz": F_HI - F_CARRIER,
           "amplitude_vpp": AMP_VPP, "sideband": "upper", "phases": phases}
    return {1: dict(one), 2: dict(one)}


def main():
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    detect_us = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    phases = [(k % 2) * np.pi for k in range(n_cycles)]     # 0, pi, 0, pi ...

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)

    plan = awg.send_waveforms(
        specs_for(phases), mode="duc",
        timing={"chirp_us": CHIRP_US, "dead_us": DEAD_US,
                "detect_us": detect_us, "n_cycles": n_cycles},
        window_frac=EDGE_FRAC, verbose=False, coherent=True)

    info = plan["channels"][AWG_SIG_CH]
    used = [e for e in plan["coherent_grid"] if e["channel"] == AWG_SIG_CH]
    carrier = used[0]["carrier_hz"] if used else F_CARRIER
    emitted = info["phases_emitted_rad"]
    print(f"\n{n_cycles} cycles x {plan['t_cycle']*1e6:.4f} us = "
          f"{plan['period_s']*1e6:.4f} us buffer "
          f"({plan['n_total']:,} complex samples)")
    print(f"carrier {carrier/1e6:.6f} MHz, baseband "
          f"{info['f_start_hz']/1e6:+.3f} to {info['f_stop_hz']/1e6:+.3f} MHz "
          f"-> chirp {F_LO/1e9:.3f}-{F_HI/1e9:.3f} GHz")
    print(f"programmed phases: "
          f"{', '.join(f'{np.degrees(p):.0f}' for p in phases)} deg")

    awg.marker_pulse(AWG_MARK_CH, segnum=AWG_MARK_CH,
                     n_wave_points=2 * plan["n_total"], marker=MARKER,
                     high_frac=MARKER_NS * 1e-9 / plan["period_s"],
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")

    npts = int(round(plan["period_s"] * SCOPE_RATE))
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

    f_mid = carrier + 0.5 * (info["f_start_hz"] + info["f_stop_hz"])
    try:
        cut = phasecycle.split_cycles(t, v, n_cycles, plan["t_cycle"],
                                      CHIRP_US * 1e-6, carrier_hz=f_mid)
    except ValueError as e:
        print(f"  {e}")
        return 1
    res = phasecycle.combine(cut["analytic"], emitted)
    print("\n".join(phasecycle.summary_lines(res)))

    try:
        _plot(t, v, cut, res, plan, n_cycles)
        print("\n  plot: marker_coherence/phase_cycling.png")
    except Exception:
        # The numbers above are the result; a broken plot must not lose them.
        # Print the traceback though -- a bare message sent me looking at the
        # wrong line once already.
        import traceback
        print("  (plot skipped)")
        traceback.print_exc()
    return 0


def _plot(t, v, cut, res, plan, n_cycles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = cut["raw"]
    dt, t0, n_win = cut["dt"], cut["t0"], cut["n_win"]
    deg = [int(round(d)) % 360 for d in res["programmed_deg"]]
    cols = plt.cm.viridis(np.linspace(0, 0.85, n_cycles))
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))

    # (a) the whole buffer, with each cycle shaded and labelled
    ax[0][0].plot(t * 1e6, v * 1e3, lw=0.25, color="0.25")
    for k in range(n_cycles):
        a = t0 + k * plan["t_cycle"]
        ax[0][0].axvspan(a * 1e6, (a + n_win * dt) * 1e6, color=cols[k],
                         alpha=0.25)
        ax[0][0].text((a + n_win * dt / 2) * 1e6, v.max() * 1e3 * 1.08,
                      f"{deg[k]}°", ha="center", color=cols[k],
                      fontweight="bold")
    ax[0][0].set_xlabel("time from trigger [us]")
    ax[0][0].set_ylabel("mV")
    ax[0][0].set_title(f"{n_cycles} chirp/FID cycles in one buffer "
                       f"({plan['t_cycle']*1e6:.3f} us each)", fontsize=10)

    # (b) the actual phase flip, a few cycles of carrier from each
    mid = n_win // 2
    span = int(round(1.6e-9 / dt))
    tt = np.arange(span) * dt * 1e9
    for k in range(n_cycles):
        ax[0][1].plot(tt, raw[k, mid:mid + span] * 1e3, lw=1.8,
                      ls="-" if deg[k] == 0 else "--", color=cols[k],
                      label=f"cycle {k}: {deg[k]}°  "
                            f"(corr {res['correlation'][k]:+.2f})")
    if n_cycles > 1:
        ax[0][1].plot(tt,
                      (raw[0, mid:mid + span] + raw[1, mid:mid + span]) * 1e3,
                      lw=2.5, color="0.45", alpha=0.8,
                      label="cycle 0 + cycle 1  (cancels)")
    ax[0][1].axhline(0, color="0.7", lw=0.5)
    ax[0][1].set_xlabel("time within the cycle [ns, offset]")
    ax[0][1].set_ylabel("mV")
    ax[0][1].set_title("same instant of each chirp: 180° cycles are "
                       "inverted", fontsize=10)
    ax[0][1].legend(fontsize=7.5, ncol=1, loc="upper right")

    # (c) measured against programmed, each unwrapped onto its own target so
    #     -1 deg is not drawn at 359 and read as a catastrophe
    meas = [res["programmed_deg"][k] + res["error_deg"][k]
            for k in range(n_cycles)]
    ax[1][0].plot(range(n_cycles), res["programmed_deg"], "o--", ms=11,
                  mfc="none", color="0.4", label="programmed")
    ax[1][0].plot(range(n_cycles), meas, "x", ms=11, mew=2.5,
                  color="crimson", label="measured")
    for k in range(n_cycles):
        ax[1][0].annotate(f"{res['error_deg'][k]:+.1f}°", (k, meas[k]),
                          textcoords="offset points", xytext=(8, 6),
                          fontsize=8, color="crimson")
    ax[1][0].set_xticks(range(n_cycles))
    ax[1][0].set_xlabel("cycle")
    ax[1][0].set_ylabel("chirp phase [deg]")
    ax[1][0].set_ylim(min(res["programmed_deg"]) - 60,
                      max(res["programmed_deg"]) + 60)
    ax[1][0].grid(alpha=0.3)
    ax[1][0].legend(fontsize=8)
    ax[1][0].set_title("emitted phase follows the programmed phase",
                       fontsize=10)

    # (d) what the two sums do
    tw = np.arange(n_win) * dt * 1e6
    ax[1][1].plot(tw, np.real(res["cycled_w"]) * 1e3, lw=0.5,
                  color="tab:green",
                  label=f"phase-cycled sum  ({res['cycled']*1e3:.0f} mV)")
    ax[1][1].plot(tw, np.real(res["straight_w"]) * 1e3, lw=0.7, color="tab:red",
                  label=f"straight sum  ({res['straight']*1e3:.0f} mV, "
                        f"{res['artifact_db']:.0f} dB)")
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
