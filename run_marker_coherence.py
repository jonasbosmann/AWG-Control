"""Does coherent averaging actually work, triggered on the AWG marker?

Wiring (as left on the bench 2026-10-19):

    AWG CH1 M1  ->  scope CH4     TRIGGER   (marker; CH1's RF output is OFF)
    AWG CH2     ->  scope CH3     SIGNAL    (DUC CW carrier)

Why the marker and not the 10 MHz
---------------------------------
The marker is played out of the same segment memory, on the same sample
clock, as the waveform. It is therefore rigid to the output in a way no
external trigger can be. That is exactly what CP-FTMW needs: every FID must
start at the same phase of the chirp.

Why this test needs a DIFFERENT control from the last one
---------------------------------------------------------
Triggering on the marker makes the AWG's reference lock almost irrelevant to
the result. If the AWG clock runs 3 ppm fast, the marker runs 3 ppm fast with
it, so the phase AT THE TRIGGER is unchanged -- the clock error shows up as a
small error in the measured FREQUENCY, not as lost coherence. Switching the
AWG to its internal TCXO is therefore NOT a valid negative control here: it
would be expected to pass, and passing would prove nothing.

The control used instead is the NCO grid condition, which is the failure mode
that genuinely applies in DUC mode. The DUC NCO free-runs; it is not reset at
segment start. So the carrier phase at the marker repeats only if the carrier
fits a whole number of cycles into one buffer:

    f_NCO * T_buffer = integer            T_buffer = n_total / (9 GS/s / 8)

    n_total = 4096  ->  T_buffer = 3.6409 us,  grid = 274658.203125 Hz

    ON  grid   k = 2944  ->  f = 808 593 750 Hz   exactly (and a whole number
                             of Hz, so the integer-Hz :NCO:CFR1 write loses
                             nothing -- k has to be a multiple of 64 for that)
    OFF grid   f + 100 kHz -> 0.364 of a grid step, i.e. the phase at segment
                             start advances 131 deg per buffer and is fully
                             randomised over the ~10^5 buffers between shots

Both states run at the same frequency to within 0.01%, through the same
trigger, the same cables and the same code path. The only difference is
whether the carrier closes the buffer. If ON passes and OFF fails, the
measurement has teeth -- and the grid-snapping note is confirmed at the same
time.

808.59 MHz is inside the 730-830 MHz IF band this spectrometer will actually
work in, so the number is the real one, not a convenient one.

What is reported
----------------
coherence = |<exp(i*phi)>| over N shots: the fraction of amplitude that
survives an N-shot coherent average. 1.0 = averaging is lossless and buys the
full sqrt(N) in SNR; the floor for random phase is 1/sqrt(N).

Three numbers per state, answering three different questions:

  trigger-referenced   phase measured from the interpolated trigger instant
                       (WFMOutpre:XZEro). This is the AWG's own coherence --
                       what PC-side averaging can reach if the shots are
                       re-aligned in software.
  sample-grid          phase measured from the first sample, i.e. ignoring
                       the sub-sample trigger position. This is what any
                       averager that simply stacks records sees.
  hardware average     the scope's own N-average, amplitude compared against
                       the mean single shot. The end-to-end answer, free of
                       every assumption in the analysis above.

The gap between the first two is the price of sub-sample trigger
quantisation; the third should land near the second if the scope stacks on
its sample grid.

Run:  python run_marker_coherence.py [n_shots]
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from awg import AWG
from scope import Scope
import reflock

# -- the waveform ------------------------------------------------------
DAC_RATE   = 9e9
INTERP     = 8
RATE_CX    = DAC_RATE / INTERP          # 1.125 GS/s complex
N_TOTAL    = 4096                       # complex samples per buffer
T_BUF      = N_TOTAL / RATE_CX          # 3.6409 us
GRID_HZ    = RATE_CX / N_TOTAL          # 274658.203125 Hz
K_ON       = 2944                       # multiple of 64 -> f lands on whole Hz
F_ON       = K_ON * GRID_HZ             # 808 593 750 Hz exactly
F_OFF      = F_ON + 100e3               # 0.364 grid steps off
AMP_VPP    = 0.5

# -- wiring ------------------------------------------------------------
AWG_SIG_CH, AWG_MARK_CH, MARKER = 2, 1, 1
SIG_CH, TRIG_CH = 3, 4

# -- acquisition -------------------------------------------------------
CAPTURE_S   = 1e-6
MAX_POINTS  = 15000
N_AVG       = 64           # hardware-average depth for the end-to-end check
MARKER_FRAC = 0.05         # 184 ns of the 3.64 us buffer


def _look(scope, ch, settle=0.15):
    """One free-running look at a channel, for ranging only.

    Returns (ptp, midpoint). Both come out of the captured record rather than
    a MEASUrement block -- fewer moving parts, and the midpoint is exactly
    what the edge trigger wants.
    """
    scope.setup(channel=ch, n_averages=1, trigger_channel=ch,
                trigger_mode="AUTO")
    time.sleep(settle)
    _, v = scope.get_waveform(ch, max_points=4000)
    return float(np.ptp(v)), float((v.max() + v.min()) / 2)


def _autorange(scope, ch, divisions=6.0, lo=0.002, hi=1.0):
    """Set a channel's V/div so the signal fills ~`divisions`."""
    ptp, mid = _look(scope, ch)
    if ptp > 0:
        with scope._lock:
            scope._dev.write(
                f"CH{ch}:SCAle {min(max(ptp / divisions, lo), hi):.4f}")
        scope._pre.clear()
        ptp, mid = _look(scope, ch)
    return ptp, mid


def collect(scope, n_shots, f_hz, span_s=0.0):
    """n_shots independent acquisitions; phase and amplitude, both ways.

    span_s spreads the shots evenly over that many seconds of WALL CLOCK
    time, and this is not cosmetic. A carrier that misses the buffer grid by
    df walks 2*pi*df radians per second of REAL time, so the phase error a
    run accumulates is set by how long the run lasts, not by how many shots
    it contains. Taken back to back, 25 shots occupy about 0.2 s, over which
    even a 0.2 Hz frequency error only walks 15 deg -- a coherence of 0.998
    measured that way says nothing about an average that runs for a minute.
    The shot times are returned so every number can be quoted with the span
    it was actually measured over.
    """
    phi_trig, phi_grid, amps, when = [], [], [], []
    f_meas = float("nan")
    # IRREGULAR schedule, fixed seed. Evenly spaced shots are a trap here: a
    # frequency error df makes phase ramp linearly in time, and if the ramp
    # happens to advance a near-whole number of cycles between evenly spaced
    # shots, every shot is sampled at nearly the same phase and a badly
    # off-grid carrier reports a coherence near 1. Random spacing cannot
    # alias. The seed is fixed so the states are compared on one schedule.
    rng = np.random.default_rng(20261020)
    sched = (np.concatenate([[0.0], np.sort(rng.uniform(0, span_s,
                                                        max(n_shots - 1, 0)))])
             if span_s > 0 else np.zeros(n_shots))
    t0 = time.perf_counter()
    for i in range(n_shots):
        due = t0 + sched[i]
        while time.perf_counter() < due:
            time.sleep(min(0.05, due - time.perf_counter()))
        when.append(time.perf_counter() - t0)
        t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS,
                                  fresh=True, absolute_time=True)
        a, p_trig = reflock.fit_tone(t, v, f_hz, t_ref=0.0)
        _, p_grid = reflock.fit_tone(t, v, f_hz, t_ref=None)
        phi_trig.append(p_trig)
        phi_grid.append(p_grid)
        amps.append(a)
        if i == 0:
            f_meas, cycles = reflock.estimate_frequency(t, v)
            print(f"    first shot: {f_meas/1e6:.6f} MHz measured vs "
                  f"{f_hz/1e6:.6f} asked  ({(f_meas/f_hz-1)*1e6:+.3f} ppm, "
                  f"{cycles:.0f} cycles, {np.ptp(v)*1000:.0f} mVpp)")
    return (np.array(phi_trig), np.array(phi_grid), np.array(amps), f_meas,
            np.array(when))


def run_state(awg, scope, label, f_hz, n_shots, trig_level, span_s=0.0):
    print(f"\n  {label}")
    awg.duc_cw_step(f_hz, channel=AWG_SIG_CH)
    awg.duc_cw_step(f_hz, channel=AWG_MARK_CH)
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")

    phi_t, phi_g, amps, f_meas, when = collect(scope, n_shots, f_hz, span_s)
    r_t, sd_t = reflock.coherence(phi_t)
    r_g, sd_g = reflock.coherence(phi_g)
    a_single = float(np.mean(amps))

    # End to end: let the scope do the averaging and see what survives.
    # In AVErage mode a SEQuence run acquires NUMAVg shots and stops, which is
    # exactly the handshake get_waveform(fresh=True) already performs.
    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                              absolute_time=True, acq_timeout=30.0)
    a_avg, _ = reflock.fit_tone(t, v, f_hz, t_ref=0.0)
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")

    floor = 1.0 / np.sqrt(n_shots)
    print(f"    trigger-referenced    coherence {r_t:5.3f}   "
          f"circular sd {sd_t:6.1f} deg")
    print(f"    sample-grid           coherence {r_g:5.3f}   "
          f"circular sd {sd_g:6.1f} deg")
    print(f"    hardware {N_AVG:>3}-average   amplitude ratio "
          f"{a_avg/a_single:5.3f}   ({a_single*2*1000:.1f} mVpp single "
          f"-> {a_avg*2*1000:.1f} mVpp averaged)")
    print(f"    random-phase floor for {n_shots} shots: {floor:.3f}"
          f"   (shots spread over {when[-1]:.1f} s of wall clock)")

    return {"label": label, "f_hz": f_hz, "f_measured_hz": f_meas,
            "n_shots": n_shots,
            "coherence_trigger_ref": r_t, "sd_trigger_ref_deg": sd_t,
            "coherence_sample_grid": r_g, "sd_sample_grid_deg": sd_g,
            "n_average": N_AVG, "amp_single_vpp": a_single * 2,
            "amp_averaged_vpp": a_avg * 2, "amp_ratio": a_avg / a_single,
            "random_floor": float(floor),
            "span_s": float(when[-1]), "shot_times_s": when.tolist(),
            "phases_trigger_ref": phi_t.tolist(),
            "phases_sample_grid": phi_g.tolist(),
            "amps_vpp": (np.array(amps) * 2).tolist()}


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    span_s = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    print(f"buffer {N_TOTAL} complex samples at {RATE_CX/1e6:.1f} MS/s "
          f"= {T_BUF*1e6:.4f} us, grid {GRID_HZ:.6f} Hz")
    print(f"  on  grid  {F_ON/1e6:.6f} MHz = {K_ON} cycles per buffer exactly")
    print(f"  off grid  {F_OFF/1e6:.6f} MHz = {F_OFF*T_BUF:.4f} cycles "
          f"-> {((F_OFF*T_BUF) % 1)*360:.1f} deg of phase walk per buffer")

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)

    # Both channels inside ONE reset, so CH1's segment (which carries the
    # marker) and CH2's segment (which carries the measured signal) start
    # together. A marker is rigid to its OWN channel's segment only.
    awg.duc_cw_setup_dual(F_ON, AMP_VPP, n_total=N_TOTAL)
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * N_TOTAL,
                     marker=MARKER, high_frac=MARKER_FRAC, level_vpp=1.0)
    # Markers play with the channel's RF off (bench-verified 2026-10-19), so
    # CH1 carries the trigger without pushing 808 MHz into whatever it feeds.
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")
    print(f"  AWG CH{AWG_MARK_CH} RF output OFF, M{MARKER} still playing")

    saved = scope.save_state()
    with scope._lock:
        v_was = {ch: scope._dev.query(f"CH{ch}:SCAle?").strip()
                 for ch in (SIG_CH, TRIG_CH)}
    results, status = [], 0
    try:
        rate = scope.set_max_sample_rate(CAPTURE_S)
        print(f"\nscope at {rate/1e9:.2f} GS/s, {CAPTURE_S*1e6:g} us record "
              f"({min(MAX_POINTS, int(rate*CAPTURE_S))} points read back)")

        m_ptp, m_mid = _autorange(scope, TRIG_CH)
        if m_ptp < 0.1:
            print(f"\nonly {m_ptp*1000:.0f} mVpp on CH{TRIG_CH} -- no marker "
                  f"there. Check AWG CH{AWG_MARK_CH} M{MARKER} -> "
                  f"scope CH{TRIG_CH}.")
            return 1
        print(f"marker on CH{TRIG_CH}: {m_ptp*1000:.0f} mVpp, "
              f"trigger at {m_mid:.3f} V (midpoint)")
        s_ptp, _ = _autorange(scope, SIG_CH)
        print(f"signal on CH{SIG_CH}: {s_ptp*1000:.0f} mVpp")

        results.append(run_state(awg, scope,
                                 "ON GRID  (carrier closes the buffer)",
                                 F_ON, n_shots, m_mid, span_s))
        results.append(run_state(awg, scope,
                                 "OFF GRID (control: 100 kHz off the grid)",
                                 F_OFF, n_shots, m_mid, span_s))
    finally:
        scope.restore_state(saved)
        with scope._lock:
            for ch, sc in v_was.items():
                scope._dev.write(f"CH{ch}:SCAle {sc}")
        awg.duc_cw_step(F_ON, channel=AWG_SIG_CH)
        awg.duc_cw_step(F_ON, channel=AWG_MARK_CH)

    print("\n" + "=" * 72)
    on, off = results
    print(f"  {'':<24}{'ON GRID':>11}{'OFF GRID':>11}"
          f"   (floor {on['random_floor']:.3f}, "
          f"{on['span_s']:.0f} s span)")
    for key, name in (("coherence_trigger_ref", "trigger-referenced"),
                      ("coherence_sample_grid", "sample-grid"),
                      ("amp_ratio", f"{N_AVG}-avg amplitude")):
        print(f"  {name:<24}{on[key]:>11.3f}{off[key]:>11.3f}")
    print("=" * 72)

    if off["coherence_trigger_ref"] > 0.5:
        print("\n  The off-grid control did NOT collapse. Either the NCO is "
              "being reset at\n  segment start after all, or the scope is not "
              "really re-arming between\n  shots -- treat the on-grid number "
              "as unproven until that is explained.")
        status = 2
    elif on["coherence_trigger_ref"] > 0.95:
        print(f"\n  Coherent averaging works on the marker trigger: "
              f"{on['coherence_trigger_ref']*100:.1f}% of amplitude\n"
              f"  survives, against {off['coherence_trigger_ref']*100:.1f}% "
              f"with the carrier taken off the buffer grid.\n"
              f"  -> every DUC carrier must sit on the {GRID_HZ:.3f} Hz grid.\n"
              "     For a CHIRP the residual can be shifted into the baseband span;\n"
              "     for CW it cannot, because a baseband tone sits on the same grid --\n"
              "     finer CW spacing needs a longer buffer, nothing else.")
    else:
        print(f"\n  On-grid coherence is only "
              f"{on['coherence_trigger_ref']:.3f}. The control works, so this "
              f"is real\n  and worth chasing before any averaging is trusted.")
        status = 3

    gap = on["coherence_trigger_ref"] - on["coherence_sample_grid"]
    if gap > 0.01:
        print(f"\n  Sub-sample trigger quantisation costs {gap*100:.1f}% of "
              f"amplitude\n  ({on['coherence_sample_grid']:.3f} on the sample "
              f"grid vs {on['coherence_trigger_ref']:.3f} referenced to the "
              f"trigger).\n  Software re-alignment on XZEro recovers it; the "
              f"scope's own averager cannot.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("marker_coherence")
    out.mkdir(exist_ok=True)
    meta = {"timestamp": stamp, "span_requested_s": span_s,
            "f_on_hz": F_ON, "f_off_hz": F_OFF,
            "grid_hz": GRID_HZ, "t_buffer_s": T_BUF, "n_total": N_TOTAL,
            "amp_vpp": AMP_VPP, "capture_s": CAPTURE_S,
            "trigger": f"AWG CH{AWG_MARK_CH} M{MARKER} -> scope CH{TRIG_CH}",
            "signal": f"AWG CH{AWG_SIG_CH} -> scope CH{SIG_CH}",
            "states": results}
    (out / f"marker_coherence_{stamp}.json").write_text(
        json.dumps(meta, indent=2))
    try:
        _plot(out / f"marker_coherence_{stamp}.png", results)
    except Exception as e:
        print(f"  (plot skipped: {e})")
    print(f"\nsaved to {out}/marker_coherence_{stamp}.*")
    print(f"AWG left on EXT 10 MHz, CH{AWG_SIG_CH} playing "
          f"{F_ON/1e6:.4f} MHz, CH{AWG_MARK_CH} RF off with M{MARKER} running.")
    return status


def _plot(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, st in enumerate(results):
        ph = np.array(st["phases_trigger_ref"])
        ax = axes[0][col]
        ax.plot(np.degrees(np.unwrap(ph)), "o-", ms=4, lw=0.8)
        ax.set_title(st["label"], fontsize=10)
        ax.set_xlabel("shot")
        ax.set_ylabel("phase at trigger [deg]")
        ax.grid(alpha=0.3)

        ax = axes[1][col]
        ax.plot(np.cos(ph), np.sin(ph), "o", ms=5, alpha=0.7)
        z = np.exp(1j * ph).mean()
        ax.arrow(0, 0, z.real, z.imag, color="crimson", width=0.015,
                 length_includes_head=True)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7"))
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(f"coherence {st['coherence_trigger_ref']:.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Marker-triggered shot-to-shot phase, "
                 f"{results[0]['f_hz']/1e6:.4f} MHz DUC CW")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
