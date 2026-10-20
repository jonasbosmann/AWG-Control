"""Is the coherent grid the BUFFER grid, or only every d-th point of it?

Why this test exists
--------------------
run_marker_coherence.py established that a DUC carrier must fit a whole
number of cycles into the buffer:  f = k * rate_cx / n_total.  That is
necessary. This asks whether it is SUFFICIENT.

It is not, because the NCO also quantises its own tuning word to
9 GS/s / 2**32 = 2.095 Hz (measured by run_nco_resolution.py). Writing
n_total = 2**a * d with d odd, a carrier can satisfy both conditions only
when k is a multiple of d, so the usable spacing is

    rate_cx / 2**a        and NOT        rate_cx / n_total

They agree only for a power-of-two buffer. The default CP-FTMW timing gives
n_total = 12512 = 2**5 * 391, where the buffer grid is 89.9 kHz but only
every 391st point of it is representable -- a usable carrier step of
35.16 MHz. A carrier snapped to the buffer grid there sits up to 1.05 Hz from
anything the NCO can produce, and 1 Hz walks a whole cycle of phase per
second.

So the prediction is sharp:

    carrier on the ACHIEVABLE grid   ->  coherent
    carrier on a BUFFER-grid point
      that is not achievable         ->  collapses over a run of seconds

The two carriers are less than 90 kHz apart, so nothing else about the
measurement differs. If the prediction holds, snapping must target
coherent_step_hz() and not buffer_grid_hz() -- which is what awg.py now does.

Wiring as for the other tests:
    AWG CH1 M1 -> scope CH4 (trigger),  AWG CH2 -> scope CH3 (signal)

Run:  python run_grid_achievable.py [n_shots] [span_s]
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
from run_pulse_averaging import (
    RATE_CX, EDGE_FRAC, AMP_VPP, AWG_SIG_CH, AWG_MARK_CH, MARKER,
    SIG_CH, TRIG_CH, CAPTURE_S, MAX_POINTS, MARKER_NS, N_AVG,
    range_on_trigger, find_pulse, measure)

N_TOTAL = 12512            # = 2**5 * 391, the default CP-FTMW timing
T_BUF = N_TOTAL / RATE_CX
PULSE_US = 2.0
F_BB = 1.234567e6
F_TARGET = 808.59375e6


def build_pulse(n_total):
    n_pulse = int(round(PULSE_US * 1e-6 * RATE_CX / 32)) * 32
    t = np.arange(n_pulse) / RATE_CX
    z = AWG._gauss_win(n_pulse, EDGE_FRAC) * np.exp(2j * np.pi * F_BB * t)
    z = z / max(np.abs(z).max(), 1e-12)
    i_arr, q_arr = np.zeros(n_total), np.zeros(n_total)
    i_arr[:n_pulse], q_arr[:n_pulse] = z.real, z.imag
    return i_arr, q_arr, n_pulse


def run_state(awg, scope, label, carrier, n_shots, trig_level, span_s,
              win_fit, win_noise):
    f_abs = carrier + F_BB
    for ch in (AWG_SIG_CH, AWG_MARK_CH):
        awg.duc_cw_step(carrier, channel=ch)
    cycles = carrier * T_BUF
    _, resid = AWG.snap_duc_carrier(carrier, N_TOTAL)
    print(f"\n  {label}")
    print(f"    carrier {carrier:.3f} Hz = {cycles:.6f} cycles per buffer, "
          f"{resid:+.3f} Hz from the nearest achievable value")

    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    rng = np.random.default_rng(20261020)
    sched = np.concatenate([[0.0], np.sort(rng.uniform(0, span_s,
                                                       max(n_shots - 1, 0)))])
    phases, amps = [], []
    t0 = time.perf_counter()
    for i in range(n_shots):
        while time.perf_counter() < t0 + sched[i]:
            time.sleep(min(0.05, t0 + sched[i] - time.perf_counter()))
        t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                                  absolute_time=True, width=2)
        a, p, _ = measure(t, v, f_abs, win_fit, win_noise)
        amps.append(a)
        phases.append(p)
    span = time.perf_counter() - t0
    r, sd = reflock.coherence(np.array(phases))
    print(f"    coherence {r:5.3f}   sd {sd:6.1f} deg   "
          f"({n_shots} shots over {span:.1f} s, floor "
          f"{1/np.sqrt(n_shots):.3f})")
    return {"label": label, "carrier_hz": carrier, "cycles_per_buffer": cycles,
            "residual_from_achievable_hz": resid, "coherence": r,
            "sd_deg": sd, "span_s": span, "n_shots": n_shots,
            "phases": list(map(float, phases))}


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    span_s = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    buf = AWG.buffer_grid_hz(N_TOTAL)
    step = AWG.coherent_step_hz(N_TOTAL)
    f_ach = round(F_TARGET / step) * step
    k = round(f_ach / buf) + 1
    if k % round(step / buf) == 0:
        k += 1
    f_buf = k * buf
    print(f"n_total {N_TOTAL} = 2^5 * {N_TOTAL >> 5},  buffer "
          f"{T_BUF*1e6:.4f} us")
    print(f"  buffer grid      {buf:12.4f} Hz")
    print(f"  achievable step  {step:12.4f} Hz  (every "
          f"{round(step/buf)}th buffer-grid point)")
    print(f"  A achievable     {f_ach:14.4f} Hz  = {f_ach*T_BUF:.4f} cycles")
    print(f"  B buffer grid    {f_buf:14.4f} Hz  = {f_buf*T_BUF:.4f} cycles "
          f"(k={k}, not a multiple of {round(step/buf)})")
    print(f"  they differ by {(f_buf-f_ach)/1e3:.3f} kHz\n")

    i_arr, q_arr, n_pulse = build_pulse(N_TOTAL)
    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)
    awg.duc_iq_setup_dual(i_arr, q_arr, f_ach, AMP_VPP)
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * N_TOTAL,
                     marker=MARKER, high_frac=MARKER_NS * 1e-9 / T_BUF,
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")

    saved = scope.save_state()
    with scope._lock:
        v_was = {ch: scope._dev.query(f"CH{ch}:SCAle?").strip()
                 for ch in (SIG_CH, TRIG_CH)}
        pos_was = scope._dev.query("HORizontal:POSition?").strip()
    results = []
    try:
        scope.set_max_sample_rate(CAPTURE_S)
        scope.set_horizontal_position(10)
        with scope._lock:
            scope._dev.write(f"CH{TRIG_CH}:SCAle 0.25")
        scope._pre.clear()
        m_ptp, m_mid = range_on_trigger(scope, TRIG_CH, 0.0)
        if m_ptp < 0.1:
            print(f"no marker on CH{TRIG_CH} -- abort")
            return 1
        range_on_trigger(scope, SIG_CH, m_mid)

        scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                    trigger_level=m_mid, trigger_mode="NORMal")
        t_r, v_r = scope.get_waveform(SIG_CH, max_points=MAX_POINTS,
                                      fresh=True, absolute_time=True, width=2)
        p0, p1 = find_pulse(t_r, v_r)
        sp = p1 - p0
        win_fit = (p0 + 0.15 * sp, p1 - 0.15 * sp)
        win_noise = (p1 + 0.3e-6, t_r[-1])
        print(f"burst {p0*1e9:+.0f} to {p1*1e9:+.0f} ns, fit window "
              f"{win_fit[0]*1e9:+.0f} to {win_fit[1]*1e9:+.0f} ns")

        results.append(run_state(awg, scope, "A  carrier on the ACHIEVABLE grid",
                                 f_ach, n_shots, m_mid, span_s, win_fit,
                                 win_noise))
        results.append(run_state(awg, scope, "B  carrier on a BUFFER-grid point "
                                 "the NCO cannot hold",
                                 f_buf, n_shots, m_mid, span_s, win_fit,
                                 win_noise))
    finally:
        scope.restore_state(saved)
        with scope._lock:
            for ch, sc in v_was.items():
                scope._dev.write(f"CH{ch}:SCAle {sc}")
            scope._dev.write(f"HORizontal:POSition {pos_was}")
        for ch in (AWG_SIG_CH, AWG_MARK_CH):
            awg.duc_cw_step(f_ach, channel=ch)

    a, b = results
    print("\n" + "=" * 70)
    print(f"  achievable grid      coherence {a['coherence']:.3f}")
    print(f"  buffer grid only     coherence {b['coherence']:.3f}")
    print("=" * 70)
    if a["coherence"] > 0.9 and b["coherence"] < 0.5:
        print("\n  Confirmed: the buffer grid is necessary but NOT sufficient.\n"
              "  Snapping must target coherent_step_hz(), which is what "
              "awg.py now does.")
    elif a["coherence"] > 0.9 and b["coherence"] > 0.9:
        print("\n  BOTH stayed coherent -- the NCO holds more bits than the\n"
              "  2.095 Hz measurement implied, and snapping to the buffer "
              "grid would\n  have been enough. The achievable-grid snap is "
              "then merely conservative,\n  but it also moves carriers by up "
              "to 17 MHz, so it should be revisited.")
    else:
        print("\n  Unexpected: check the setup before reading anything into "
              "this.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("marker_coherence")
    out.mkdir(exist_ok=True)
    (out / f"grid_achievable_{stamp}.json").write_text(json.dumps(
        {"timestamp": stamp, "n_total": N_TOTAL, "t_buffer_s": T_BUF,
         "buffer_grid_hz": buf, "achievable_step_hz": step,
         "f_baseband_hz": F_BB, "states": results}, indent=2))
    print(f"\nsaved to {out}/grid_achievable_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
