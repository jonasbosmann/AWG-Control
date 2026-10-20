"""How fine is the Proteus DUC NCO's frequency quantum, really?

Why this is worth measuring
---------------------------
run_marker_coherence.py showed that a DUC carrier keeps 99.8% of its
amplitude through a 60 s average when it closes the buffer exactly

    f * T_buffer = integer,     T_buffer = n_total / (9 GS/s / 8) = 3.6409 us
    grid = 274658.203125 Hz

and 21% (the random-phase floor) when it misses by 100 kHz. So the carrier
has to be put ON that grid. The question left over is how precisely the
instrument has to be TOLD, because awg.py writes :NCO:CFR1 with '%.0f' --
whole Hz -- and the grid points are not whole Hz for most k.

A first attempt treated that rounding as the thing to test and produced a
negative control that refused to fail: k = 2945 written as '808868408'
instead of '808868408.203125', a 0.203 Hz error that should walk 12 cycles
of phase in 60 s, stayed at coherence 0.998. The only way that happens is if
the instrument never produced the 0.203 Hz error in the first place -- i.e.
the NCO quantises its own tuning word coarsely enough to swallow it. A
32-bit word at 9 GS/s would give 9e9 / 2^32 = 2.095 Hz, which fits.

This script measures that quantum instead of assuming it.

Method
------
Phase walk is a far better frequency discriminator than any spectrum: an
error df shifts the phase at the marker by 2*pi*df per second of wall clock,
so over a 20 s run even 0.05 Hz is a visible whole cycle. So: sit on an exact
grid point, ask for grid + delta for a series of small deltas, and watch
where coherence falls over.

    delta rounds to zero         -> the NCO lands back on the grid -> coherent
    delta rounds to one quantum  -> real error of ~q - delta       -> collapses

The changeover happens at delta = q/2, which brackets the quantum without
needing to resolve it in frequency at all.

Note that every grid point is exactly representable whatever the word length:
grid k is k * 9e9 / 2^15, so in an N-bit word its tuning value is k * 2^(N-15),
an integer for any N >= 15. Only the OFFSETS get quantised.

Wiring, as for run_marker_coherence.py:
    AWG CH1 M1 -> scope CH4 (trigger),  AWG CH2 -> scope CH3 (signal)

Run:  python run_nco_resolution.py [n_shots] [span_s]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from awg import AWG
from scope import Scope
import reflock
from run_marker_coherence import (
    GRID_HZ, T_BUF, N_TOTAL, F_ON, AMP_VPP, AWG_SIG_CH, AWG_MARK_CH, MARKER,
    SIG_CH, TRIG_CH, CAPTURE_S, MARKER_FRAC, _autorange, collect)

# Offsets from the exact grid point, in Hz. 1000 Hz leads as a gross control:
# it must collapse, or the run proves nothing about the small ones.
DELTAS_HZ = [1000.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
COHERENT = 0.80          # above this, call it coherent
COLLAPSED = 0.45         # below this, call it collapsed


def nco_write(awg, channel, f_hz):
    """Set the NCO with full precision and report what the AWG echoes back."""
    with awg._lock:
        awg._dev.write(f":INST:CHAN {channel}")
        awg._dev.write(f":NCO:CFR1 {f_hz:.6f}")
        awg._dev.query("*OPC?")
        back = awg._dev.query(":NCO:CFR1?").strip()
        awg._dev.query(":SYST:ERR?")
    return back


def run_delta(awg, scope, delta, n_shots, trig_level, span_s):
    f_hz = F_ON + delta
    back = nco_write(awg, AWG_SIG_CH, f_hz)
    nco_write(awg, AWG_MARK_CH, f_hz)
    print(f"\n  grid {delta:+9.3f} Hz   (asked {f_hz:.6f}, echo {back})")

    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    phi_t, _, _, _, when = collect(scope, n_shots, f_hz, span_s)
    r, sd = reflock.coherence(phi_t)

    # If the AWG really produced the offset, the phase ramps; fit the ramp to
    # recover the frequency error it actually applied. Only meaningful while
    # the walk stays under half a cycle between the closest pair of shots.
    slope = np.polyfit(when, np.unwrap(phi_t), 1)[0] / (2 * np.pi)
    verdict = ("coherent " if r > COHERENT else
               "COLLAPSED" if r < COLLAPSED else "marginal ")
    print(f"    coherence {r:5.3f}  sd {sd:6.1f} deg  {verdict}"
          f"   over {when[-1]:.1f} s   (apparent drift {slope*1e3:+.1f} mHz)")
    return {"delta_hz": delta, "f_hz": f_hz, "echo": back,
            "coherence": r, "sd_deg": sd, "span_s": float(when[-1]),
            "apparent_drift_hz": float(slope), "n_shots": n_shots,
            "phases": phi_t.tolist(), "shot_times_s": when.tolist()}


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    span_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    print(f"grid {GRID_HZ:.6f} Hz, buffer {T_BUF*1e6:.4f} us")
    print(f"base {F_ON:.6f} Hz = {F_ON/GRID_HZ:.0f} grid steps, exactly on grid")
    print(f"{n_shots} shots over {span_s:g} s per point; "
          f"{1/span_s*1e3:.0f} mHz of error is one cycle of walk")

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)
    awg.duc_cw_setup_dual(F_ON, AMP_VPP, n_total=N_TOTAL)
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * N_TOTAL,
                     marker=MARKER, high_frac=MARKER_FRAC, level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")

    saved = scope.save_state()
    with scope._lock:
        v_was = {ch: scope._dev.query(f"CH{ch}:SCAle?").strip()
                 for ch in (SIG_CH, TRIG_CH)}
    results = []
    try:
        rate = scope.set_max_sample_rate(CAPTURE_S)
        m_ptp, m_mid = _autorange(scope, TRIG_CH)
        if m_ptp < 0.1:
            print(f"no marker on CH{TRIG_CH} ({m_ptp*1000:.0f} mVpp) -- abort")
            return 1
        _autorange(scope, SIG_CH)
        print(f"scope {rate/1e9:.2f} GS/s, trigger CH{TRIG_CH} @ {m_mid:.3f} V")

        for i, d in enumerate(DELTAS_HZ):
            results.append(run_delta(awg, scope, d, n_shots, m_mid, span_s))
            if i == 0 and results[0]["coherence"] > COLLAPSED:
                print(f"\n  The {DELTAS_HZ[0]:g} Hz gross control did not "
                      f"collapse (coherence "
                      f"{results[0]['coherence']:.3f}).\n  Something is wrong "
                      f"with the trigger or the NCO write -- stopping, "
                      f"nothing here would mean anything.")
                return 2
    finally:
        scope.restore_state(saved)
        with scope._lock:
            for ch, sc in v_was.items():
                scope._dev.write(f"CH{ch}:SCAle {sc}")
        nco_write(awg, AWG_SIG_CH, F_ON)
        nco_write(awg, AWG_MARK_CH, F_ON)

    fine = [r for r in results if r["delta_hz"] < 100]
    print("\n" + "=" * 62)
    print(f"  {'offset from grid':<22}{'coherence':>11}")
    for r in results:
        tag = "  <- gross control" if r["delta_hz"] >= 100 else ""
        print(f"  {r['delta_hz']:+9.3f} Hz          {r['coherence']:>8.3f}{tag}")
    print("=" * 62)

    ok = [r["delta_hz"] for r in fine if r["coherence"] > COHERENT]
    bad = [r["delta_hz"] for r in fine if r["coherence"] < COLLAPSED]
    if ok and bad:
        lo, hi = max(ok), min(bad)
        print(f"\n  Coherence survives an asked-for offset of {lo:g} Hz and "
              f"dies by {hi:g} Hz.\n  The NCO therefore quantises its tuning "
              f"to q with {2*lo:g} Hz < q < {2*hi:g} Hz\n  (the changeover is "
              f"at q/2, where the request starts rounding to the next step).")
        for bits in range(28, 41):
            q = 9e9 / 2 ** bits
            if 2 * lo < q < 2 * hi:
                print(f"  -> consistent with a {bits}-bit tuning word: "
                      f"9 GS/s / 2^{bits} = {q:.3f} Hz")
        print(f"\n  Consequence for awg.py: every grid point k*{GRID_HZ:.6f} Hz "
              f"is exactly\n  representable, and the '%.0f' integer-Hz write "
              f"rounds by at most 0.5 Hz --\n  less than q/2, so it lands back "
              f"on the grid point. Writing whole Hz is SAFE,\n  provided the "
              f"frequency asked for is a grid point in the first place.")
    elif ok and not bad:
        print(f"\n  Nothing up to {max(ok):g} Hz off the grid disturbed the "
              f"phase, while the\n  {DELTAS_HZ[0]:g} Hz control did. The "
              f"quantum is coarser than {2*max(ok):g} Hz; widen DELTAS_HZ.")
    else:
        print("\n  No offset stayed coherent, including zero. Re-check the "
              "setup before\n  reading anything into the rest.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("marker_coherence")
    out.mkdir(exist_ok=True)
    (out / f"nco_resolution_{stamp}.json").write_text(json.dumps(
        {"timestamp": stamp, "grid_hz": GRID_HZ, "t_buffer_s": T_BUF,
         "base_hz": F_ON, "span_requested_s": span_s, "n_shots": n_shots,
         "states": results}, indent=2))
    print(f"\nsaved to {out}/nco_resolution_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
