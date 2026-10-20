"""Set the bench up to WATCH coherent averaging on a scope math channel.

Unlike the measurement scripts, this one leaves the instruments running and
does NOT restore the scope afterwards -- the whole point is that the display
keeps updating after it exits.

    AWG CH1 M1  ->  scope CH4   trigger   (CH1 RF output OFF)
    AWG CH2     ->  scope CH3   signal
    MATH1 = AVG(CH3)            the averaged trace to watch

What to look for
----------------
CH3 shows one shot and never changes much. MATH1 is the average. Switch the
carrier on and off the coherent grid and MATH1 is where the difference shows:

    carrier ON grid   MATH1 looks like CH3 at full amplitude  (0.99 of it)
    carrier OFF grid  MATH1 collapses towards zero while CH3 stays full size

That collapse is the whole point: averaging the wrong signal does not just
fail to help, it destroys what you had. Measured 2026-10-20, a 64-average of
an off-grid carrier gives an SNR gain of 0.6-0.9x, i.e. worse than one shot.

Usage
-----
    python demo_averaging.py [pulse|cw] [on|off|flip]

    pulse   2 us gated burst, then 12.6 us of nothing   (default)
    cw      continuous carrier
    on      carrier on the coherent grid                (default)
    off     carrier 25 kHz off the grid
    flip    alternate on/off every 10 s until Ctrl-C, so the effect can just
            be watched rather than re-run

Switching only the carrier (on/off) is a single NCO write and takes a moment;
switching the waveform re-uploads and takes a few seconds.
"""
import sys
import time

import numpy as np

from awg import AWG
from scope import Scope
import reflock
from run_pulse_averaging import (
    RATE_CX, N_TOTAL, T_BUF, GRID_HZ, F_NCO_ON, F_NCO_OFF, F_BB, AMP_VPP,
    AWG_SIG_CH, AWG_MARK_CH, MARKER, SIG_CH, TRIG_CH, MARKER_NS, build_pulse)

WINDOW_S = 8e-6            # shows the pulse and a long quiet stretch
PRETRIG_PCT = 10
MATH_AVGS = 64
READ_POINTS = 40000
FLIP_PERIOD_S = 10.0


def load_waveform(awg, kind, carrier):
    """Upload pulse or CW to CH1+CH2, then put the marker back on CH1."""
    if kind == "pulse":
        i_arr, q_arr, n_pulse = build_pulse()
        awg.duc_iq_setup_dual(i_arr, q_arr, carrier, AMP_VPP)
        what = (f"pulse {n_pulse/RATE_CX*1e6:.2f} us in a "
                f"{T_BUF*1e6:.2f} us buffer, line at "
                f"{(carrier+F_BB)/1e6:.4f} MHz")
    else:
        awg.duc_cw_setup_dual(carrier, AMP_VPP, n_total=N_TOTAL)
        what = f"CW at {carrier/1e6:.4f} MHz"
    # The marker lives in the segment, so it has to be re-sent after any
    # upload. Its channel's RF stays off -- markers play regardless.
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * N_TOTAL,
                     marker=MARKER, high_frac=MARKER_NS * 1e-9 / T_BUF,
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")
    return what


def set_carrier(awg, carrier):
    for ch in (AWG_SIG_CH, AWG_MARK_CH):
        awg.duc_cw_step(carrier, channel=ch)


def setup_math(scope, source_ch, n_avg):
    """Define MATH1 as a running average of a channel. Returns a status line.

    The user's MATH1 was present but with an EMPTY definition, so it was not
    computing anything. Averaging on this scope is an ADVanced-type math
    expression, not a checkbox.
    """
    with scope._lock:
        scope._dev.write("*CLS")
        if "MATH1" not in scope._dev.query("MATH:LIST?").upper():
            scope._dev.write("MATH:ADDNew 'MATH1'")
            scope._dev.query("*OPC?")
        scope._dev.write("MATH:MATH1:TYPe ADVanced")
        scope._dev.write(f"MATH:MATH1:DEFine \"AVG(CH{source_ch})\"")
        scope._dev.query("*OPC?")
        # AVG:WEIGHT, not AVG:WEIGHTing -- the latter is a command error
        # (ESR 32) and leaves the depth at whatever it was.
        scope._dev.write(f"MATH:MATH1:AVG:WEIGHT {n_avg}")
        scope._dev.write("MATH:MATH1:AVG:MODE 1")
        scope._dev.write("DISplay:GLObal:MATH1:STATE ON")
        defn = scope._dev.query("MATH:MATH1:DEFine?").strip()
        try:
            w = scope._dev.query("MATH:MATH1:AVG:WEIGHT?").strip()
        except Exception:
            w = "?"
        esr = scope._dev.query("*ESR?").strip()
    return defn, w, esr


def report(scope, carrier, label):
    """Measure CH3 and MATH1 and say what the averaging is doing."""
    f_abs = carrier + F_BB
    out = {}
    for src in (f"CH{SIG_CH}", "MATH1"):
        try:
            t, v = scope.get_waveform(src, max_points=READ_POINTS,
                                      fresh=False, width=2,
                                      absolute_time=True)
            env = np.convolve(np.abs(v - np.median(v)), np.ones(201) / 201,
                              mode="same")
            m = env > 0.5 * env.max()
            if m.sum() > 500:
                a, _ = reflock.fit_tone(t[m], v[m], f_abs, t_ref=0.0)
            else:
                a = 0.0
            out[src] = (float(np.ptp(v)), float(a) * 2)
        except Exception as e:
            out[src] = (float("nan"), float("nan"))
            print(f"    could not read {src}: {e}")
    c_ptp, c_amp = out[f"CH{SIG_CH}"]
    m_ptp, m_amp = out["MATH1"]
    ratio = m_amp / c_amp if c_amp > 0 else float("nan")
    print(f"    {label:<9} CH{SIG_CH} {c_amp*1e3:6.1f} mVpp    "
          f"MATH1 {m_amp*1e3:6.1f} mVpp    MATH1/CH{SIG_CH} = {ratio:5.3f}")
    return ratio


def main():
    kind = sys.argv[1].lower() if len(sys.argv) > 1 else "pulse"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "on"
    if kind not in ("pulse", "cw") or mode not in ("on", "off", "flip"):
        print(__doc__)
        return 2
    carrier = F_NCO_ON if mode != "off" else F_NCO_OFF

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)

    what = load_waveform(awg, kind, carrier)
    print(f"\nAWG: {what}")
    print(f"     carrier {carrier/1e6:.6f} MHz = {carrier/GRID_HZ:.4f} grid "
          f"steps ({'ON' if abs(carrier-F_NCO_ON) < 1 else 'OFF'} grid, "
          f"grid {GRID_HZ/1e3:.3f} kHz)")

    # Scope: trigger on the MARKER. This is what restore_state() undid after
    # the last measurement run -- the source had gone back to CH1, so the
    # scope was not triggering on anything and the math average was averaging
    # whatever happened to be in the buffer.
    rate = scope.set_max_sample_rate(WINDOW_S)
    scope.set_horizontal_position(PRETRIG_PCT)
    with scope._lock:
        scope._dev.write(f"CH{TRIG_CH}:SCAle 0.25")
        scope._dev.write(f"CH{TRIG_CH}:TERmination 50")
        scope._dev.write(f"CH{SIG_CH}:TERmination 50")
        scope._dev.write(f"SELect:CH{SIG_CH} ON")
        scope._dev.write(f"SELect:CH{TRIG_CH} ON")
    scope._pre.clear()
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=0.0, trigger_mode="NORMal")

    t, v = scope.get_waveform(TRIG_CH, max_points=READ_POINTS, fresh=True,
                              width=2, acq_timeout=10.0)
    m_ptp = float(np.ptp(v))
    m_mid = float((v.max() + v.min()) / 2)
    if m_ptp < 0.1:
        print(f"\nno marker on CH{TRIG_CH} ({m_ptp*1e3:.0f} mVpp) -- check "
              f"AWG CH{AWG_MARK_CH} M{MARKER} -> scope CH{TRIG_CH}")
        return 1
    with scope._lock:
        scope._dev.write(f"TRIGger:A:LEVel:CH{TRIG_CH} {m_mid:.3f}")
    t, v = scope.get_waveform(SIG_CH, max_points=READ_POINTS, fresh=True,
                              width=2)
    with scope._lock:
        scope._dev.write(f"CH{SIG_CH}:SCAle "
                         f"{min(max(np.ptp(v)/6.0, 0.002), 1.0):.4f}")
    scope._pre.clear()
    print(f"Scope: {rate/1e9:.2f} GS/s, {WINDOW_S*1e6:g} us window, "
          f"{PRETRIG_PCT}% pretrigger, SAMple mode,")
    print(f"       trigger CH{TRIG_CH} NORMal @ {m_mid:.3f} V "
          f"(marker {m_ptp*1e3:.0f} mVpp)")

    defn, w, esr = setup_math(scope, SIG_CH, MATH_AVGS)
    print(f"MATH1: {defn}   averages={w}   (ESR {esr})")
    if "AVG" not in defn.upper():
        print("       !! MATH1 did not take the AVG definition. Set it by "
              "hand:\n"
              "          Math > Add Math > Advanced, expression AVG(CH3),\n"
              "          and make sure Acquisition mode is Sample, not "
              "Average.")

    # free-running so the math average keeps accumulating after this exits
    with scope._lock:
        scope._dev.write("ACQuire:MODe SAMple")
        scope._dev.write("ACQuire:STOPAfter RUNSTop")
        scope._dev.write("ACQuire:STATE RUN")

    print()
    if mode == "flip":
        print(f"Flipping the carrier on/off the grid every "
              f"{FLIP_PERIOD_S:g} s. Watch MATH1. Ctrl-C to stop.\n")
        try:
            while True:
                for lbl, f in (("ON grid", F_NCO_ON), ("OFF grid", F_NCO_OFF)):
                    set_carrier(awg, f)
                    # A 64-deep running average needs ~64 acquisitions to
                    # flush the previous state. Measured: 3 s after a switch
                    # MATH1 still read 0.80 of CH3 on its way back up to
                    # 0.99, so a short settle reports a half-converged number.
                    time.sleep(6.0)
                    report(scope, f, lbl)
                    time.sleep(max(FLIP_PERIOD_S - 6.0, 0))
        except KeyboardInterrupt:
            set_carrier(awg, F_NCO_ON)
            print("\nstopped; carrier left ON grid")
    else:
        time.sleep(6.0)       # see the note in flip mode: the average settles
        report(scope, carrier, "ON grid" if mode == "on" else "OFF grid")
        print(f"\nLeft running. Re-run with a different carrier to compare:")
        print(f"    python demo_averaging.py {kind} "
              f"{'off' if mode == 'on' else 'on'}")
        print(f"    python demo_averaging.py {kind} flip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
