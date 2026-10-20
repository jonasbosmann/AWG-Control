"""How many averaged shots per second can this scope actually deliver?

The AWG repeats every 11.12 us, so 90,000 shots per second are available.
Normal triggered acquisition uses about 32 of them. This measures where the
rest go and whether segmented acquisition (FastFrame) recovers them.

  NORMAL      one trigger, one record, re-arm. Measured cost per shot is
              ~21 ms of fixed overhead plus ~0.084 us per point, so the 10 us
              of signal is under 0.05% of the time spent.

  FASTFRAME   N frames captured back to back into deep memory, then one
              readout. If the inter-frame dead time is small the capture
              approaches the trigger rate. This scope's firmware REJECTS
              HORizontal:FASTframe:SUMFrame (ESR 32), so there is no
              hardware-averaged summary frame -- the frames have to come out
              and be averaged on the PC, which is why the transfer cost is
              measured here too.

Run:  python run_acq_throughput.py [window_us] [rate_GSs]
"""
import sys
import time

import numpy as np

from scope import Scope

SIG_CH, TRIG_CH = 3, 4


def timed_sequence(s, timeout=120.0):
    """Arm one SEQuence and time it to completion."""
    with s._lock:
        s._dev.write("ACQuire:STATE STOP")
        s._dev.write("ACQuire:STOPAfter SEQuence")
        t0 = time.perf_counter()
        s._dev.write("ACQuire:STATE RUN")
        while time.perf_counter() - t0 < timeout:
            if s._dev.query("ACQuire:STATE?").strip() in ("0", "0.0"):
                return time.perf_counter() - t0
            time.sleep(0.002)
    return float("nan")


def main():
    window = (float(sys.argv[1]) if len(sys.argv) > 1 else 10.0) * 1e-6
    rate = (float(sys.argv[2]) if len(sys.argv) > 2 else 12.5) * 1e9
    npts = int(round(window * rate))
    s = Scope()

    with s._lock:
        was = {k: s._dev.query(q).strip() for k, q in (
            ("mode", "HORizontal:MODE?"), ("sr", "HORizontal:MODE:SAMPLERate?"),
            ("rl", "HORizontal:MODE:RECOrdlength?"), ("acq", "ACQuire:MODe?"),
            ("navg", "ACQuire:NUMAVg?"), ("stop", "ACQuire:STOPAfter?"),
            ("tmode", "TRIGger:A:MODe?"), ("tsrc", "TRIGger:A:EDGE:SOUrce?"),
            ("ff", "HORizontal:FASTframe:STATE?"),
            ("ffc", "HORizontal:FASTframe:COUNt?"))}
    print(f"window {window*1e6:g} us at {rate/1e9:g} GS/s = {npts:,} points\n")

    try:
        # trigger on the marker, as the real measurement would
        with s._lock:
            s._dev.write("HORizontal:FASTframe:STATE OFF")
            s._dev.write("HORizontal:MODE MANual")
            s._dev.write(f"HORizontal:MODE:SAMPLERate {rate:.0f}")
            s._dev.write(f"HORizontal:MODE:RECOrdlength {npts}")
            s._dev.write(f"CH{TRIG_CH}:TERmination 50")
            s._dev.write("TRIGger:A:TYPe EDGE")
            s._dev.write(f"TRIGger:A:EDGE:SOUrce CH{TRIG_CH}")
            s._dev.write(f"TRIGger:A:LEVel:CH{TRIG_CH} 0")
            s._dev.write("TRIGger:A:MODe NORMal")
            s._dev.write("ACQuire:MODe SAMple")
            s._dev.query("*OPC?")
        s._pre.clear()
        el = timed_sequence(s, 15)
        if not np.isfinite(el):
            print("no trigger -- is the marker on AWG CH1 M1 reaching "
                  f"scope CH{TRIG_CH}?")
            return 1
        print(f"trigger OK (single acquisition in {el*1e3:.0f} ms)\n")

        print("=" * 66)
        print("NORMAL triggered acquisition, hardware average")
        print("=" * 66)
        print(f"  {'shots':>7}{'time':>9}{'per shot':>11}{'shots/s':>10}")
        for n in (16, 64, 128):
            with s._lock:
                s._dev.write("ACQuire:MODe AVErage")
                s._dev.write(f"ACQuire:NUMAVg {n}")
                s._dev.query("*OPC?")
            el = timed_sequence(s)
            print(f"  {n:>7}{el:>8.2f}s{el/n*1e3:>10.1f}ms{n/el:>10.1f}")
        normal_rate = n / el

        print("\n" + "=" * 66)
        print("FASTFRAME segmented acquisition")
        print("=" * 66)
        with s._lock:
            s._dev.write("ACQuire:MODe SAMple")
            s._dev.write("HORizontal:FASTframe:STATE ON")
            s._dev.query("*OPC?")
            s._dev.write("HORizontal:FASTframe:COUNt 1000000")
            s._dev.query("*OPC?")
            fmax = int(float(s._dev.query("HORizontal:FASTframe:COUNt?")))
        print(f"  max {fmax} frames of {npts:,} points "
              f"= {fmax*npts/1e6:.1f} Mpts\n")
        print(f"  {'frames':>7}{'time':>9}{'per frame':>12}{'frames/s':>11}")
        ff_rate = float("nan")
        for n in (10, 100, min(fmax, 500)):
            with s._lock:
                s._dev.write(f"HORizontal:FASTframe:COUNt {n}")
                s._dev.query("*OPC?")
            el = timed_sequence(s)
            if np.isfinite(el):
                ff_rate = n / el
                print(f"  {n:>7}{el:>8.3f}s{el/n*1e6:>10.0f}us{n/el:>11.0f}")
            else:
                print(f"  {n:>7}   timed out")

        # what it costs to get the frames out, since there is no summary frame
        print("\n  readout (ASCII, what scope.py does today):")
        t0 = time.perf_counter()
        t, v = s.get_waveform(SIG_CH, max_points=npts, fresh=False, width=2)
        dt_one = time.perf_counter() - t0
        print(f"    one {npts:,}-point frame: {dt_one:.2f} s "
              f"-> {npts/dt_one/1e6:.2f} Mpts/s")
        print(f"    all {min(fmax,500)} frames would be "
              f"{dt_one*min(fmax,500)/60:.1f} min at that rate")

        print("\n" + "=" * 66)
        if np.isfinite(ff_rate):
            print(f"  acquisition:  normal {normal_rate:,.0f} shots/s   "
                  f"FastFrame {ff_rate:,.0f} shots/s   "
                  f"({ff_rate/normal_rate:.0f}x)")
            print(f"  AWG offers 89,900 shots/s; FastFrame uses "
                  f"{ff_rate/89900*100:.1f}% of them, normal "
                  f"{normal_rate/89900*100:.3f}%")
            for target in (1e4, 1e6):
                print(f"  {target:.0e} shots: normal "
                      f"{target/normal_rate/60:.1f} min, FastFrame capture "
                      f"{target/ff_rate:.1f} s (+ readout)")
        print("=" * 66)
    finally:
        with s._lock:
            s._dev.write(f"HORizontal:FASTframe:COUNt {was['ffc']}")
            s._dev.write(f"HORizontal:FASTframe:STATE {was['ff']}")
            s._dev.write("HORizontal:MODE MANual")
            s._dev.write(f"HORizontal:MODE:SAMPLERate {was['sr']}")
            s._dev.write(f"HORizontal:MODE:RECOrdlength {was['rl']}")
            s._dev.write(f"HORizontal:MODE {was['mode']}")
            s._dev.write(f"ACQuire:MODe {was['acq']}")
            s._dev.write(f"ACQuire:NUMAVg {was['navg']}")
            s._dev.write(f"ACQuire:STOPAfter {was['stop']}")
            s._dev.write(f"TRIGger:A:MODe {was['tmode']}")
            s._dev.write(f"TRIGger:A:EDGE:SOUrce {was['tsrc']}")
            s._dev.write("ACQuire:STATE RUN")
        s._pre.clear()
        print("\nrestored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
