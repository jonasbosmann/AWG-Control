"""Verify that the AWG is really locked to the external 10 MHz reference.

Why a dedicated script: the Proteus has NO lock-status query. `:ROSC:SOUR?`
reports what was accepted, not whether the PLL acquired -- and fw 1.237.0 is
known to accept commands and silently ignore them. The only honest test is to
compare the AWG's own output against the reference and look for a beat.

Wiring
------
    Wenzel OCXO -> ZX10R-2-183-S+ -> arm 1 -> scope rear Ref In (NO terminator)
                                  -> arm 2 -> 6 dB pad -> AWG REF IN (SMP)

    scope rear AUX OUT -> scope CH3 (50 ohm)
        Utility > I/O > AUX OUT > Reference Clock. With the scope's timebase
        already locked to the OCXO this is a Wenzel-derived 10 MHz, which is
        why no third splitter arm is needed.

    AWG CH1 -> scope CH1 (50 ohm)

    Scope: trigger on CH3, ~20 ns/div, both channels 50 ohm.

What to watch
-------------
CH1 carries a 10 MHz sine from the AWG. Locked, CH1 and CH3 descend from the
same physical oscillator and CH1 stands still. Unlocked, CH1 crawls at
delta * 10 MHz:

    delta = 1 ppm    ->  10 Hz beat  ->  one cycle slip every 0.1 s
    delta = 2.5 ppm  ->  25 Hz beat  ->  one cycle slip every 0.04 s
    locked           ->   0 Hz       ->  stationary

Step 4 drives the AWG between INT and EXT on command so the crawl starts and
stops while you watch. That toggle, not the readback, is the actual proof.

Open questions this also settles
--------------------------------
  * does `*RST` drop the lock? (step 2 -- matters for how the GUI sequences
    a reset against set_reference_external)

`:ROSC:BYP` is already settled and needs no test: it does NOT exist on fw
1.237.0. The query gets no reply, times out, and queues '208, syntax error
in scpi' (bench-confirmed 2026-10-19). The [1]/[0] keys below say so rather
than hanging.

Run:  python check_reference.py
"""
import sys
import time

from awg import AWG
from scope import Scope

# ── configuration ─────────────────────────────────────────────────────────
TEST_FREQ_HZ = 10e6      # match the reference so the beat is a slow crawl
AMP_VPP      = 0.50      # modest; scope CH1 at 50 ohm
CHANNEL      = 1
REF_MHZ      = 10


def show(tag, st):
    print(f"  {tag:<22} SOUR={st['source']:<5} FREQ={st['freq_raw']:<12} "
          f"BYP={st['bypass']}")


def main():
    awg = AWG()

    # ── 1. as found ───────────────────────────────────────────────────
    print("\n1. reference as found")
    show("AWG as found", awg.reference_status())

    # The scope, unlike the Proteus, has a real lock query. Check it first --
    # if the scope is not locked there is nothing to test the AWG against.
    sc, was = None, None
    try:
        sc = Scope()
        st = sc.reference_status()
        print(f"  {'scope':<22} SOUR={st['source']}  STATE={st['state']}")
        if not (st["external"] and st["locked"]):
            print("   -> scope is NOT locked to the external 10 MHz. Fix first:")
            print("      Acquisition badge > Timebase Reference Source >"
                  " External (1 ppm)")
            return 1
        # Put a Wenzel-derived 10 MHz on the rear AUX OUT so it can be fed to
        # a scope channel as the phase reference -- no third splitter arm.
        was = sc.aux_out_source()
        sc.set_aux_out("REFOut")
        print(f"  {'scope AUX OUT':<22} {was} -> REFOUT (10 MHz ref clock)")
        if was != "REFOUT":
            print("                         (restored at the end)")
    except Exception as e:
        print(f"   (scope unreachable: {e}; continuing with the AWG alone)")

    # ── 2. does *RST drop the lock? ───────────────────────────────────
    print("\n2. probing whether *RST clears the reference setting")
    try:
        awg.set_reference_external(REF_MHZ)
    except RuntimeError as e:
        print(f"   could not set EXT before the probe: {e}")
        print("   is the 10 MHz cable actually on AWG REF IN?")
        return 1
    awg._dev.write("*CLS; *RST")
    time.sleep(0.5)
    after = awg.reference_status()
    show("after *RST", after)
    if "EXT" in after["source"]:
        print("   -> *RST PRESERVES the reference.")
    else:
        print("   -> *RST CLEARS the reference back to internal.")
    print("      Either way AWG._setup() now re-applies and re-verifies the")
    print("      reference after every reset=True, so call sites do not have")
    print("      to sequence set_reference_external() against their resets.")

    # ── 3. test tone ──────────────────────────────────────────────────
    print(f"\n3. {TEST_FREQ_HZ/1e6:g} MHz test tone on CH{CHANNEL} "
          f"(resets the instrument, so the reference is set after)")
    actual = awg.send_sine(TEST_FREQ_HZ, AMP_VPP, channel=CHANNEL, exact=True)
    err = actual - TEST_FREQ_HZ
    print(f"   commanded {TEST_FREQ_HZ/1e6:.6f} MHz, synthesised "
          f"{actual/1e6:.6f} MHz (err {err:+.3f} Hz)")
    if abs(err) > 1.0:
        print("   WARNING: the tone is not exactly 10 MHz, so there will be a")
        print("   beat even when locked. Expected crawl from this alone: "
              f"{abs(err):.1f} Hz.")

    print("\n   setting the external reference (FREQ before SOUR)")
    awg.set_reference_external(REF_MHZ)
    show("after set EXT", awg.reference_status())

    # ── 4. the toggle test ────────────────────────────────────────────
    print("\n4. beat test")
    print("   Set the scope up now (AUX OUT is already switched for you):")
    print("     CH3 = rear AUX OUT -> CH3 input, 50 ohm, ~200 mV/div")
    print("     CH1 = AWG CH1, 50 ohm")
    print("     trigger on CH3, ~20 ns/div")
    print("   Then watch CH1 while toggling below.")
    print("     locked   -> CH1 stationary")
    print("     unlocked -> CH1 crawls, ~10 Hz per ppm (one slip per 0.1 s)")

    actions = {
        "e": ("EXTERNAL", lambda: awg.set_reference_external(REF_MHZ)),
        "i": ("INTERNAL", awg.set_reference_internal),
        "1": ("EXTERNAL, BYP ON",
              lambda: awg.set_reference_external(REF_MHZ, bypass=True)),
        "0": ("EXTERNAL, BYP OFF",
              lambda: awg.set_reference_external(REF_MHZ, bypass=False)),
    }
    print("\n   [e] external   [i] internal   [1] ext+BYP ON   [0] ext+BYP OFF"
          "   [q] quit")
    while True:
        try:
            key = input("   > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if key in ("q", "quit", ""):
            break
        if key not in actions:
            print("   unknown key")
            continue
        label, fn = actions[key]
        print(f"   -> {label}")
        try:
            show("now", fn())
        except RuntimeError as e:
            print(f"   FAILED: {e}")

    # ── 5. leave it locked ────────────────────────────────────────────
    print("\n5. leaving the AWG on the external reference")
    try:
        awg.set_reference_external(REF_MHZ)
    except RuntimeError as e:
        print(f"   WARNING: could not restore EXT: {e}")
    show("final", awg.reference_status())
    if sc is not None and was and was != "REFOUT":
        try:
            print(f"   scope AUX OUT restored to {sc.set_aux_out(was)}")
        except Exception as e:
            print(f"   WARNING: could not restore AUX OUT to {was}: {e}")

    print("\nRecord what you saw: whether CH1 was stationary on EXT, whether the")
    print("crawl appeared on INT, and whether BYP ON/OFF changed anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
