"""SUPERSEDED 2026-10-20 by run_marker_coherence.py -- do not use.

Two reasons. It triggers on the scope's AUX OUT 10 MHz, which is independent
of the AWG and so cannot resolve the sub-10-ns alignment the marker gives;
and it calls get_waveform() without fresh=True, which is what produced the
invalid "coherence 1.000 for both states" on 2026-10-19 (the scope handed
back the same stale frame 25 times). Kept only for the record.

Does coherent averaging actually work? Measure shot-to-shot phase.

Every phase check so far has been self-triggered on the AWG's own output,
which makes phase stability true by construction and proves nothing. This
uses an INDEPENDENT trigger that already exists on the bench: the scope's
rear AUX OUT, set to the 10 MHz reference clock, wired to CH2.

    AUX OUT (10 MHz, Wenzel-derived)  ->  scope CH2   <- trigger
    AWG CH2 (DUC CW)                  ->  scope CH3   <- measured

Why 500 MHz exactly
-------------------
The trigger fires on a 10 MHz edge, and which edge it catches is arbitrary.
The tone's phase is only repeatable across triggers if the tone is an exact
integer multiple of 10 MHz -- 500 MHz = 50 x 10 MHz. At 2.5 GS/s the Direct
mode grid (1.2207 MHz) has no usable multiple below Nyquist, so this runs in
DUC mode where :NCO:CFR1 can be set exactly. DUC at 9 GS/s also preserves the
external reference, unlike 9 GS/s Direct.

What it reports
---------------
coherence = |<exp(i*phi)>| over N acquisitions. That is literally the
fraction of amplitude surviving an N-shot coherent average: 1.0 means
averaging is lossless and buys the full sqrt(N) in SNR, 0 means the shots
cancel. For randomly distributed phase it floors at about 1/sqrt(N).

Prediction
----------
    EXTERNAL  AWG locked to the same 10 MHz as the trigger -> coherence ~1
    INTERNAL  ~40 ppb offset = 20 Hz beat at 500 MHz; acquisitions are
              ~0.5 s apart, so the phase walks many whole cycles between
              shots -> coherence at the 1/sqrt(N) floor

Unlike the frequency test, this discriminates easily, because phase
integrates the frequency error over the time between shots.

Run:  python run_phase_coherence.py [n_shots]
"""
import sys

import numpy as np

from awg import AWG
from scope import Scope
import reflock

TONE_HZ = 500e6          # exactly 50 x 10 MHz
AMP_VPP = 0.4
AWG_CH, SIG_CH, TRIG_CH = 2, 3, 2
TRIG_LEVEL_V = 0.6       # AUX OUT swings ~0.25 to 1.0 V into 50 ohm
CAPTURE_S = 20e-6
MAX_POINTS = 200000


def run_state(awg, scope, label, n_shots):
    awg.duc_cw_setup(TONE_HZ, AMP_VPP, channel=AWG_CH)
    scope.set_max_sample_rate(CAPTURE_S)
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=TRIG_LEVEL_V, trigger_mode="NORMal")

    phases, amps = [], []
    for _ in range(n_shots):
        t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS)
        phases.append(reflock.estimate_phase(t, v, TONE_HZ))
        amps.append(np.ptp(v))
    r, sd = reflock.coherence(phases)
    floor = 1.0 / np.sqrt(len(phases))
    print(f"  {label:<22} coherence {r:5.3f}   circular sd {sd:6.1f} deg"
          f"   (random-phase floor {floor:.3f})")
    print(f"  {'':<22} single-shot Vpp {np.mean(amps):.4f} "
          f"-> averaged would be {np.mean(amps) * r:.4f}")
    return {"label": label, "coherence": r, "sd_deg": sd,
            "floor": floor, "vpp": float(np.mean(amps))}


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    awg, scope = AWG(), Scope()

    ref = scope.reference_status()
    if not (ref["external"] and ref["locked"]):
        print(f"scope not locked to the external reference ({ref}) -- abort")
        return 1
    was_aux = scope.aux_out_source()
    scope.set_aux_out("REFOut")
    saved = scope.save_state()
    # AUX OUT swings 0.25-1.0 V into 50 ohm; at a tight V/div it clips and the
    # trigger gets unreliable. Give it room, and put it back afterwards.
    was_vert = scope._dev.query(f"CH{TRIG_CH}:SCAle?").strip()
    scope.set_vertical(TRIG_CH, 0.25)
    print(f"trigger: AUX OUT 10 MHz on CH{TRIG_CH} @ {TRIG_LEVEL_V} V\n"
          f"signal : AWG CH{AWG_CH} DUC {TONE_HZ/1e6:g} MHz on CH{SIG_CH}\n"
          f"{n_shots} shots per state\n")

    out = []
    try:
        awg.set_reference_external(10)
        out.append(run_state(awg, scope, "EXTERNAL (Wenzel)", n_shots))
        print()
        awg.set_reference_internal()
        out.append(run_state(awg, scope, "INTERNAL TCXO", n_shots))
    finally:
        scope.restore_state(saved)
        scope._dev.write(f"CH{TRIG_CH}:SCAle {was_vert}")
        if was_aux != "REFOUT":
            scope.set_aux_out(was_aux)
        awg.set_reference_internal()

    print()
    if len(out) == 2 and out[1]["coherence"] > 0:
        print(f"  EXTERNAL keeps {out[0]['coherence']/out[1]['coherence']:.1f}x "
              f"more amplitude through averaging than INTERNAL")
    print("\nAWG left on INT; re-tick Ext 10 MHz in the GUI when you want it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
