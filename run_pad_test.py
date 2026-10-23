"""Is the chirp's envelope ripple a reflection, or is it internal?

LABBOOK 18-19: the envelope of a 2-2.5 GHz chirp carries a ~135 MHz ripple,
which reads as a 7.4 ns round trip. That is about 2.6 ns longer than a 50 cm
cable can explain, and the delay on its own cannot say whether the extra
length is inside the AWG output stage, inside the scope front end, or
whether the ripple is a reflection at all.

A pad settles it without needing any extra bandwidth. The direct wave crosses
the attenuator once; the reflected wave crosses it three times (out, back,
out). So the standing-wave ratio falls by TWICE the pad value while the
direct signal falls by one:

    pad      round-trip loss     ripple from 0.85 dB pp      |G1.G2|
    none            -                   0.85 dB              0.049
    7 dB         -14 dB                 0.17 dB              0.0097
    10 dB        -20 dB                 0.09 dB              0.0049

If the ripple drops by about that much, it is a reflection in the cable path
and the remaining question is only where. If it does NOT drop, it is behind
the pad -- generated inside one instrument -- and that is the more
interesting answer.

What this script controls, so the two runs are comparable:

  - the AWG amplitude is FIXED across runs. Changing it would change the
    DAC's own compression and output match.
  - the scope's vertical scale is FIXED across runs (--scale). The front-end
    attenuator setting changes the scope's input match, which is one of the
    two reflection coefficients being measured -- so it must not move between
    pad-in and pad-out. Pass the same --scale to both runs; it has to be set
    for the LARGER (no-pad) signal, so the padded run simply uses less of the
    ADC range.
  - REPEATS. One capture gives a delay with no error bar, which is how the
    "+-2.5 ns" guess got into the lab book. Each run takes several
    independent acquisitions and reports the scatter.

Run:  python run_pad_test.py <label> [pad_db] [repeats] [--scale V/div]

      python run_pad_test.py pad7  7      # with the 7 dB pad fitted
      python run_pad_test.py nopad 0      # pad removed, everything else same

Each run writes marker_coherence/pad_test_<label>.json and compares itself
against every other label already saved.
"""
import json
import sys
from pathlib import Path

import numpy as np

from awg import AWG
from scope import Scope
from run_chirp_flatness import analyse, plot

# -- the chirp: 2000-2500 MHz, as in LABBOOK 18-19 ----------------------
F_LO, F_HI = 2000e6, 2500e6
F_CARRIER = 0.5 * (F_LO + F_HI)                 # 2250 MHz NCO
CHIRP_US, DEAD_US, DETECT_US = 3.4, 0.1, 0.5
AMP_VPP = 0.5
MARKER_NS = 100

# Cabling, as established in run_chirp_averaging.py:
#     AWG CH1 M1 -> scope CH4   trigger   (CH1 RF off)
#     AWG CH2    -> scope CH3   signal
# Both AWG channels get the SAME chirp so their segments are the same length
# and the marker stays rigid to the measured waveform; CH1's RF is then shut
# off so only its marker plays.
AWG_SIG_CH, AWG_MARK_CH, MARKER = 2, 1, 1
SIG_CH, TRIG_CH = 3, 4
SCOPE_RATE = 50e9
REC_LEN = 200000                 # 4.0 us at 50 GS/s -- the whole chirp, undecimated
N_AVG = 16
TRIG_POS_PCT = 5                 # keep the record post-trigger
OUT = Path("marker_coherence")


def specs_for(carrier):
    one = {"kind": "chirp", "gate": "chirp", "carrier_hz": carrier,
           "f_start_hz": F_LO - carrier, "f_stop_hz": F_HI - carrier,
           "amplitude_vpp": AMP_VPP, "sideband": "upper"}
    return {1: dict(one), 2: dict(one)}


def send_chirp(awg):
    plan = awg.send_waveforms(specs_for(F_CARRIER), mode="duc",
                              timing={"chirp_us": CHIRP_US,
                                      "dead_us": DEAD_US,
                                      "detect_us": DETECT_US},
                              verbose=False, coherent=True)
    n_total, period_s = plan["n_total"], plan["period_s"]
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * n_total,
                     marker=MARKER, high_frac=MARKER_NS * 1e-9 / period_s,
                     level_vpp=1.0)
    with awg._lock:                       # marker only, no RF, from CH1
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")
    used = [e for e in plan["coherent_grid"] if e["channel"] == AWG_SIG_CH]
    car = used[0]["carrier_hz"] if used else F_CARRIER
    print(f"  carrier {car/1e6:.6f} MHz, buffer {period_s*1e6:.3f} us "
          f"-> {1/period_s/1e3:.1f} kHz rep, chirp on AWG CH{AWG_SIG_CH}, "
          f"marker on CH{AWG_MARK_CH} M{MARKER}")
    return period_s


def setup_scope(s, scale=None):
    s.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
            trigger_level=0.25, trigger_mode="NORMal")
    with s._lock:
        s._dev.write("HORizontal:MODE MANual")
        s._dev.write(f"HORizontal:MODE:SAMPLERate {SCOPE_RATE:.0f}")
        s._dev.write(f"HORizontal:MODE:RECOrdlength {REC_LEN}")
        s._dev.query("*OPC?")
    s.set_horizontal_position(TRIG_POS_PCT)
    if scale is not None:
        with s._lock:
            s._dev.write(f"CH{SIG_CH}:SCAle {scale:.4e}")
    s._pre.clear()
    with s._lock:
        sr = float(s._dev.query("HORizontal:SAMPLERate?"))
        rl = int(float(s._dev.query("HORizontal:MODE:RECOrdlength?")))
        sc = float(s._dev.query(f"CH{SIG_CH}:SCAle?"))
    print(f"  scope {rl} pts at {sr/1e9:g} GS/s = {rl/sr*1e6:.3f} us, "
          f"CH{SIG_CH} {sc*1e3:.1f} mV/div, AVE x{N_AVG}, trig CH{TRIG_CH} NORM")
    return sc


def capture(s):
    t, v = s.get_waveform(SIG_CH, max_points=REC_LEN, fresh=True, width=2,
                          absolute_time=True, acq_timeout=30.0)
    return t, v


def summarise(runs):
    """mean +- sd over repeats, which is the only honest error bar here."""
    def col(key, sub=None):
        vals = [(r[key] if sub is None else r[key][sub]) for r in runs
                if (r[key] is not None)]
        return np.array(vals, dtype=float)
    out = {"n": len(runs)}
    for name, key, sub in (("delay_ns", "ripple", "delay_s"),
                           ("ripple_pp_db", "ripple", "pp_db"),
                           ("gamma_product", "ripple", "gamma_product"),
                           ("contrast", "ripple", "contrast"),
                           ("flat_pp_db", "flat_pp_db", None),
                           ("tilt_db_per_ghz", "tilt_db_per_hz", None),
                           ("ptp_mv", "ptp_v", None)):
        a = col(key, sub)
        if a.size == 0:
            continue
        if name == "delay_ns":
            a = a * 1e9
        if name == "tilt_db_per_ghz":
            a = a * 1e9
        if name == "ptp_mv":
            a = a * 1e3
        out[name] = {"mean": float(a.mean()),
                     "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                     "vals": [float(x) for x in a]}
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scale = None
    for a in sys.argv[1:]:
        if a.startswith("--scale"):
            scale = float(a.split("=", 1)[1]) if "=" in a else None
    if scale is None and "--scale" in sys.argv:
        scale = float(sys.argv[sys.argv.index("--scale") + 1])
    if not args:
        print(__doc__)
        return 2
    label = args[0]
    pad_db = float(args[1]) if len(args) > 1 else 0.0
    repeats = int(args[2]) if len(args) > 2 else 5

    OUT.mkdir(exist_ok=True)
    print(f"=== pad test '{label}': {pad_db:g} dB in the path, "
          f"{repeats} repeats ===\n")
    print(f"chirp {F_LO/1e9:g}-{F_HI/1e9:g} GHz "
          f"(NCO {F_CARRIER/1e6:g} MHz, baseband "
          f"{(F_LO-F_CARRIER)/1e6:+g} to {(F_HI-F_CARRIER)/1e6:+g} MHz), "
          f"{CHIRP_US} us, {AMP_VPP} Vpp")

    awg = AWG()
    send_chirp(awg)
    s = Scope()
    used_scale = setup_scope(s, scale)
    if scale is None:
        print("\n  NOTE: no --scale given, so the scope auto-kept its vertical "
              "setting.\n        Pass the SAME --scale to the paired run or the "
              "comparison is\n        confounded by the front-end attenuator.")

    runs = []
    for i in range(repeats):
        t, v = capture(s)
        r = analyse(t, v, verbose=(i == 0))
        if r is None:
            print(f"  repeat {i}: no chirp in the record")
            return 1
        if i == 0:
            plot(r, OUT / f"pad_test_{label}.png",
                 title=f"chirp {F_LO/1e9:g}-{F_HI/1e9:g} GHz, "
                       f"{pad_db:g} dB pad ({label})")
            print(f"\nrepeats (delay / depth / ptp):")
        rip = r["ripple"]
        print(f"  {i}: {rip['delay_s']*1e9:7.3f} ns  {rip['pp_db']:6.3f} dB pp"
              f"  {r['ptp_v']*1e3:7.1f} mVpp  contrast {rip['contrast']:5.1f}")
        runs.append({k: v2 for k, v2 in r.items() if not k.startswith("_")})

    summ = summarise(runs)
    rec = {"label": label, "pad_db": pad_db, "scale_v_per_div": used_scale,
           "f_lo_hz": F_LO, "f_hi_hz": F_HI, "chirp_us": CHIRP_US,
           "amp_vpp": AMP_VPP, "scope_rate": SCOPE_RATE, "n_avg": N_AVG,
           "summary": summ, "runs": runs}
    path = OUT / f"pad_test_{label}.json"
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8")

    print(f"\n  {'':18}{'mean':>10}{'sd':>9}")
    for k in ("delay_ns", "ripple_pp_db", "gamma_product", "contrast",
              "flat_pp_db", "tilt_db_per_ghz", "ptp_mv"):
        if k in summ:
            print(f"  {k:18}{summ[k]['mean']:>10.4f}{summ[k]['sd']:>9.4f}")
    print(f"\nsaved {path}")

    # -- compare against anything else already measured -------------------
    others = sorted(p for p in OUT.glob("pad_test_*.json") if p != path)
    if others:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        g_here = summ["gamma_product"]["mean"]
        for p in others:
            o = json.loads(p.read_text(encoding="utf-8"))
            os_ = o["summary"]
            d_pad = pad_db - o["pad_db"]
            pred = os_["gamma_product"]["mean"] * 10 ** (-2 * d_pad / 20)
            print(f"\n  '{o['label']}' ({o['pad_db']:g} dB) -> "
                  f"'{label}' ({pad_db:g} dB), a {d_pad:+g} dB change")
            if abs(o["scale_v_per_div"] - used_scale) > 1e-9:
                print(f"    WARNING vertical scale differs "
                      f"({o['scale_v_per_div']*1e3:.1f} vs "
                      f"{used_scale*1e3:.1f} mV/div) -- the scope's input match "
                      f"may have changed with it, so this comparison is not "
                      f"clean")
            print(f"    ripple depth  {os_['ripple_pp_db']['mean']:.3f} "
                  f"+-{os_['ripple_pp_db']['sd']:.3f}  ->  "
                  f"{summ['ripple_pp_db']['mean']:.3f} "
                  f"+-{summ['ripple_pp_db']['sd']:.3f} dB pp")
            print(f"    |G1.G2|       {os_['gamma_product']['mean']:.5f}"
                  f"  ->  {g_here:.5f}   "
                  f"(a pure reflection predicts {pred:.5f})")
            if pred > 0:
                ratio = g_here / pred
                drop_db = 20 * np.log10(max(os_["gamma_product"]["mean"], 1e-12)
                                        / max(g_here, 1e-12))
                print(f"    measured round-trip change {drop_db:+.1f} dB, "
                      f"a reflection predicts {2*d_pad:+.1f} dB")
                if 0.5 < ratio < 2.0:
                    print("    -> CONSISTENT with a reflection in the padded "
                          "path")
                elif ratio >= 2.0:
                    print("    -> the ripple did NOT fall as a reflection "
                          "would. Part of it is behind\n       the pad, i.e. "
                          "generated inside one instrument.")
                else:
                    print("    -> fell by MORE than a reflection predicts; "
                          "check the pad value and\n       that nothing else "
                          "moved between runs.")
            print(f"    delay         {os_['delay_ns']['mean']:.3f} "
                  f"+-{os_['delay_ns']['sd']:.3f}  ->  "
                  f"{summ['delay_ns']['mean']:.3f} "
                  f"+-{summ['delay_ns']['sd']:.3f} ns")
            print(f"    direct signal {os_['ptp_mv']['mean']:.1f} -> "
                  f"{summ['ptp_mv']['mean']:.1f} mVpp "
                  f"({20*np.log10(summ['ptp_mv']['mean']/max(os_['ptp_mv']['mean'],1e-9)):+.2f} dB, "
                  f"the pad alone predicts {-d_pad:+.1f} dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
