"""Does the external reference actually buy anything? Measure the wander.

A single frequency measurement cannot tell the Wenzel from this unit's TCXO:
the TCXO happens to sit ~40 ppb away, which is at the measurement noise floor
(2026-10-19). The OCXO's advantage is not in the instantaneous offset, it is
in how little that offset MOVES. So measure the offset repeatedly over tens of
minutes in both states and compare the SPREAD, not the mean.

Design
------
The two states are INTERLEAVED, not run in sequence. Room temperature drifting
over the run would otherwise land entirely on whichever state went second and
masquerade as wander. Interleaving makes it common-mode. The order within each
pair alternates as well, so a temperature ramp inside a single pair cannot bias
one state either.

Each point is a reflock.check_lock() with n_reps captures, which also returns
its own within-point scatter. That is the short-term noise floor, and it is the
thing the between-point spread has to beat before "wander" means anything.

What to expect if the reference is doing its job: EXTERNAL spread at or near
the noise floor, INTERNAL spread several times larger and correlated with the
room.

Wiring: AWG CH2 -> scope CH3 (50 ohm). Scope locked to the external 10 MHz.
Leaves the AWG on EXT with the DUC 10 MHz tone restored on CH2.

Run:  python run_ref_drift.py [minutes]
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awg import AWG
from scope import Scope
import reflock

# ── configuration ─────────────────────────────────────────────────────────
DURATION_MIN = 45.0        # total; each state gets about half the points
PAIR_PERIOD_S = 30.0       # one INT + one EXT measurement per period
N_REPS = 3                 # captures per point (noise floor ~0.03 ppm)
AWG_CH, SCOPE_CH = 2, 3
RESTORE_FREQ_HZ, RESTORE_AMP = 10e6, 0.3


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else DURATION_MIN
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir = os.path.join("ref_drift", stamp)
    os.makedirs(outdir, exist_ok=True)

    awg, scope = AWG(), Scope()
    ref = scope.reference_status()
    if not (ref["external"] and ref["locked"]):
        print(f"scope is not locked to the external reference ({ref}) -- abort")
        return 1

    rows = []
    t0 = time.time()
    deadline = t0 + minutes * 60.0
    csv = open(os.path.join(outdir, "data.csv"), "w", buffering=1)
    csv.write("t_s,state,ppm,ppm_sd,f_measured_hz,f_nominal_hz\n")
    print(f"{minutes:.0f} min, one INT+EXT pair every {PAIR_PERIOD_S:.0f} s "
          f"-> {outdir}\n")
    print(f"{'t (min)':>8}  {'INT (ppm)':>18}  {'EXT (ppm)':>18}")

    pair = 0
    while time.time() < deadline:
        pair_start = time.time()
        # alternate which state goes first so an intra-pair temperature ramp
        # cannot systematically favour one of them
        order = ("int", "ext") if pair % 2 == 0 else ("ext", "int")
        got = {}
        for state in order:
            if state == "int":
                awg.set_reference_internal()
            else:
                awg.set_reference_external(10)
            try:
                r = reflock.check_lock(awg, scope, awg_channel=AWG_CH,
                                       scope_channel=SCOPE_CH, n_reps=N_REPS,
                                       verbose=False)
            except Exception as e:
                print(f"  point failed ({state}): {e}")
                continue
            t = time.time() - t0
            got[state] = r
            rows.append({"t_s": t, "state": state, "ppm": r["ppm"],
                         "ppm_sd": r["ppm_sd"], "f": r["f_measured"],
                         "f_nom": r["f_nominal"]})
            csv.write(f"{t:.1f},{state},{r['ppm']:.6f},{r['ppm_sd']:.6f},"
                      f"{r['f_measured']:.4f},{r['f_nominal']:.4f}\n")
        fmt = lambda s: (f"{got[s]['ppm']:+8.3f} ± {got[s]['ppm_sd']:.3f}"
                         if s in got else " " * 18)
        print(f"{(time.time()-t0)/60:8.1f}  {fmt('int')}  {fmt('ext')}")
        pair += 1
        time.sleep(max(0.0, PAIR_PERIOD_S - (time.time() - pair_start)))
    csv.close()

    # ── summary ──────────────────────────────────────────────────────
    summary = {"started": stamp, "minutes": minutes, "n_pairs": pair,
               "awg_channel": AWG_CH, "scope_channel": SCOPE_CH,
               "n_reps_per_point": N_REPS, "states": {}}
    print("\n" + "=" * 66)
    for state in ("int", "ext"):
        p = np.array([r["ppm"] for r in rows if r["state"] == state])
        n = np.array([r["ppm_sd"] for r in rows if r["state"] == state])
        if p.size < 2:
            continue
        d = {"n": int(p.size), "mean_ppm": float(p.mean()),
             "wander_sd_ppm": float(p.std(ddof=1)),
             "range_ppm": float(p.max() - p.min()),
             "noise_floor_ppm": float(np.median(n))}
        d["wander_over_noise"] = d["wander_sd_ppm"] / d["noise_floor_ppm"]
        summary["states"][state] = d
        label = "INTERNAL TCXO" if state == "int" else "EXTERNAL (Wenzel)"
        print(f"{label:<20} n={d['n']:3d}  mean {d['mean_ppm']:+.3f} ppm")
        print(f"{'':<20} wander (sd)      {d['wander_sd_ppm']:.4f} ppm")
        print(f"{'':<20} peak-to-peak     {d['range_ppm']:.4f} ppm")
        print(f"{'':<20} noise floor      {d['noise_floor_ppm']:.4f} ppm")
        print(f"{'':<20} wander / noise   {d['wander_over_noise']:.2f}"
              f"   {'<- real wander' if d['wander_over_noise'] > 2 else '<- at the noise floor'}")
        print()
    if len(summary["states"]) == 2:
        ratio = (summary["states"]["int"]["wander_sd_ppm"] /
                 summary["states"]["ext"]["wander_sd_ppm"])
        summary["int_over_ext_wander"] = ratio
        print(f"INTERNAL wanders {ratio:.1f}x more than EXTERNAL")
    print("=" * 66)

    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for state, colour, label in (("int", "#c2492f", "internal TCXO"),
                                 ("ext", "#1f4e6b", "external (Wenzel)")):
        t = [r["t_s"] / 60 for r in rows if r["state"] == state]
        p = [r["ppm"] for r in rows if r["state"] == state]
        e = [r["ppm_sd"] for r in rows if r["state"] == state]
        ax.errorbar(t, p, yerr=e, fmt="o-", ms=3, lw=1, capsize=2,
                    color=colour, label=label, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("AWG clock error vs reference-locked scope (ppm)")
    ax.set_title("10 MHz reference: does locking reduce the wander?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drift.png"), dpi=130)
    print(f"\nsaved {outdir}/data.csv, summary.json, drift.png")

    # ── put the bench back ───────────────────────────────────────────
    awg.duc_cw_setup(RESTORE_FREQ_HZ, RESTORE_AMP, channel=AWG_CH)
    awg.set_reference_external(10)
    print(f"restored: AWG CH{AWG_CH} DUC {RESTORE_FREQ_HZ/1e6:g} MHz "
          f"@ {RESTORE_AMP} Vpp, reference EXT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
