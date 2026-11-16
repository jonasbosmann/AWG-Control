"""Offline checks for multi-cycle, phase-cycled buffer building.

No instruments. Uses AWG.__new__(AWG) the way gui.py:101 already does for its
preview planner, so nothing here touches hardware or the network.

Run:  python tests/test_phasecycle_build.py [--bless]

      --bless  regenerate tests/baseline_build.npz from the CURRENT code.
               Do this ONCE before changing _build_channel, never after.

What it checks
--------------
1. _plan_timing with n_cycles=1 is unchanged from the recorded baseline.
2. n_cycles=K gives n_total == K * n_cycle, cycles aligned to granularity.
3. _build_channel with n_cycles=1 and no phases is BIT-IDENTICAL to the
   baseline, for every kind and both modes.
4. The NCO de-rotation: with K cycles on a grid-snapped carrier, the phase
   actually emitted at the start of cycle k equals the phase programmed for
   cycle k. This is the trap from LABBOOK 21 that cost a measurement, and it
   is testable without a scope.
6. The same for the LOWER sideband, on a carrier with a fractional number of
   turns per cycle. Until 2026-11-16 the correction had the upper-sideband
   sign there and a programmed 0/180 was emitted as 0/0.
7. phase_multiplier: for a xN chain the AWG emits phase/N.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from awg import AWG                                           # noqa: E402

BASELINE = Path(__file__).resolve().parent / "baseline_build.npz"

TIMING = {"chirp_us": 1.0, "dead_us": 0.1, "detect_us": 10.0}
RATE_CX = 9e9 / 8

# one spec per (mode, kind) worth pinning
SPECS = {
    "duc_chirp":  ("duc", {"kind": "chirp", "gate": "chirp", "carrier_hz": 2.25e9,
                           "f_start_hz": -250e6, "f_stop_hz": 250e6,
                           "sideband": "upper"}),
    "duc_cw":     ("duc", {"kind": "cw", "gate": "continuous", "carrier_hz": 2.25e9,
                           "freq_hz": 100e6, "sideband": "upper"}),
    "duc_cw_det": ("duc", {"kind": "cw", "gate": "detect", "carrier_hz": 2.25e9,
                           "freq_hz": 100e6, "sideband": "lower"}),
    "duc_square": ("duc", {"kind": "square", "gate": "chirp", "carrier_hz": 2.25e9,
                           "freq_hz": 50e6, "sideband": "upper"}),
    "duc_ramp":   ("duc", {"kind": "ramp", "gate": "chirp", "carrier_hz": 2.25e9,
                           "freq_hz": 50e6, "sideband": "both"}),
    "dir_chirp":  ("direct", {"kind": "chirp", "gate": "chirp",
                              "f_start_hz": 100e6, "f_stop_hz": 300e6}),
    "dir_cw":     ("direct", {"kind": "cw", "gate": "continuous",
                              "freq_hz": 100e6}),
    "dir_square": ("direct", {"kind": "square", "gate": "chirp",
                              "freq_hz": 50e6}),
}
RATE_DIRECT = 2.5e9


def planner():
    return AWG.__new__(AWG)


def build_one(a, mode, spec, timing):
    rate_eff = RATE_CX if mode == "duc" else RATE_DIRECT
    plan = a._plan_timing(mode, timing, rate_eff)
    plan["rate_dac"] = 9e9 if mode == "duc" else rate_eff
    data, info = a._build_channel(mode, spec, plan, 0.05)
    return plan, data, info


def bless():
    a = planner()
    out = {}
    for name, (mode, spec) in SPECS.items():
        plan, data, _ = build_one(a, mode, spec, TIMING)
        out[f"{name}__data"] = data
        out[f"{name}__ntotal"] = np.array([plan["n_total"]])
    pl = a._plan_timing("duc", TIMING, RATE_CX)
    out["plan_duc"] = np.array([pl["n_chirp"], pl["n_dead"], pl["n_detect"],
                                pl["n_total"]])
    np.savez_compressed(BASELINE, **out)
    print(f"blessed {BASELINE} from the current code "
          f"({len(SPECS)} specs)")


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def main():
    if "--bless" in sys.argv:
        bless()
        return 0
    if not BASELINE.exists():
        print(f"no baseline at {BASELINE} -- run with --bless on the UNCHANGED "
              f"code first")
        return 2
    base = np.load(BASELINE)
    a = planner()
    ok = True

    print("\n1. _plan_timing unchanged for n_cycles=1")
    pl = a._plan_timing("duc", TIMING, RATE_CX)
    want = base["plan_duc"]
    got = np.array([pl["n_chirp"], pl["n_dead"], pl["n_detect"], pl["n_total"]])
    ok &= check("chirp/dead/detect/total", np.array_equal(got, want),
                f"{got.tolist()} vs {want.tolist()}")

    print("\n2. n_cycles=K multiplies the buffer, cycles stay aligned")
    for k in (1, 2, 3, 4, 8):
        p = a._plan_timing("duc", dict(TIMING, n_cycles=k), RATE_CX)
        good = (p["n_total"] == k * p["n_cycle"] and p["n_cycles"] == k
                and p["n_cycle"] % a.GRAN_DUC == 0
                and abs(p["t_cycle"] - p["n_cycle"] / RATE_CX) < 1e-18)
        ok &= check(f"K={k}", good,
                    f"n_cycle {p['n_cycle']:,} n_total {p['n_total']:,} "
                    f"t_cycle {p['t_cycle']*1e6:.4f} us")
    p1 = a._plan_timing("duc", dict(TIMING, n_cycles=1), RATE_CX)
    ok &= check("K=1 equals no n_cycles at all",
                p1["n_total"] == pl["n_total"])

    print("\n3. _build_channel bit-identical for n_cycles=1, no phases")
    for name, (mode, spec) in SPECS.items():
        _, data, _ = build_one(a, mode, spec, TIMING)
        ok &= check(name, np.array_equal(data, base[f"{name}__data"]),
                    f"{len(data):,} samples")

    print("\n4. NCO de-rotation: emitted phase == programmed phase")
    # A chirp is awkward to read a phase off, so use a gated CW burst: its
    # analytic phase at the window start IS the programmed phase.
    K = 4
    phases_deg = [0.0, 180.0, 90.0, 270.0]
    phases = [np.radians(d) for d in phases_deg]
    rate_eff = RATE_CX
    pl = a._plan_timing("duc", dict(TIMING, n_cycles=K), rate_eff)
    pl["rate_dac"] = 9e9
    carrier = a.snap_duc_carrier(2.25e9, pl["n_total"], 9e9, 8)[0]
    spec = {"kind": "cw", "gate": "chirp", "carrier_hz": carrier,
            "freq_hz": 100e6, "sideband": "upper", "phases": phases}
    data, info = a._build_channel("duc", spec, pl, 0.0)   # no taper: clean edges
    iq = data.astype(np.float64)
    i = iq[0::2] - 32768.0
    q = iq[1::2] - 32768.0
    z = i + 1j * q
    start, n_win = pl["windows"]["chirp"]
    for k in range(K):
        s = start + k * pl["n_cycle"]
        # measured phase of the baseband at this cycle's first sample, with
        # the carrier's own advance to that instant added back on
        bb = float(np.angle(z[s]))
        adv = 2 * np.pi * carrier * (k * pl["t_cycle"])
        emitted = (bb + adv) % (2 * np.pi)
        want = phases[k] % (2 * np.pi)
        err = np.degrees((emitted - want + np.pi) % (2 * np.pi) - np.pi)
        ok &= check(f"cycle {k} programmed {phases_deg[k]:>5.0f} deg",
                    abs(err) < 0.05, f"error {err:+.4f} deg")

    print("\n5. phases list shorter than K wraps")
    spec2 = dict(spec, phases=[0.0, np.pi])
    d2, _ = a._build_channel("duc", spec2, pl, 0.0)
    z2 = (d2.astype(np.float64)[0::2] - 32768.0) + \
         1j * (d2.astype(np.float64)[1::2] - 32768.0)
    same = []
    for k in range(K):
        s = start + k * pl["n_cycle"]
        adv = 2 * np.pi * carrier * (k * pl["t_cycle"])
        same.append(np.degrees((float(np.angle(z2[s])) + adv) % (2 * np.pi)))
    alt = abs(((same[2] - same[0] + 180) % 360) - 180) < 0.05 and \
        abs(((same[3] - same[1] + 180) % 360) - 180) < 0.05
    ok &= check("0,pi wraps to 0,pi,0,pi", alt,
                " ".join(f"{s:.1f}" for s in same))

    def rf_phases(spec_x, gate="chirp"):
        """RF phase of each cycle relative to cycle 0, in degrees, plus info.

        Compares whole windows (complex inner product) with the NCO's advance
        to each cycle start put back on, so it works for either sideband.
        """
        d, inf = a._build_channel("duc", spec_x, pl, 0.0)
        zz = (d.astype(np.float64)[0::2] - 32768.0) + \
            1j * (d.astype(np.float64)[1::2] - 32768.0)
        st, nw = pl["windows"][gate]
        fc = spec_x["carrier_hz"]
        z0 = zz[st:st + nw]
        out = []
        for k in range(K):
            s0 = st + k * pl["n_cycle"]
            zk = zz[s0:s0 + nw] * np.exp(
                2j * np.pi * ((fc * k * pl["t_cycle"]) % 1.0))
            out.append(float(np.degrees(np.angle(np.vdot(z0, zk)))))
        return out, inf

    def worst_err(got_deg, want_rad):
        want = [np.degrees(w - want_rad[0]) for w in want_rad]
        return max(abs((g - w + 180.0) % 360.0 - 180.0)
                   for g, w in zip(got_deg, want))

    print("\n6. lower sideband: emitted phase == info['phases_emitted_rad']")
    # Needs a carrier with a FRACTIONAL number of turns per cycle, or the NCO
    # correction is zero and its sign cannot be wrong. Step down the coherent
    # grid from 2.25 GHz until there is one.
    step = a.coherent_step_hz(pl["n_total"], 9e9, 8)
    car_f = carrier
    for _ in range(64):
        if 0.05 < (car_f * pl["t_cycle"]) % 1.0 < 0.95:
            break
        car_f -= step
    frac = (car_f * pl["t_cycle"]) % 1.0
    ok &= check("found a carrier with fractional turns per cycle",
                0.05 < frac < 0.95, f"{car_f/1e6:.6f} MHz, {frac:.4f} turns")
    for pat_deg in ([0.0, 180.0], [0.0, 90.0, 180.0, 270.0]):
        pat = [np.radians(x) for x in pat_deg]
        for sb in ("upper", "lower"):
            sp = dict(spec, carrier_hz=car_f, sideband=sb, phases=pat)
            got, inf = rf_phases(sp)
            e = worst_err(got, inf["phases_emitted_rad"])
            ok &= check(f"{sb:5s} {len(pat)}-step", e < 0.05,
                        f"emitted {' '.join(f'{g:+.1f}' for g in got)}  "
                        f"worst error {e:.4f} deg")

    print("\n7. chain multiplier: the AWG emits phase/N, info reports x N")
    for n_mult in (1, 2, 144):
        pat = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        sp = dict(spec, carrier_hz=car_f, phases=pat, phase_multiplier=n_mult)
        got, inf = rf_phases(sp)
        e_awg = worst_err(got, [x / n_mult for x in pat])
        e_out = worst_err([g * n_mult for g in got], inf["phases_emitted_rad"])
        ok &= check(f"N={n_mult}", e_awg < 0.02 and e_out < 0.5,
                    f"AWG steps {' '.join(f'{g:+.3f}' for g in got)} deg, "
                    f"after xN error {e_out:.3f} deg")

    print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}\n")
    return 0 if ok else 1


def test_phasecycle_build():
    """pytest entry point: the whole script must report ALL PASS."""
    assert BASELINE.exists(), f"no baseline at {BASELINE}"
    assert main() == 0, "see the FAIL lines printed above"


if __name__ == "__main__":
    sys.exit(main())
