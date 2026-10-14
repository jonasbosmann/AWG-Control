"""Propagate a stage-A output level through the whole x8 chain and check every
part's input window at BOTH band edges.

Why this exists separately from multiplier_chain.full_chain(): that function
tracks ONE level down the chain (the band's weak end) and solves for the AWG
drive itself. This one takes the stage-A amplifier output as the INPUT --
because that is what run_drive_level.py actually sets at the bench -- and
carries both band edges through, so an under-driven part at one end of the
band cannot hide behind an in-spec number at the other.

That distinction matters more than it sounds. The chain is not flat: the AWG
is strongest at the low end and every doubler and amplifier adds its own tilt,
so "the level" is really two levels that drift apart. A part whose input
window is only 5 dB wide (the ZX90-2-24-S+ is +11 to +16 dBm) can be inside it
at one band edge and outside at the other.

CONVENTIONS -- read this before comparing numbers
-------------------------------------------------
run_drive_level.py sets the band MAXIMUM to its target, so that nothing in the
band exceeds it. multiplier_chain.full_chain() quotes the band MINIMUM (the
weak end). The two differ by the stage's tilt, ~1.2 dB. So the design's
+16.13 dBm quoted at the weak end is a band MAXIMUM of about +17.3 dBm, and
asking run_drive_level.py for +15 dBm is NOT the design point -- it is about
2.3 dB below it.

Run:  python x8_chain/verify_chain.py [stage_A_band_max_dBm ...]
      defaults to the experiment level and the design point.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import multiplier_chain as mc


FILTER_IL_DB = mc.FILTER_IL_DB      # 1.5, allowance for a real bandpass
CABLE_IL_DB = mc.CABLE_IL_DB        # 0.5 per hop

B_LO, B_HI = config.BAND_A_HZ


def _b(mult):
    """Band label after `mult` doublings of the AWG band."""
    return f"{mult*B_LO/1e9:.3f}-{mult*B_HI/1e9:.3f}"


# The chain after the stage-A amplifier. Each entry either passes both band
# edges through a fixed loss/gain, or is a part with an input window to check.
#   (name, delta_db, win_lo, win_hi, note, band_ghz, stock)
# win_* are the INPUT window of the part, checked against the level ARRIVING.
AFTER_STAGE_A = [
    # MEASURED 2026-07-31 (PE87FL1012): 0.64 dB at 1.614 GHz, 1.50 dB at
    # 1.927 GHz. Per-edge rather than the flat 1.5 dB allowance, because this
    # filter's loss is not flat -- 1.927 GHz sits on its upper passband skirt,
    # which is where the extra 0.86 dB of tilt comes from.
    ("BPF 1.6-1.95 GHz",      (-0.64, -1.50), None, None,
     "MEASURED", _b(1), "on hand"),
    # No cable here: the filter bolts straight onto the doubler. Worth 0.5 dB
    # exactly where the chain is shortest, plus one less connector pair of
    # mismatch ripple.
    ("HMC188 doubler",        -mc.HMC188_CL_TYP_DB,
     mc.HMC188_PIN_MIN_DBM, mc.HMC188_PIN_MAX_DBM,
     f"nominal {mc.HMC188_PIN_NOM_DBM:+.0f} dBm", f"-> {_b(2)}", "on hand"),
    ("BPF 3.2-3.9 GHz",       -FILTER_IL_DB, None, None, "", _b(2), "BUY"),
    ("cable",                 -CABLE_IL_DB,  None, None, "", _b(2), ""),
    # DATASHEET gain, 15.0 dB -- not the 16.9 dB that multiplier_chain.py
    # prints here. That function back-solves the gain needed to land the
    # MMD-0415HS at its +14 dBm nominal and labels it "16.9 dB needed"; it
    # never checks the number against the part actually in the drawer. The
    # part has 15.0 dB, so the chain is 1.9 dB short at this point and
    # everything downstream inherits the shortfall.
    ("ZX60-123LN-S+",         mc.AMP_123LN_GAIN_DB, None, None,
     "DATASHEET 15.0 dB -- the design back-solves 16.9 dB here and never "
     "checks it; see the module docstring", _b(2), "on hand"),
    ("MMD-0415HS doubler",    -mc.MMD_CL_TYP_DB,
     mc.MMD_PIN_MIN_DBM, mc.MMD_PIN_MAX_DBM,
     f"nominal {mc.MMD_PIN_NOM_DBM:+.0f} dBm", f"-> {_b(4)}", "on hand"),
    ("BPF 6.4-7.8 GHz",       -FILTER_IL_DB, None, None, "", _b(4), "BUY"),
    ("cable",                 -CABLE_IL_DB,  None, None, "", _b(4), ""),
    ("ZX60-04183LN+",         mc.AMP_KU_GAIN_TYP_DB, None, None, "",
     _b(4), "on hand (1 of 2)"),
    ("ZX90-2-24-S+ doubler",  -mc.ZX90_CL_AT_11DBM_MEAN,
     mc.ZX90_PIN_MIN_DBM, mc.ZX90_PIN_MAX_DBM,
     "the tightest window in the chain, only 5 dB wide",
     f"-> {_b(8)}", "on hand"),
    ("BPF 12.9-15.5 GHz",     -FILTER_IL_DB, None, None, "", _b(8), "BUY"),
    ("cable",                 -CABLE_IL_DB,  None, None, "", _b(8), ""),
    ("pad (final trim)",      -3.9,          None, None,
     "DC-18 GHz part (BW-S); a DC-6 GHz VAT will not work here", _b(8), "BUY"),
    ("ZX60-04183LN+ output",  mc.AMP_KU_GAIN_TYP_DB, None, None, "",
     _b(8), "on hand (2 of 2)"),
    ("isolator PE83IR1023",   -mc.ISOLATOR_IL_DB, None, None, "",
     _b(8), "on hand"),
]

# Amps whose own P1dB has to be respected too, keyed by the stage name above.
AMP_P1DB = {
    "ZX60-04183LN+": mc.AMP_KU_P1DB_DBM,
    "ZX60-04183LN+ output": mc.AMP_KU_P1DB_DBM,
}

AMP_A_P1DB = mc.ZX602522MA_P1DB_DBM


def awg_band_edges(vpp=None):
    """Measured AWG level at the x8 band edges (strong, weak), in dBm."""
    resp = mc.load_scope_band_response()
    lo, hi = config.BAND_A_HZ
    if resp is None:
        v = config.dbm_from_vpp(vpp or config.AWG_MAX_VPP)
        return v, v
    f, a = resp[0], resp[1]
    strong, weak = float(np.interp(lo, f, a)), float(np.interp(hi, f, a))
    if vpp:
        d = 20 * np.log10(vpp / mc.AWG_OPERATING_AMP_VPP)
        strong, weak = strong + d, weak + d
    return strong, weak


def check(band_max_dbm, tilt_db=None, label=""):
    """Carry both band edges from the stage-A amp output to the AMC."""
    if tilt_db is None:
        s, w = awg_band_edges()
        tilt_db = s - w              # the AWG's own tilt, 1.17 dB measured
    hi = band_max_dbm                # strong edge (band low frequency)
    lo = band_max_dbm - tilt_db      # weak edge

    # Back-solve the AWG level this stage-A output implies, so the listing
    # shows the amplitude to actually dial in rather than a bare reference.
    # Uses the DATASHEET gain -- run_drive_level.py replaces it with a closed
    # loop on the measured output, and the two are worth comparing.
    drive_pad = config.STAGES["A"]["drive_pad_db"]
    gain_a = config.STAGES["A"]["gain_typ_db"]
    awg_need_s = band_max_dbm - gain_a + drive_pad + CABLE_IL_DB
    awg_ref_s, awg_ref_w = awg_band_edges()
    vpp = float(np.clip(config.AWG_MAX_VPP * 10 ** ((awg_need_s - awg_ref_s) / 20),
                        0.001, config.AWG_MAX_VPP))
    awg_s = awg_ref_s + 20 * np.log10(vpp / config.AWG_MAX_VPP)
    awg_w = awg_ref_w + 20 * np.log10(vpp / config.AWG_MAX_VPP)

    print("=" * 100)
    print(f"STAGE-A OUTPUT {band_max_dbm:+.2f} dBm (band max)"
          + (f"   -- {label}" if label else ""))
    print(f"  weak edge {lo:+.2f} dBm; tilt {tilt_db:.2f} dB carried through")
    print("=" * 100)
    print(f"  {'#':>2} {'part':<24}{'band GHz':>16}{'dB':>7}{'strong':>8}"
          f"{'weak':>8}  {'stock':<16} check")

    fails = []
    n = 1
    print(f"  {n:>2} {'AWG CH1 DUC CW':<24}{_b(1):>16}{'':>7}"
          f"{awg_s:>+8.2f}{awg_w:>+8.2f}  {vpp:.3f} Vpp        "
          f"measured curve, 2 runs; {config.AWG_MAX_VPP} Vpp ceiling "
          f"({20*np.log10(vpp/config.AWG_MAX_VPP):+.1f} dB headroom)")
    n += 1
    a_s, a_w = awg_s - CABLE_IL_DB, awg_w - CABLE_IL_DB
    print(f"  {n:>2} {'cable (keep SHORT)':<24}{_b(1):>16}"
          f"{-CABLE_IL_DB:>+7.1f}{a_s:>+8.2f}{a_w:>+8.2f}  {'':<16} "
          f"lowest-level point in the chain")
    n += 1
    a_s, a_w = a_s - drive_pad, a_w - drive_pad
    print(f"  {n:>2} {f'{drive_pad:g} dB pad':<24}{_b(1):>16}"
          f"{-drive_pad:>+7.1f}{a_s:>+8.2f}{a_w:>+8.2f}  "
          f"{'on hand':<16} in front of the drive amp for every measurement")
    n += 1
    back = AMP_A_P1DB - hi
    tag = "OK" if back >= 3.0 else ("TIGHT" if back >= 1.0 else "** PAST P1dB **")
    if back < 3.0:
        fails.append(f"stage-A amp only {back:.2f} dB below P1dB")
    print(f"  {n:>2} {'ZX60-2522MA-S+ (amp)':<24}{_b(1):>16}{gain_a:>+7.1f}"
          f"{hi:>+8.2f}{lo:>+8.2f}  {'on hand':<16} "
          f"{back:.1f} dB below P1dB {AMP_A_P1DB:+.1f}  {tag}")

    for name, d, wlo, whi, note, band, stock in AFTER_STAGE_A:
        n += 1
        # A stage's loss/gain may differ between the band edges (a filter skirt
        # does exactly this), so entries may give a (strong, weak) pair.
        d_s, d_w = d if isinstance(d, tuple) else (d, d)
        msg = ""
        if wlo is not None:
            # The window is checked on the level ARRIVING, i.e. before this
            # part's own conversion loss or gain is applied.
            marg_lo, marg_hi = lo - wlo, whi - hi
            if marg_lo < 0:
                msg = (f"** {-marg_lo:.2f} dB UNDER the {wlo:+.0f} dBm floor "
                       f"at the weak edge **")
                fails.append(f"{name}: {msg}")
            elif marg_hi < 0:
                msg = (f"** {-marg_hi:.2f} dB OVER the {whi:+.0f} dBm ceiling "
                       f"at the strong edge **")
                fails.append(f"{name}: {msg}")
            else:
                msg = (f"in {wlo:+.0f}..{whi:+.0f} dBm, margin "
                       f"{marg_lo:.2f}/{marg_hi:.2f} dB")
            if note:
                msg += f"  ({note})"
        elif note:
            msg = note
        hi, lo = hi + d_s, lo + d_w
        if name in AMP_P1DB:
            back = AMP_P1DB[name] - hi
            msg = f"out {hi:+.2f} dBm, {back:.1f} dB below P1dB {AMP_P1DB[name]:+.0f}"
            if back < mc.AMP_BACKOFF_MIN_DB:
                msg += f"  ** under the {mc.AMP_BACKOFF_MIN_DB:g} dB minimum **"
                fails.append(f"{name}: only {back:.2f} dB below P1dB")
        dtxt = f"{d_s:+.1f}/{d_w:+.1f}" if d_s != d_w else f"{d_s:+.1f}"
        print(f"  {n:>2} {name:<24}{band:>16}{dtxt:>11}{hi:>+8.2f}{lo:>+8.2f}  "
              f"{stock:<16} {msg}")

    print()
    amc_lo, amc_hi = 0.0, config.AMC_WINDOW_DB
    print(f"  AMC input window {amc_lo:+.0f}..{amc_hi:+.0f} dBm  ->  "
          f"DELIVERED {hi:+.2f} (strong) / {lo:+.2f} (weak) dBm")
    if lo < amc_lo or hi > amc_hi:
        m = (f"AMC input {lo:+.2f}..{hi:+.2f} dBm falls outside its "
             f"{amc_lo:+.0f}..{amc_hi:+.0f} dBm window")
        fails.append(m)
        print(f"  ** {m} -- retrim the pad **")
    else:
        print(f"     inside, margin {lo-amc_lo:.2f} dB below / "
              f"{amc_hi-hi:.2f} dB above")

    print()
    if fails:
        print(f"  VERDICT: {len(fails)} problem(s)")
        for f_ in fails:
            print(f"    - {f_}")
    else:
        print("  VERDICT: every part inside its window at both band edges")
    print()
    return not fails


def main():
    print(__doc__.split("Run:")[0])
    print("All levels are DATASHEET typicals downstream of stage A -- that is\n"
          "exactly what this bench campaign exists to replace. Treat the\n"
          "margins as provisional until each stage has been measured.\n")

    s, w = awg_band_edges()
    print(f"measured AWG at {mc.AWG_OPERATING_AMP_VPP} Vpp across the x8 band: "
          f"{s:+.2f} (1.614 GHz) to {w:+.2f} dBm (1.927 GHz), tilt {s-w:.2f} dB\n")

    if len(sys.argv) > 1:
        levels = [(float(a), "") for a in sys.argv[1:]]
    else:
        design_max = 17.26   # multiplier_chain quotes +16.13 at the WEAK edge
        levels = [
            (config.TARGET_OUT_DBM,
             f"the experiment level run_drive_level.py targets"),
            (design_max,
             "the design point (+16.13 dBm quoted at the weak edge)"),
        ]
    ok = [check(lvl, label=lab) for lvl, lab in levels]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
