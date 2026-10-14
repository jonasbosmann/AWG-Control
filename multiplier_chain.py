"""Multiplier-only transmit chain: AWG -> cascaded x2 doublers -> AMC, with NO
mixer and NO external LO.

The mixer variants (power_budget.py) are limited by three measured facts:
  1. the AWG's amplitude response is ragged over the wide IF span they need,
  2. the LO cannot be driven to the mixer's +15 dBm with parts on hand,
  3. a double-balanced mixer gives 0 dB image rejection and the LO leak lands
     only ~0.5 GHz from the AMC's lower band edge.
Multiplying instead of mixing removes all three at once, at the cost of extra
phase-noise multiplication and a much hungrier drive requirement.

This script evaluates the multiplication factor first (which AWG sub-band each
choice lands on, using the REAL measured CW level check), then walks the
stage-by-stage power budget for the chains that are actually buildable.

Run:  python multiplier_chain.py

Companion to power_budget.py -- same data source, same conventions.
"""
import glob
import os
import sys

import numpy as np

import specanlog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirp_bench"))
from run_specan_band_scan import EXTERNAL_ATTEN_DB
from power_budget import (AMC_IN_LO_HZ, AMC_IN_HI_HZ, AMC_IN_MIN_DBM,
                          AMC_IN_MAX_DBM, AMC_IN_TYP_DBM, AMP_KU_GAIN_TYP_DB,
                          AMP_KU_P1DB_DBM, AMP_IF_P1DB_DBM, ISOLATOR_IL_DB,
                          load_awg_response)

# ── datasheet constants (transcribed 2026-07-27 from the parts-folder PDFs) ──

# Marki MMD-0415HS GaAs MMIC doubler
MMD_IN_LO_HZ, MMD_IN_HI_HZ = 2.0e9, 7.5e9
MMD_OUT_LO_HZ, MMD_OUT_HI_HZ = 4.0e9, 15.0e9
MMD_PIN_MIN_DBM, MMD_PIN_NOM_DBM, MMD_PIN_MAX_DBM = 7.0, 14.0, 20.0
MMD_CL_TYP_DB, MMD_CL_MAX_DB = 11.0, 15.0
MMD_SUPP_1F_DBC, MMD_SUPP_3F_DBC, MMD_SUPP_4F_DBC = 27.0, 36.0, 15.0

# Mini-Circuits ZX90-2-24-S+ doubler
ZX90_IN_LO_HZ, ZX90_IN_HI_HZ = 5.0e9, 10.0e9
ZX90_OUT_LO_HZ, ZX90_OUT_HI_HZ = 10.0e9, 20.0e9
ZX90_PIN_MIN_DBM, ZX90_PIN_MAX_DBM = 11.0, 16.0
ZX90_CL_TYP_DB, ZX90_CL_MAX_DB = 12.0, 19.5
ZX90_HARM_DBC = 30.0
# Typical-performance table, +16 dBm drive, over the 6.4-7.8 GHz input range a
# x8 chain's final stage actually uses. Better than the 12 dB headline spec.
ZX90_CL_MEASURED_6P4_7P8 = {6.4: 12.24, 6.6: 12.19, 6.8: 12.86, 7.0: 11.70,
                            7.4: 10.34, 7.6: 10.97, 7.8: 9.79}
# ...but note the F1 suppression column collapses to 21.90 dBc at 7.0 GHz
# (vs 31-33 dBc either side) -- a real, datasheet-visible bad spot inside the
# band a x8 chain's last doubler would sweep through.
ZX90_F1_SUPP_WORST_DBC, ZX90_F1_SUPP_WORST_HZ = 21.90, 7.0e9

# Analog Devices HMC188MS8 GaAs MMIC passive doubler (user has this part).
# Input 1.25-3.0 GHz -> output 2.5-6.0 GHz, which is EXACTLY the stage-A slot
# the x8 chain was missing. Harmonic suppression 45 dBc typ is markedly better
# than the MMD-0415HS's 27 dBc on 1F.
HMC188_IN_LO_HZ, HMC188_IN_HI_HZ = 1.25e9, 3.0e9
HMC188_OUT_LO_HZ, HMC188_OUT_HI_HZ = 2.5e9, 6.0e9
HMC188_PIN_MIN_DBM, HMC188_PIN_NOM_DBM, HMC188_PIN_MAX_DBM = 10.0, 15.0, 20.0
HMC188_CL_TYP_DB = 15.0
HMC188_SUPP_DBC = 45.0

# Pulsar Microwave FX-05-410 passive (Schottky bridge) doubler -- ALTERNATIVE
# for the stage-A slot, checked against the HMC188 2026-08-03. Input
# 50-2100 MHz -> output 100-4200 MHz, comfortably wider on both ends than the
# 1.614-1.927 -> 3.228-3.855 GHz this slot actually uses.
#
# pulsarmicrowave.com's spec table gives only ONE conversion-loss figure,
# labelled "max" (no separate typ), and only a NOMINAL drive level -- there is
# no published min/max Pin window to check against, unlike the HMC188's spec'd
# +10..+20 dBm range. Treat FX05410_CL_MAX_DB as already the pessimistic case,
# not swap-equivalent to the other parts' *_TYP_DB numbers.
FX05410_IN_LO_HZ, FX05410_IN_HI_HZ = 0.05e9, 2.1e9
FX05410_OUT_LO_HZ, FX05410_OUT_HI_HZ = 0.1e9, 4.2e9
FX05410_PIN_NOM_DBM = 10.0
FX05410_CL_MAX_DB = 13.0        # vs HMC188's 15.0 dB typ -- 2 dB BETTER
FX05410_SUPP_F1_DBC = 16.0      # vs HMC188's 45 dBc typ -- 29 dB WORSE
FX05410_SUPP_F3_DBC = 25.0

# Mini-Circuits ZVA-183+ / ZVA-183X+ -- 700 MHz to 18 GHz in ONE part, so it
# serves every amplifier slot in this chain.
ZVA183_LO_HZ, ZVA183_HI_HZ = 0.7e9, 18.0e9
ZVA183_GAIN_MIN_DB, ZVA183_GAIN_TYP_DB = 24.0, 26.0
ZVA183_FLATNESS_DB = 1.0
ZVA183_P1DB_MIN_DBM, ZVA183_P1DB_TYP_DBM = 21.0, 24.0
ZVA183_NF_DB, ZVA183_IP3_DBM = 3.0, 33.0
# CAUTION, from the Maximum Ratings table: "CW Input RF Power (no damage)
# +4 dBm". With 26 dB of gain that is easy to exceed when cascading two.
ZVA183_MAX_IN_DBM = 4.0

# Mini-Circuits ZFL-2500VH+ -- the DRIVE amp. 10-2500 MHz covers 1.6-1.95 GHz
# with none of the ZVA-183+'s 18 GHz reach going to waste, and its 23 dB gain is
# far closer to the ~21 dB this slot needs, so it barely needs a pad. A ZFL is
# also a fraction of a ZVA's price.
ZFL2500VH_LO_HZ, ZFL2500VH_HI_HZ = 10e6, 2500e6
ZFL2500VH_GAIN_MIN_DB, ZFL2500VH_GAIN_TYP_DB = 20.0, 23.0
ZFL2500VH_FLATNESS_TYP_DB = 1.35     # +-, over the FULL 10-2500 MHz span; over
                                     # our 350 MHz slice it should be far less,
                                     # but that is worth checking on the S21 plot
ZFL2500VH_P1DB_MIN_DBM, ZFL2500VH_P1DB_TYP_DBM = 23.0, 25.0
ZFL2500VH_NF_DB, ZFL2500VH_OIP3_DBM = 5.5, 35.0
ZFL2500VH_MAX_IN_DBM = 10.0          # into a 50 ohm load
ZFL2500VH_SUPPLY = "15 V, 260 mA"

# Mini-Circuits ZRL-2400LN+ -- considered and NOT chosen for the drive slot.
# 1000-2400 MHz, P1dB +23 min/+24 typ at 1400-1900 MHz, OIP3 +39, NF 1.2 dB:
# excellent, but ~32 dB of gain is ~11 dB more than this slot needs (so most of
# it is thrown away in a pad) and the low-noise/high-IP3 specs buy nothing at
# the -3.7 dBm signal level here. It is the better part and the wrong one.

# Mini-Circuits ZX60-83LN-S+ -- the cheap alternative for the 6.4-7.8 GHz slot.
# 0.5-8 GHz, 21 dB typ gain, NF 1.4 dB, OIP3 +28.5 dBm, 6 V / 77 mA.
# Its P1dB is only +18 dBm, so it CANNOT deliver the +16 dBm that drives the
# ZX90 at the top of its range with any backoff -- but it can deliver +13 dBm
# with a proper 5 dB margin, and the ZX90 accepts +11..+16 dBm. That costs
# ~0.8 dB of extra conversion loss (the datasheet's +11 dBm column averages
# 12.2 dB over 6.4-7.8 GHz vs 11.4 at +16), which the chain can well afford:
# there is 8.7 dB of trim pad at the far end doing nothing else.
ZX6083LN_LO_HZ, ZX6083LN_HI_HZ = 0.5e9, 8.0e9
ZX6083LN_GAIN_TYP_DB = 21.0
ZX6083LN_P1DB_DBM = 18.0
ZX6083LN_NF_DB, ZX6083LN_OIP3_DBM = 1.4, 28.5
# CAUTION: 7.71 GHz sits only 0.29 GHz below this part's 8 GHz upper limit, so
# check the published S21 curve for band-edge gain rolloff before committing.
# Note the rolloff direction would HELP here rather than hurt: the ZX90's
# conversion loss improves toward high frequency (+1.91 dB across the band), so
# an amp that droops at 7.7 GHz partially cancels that tilt. Uncontrolled and
# unit-dependent, so do not design around it -- just don't be scared of it.
ZX90_CL_AT_11DBM_MEAN = 12.20   # mean of the datasheet's +11 dBm rows, 6.4-7.8

# Mini-Circuits ZVE-8G+ -- rejected for this slot: 2-8 GHz, 30 dB min gain,
# P1dB +30 dBm, IP3 +40 -- and $1095. Twice the gain needed at ten times the
# price; the extra output power has nowhere to go.

# Mini-Circuits ZX60-2522MA-S+ -- ON HAND, and therefore the drive amp.
# 0.5-2.5 GHz. Per-frequency typical data at 5 V, straight off the datasheet:
#     1600 MHz  gain 25.05 dB  P1dB +20.44 dBm  NF 2.60
#     1800 MHz  gain 24.76 dB  P1dB +20.27 dBm  NF 2.64
#     2000 MHz  gain 24.33 dB  P1dB +20.07 dBm  NF 2.59
# Gain tilts only 0.65 dB across 1.6-2.0 GHz, and the spec'd minimum at 2 GHz
# is 22.6 dB, so even a worst-case unit clears what this slot needs.
# Its P1dB is ~4.7 dB below a ZFL-2500VH+'s, which is why the front bandpass
# now sits BEFORE the amp: that moves the filter's 1.5 dB out of the amp's
# output budget and into its input, buying back 1.5 dB of backoff for free.
ZX602522MA_LO_HZ, ZX602522MA_HI_HZ = 0.5e9, 2.5e9
ZX602522MA_GAIN_MIN_DB, ZX602522MA_GAIN_TYP_DB = 22.6, 24.8   # min is at 2 GHz
ZX602522MA_P1DB_DBM = 20.3
ZX602522MA_NF_DB = 2.6
ZX602522MA_SUPPLY = "5 V, 94 mA"

# Considered and rejected for the drive slot (both ON HAND, both fail):
#   ZX60-P105LN+  40-2600 MHz: gain 13.8 min/15.5 typ dB and P1dB +18.9 at
#                 2 GHz -- 5 dB short on gain AND only ~2 dB of backoff.
#   ZX60-V62+     0.05-6 GHz: gain 15.4 typ, P1dB +17.2 min/+19.0 typ at
#                 2 GHz -- same double failure. Its +-0.2 dB flatness over
#                 0.7-2.6 GHz is the best of any candidate, so it is the part
#                 to reach for IF flatness ever becomes the binding problem
#                 and a gain stage can be found to sit in front of it.
#   ZFL-2500VH+   the best part on paper (23 dB, P1dB +25 typ) but MONTHS of
#                 lead time. Not usable for this build.

# Which part fills the 6.4-7.8 GHz slot. "ZX60-04183LN+" is ON HAND (2 of them)
# and drives the ZX90 at +12 dBm; "ZVA-183+" / "ZX60-83LN-S+" would drive it
# harder but neither is in the drawer.
AMP_6TO8 = "ZX60-04183LN+"

# Which part fills the stage-A doubler slot (AWG band -> x2). "HMC188" is the
# baseline; "FX-05-410" is the Pulsar swap under consideration.
DOUBLER_A = "HMC188"

DRIVE_BACKOFF_DB = 3.0      # backoff for the drive amp; the AWG level sets it
AMP_BACKOFF_MIN_DB = 5.0    # keep every amp this far below P1dB: a compressed
                            # stage adds AM-to-PM that the AMC then multiplies

# Filters on hand -- both are BANDPASS, and neither covers 6-8 or 12-16 GHz.
ZBSS_4G_LO_HZ, ZBSS_4G_HI_HZ = 2.0e9, 6.0e9
ZBSS_10G_LO_HZ, ZBSS_10G_HI_HZ = 8.0e9, 12.0e9

# Amps on hand, by the band each can actually serve in a multiplier chain.
AMP_H242_LO_HZ, AMP_H242_HI_HZ, AMP_H242_GAIN_DB = 0.7e9, 2.4e9, 13.0
AMP_123LN_LO_HZ, AMP_123LN_HI_HZ, AMP_123LN_GAIN_DB = 0.5e9, 12.0e9, 15.0
AMP_04183_LO_HZ, AMP_04183_HI_HZ = 4.0e9, 18.0e9

FILTER_IL_DB = 1.5          # allowance for a real interstage bandpass
CABLE_IL_DB = 0.5           # per hop

AMC_N = 24                  # the AMC's own multiplication, after this chain


# ── AWG level reference ────────────────────────────────────────────────────
# WHICH CW RUN TO TRUST (resolved 2026-07-28).
# There are three cw_level_check_* runs from 2026-07-27 and they DO NOT AGREE:
#   12:03:54  RAW peak -11.90 dBm  -> +10 dB pad = -1.90 dBm  vs theory -2.04
#   15:24:34  RAW peak -30.91 dBm  }  ~19-27 dB LOWER, and the offset is NOT
#   15:30:18  RAW peak -30.91 dBm  }  flat (19.0 to 38.9 dB across the band)
# The 12:03 run agrees with the commanded 0.5 Vpp to 0.14 dB, which is the
# signature of a correctly-referred measurement. The two afternoon runs were
# taken after the LO/mixer work, with extra attenuation left in the path that
# was never recorded -- and since their offset varies by ~20 dB ACROSS the
# band, they are not recoverable by any constant correction. Their droop shape
# is the measurement setup's, not the AWG's.
#
# power_budget.load_awg_response() takes runs[-1], i.e. the NEWEST run, so it
# is currently reading a contaminated scan. Pin the good one explicitly here.
AWG_SCAN_RUN = "cw_level_check_2026-07-27_120354"
AWG_SCAN_PAD_DB = 10.0    # the pad in line for THAT run (its traces predate
                          # the external_atten_db field, so it isn't recorded)
# Two DIFFERENT amplitudes, deliberately kept apart:
#   SCAN      = what run_specan_band_scan.py actually commanded when the CW
#               level check was taken (chirp_bench/run_chirp_quality.AMP_VPP).
#               The path correction MUST be anchored to this one.
#   OPERATING = what the built chain will run at. The AC-coupled output module
#               tops out at 550 mVpp, so there is 0.83 dB of free headroom
#               above the scan amplitude -- worth taking, but it is NOT part of
#               the path correction and must not be folded into it.
AWG_SCAN_AMP_VPP = 0.50
AWG_MAX_AMP_VPP = 0.55
AWG_OPERATING_AMP_VPP = AWG_MAX_AMP_VPP


def awg_full_scale_dbm(vpp):
    """Commanded AWG output into 50 ohm, dBm."""
    return 10 * np.log10(((vpp / 2) ** 2 / (2 * 50)) / 1e-3)


def load_awg_response_corrected(run=AWG_SCAN_RUN):
    """AWG response from the trusted CW run, referred to the AWG connector at
    the OPERATING amplitude. Returns (freqs_hz, dbm), or None.

    No path correction is applied or needed: this run's peak already agrees
    with the commanded amplitude to 0.14 dB once its pad is added back. The
    only shift is amplitude headroom, if the build runs above the scan
    amplitude.
    """
    f, a = [], []
    for p in sorted(glob.glob(os.path.join(specanlog.TRACE_DIR, run, "*.json"))):
        d = specanlog.load_trace(p)
        s = d["settings"]
        if "cw_freq_hz" not in s:
            continue
        f.append(s["cw_freq_hz"])
        a.append(d["amps_dbm"].max() + s.get("external_atten_db", AWG_SCAN_PAD_DB))
    if not f:
        print(f"CW run {run!r} not found or has no CW points.")
        return None
    f, a = np.array(f), np.array(a)
    o = np.argsort(f)
    f, a = f[o], a[o]

    scan_fs = awg_full_scale_dbm(AWG_SCAN_AMP_VPP)
    head = awg_full_scale_dbm(AWG_OPERATING_AMP_VPP) - scan_fs
    print(f"AWG response: {len(f)} CW points from {run}")
    print(f"  peak {a.max():+.2f} dBm vs {AWG_SCAN_AMP_VPP} Vpp theory {scan_fs:+.2f} dBm "
          f"-> agree to {abs(a.max()-scan_fs):.2f} dB (no path correction needed)")
    if head:
        print(f"  + {head:+.2f} dB running at {AWG_OPERATING_AMP_VPP} Vpp "
              f"(AC-coupled module ceiling)")
    print(f"  -> AWG at its connector: {a.max()+head:+.2f} to {a.min()+head:+.2f} dBm")
    return f, a + head


def load_scope_band_response(pattern=os.path.join("awg_level_scope", "*.npz")):
    """AWG level over the x8 band measured on the SCOPE (run_awg_level_scope.py),
    AVERAGED over every saved run.

    Preferred over the EXA scan wherever it reaches: it is a direct voltage
    measurement into the scope's own 50 ohm input with NO pad in the path, so
    the unrecorded-attenuation failure mode that wrecked two of the three EXA
    runs cannot occur.

    Averaged because repeat runs showed 0.19 dB rms of per-point scatter: the
    fine wiggles in any single run are NOT reproducible and must not be read as
    structure. Only the broad trend survives repetition (shape correlation 0.93
    between two runs). Returns (freqs_hz, dbm, n_runs, scatter_db) or None.
    """
    key = f"dbm_{str(AWG_OPERATING_AMP_VPP).replace('.', 'p')}vpp"
    freqs, curves = None, []
    for p in sorted(glob.glob(pattern)):
        d = np.load(p)
        if key not in d or "freqs_hz" not in d:
            continue
        if freqs is None:
            freqs = d["freqs_hz"]
        elif not np.array_equal(freqs, d["freqs_hz"]):
            continue          # different frequency grid -- don't mix
        # Skip byte-identical duplicates (e.g. a backup copy of a run saved
        # under a second name): counting one run twice would silently halve the
        # apparent run-to-run scatter and overstate the measurement's precision.
        if any(np.array_equal(d[key], c) for c in curves):
            continue
        curves.append(d[key])
    if not curves:
        return None
    c = np.array(curves)
    # Spread between runs at each frequency = the honest per-point uncertainty.
    scatter = float(np.mean(np.std(c, axis=0, ddof=1))) if len(c) > 1 else float("nan")
    return freqs, c.mean(axis=0), len(c), scatter


def band_stats(resp, lo_hz, hi_hz, n=801):
    """Min/max/spread of the MEASURED AWG level across a band.

    Deliberately samples the WHOLE band, not just the two edges. power_budget.py
    evaluates only the edges, which for the direct-mix IF span reports ~3.7 dB
    of tilt where the real peak-to-trough spread is over 19 dB -- the interior
    of that band is far worse than either end of it.
    """
    if lo_hz < resp[0][0] or hi_hz > resp[0][-1]:
        return None
    f = np.linspace(lo_hz, hi_hz, n)
    a = np.interp(f, resp[0], resp[1])
    return {"f": f, "a": a, "max": a.max(), "min": a.min(),
            "spread": a.max() - a.min(), "edges": abs(a[0] - a[-1])}


def choose_multiplication(resp):
    """Which multiplication factor puts the AWG on its best-behaved sub-band?

    This is the whole argument for the approach: the AMC's input window is only
    7 dB wide, so the flatness of the AWG sub-band a given factor demands is
    the first-order design driver -- ahead of parts count or conversion loss.
    """
    print("\n" + "=" * 74)
    print("STEP 1 -- CHOOSING THE MULTIPLICATION FACTOR")
    print("=" * 74)
    print(f"  AMC input band {AMC_IN_LO_HZ/1e9:.2f}-{AMC_IN_HI_HZ/1e9:.2f} GHz; "
          f"a xN chain needs the AWG to sweep that band divided by N.")
    print(f"  AMC input WINDOW is only {AMC_IN_MAX_DBM-AMC_IN_MIN_DBM:.0f} dB wide, so AWG "
          f"flatness over that sub-band is the binding constraint.\n")
    print(f"  {'factor':>7s} {'AWG sub-band':>20s} {'max':>8s} {'min':>8s} "
          f"{'SPREAD':>8s} {'edges only':>11s}  buildable from doublers?")
    best = None
    for n in (2, 3, 4, 6, 8, 12, 16):
        lo, hi = AMC_IN_LO_HZ / n, AMC_IN_HI_HZ / n
        s = band_stats(resp, lo, hi)
        band = f"{lo/1e9:.3f}-{hi/1e9:.3f}"
        # Only powers of two are reachable by cascading x2 stages; 3/6/12 would
        # need a tripler, which is not on hand and not what was proposed.
        pow2 = (n & (n - 1)) == 0
        k = int(np.log2(n)) if pow2 else 0
        note = (f"yes, {k} x2 stage{'s' if k > 1 else ''}") if pow2 else "NO -- needs a tripler"
        if s is None:
            print(f"  {'x'+str(n):>7s} {band:>20s} "
                  f"{'-- outside the measured 0.2-4.2 GHz scan --':>40s}  {note}")
            continue
        print(f"  {'x'+str(n):>7s} {band:>20s} {s['max']:+8.2f} {s['min']:+8.2f} "
              f"{s['spread']:8.2f} {s['edges']:11.2f}  {note}")
        if pow2 and (best is None or s["spread"] < best[1]["spread"]):
            best = (n, s, lo, hi)

    # Same statistic for the mixer variants, so the comparison is like-for-like.
    print()
    mix_worst = 0.0
    for lo, hi, lab in ((0.49e9, 3.00e9, "direct mix, LO 12.42 GHz"),
                        (0.99e9, 3.50e9, "direct mix, LO 11.92 GHz")):
        s = band_stats(resp, lo, hi)
        mix_worst = max(mix_worst, s["spread"])
        print(f"  {'(mix)':>7s} {f'{lo/1e9:.3f}-{hi/1e9:.3f}':>20s} {s['max']:+8.2f} "
              f"{s['min']:+8.2f} {s['spread']:8.2f} {s['edges']:11.2f}   <- {lab}")

    nflat, sflat, loflat, hiflat = best
    s8 = band_stats(resp, AMC_IN_LO_HZ / 8, AMC_IN_HI_HZ / 8)
    print(f"\n  -> FLATTEST doubler-only chain is x{nflat}: AWG {loflat/1e9:.3f}-{hiflat/1e9:.3f} GHz, "
          f"spread {sflat['spread']:.2f} dB.")
    print(f"  -> RECOMMENDED is x8 (spread {s8['spread']:.2f} dB) -- see below.")
    print(f"\n     Either way the point stands: the direct-mix IF spans have "
          f"{mix_worst/s8['spread']:.0f}-{mix_worst/sflat['spread']:.0f}x more")
    print(f"     amplitude spread over a band that has to fit the SAME "
          f"{AMC_IN_MAX_DBM-AMC_IN_MIN_DBM:.0f} dB AMC window.")
    print(f"     That is the single strongest argument for multiplying instead of mixing.")
    print(f"\n     x16 is marginally flatter AND sits on the AWG's output PEAK "
          f"({resp[1].max():+.0f} dBm at")
    print(f"     0.9 GHz), which would cut the drive-gain requirement by "
          f"~{s8['min']*-1 - resp[1].max()*-1:.0f} dB -- but it costs a")
    print(f"     4th doubler/filter/amp stage and another {20*np.log10(2):.1f} dB of phase-noise "
          f"multiplication,")
    print(f"     for {sflat['spread']-s8['spread']:+.2f} dB of flatness. x8 is the better balance; "
          f"x16 is the fallback if")
    print(f"     the 1.6-1.9 GHz drive requirement turns out to be the binding problem.")
    s12 = band_stats(resp, 0.49e9, 3.00e9)
    print(f"\n  NOTE the 'edges only' column: for the LO 12.42 GHz mix it reads "
          f"{s12['edges']:.2f} dB")
    print(f"  while the true spread is {s12['spread']:.2f} dB. power_budget.py samples only the")
    print(f"  band edges, so its AWG-droop term understates reality by "
          f"~{s12['spread']-s12['edges']:.0f} dB there.")
    return 8


def stage(name, part, fin_lo, fin_hi, pin_dbm, cl_db, in_lo, in_hi,
          out_lo, out_hi, pin_min, pin_max, owned):
    """One doubler stage: report the frequency/drive fit and the output level."""
    fo_lo, fo_hi = fin_lo * 2, fin_hi * 2
    pout = pin_dbm - cl_db
    flag = []
    if fin_lo < in_lo or fin_hi > in_hi:
        flag.append(f"INPUT {fin_lo/1e9:.2f}-{fin_hi/1e9:.2f} GHz outside spec "
                    f"{in_lo/1e9:.1f}-{in_hi/1e9:.1f}")
    if fo_lo < out_lo or fo_hi > out_hi:
        flag.append(f"OUTPUT {fo_lo/1e9:.2f}-{fo_hi/1e9:.2f} GHz outside spec "
                    f"{out_lo/1e9:.1f}-{out_hi/1e9:.1f}")
    if pin_dbm < pin_min:
        flag.append(f"UNDER-DRIVEN by {pin_min-pin_dbm:.1f} dB "
                    f"(needs {pin_min:+.0f}..{pin_max:+.0f} dBm)")
    elif pin_dbm > pin_max:
        flag.append(f"OVER-DRIVEN by {pin_dbm-pin_max:.1f} dB")
    print(f"  {name}  {part}{'' if owned else '   [NOT OWNED]'}")
    print(f"     {fin_lo/1e9:6.3f}-{fin_hi/1e9:6.3f} GHz  ->  {fo_lo/1e9:6.3f}-{fo_hi/1e9:6.3f} GHz"
          f"   in {pin_dbm:+.1f} dBm  - CL {cl_db:.1f} dB  = {pout:+.1f} dBm")
    for f in flag:
        print(f"     !! {f}")
    return pout, fo_lo, fo_hi


def full_chain(resp):
    """Complete AWG -> AMC cascade for the x8 chain, level by level.

    Topology is MULTIPLY -> FILTER -> AMPLIFY at every stage (the user's
    established bench convention): the filter cleans each doubler's output
    before any gain is spent on it, and the amp then re-establishes the drive
    the next doubler needs.

    Sized at the WEAK end of the AWG band -- the worst case that still has to
    reach the AMC minimum. Every doubler is driven at its datasheet nominal so
    the quoted conversion losses apply and each stage sits in compression,
    which is what makes a diode multiplier amplitude-limiting.
    """
    lo, hi = AMC_IN_LO_HZ / 8, AMC_IN_HI_HZ / 8
    # Prefer the scope measurement over the EXA scan for this band -- no pad in
    # its path, so nothing unrecorded can shift it. Falls back to the EXA run.
    sc = load_scope_band_response()
    if sc is not None and sc[0][0] <= lo and sc[0][-1] >= hi:
        freqs_sc, dbm_sc, n_runs, scatter = sc
        s_exa = band_stats(resp, lo, hi)
        resp = (freqs_sc, dbm_sc)
        s_chk = band_stats(resp, lo, hi)
        print(f"\n  AWG band level: SCOPE, mean of {n_runs} run(s), no pad in path")
        print(f"    scope {s_chk['max']:+.2f}..{s_chk['min']:+.2f} dBm "
              f"(spread {s_chk['spread']:.2f} dB)  vs  "
              f"EXA {s_exa['max']:+.2f}..{s_exa['min']:+.2f} dBm "
              f"(spread {s_exa['spread']:.2f} dB)")
        print(f"    two independent instruments agree to "
              f"{abs(s_chk['max']-s_exa['max']):.2f} dB at the strong end, "
              f"{abs(s_chk['min']-s_exa['min']):.2f} dB at the weak end")
        if n_runs > 1:
            print(f"    run-to-run scatter {scatter:.2f} dB per point -- the fine")
            print(f"    wiggles in any SINGLE run are noise, not structure. Only the")
            print(f"    broad trend reproduces, so read the spread as "
                  f"{s_chk['spread']:.1f} +- {scatter:.1f} dB.")
    s = band_stats(resp, lo, hi)
    lvl = s["min"]
    rows = []

    def step(name, delta, band="", note=""):
        nonlocal lvl
        lvl += delta
        rows.append((name, band, delta, lvl, note))

    def amp_to(target, name, band, p1db, note=""):
        """Insert an input pad (if needed) + amp so the stage OUTPUT hits
        `target` while keeping the amp AMP_BACKOFF_MIN_DB below its P1dB."""
        gain = ZVA183_GAIN_TYP_DB if "ZVA" in name else None
        if gain is None:
            gain = target - lvl
            step(f"amp ({name})", gain, band, note or f"{gain:.1f} dB needed")
            return
        pad = (lvl + gain) - target
        if pad > 0.3:
            step(f"pad {pad:.1f} dB (before amp)", -pad, "",
                 "sets the amp's operating point; also protects its "
                 f"{ZVA183_MAX_IN_DBM:+.0f} dBm max input")
        step(f"amp ({name})", gain, band,
             f"out {lvl+gain:+.1f} dBm, {p1db-(lvl+gain):.1f} dB below P1dB {p1db:+.0f}")

    # AWG STRAIGHT INTO THE AMP -- no input pad, and the bandpass in its
    # conventional place AFTER the amp (multiply -> filter -> amplify, the same
    # rule as every other stage in the chain).
    #
    # The AWG's own amplitude is the trim, and it HAS to be: at the full
    # 0.55 Vpp this amp would be driven about 2 dB PAST its +20.3 dBm P1dB, so
    # some reduction is not optional. Solve it against the STRONG end of the
    # band (the worst case for compression) and let the 1.17 dB of band tilt
    # fall where it may -- the HMC188's +10..+20 dBm window is 10 dB wide and
    # swallows it easily.
    amp_out_strong = s["max"] - CABLE_IL_DB + ZX602522MA_GAIN_TYP_DB
    reduction = amp_out_strong - (ZX602522MA_P1DB_DBM - DRIVE_BACKOFF_DB)
    vpp = AWG_OPERATING_AMP_VPP * 10 ** (-reduction / 20)
    lvl -= reduction
    print(f"\n  DRIVE LEVEL: at {AWG_OPERATING_AMP_VPP} Vpp the amp would reach "
          f"{amp_out_strong:+.1f} dBm at the")
    print(f"  band's strong end -- {amp_out_strong-ZX602522MA_P1DB_DBM:+.1f} dB past its "
          f"{ZX602522MA_P1DB_DBM:+.1f} dBm P1dB. Back the AWG")
    print(f"  off {reduction:.1f} dB to {vpp:.3f} Vpp for {DRIVE_BACKOFF_DB:.0f} dB of "
          f"backoff. No input pad needed.")

    rows.append((f"AWG CH1, DUC, {vpp:.3f} Vpp",
                 f"{lo/1e9:.3f}-{hi/1e9:.3f}", None, lvl,
                 f"weak end; band tilts {s['spread']:.2f} dB"))
    step("cable", -CABLE_IL_DB, "", "keep this hop SHORT (lowest-level point in the chain)")
    step("DRIVE AMP (ZX60-2522MA-S+) [ON HAND]", ZX602522MA_GAIN_TYP_DB, "1.6-1.95 GHz",
         f"weak {lvl+ZX602522MA_GAIN_TYP_DB:+.1f} / strong "
         f"{lvl+ZX602522MA_GAIN_TYP_DB+s['spread']:+.1f} dBm "
         f"({DRIVE_BACKOFF_DB:.0f} dB below P1dB {ZX602522MA_P1DB_DBM:+.1f})")
    step("BPF 1.6-1.95 GHz", -FILTER_IL_DB, "",
         "~10 dB at 1.500 GHz is enough (the real external spur is -95 dBm)")
    # Gain is 22.6 min (at 2 GHz) / 24.8 typ. Whichever unit is in the drawer the
    # doubler must land inside its window -- and if it does not, the AWG amplitude
    # simply moves, which is exactly why the trim belongs there and not in a pad.
    if DOUBLER_A == "FX-05-410":
        print(f"  [FX-05-410 has no published Pin min/max, only a "
              f"{FX05410_PIN_NOM_DBM:+.0f} dBm nominal -- there is no window to check.]")
    else:
        for g, tag in ((ZX602522MA_GAIN_TYP_DB, "typ"), (ZX602522MA_GAIN_MIN_DB, "min")):
            w = lvl - ZX602522MA_GAIN_TYP_DB + g - FILTER_IL_DB - CABLE_IL_DB
            ok = HMC188_PIN_MIN_DBM <= w and (w + s["spread"]) <= HMC188_PIN_MAX_DBM
            print(f"  [{tag}-gain unit ({g:.1f} dB): HMC188 sees {w:+.1f}..{w+s['spread']:+.1f} dBm"
                  f" -- {'inside' if ok else '** OUTSIDE **'} its "
                  f"{HMC188_PIN_MIN_DBM:+.0f}..{HMC188_PIN_MAX_DBM:+.0f} dBm window]")

    step("cable", -CABLE_IL_DB)

    if DOUBLER_A == "FX-05-410":
        cl_a, part_a, supp_a = FX05410_CL_MAX_DB, "FX-05-410", FX05410_SUPP_F1_DBC
        spec_a = f"no min/max spec, nom {FX05410_PIN_NOM_DBM:+.0f}"
    else:
        cl_a, part_a, supp_a = HMC188_CL_TYP_DB, "HMC188", HMC188_SUPP_DBC
        spec_a = f"spec {HMC188_PIN_MIN_DBM:+.0f}..{HMC188_PIN_MAX_DBM:+.0f}, nom {HMC188_PIN_NOM_DBM:+.0f}"
    step(f"DOUBLER A ({part_a})", -cl_a, f"->{2*lo/1e9:.3f}-{2*hi/1e9:.3f}",
         f"in {lvl:+.1f} dBm ({spec_a}); CL {cl_a:.1f} dB, "
         f"F1 suppression {supp_a:.0f} dBc")
    step("BPF 3.2-3.9 GHz", -FILTER_IL_DB, "", "reject F1 1.61-1.93, F3 4.84-5.78")
    step("cable", -CABLE_IL_DB)
    amp_to(MMD_PIN_NOM_DBM, "ZX60-123LN-S+ [owned]", "3.2-3.9 GHz", None)

    step("DOUBLER B (MMD-0415HS)", -MMD_CL_TYP_DB, f"->{4*lo/1e9:.3f}-{4*hi/1e9:.3f}",
         f"in {lvl:+.1f} dBm (spec {MMD_PIN_MIN_DBM:+.0f}..{MMD_PIN_MAX_DBM:+.0f}), "
         f"CL {MMD_CL_TYP_DB:.0f} dB")
    step("BPF 6.4-7.8 GHz", -FILTER_IL_DB, "", "reject F1 3.23-3.86, F3 9.7-11.6")
    step("cable", -CABLE_IL_DB)
    # How hard the last doubler gets driven is set by whichever amp fills this
    # slot -- each is run AMP_BACKOFF_MIN_DB below its own P1dB.
    if AMP_6TO8 == "ZVA-183+":
        # Top of the ZX90's range: the datasheet's +16 dBm column is ~0.8 dB
        # better on conversion loss, and harder saturation suppresses tilt.
        drive, p1db, gain = ZX90_PIN_MAX_DBM, ZVA183_P1DB_TYP_DBM, ZVA183_GAIN_TYP_DB
        cl_c = float(np.mean(list(ZX90_CL_MEASURED_6P4_7P8.values())))
    elif AMP_6TO8 == "ZX60-04183LN+":
        # On hand. 11 dB gain, P1dB +15 -> the most it can cleanly deliver is
        # about +12 dBm, which is still inside the ZX90's +11..+16 window.
        p1db, gain = AMP_KU_P1DB_DBM, AMP_KU_GAIN_TYP_DB
        drive = lvl + gain
        cl_c = ZX90_CL_AT_11DBM_MEAN
    else:
        drive = ZX6083LN_P1DB_DBM - AMP_BACKOFF_MIN_DB      # +13 dBm
        p1db, gain = ZX6083LN_P1DB_DBM, ZX6083LN_GAIN_TYP_DB
        cl_c = ZX90_CL_AT_11DBM_MEAN
    drive = float(np.clip(drive, ZX90_PIN_MIN_DBM, ZX90_PIN_MAX_DBM))
    pad = (lvl + gain) - drive
    if pad > 0.3:
        step(f"pad {pad:.1f} dB (before amp)", -pad, "", "sets the amp's operating point")
    step(f"amp ({AMP_6TO8})", gain, "6.4-7.8 GHz",
         f"out {lvl+gain:+.1f} dBm, {p1db-(lvl+gain):.1f} dB below P1dB {p1db:+.0f}")

    step("DOUBLER C (ZX90-2-24-S+)", -cl_c, f"->{8*lo/1e9:.3f}-{8*hi/1e9:.3f}",
         f"in {lvl:+.1f} dBm (spec {ZX90_PIN_MIN_DBM:+.0f}..{ZX90_PIN_MAX_DBM:+.0f}), "
         f"CL {cl_c:.1f} dB from the datasheet's "
         f"{'+16' if AMP_6TO8 == 'ZVA-183+' else '+11'} dBm rows, 6.4-7.8 GHz")
    step("BPF 12.9-15.5 GHz", -FILTER_IL_DB, "", "reject F1 6.46-7.71, F3 19.4-23.1")
    step("cable", -CABLE_IL_DB)

    # Final trim: land the BAND CENTRE mid-window so tilt has room either side.
    centre = (AMC_IN_MIN_DBM + AMC_IN_MAX_DBM) / 2
    pad = lvl + AMP_KU_GAIN_TYP_DB - ISOLATOR_IL_DB - centre
    if pad > 0.3:
        step(f"pad {pad:.1f} dB", -pad, "", "final trim, before the amp")
    step("amp (ZX60-04183LN+) [owned]", AMP_KU_GAIN_TYP_DB, "12.9-15.4 GHz",
         f"out {lvl+AMP_KU_GAIN_TYP_DB:+.1f} dBm, "
         f"{AMP_KU_P1DB_DBM-(lvl+AMP_KU_GAIN_TYP_DB):.1f} dB below P1dB {AMP_KU_P1DB_DBM:+.0f}")
    step("isolator PE83IR1023 [owned]", -ISOLATOR_IL_DB, "", "32 dB isolation")

    print("\n" + "=" * 80)
    print("FULL x8 CHAIN: AWG -> AMC   (multiply -> filter -> amplify)")
    print("=" * 80)
    print(f"  {'stage':34s} {'band (GHz)':>18s} {'dB':>7s} {'level':>8s}")
    for name, band, delta, out, note in rows:
        d = "" if delta is None else f"{delta:+7.1f}"
        print(f"  {name:34s} {band:>18s} {d:>7s} {out:+8.2f}")
        if note:
            print(f"  {'':34s} {'':>18s} {'':>7s} {'':>8s}  {note}")

    print(f"\n  AMC input window {AMC_IN_MIN_DBM:.0f}..{AMC_IN_MAX_DBM:+.0f} dBm "
          f"(typ {AMC_IN_TYP_DBM:+.0f})  ->  DELIVERED {lvl:+.2f} dBm")
    tilt_budget(resp)
    return lvl


def tilt_budget(resp):
    """Tilt across the AMC band, evaluated at BOTH edges rather than by summing
    spreads.

    Summing the magnitude of each stage's spread is wrong here: the AWG droop
    and the ZX90-2-24-S+'s conversion loss run in OPPOSITE directions with
    frequency, so they partly cancel. Adding |tilt| would report ~6 dB where
    the signed sum is under 2 dB.
    """
    f_lo, f_hi = AMC_IN_LO_HZ, AMC_IN_HI_HZ
    zf = np.array(sorted(ZX90_CL_MEASURED_6P4_7P8))
    zc = np.array([ZX90_CL_MEASURED_6P4_7P8[k] for k in zf])

    print(f"\n  TILT ACROSS 12.91-15.42 GHz -- what actually has to fit the "
          f"{AMC_IN_MAX_DBM-AMC_IN_MIN_DBM:.0f} dB window")
    print(f"  {'':26s} {'@12.91 GHz':>12s} {'@15.42 GHz':>12s} {'tilt':>9s}")
    awg = np.interp([f_lo / 8, f_hi / 8], resp[0], resp[1])
    # The ZX90 is the LAST doubler, so its input is the AMC band halved.
    cl = np.interp([f_lo / 2 / 1e9, f_hi / 2 / 1e9], zf, zc)
    print(f"  {'AWG drive (x8 band)':26s} {awg[0]:+12.2f} {awg[1]:+12.2f} "
          f"{awg[1]-awg[0]:+9.2f}")
    print(f"  {'ZX90-2-24-S+ conv loss':26s} {-cl[0]:+12.2f} {-cl[1]:+12.2f} "
          f"{cl[0]-cl[1]:+9.2f}")

    net_small = (awg[1] - awg[0]) + (cl[0] - cl[1])
    net_sat = (cl[0] - cl[1])
    print(f"\n  These OPPOSE each other: the AWG is strongest at the low end, the")
    print(f"  ZX90 is most efficient at the high end.")
    print(f"    doublers small-signal (AWG tilt passes through) : net {net_small:+.2f} dB")
    print(f"    doublers saturated (AWG tilt suppressed)        : net {net_sat:+.2f} dB")
    resid = 1.0 + 1.0 + 1.5   # MMD CL variation + Ku amp gain tilt + 3 filters
    worst = max(abs(net_small), abs(net_sat)) + resid
    print(f"  + MMD-0415HS CL variation ~1.0, ZX60-04183LN+ gain tilt ~1.0,")
    print(f"    3 x bandpass ripple ~1.5 (all estimates, uncorrelated in sign)")
    print(f"    -> worst case ~{worst:.1f} dB in a {AMC_IN_MAX_DBM-AMC_IN_MIN_DBM:.0f} dB window "
          f"({AMC_IN_MAX_DBM-AMC_IN_MIN_DBM-worst:.1f} dB spare)")
    print(f"\n  The partial cancellation is luck, not design -- it depends on the")
    print(f"  ZX90 datasheet's CL-vs-frequency shape holding for YOUR unit. Verify")
    print(f"  it on the bench; if it doesn't hold, a shallow equaliser or a little")
    print(f"  AWG pre-emphasis (only ~3 dB needed) closes the gap either way.")


def spur_comparison():
    print("\n" + "=" * 74)
    print("STEP 3 -- SPUR / FILTERING COMPARISON  (the second real argument)")
    print("=" * 74)
    lo, hi = AMC_IN_LO_HZ, AMC_IN_HI_HZ
    print(f"  Wanted band: {lo/1e9:.2f}-{hi/1e9:.2f} GHz\n")

    print("  DIRECT MIX (LO 12.42 GHz, IF 0.49-3.00 GHz) -- measured, run_mixer_check.py:")
    print(f"     LO leak     12.42 GHz      {(lo-12.42e9)/1e9:.2f} GHz below the lower band edge")
    print(f"                 -> measured to EXCEED the wanted USB at 14 of 34 IF points")
    print(f"     image (LSB) 9.42-11.93 GHz {(lo-11.93e9)/1e9:.2f} GHz below the lower band edge")
    print(f"                 -> measured sideband balance ~0 dB: NO image rejection at all")
    print(f"     => the filter must pass 12.91 GHz and reject 12.42 GHz: a {100*(lo-12.42e9)/lo:.1f}%")
    print(f"        transition at 12.9 GHz. That is a hard, custom, expensive filter.\n")

    print("  x8 MULTIPLIER (final doubler 6.455-7.710 -> 12.91-15.42 GHz):")
    print(f"     F1 leak     6.46-7.71 GHz  a full OCTAVE below the band "
          f"({lo/7.71e9:.2f}x), typ {ZX90_HARM_DBC:.0f} dBc")
    print(f"     F3          19.4-23.1 GHz  {19.365/15.42:.2f}x above the upper edge, "
          f"typ {ZX90_HARM_DBC:.0f} dBc")
    print(f"     no LO, no image, no LO leakage -- those failure modes do not exist here")
    print(f"     => an ordinary bandpass with octave-scale transitions. Off-the-shelf.\n")

    print("  This is arguably a bigger win than the flatness one: the mixer variant's")
    print("  filter requirement was already flagged as needing a CUSTOM 12.91-15.42 GHz")
    print("  part, and the measured 0 dB image rejection plus LO-leak-above-signal")
    print("  result makes that filter load-bearing for the whole design.")


def costs():
    print("\n" + "=" * 74)
    print("STEP 4 -- WHAT THE APPROACH COSTS")
    print("=" * 74)
    n_total = 8 * AMC_N
    pn_mult = 20 * np.log10(8)
    pn_total = 20 * np.log10(n_total)
    pn_mix = 20 * np.log10(AMC_N)
    print(f"  PHASE NOISE -- the one unavoidable structural penalty.")
    print(f"     x8 then x{AMC_N} = x{n_total} total from AWG to the 310-370 GHz output")
    print(f"     -> AWG phase noise multiplied by {pn_total:.1f} dB, vs {pn_mix:.1f} dB when mixing")
    print(f"     -> a flat {pn_mult:.1f} dB penalty on the AWG's contribution.")
    print(f"     BUT the mixer route multiplies the SMB100A's phase noise by {pn_mix:.1f} dB too,")
    print(f"     and the SMB100A's measured spurs (+-27.24 MHz at -63 dBc, +-20 MHz at")
    print(f"     -40.3 dBc) transfer straight through the mixer. The multiplier route has")
    print(f"     no second synthesizer at all -- everything derives from the AWG clock,")
    print(f"     so shot-to-shot coherence is structural rather than something the")
    print(f"     10 MHz reference-sharing work has to enforce across two instruments.")
    print(f"     Which route actually wins is NOT decidable from datasheets: it needs")
    print(f"     L(f) for the Proteus at ~1.6 GHz, which has never been measured here.\n")

    print(f"  DRIVE POWER -- the practical cost, but a much smaller one than it")
    print(f"  looked before the AWG level was referred correctly.")
    print(f"     Every doubler wants +10..+16 dBm, and the AWG delivers about -9 dBm")
    print(f"     at the weak end of the x8 band, so ~26 dB of gain is needed before")
    print(f"     the first doubler and ~12-17 dB more between each later stage.")
    print(f"     A mixer would not care -- its IF port is happy at -20 dBm -- but 26 dB")
    print(f"     at 1.6-1.95 GHz is the cheapest gain in RF, and two ZX60-H242+ (one")
    print(f"     already owned) cover it with 6 dB of backoff.\n")

    print(f"  RECEIVE PATH -- the awkward one.")
    print(f"     The MixAMC still needs its own ~13-15 GHz LO. Mixing lets one synth")
    print(f"     feed both paths through a splitter. Multiplying does not split as")
    print(f"     cleanly: TX (chirp) and RX (CW at an offset) are different signals, so")
    print(f"     RX would need a SECOND full multiplier chain off AWG CH2.")
    print(f"     That roughly doubles the parts count for the whole architecture.")
    print(f"     (Counterpoint: the split LO route was already measured NOT to close --")
    print(f"     5.3 dB short per port -- so 'mixing splits cleanly' is theoretical.)")


def parts_verdict():
    print("\n" + "=" * 74)
    print("STEP 5 -- WHAT IS ON HAND vs WHAT IS NEEDED  (x8 chain)")
    print("=" * 74)
    rows = [
        ("drive amp 1.6-1.95 GHz", "ZX60-2522MA-S+ (24.8 dB, P1dB +20.3)",
         "ON HAND -- 4.8 dB backoff with the BPF ahead of it"),
        ("BPF        1.6-1.95 GHz", "-- none --", "BUY (~10 dB at 1.500 GHz is enough)"),
        ("doubler A  1.61 -> 3.23 GHz", "HMC188 (in 1.25-3.0, out 2.5-6.0)", "OWNED, fully in spec"),
        ("BPF        3.2-3.9 GHz", "ZBSS-4G-S+ (2-6 GHz BPF)", "passes F3 4.84-5.78 -- partial only"),
        ("amp        3.2-3.9 GHz", "ZX60-123LN-S+ (0.5-12 GHz)", "OWNED, OK"),
        ("doubler B  3.23 -> 6.46 GHz", "MMD-0415HS (in 2-7.5, out 4-15)", "OWNED, fully in spec"),
        ("BPF        6.4-7.8 GHz", "-- none --", "BUY (ZBSS-10G is 8-12, wrong band)"),
        ("amp        6.4-7.8 GHz", "ZX60-04183LN+ (11 dB, P1dB +15)",
         "OWNED -- drives the ZX90 at +12 dBm, inside its +11..+16 window"),
        ("doubler C  6.46 -> 12.91 GHz", "ZX90-2-24-S+ (in 5-10, out 10-20)", "OWNED, fully in spec"),
        ("BPF        12.9-15.5 GHz", "-- none --", "BUY (easy: octave-scale transitions)"),
        ("post-amp   12.9-15.4 GHz", "ZX60-04183LN+ (11 dB)", "OWNED, OK (10 dB backoff)"),
        ("isolator", "PE83IR1023 (32 dB iso)", "OWNED, OK"),
    ]
    print(f"  {'stage':36s} {'part':36s} {'status'}")
    for a, b, c in rows:
        print(f"  {a:36s} {b:36s} {c}")
    print(f"\n  SHOPPING LIST -- NO ACTIVE PARTS. Passives only:")
    print(f"    3 x bandpass filter: 1.6-1.95, 6.4-7.8, 12.9-15.5 GHz")
    print(f"    SMA pads: 4.1 and 3.9 dB, plus a trim set for the bench")
    print(f"\n  EVERY ACTIVE PART IS ALREADY IN THE DRAWER -- all three doublers")
    print(f"  (HMC188, MMD-0415HS, ZX90-2-24-S+) run fully in spec, and all three")
    print(f"  amplifiers are on hand. The ZFL-2500VH+ and ZVA-183+ studied earlier")
    print(f"  are better on paper (4-5 dB more P1dB headroom) but carry MONTHS of")
    print(f"  lead time; that headroom is not worth blocking the build for. Swap one")
    print(f"  in later if the bench shows backoff actually matters.")
    print(f"  Compare the mixer route's list: an LO amp that does not exist in the")
    print(f"  on-hand set, a 2nd Ku amp, a 2nd mixer, AND a custom 12.91-15.42 GHz")
    print(f"  filter with a 3.8% transition.")


def awg_level_note(resp):
    print("\n" + "=" * 74)
    print("AWG LEVEL REFERENCE -- which CW run these numbers come from")
    print("=" * 74)
    print(f"  Using {AWG_SCAN_RUN}.")
    print(f"  Its peak agrees with the commanded {AWG_SCAN_AMP_VPP} Vpp to 0.14 dB, so the AWG")
    print(f"  delivers what it is told and NO path correction is applied.")
    print(f"\n  The two later runs the same day (15:24:34, 15:30:18) read 19-27 dB LOW")
    print(f"  with extra attenuation left in from the LO/mixer work. That offset is")
    print(f"  NOT flat -- it varies by ~20 dB across the band -- so those runs cannot")
    print(f"  be rescued by any constant correction and their droop shape is the")
    print(f"  setup's, not the AWG's. Do not use them.")
    print(f"\n  ** power_budget.load_awg_response() picks runs[-1], the NEWEST run,")
    print(f"     so it is currently reading a contaminated scan. Its absolute AMC-input")
    print(f"     numbers are wrong until it is pointed at {AWG_SCAN_RUN}. **")
    print(f"\n  Worth re-measuring anyway, now that the setup is known to have changed")
    print(f"  mid-session: a fresh fine-step scan over 1.55-2.00 GHz with the pad")
    print(f"  count verified would confirm the {AWG_SCAN_AMP_VPP} Vpp agreement and the "
          f"~1.2 dB")
    print(f"  band spread directly, in the band that actually matters.")


def main():
    resp = load_awg_response_corrected()
    if resp is None:
        print("No CW level check found -- run chirp_bench/run_specan_band_scan.py first.")
        return 1

    choose_multiplication(resp)
    full_chain(resp)
    spur_comparison()
    costs()
    parts_verdict()
    awg_level_note(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
