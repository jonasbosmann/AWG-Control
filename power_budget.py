"""Transmit-chain power budget for the CP-FTMW spectrometer, computed from
REAL measured AWG output (chirp_bench/run_specan_band_scan.py's CW level
check) plus datasheet numbers transcribed from the PDFs in
"C:\\Users\\Admin\\Documents\\manuals MW parts".

Answers the gating question before anything gets cabled up: does the chain
actually land the AMC inside its 0..+7 dBm input window, across the whole
chirp band?

Run:  python power_budget.py

The AWG term is not a guess -- it reads the most recent CW level check
(absolute dBm at the AWG output, external pad already corrected out) and
uses the real measured level at each IF frequency, including the ~7 dB
droop, rather than assuming a flat source.
"""
import glob
import os
import sys

import numpy as np

import specanlog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirp_bench"))
from run_specan_band_scan import EXTERNAL_ATTEN_DB

# ── datasheet constants (transcribed 2026-07-27 from the PDFs) ─────

# ACST 1211B-24AL AMC, p.1 "Technical Specification"
AMC_IN_MIN_DBM, AMC_IN_TYP_DBM, AMC_IN_MAX_DBM = 0.0, +5.0, +7.0
AMC_IN_LO_HZ, AMC_IN_HI_HZ = 12.91e9, 15.42e9

# Mini-Circuits ZMDB-24H-K+ mixer, Level 15.
MIXER_LO_DRIVE_DBM = +15.0        # what "Level 15" requires
MIXER_CL_SPEC_TYP_DB = 8.5        # spec table, typ
MIXER_CL_SPEC_MAX_DB = 10.8       # spec table, max
# Typical-performance table at 13/14/15 GHz -- BUT the table footnote says
# "Conversion loss at 30 MHz IF. Increases with IF frequency." Our IF is
# 1.0-3.5 GHz, three orders of magnitude beyond that test condition, so
# these are a FLOOR, not an estimate.
MIXER_CL_TABLE_13_15GHZ_DB = 7.2
MIXER_LR_ISOLATION_DB = 25.6      # worst of 13/14/15 GHz rows -> LO leakage
MIXER_RF_P1DB_DBM = +10.0

# Pasternack PE83IR1023 isolator, 8-18 GHz (dual junction)
ISOLATOR_IL_DB = 1.4              # datasheet maximum
ISOLATOR_ISO_DB = 32.0
# Pasternack PE8304 isolator, 8-18 GHz (single junction) -- the SECOND
# isolator on hand: lower loss but half the isolation, so it belongs on the
# less reflection-sensitive path.
ISOLATOR2_IL_DB = 0.6
ISOLATOR2_ISO_DB = 16.0

# Mini-Circuits ZC2PD-06263-S+ 2-way power divider, 6-26.5 GHz -- the part
# that lets ONE synth feed both the TX mixer and the RX MixAMC LO.
SPLITTER_SPLIT_DB = 3.0           # unavoidable 2-way power division
SPLITTER_IL_DB = 0.7              # typ insertion loss on top of the split
SPLITTER_ISO_DB = 26.0
SPLITTER_FREQ_LO_HZ, SPLITTER_FREQ_HI_HZ = 6.0e9, 26.5e9

# Mini-Circuits ZFRSC-4-842+ 4-way RESISTIVE splitter: DC-8400 MHz only, so
# it cannot be used anywhere in the ~12 GHz LO distribution. IF side only.
SPLITTER4_FREQ_HI_HZ = 8.4e9

# ACST MixAMC WR2.8 250-400 GHz, N=x24, DSB conversion loss 10 dB typ -- the
# FID receiver. Quote A-241209-2 confirms the part but gives NO electrical
# spec: its required LO drive power and LO frequency range are UNKNOWN.
# x24 into 250-400 GHz implies an LO somewhere around 10.4-16.7 GHz.
MIXAMC_LO_DRIVE_DBM = None        # <- unknown; ask ACST
MIXAMC_N = 24

# Mini-Circuits ZX60-04183LN+, 4-18 GHz (the only on-hand amp covering 13-15 GHz)
AMP_KU_GAIN_TYP_DB, AMP_KU_GAIN_MIN_DB = 11.0, 9.0   # 12-18 GHz rows
AMP_KU_P1DB_DBM = +15.0

# Mini-Circuits ZX60-123LN-S+, 0.5-12 GHz (LO booster)
AMP_LO_GAIN_12GHZ_DB = 14.4
AMP_LO_P1DB_12GHZ_DBM = +13.4
AMP_LO_P1DB_10GHZ_DBM = +14.8
AMP_LO_FREQ_MAX_HZ = 12.0e9

# Mini-Circuits ZX60-H242+ -- 700-2400 MHz ONLY, so it cannot touch the
# 13-15 GHz path; usable only on the IF side and only below 2.4 GHz.
AMP_IF_FREQ_LO_HZ, AMP_IF_FREQ_HI_HZ = 700e6, 2400e6
AMP_IF_P1DB_DBM = +23.0

# R&S SMB100A test LO (firmware-enforced ceiling, confirmed on the bench)
SYNTH_MAX_DBM = +13.0

# Practical rule of thumb: to keep an upconverting mixer linear, hold the
# IF drive well below the LO level. Used only to flag over-driving.
IF_BELOW_LO_DB = 8.0


def load_awg_response():
    """Measured AWG output vs frequency (dBm, pad-corrected) from the most
    recent CW level check. Returns (freqs_hz, dbm) or None."""
    runs = sorted(glob.glob(os.path.join(specanlog.TRACE_DIR, "cw_level_check_*")))
    if not runs:
        return None
    files = sorted(glob.glob(os.path.join(runs[-1], "*.json")))
    f, a = [], []
    missing_pad = 0
    for p in files:
        d = specanlog.load_trace(p)
        s = d["settings"]
        if "cw_freq_hz" not in s:
            continue
        f.append(s["cw_freq_hz"])
        # Traces are saved RAW; the external pad must be added back to refer
        # levels to the AWG output. Traces taken before external_atten_db was
        # recorded don't carry it -- fall back to the CURRENT configured pad
        # rather than 0.0, since defaulting to zero silently under-reports the
        # source by the full pad value (10 dB) and quietly wrecks the budget.
        if "external_atten_db" in s:
            pad = s["external_atten_db"]
        else:
            pad = EXTERNAL_ATTEN_DB
            missing_pad += 1
        a.append(d["amps_dbm"].max() + pad)
    if not f:
        return None
    f, a = np.array(f), np.array(a)
    o = np.argsort(f)
    print(f"AWG response: {len(f)} CW points from {os.path.basename(runs[-1])}")
    if missing_pad:
        print(f"  NOTE: {missing_pad} trace(s) predate the external_atten_db field; "
              f"assumed the currently configured {EXTERNAL_ATTEN_DB:.0f} dB pad was "
              f"in place. Re-run the CW check to record it explicitly.")
    return f[o], a[o]


def awg_dbm_at(resp, freqs):
    return np.interp(freqs, resp[0], resp[1])


def load_measured_conv_loss():
    """MEASURED conversion loss vs IF from the newest run_mixer_check.py run.

    This replaces the datasheet figure, which is quoted at 30 MHz IF with an
    explicit note that loss rises with IF -- the single unknown that
    previously made this budget uncloseable. Returns (if_hz, cl_db, fit) or
    None. `fit` is a linear least-squares (slope, intercept): individual
    points carry ~0.9 dB rms of mismatch ripple because the measurement was
    taken with NO isolator, so the trend is the trustworthy part, not any
    single point."""
    import re
    runs = sorted(glob.glob(os.path.join(specanlog.TRACE_DIR, "mixer_check_*")))
    if not runs:
        return None
    ref = load_awg_response()
    if ref is None:
        return None
    f, cl = [], []
    for p in sorted(glob.glob(os.path.join(runs[-1], "*usb_*.json"))):
        g = re.search(r"usb_(\d+)p(\d+)GHz", os.path.basename(p))
        if not g:
            continue
        f_if = float(f"{g.group(1)}.{g.group(2)}") * 1e9
        d = specanlog.load_trace(p)
        usb = d["amps_dbm"].max() + d["settings"].get("external_atten_db", EXTERNAL_ATTEN_DB)
        f.append(f_if)
        cl.append(float(np.interp(f_if, ref[0], ref[1])) - usb)
    if not f:
        return None
    f, cl = np.array(f), np.array(cl)
    o = np.argsort(f)
    f, cl = f[o], cl[o]
    fit = np.polyfit(f / 1e9, cl, 1)
    print(f"Measured conversion loss: {len(f)} points from {os.path.basename(runs[-1])}")
    print(f"  {cl.min():.2f}-{cl.max():.2f} dB, trend {fit[0]:+.2f} dB/GHz "
          f"(datasheet 8.5 typ AT 30 MHz IF; this fit -> {np.polyval(fit,0.03):.2f} dB there)")
    return f, cl, fit


def check_lo_chain(split=True, lo_hz=11.92e9):
    """LO distribution: SMB100A -> ZX60-123LN-S+ -> [2-way splitter] -> mixer(s).

    split=True models feeding BOTH the TX mixer and the RX MixAMC LO from one
    synth via the ZC2PD-06263-S+, which is what a full transmit+receive chain
    needs. Returns the per-port LO power actually available."""
    print("\n" + "=" * 72)
    print(f"LO DISTRIBUTION  (LO {lo_hz/1e9:.2f} GHz, "
          f"{'SPLIT 2 ways for TX + RX' if split else 'single path'})")
    print("=" * 72)
    print(f"  SMB100A ceiling             {SYNTH_MAX_DBM:+.1f} dBm")
    print(f"  ZX60-123LN-S+  range 0.5-{AMP_LO_FREQ_MAX_HZ/1e9:.0f} GHz, "
          f"P1dB {AMP_LO_P1DB_12GHZ_DBM:+.1f} dBm @12 GHz")
    if lo_hz > AMP_LO_FREQ_MAX_HZ:
        print(f"  !! LO {lo_hz/1e9:.2f} GHz is ABOVE the booster's {AMP_LO_FREQ_MAX_HZ/1e9:.0f} GHz "
              f"limit -- no on-hand amp covers it")
    else:
        print(f"     (LO {lo_hz/1e9:.2f} GHz sits at the very top of its range)")

    lvl = AMP_LO_P1DB_12GHZ_DBM
    print(f"\n  booster output (at P1dB)    {lvl:+.1f} dBm")
    if split:
        if not (SPLITTER_FREQ_LO_HZ <= lo_hz <= SPLITTER_FREQ_HI_HZ):
            print(f"  !! splitter range is {SPLITTER_FREQ_LO_HZ/1e9:.0f}-"
                  f"{SPLITTER_FREQ_HI_HZ/1e9:.1f} GHz -- LO outside it")
        lvl -= SPLITTER_SPLIT_DB + SPLITTER_IL_DB
        print(f"  - ZC2PD 2-way split         -{SPLITTER_SPLIT_DB:.1f} dB")
        print(f"  - ZC2PD insertion loss      -{SPLITTER_IL_DB:.1f} dB")
        print(f"  = per-port LO              {lvl:+.1f} dBm  (each of TX / RX)")

    deficit = MIXER_LO_DRIVE_DBM - lvl
    print(f"\n  ZMDB-24H-K+ needs           {MIXER_LO_DRIVE_DBM:+.1f} dBm (Level 15)")
    if deficit > 0:
        print(f"  -> LO DEFICIT {deficit:.1f} dB per mixer port")
        print(f"     and that assumes running the booster AT P1dB (compressed);")
        print(f"     a clean LO wants several dB more back-off, so the real")
        print(f"     deficit is worse. Under-driving raises conversion loss")
        print(f"     ~1-3 dB and degrades IP3/linearity on EVERY path fed here.")
    else:
        print(f"  -> OK, {-deficit:.1f} dB of margin")
    return lvl


def lo_amp_requirement(lo_hz=11.92e9, cable_loss_db=1.0, backoff_db=4.0,
                        synth_levels=(5.0, 8.0, 10.0, 13.0)):
    """SPLIT-FIRST LO distribution -- synth -> 2-way splitter -> one amp per
    branch -> mixer LO port. Computes what each branch amplifier must do.

    Splitting before amplifying (rather than after) is the better topology
    here: each amp then only has to reach the mixer's +15 dBm on its own,
    instead of one amp having to reach +18.7 dBm to feed both ports through
    the split. It also decouples the branches -- an amp compressing on one
    path can't pull the other down.

    backoff_db: how far below P1dB the amp should actually run. An LO driven
    into compression generates harmonics and raises the mixer's spurious
    output, so the amp is sized to deliver +15 dBm while sitting this far
    below its own P1dB -- i.e. required P1dB = 15 + cable + backoff.
    """
    print("\n" + "=" * 72)
    print(f"LO BRANCH AMPLIFIER REQUIREMENT  (split first, then amplify)")
    print("=" * 72)
    print(f"  topology: SMB100A -> ZC2PD-06263-S+ (2-way) -> [AMP] -> mixer LO")
    print(f"  target at each mixer LO port : {MIXER_LO_DRIVE_DBM:+.1f} dBm (Level 15)")
    print(f"  splitter                     : -{SPLITTER_SPLIT_DB:.1f} dB split "
          f"-{SPLITTER_IL_DB:.1f} dB insertion = -{SPLITTER_SPLIT_DB+SPLITTER_IL_DB:.1f} dB")
    print(f"  cable/connector allowance    : {cable_loss_db:.1f} dB per hop "
          f"(synth->splitter->amp->mixer)")

    # Amp must deliver the mixer drive plus the loss of the last hop.
    amp_out_needed = MIXER_LO_DRIVE_DBM + cable_loss_db
    p1db_needed = amp_out_needed + backoff_db
    print(f"\n  amp OUTPUT needed            : {amp_out_needed:+.1f} dBm "
          f"(+{cable_loss_db:.1f} dB for the amp->mixer hop)")
    print(f"  amp P1dB needed              : {p1db_needed:+.1f} dBm "
          f"(running {backoff_db:.0f} dB backed off for a clean, harmonic-free LO)")

    print(f"\n  {'synth level':>12s} {'after split':>12s} {'at amp in':>11s} "
          f"{'GAIN NEEDED':>12s}")
    for s in synth_levels:
        after_split = s - SPLITTER_SPLIT_DB - SPLITTER_IL_DB - cable_loss_db
        at_amp_in = after_split - cable_loss_db
        gain = amp_out_needed - at_amp_in
        flag = "" if s <= SYNTH_MAX_DBM else "  (ABOVE SYNTH MAX)"
        print(f"  {s:+11.1f} {after_split:+12.1f} {at_amp_in:+11.1f} "
              f"{gain:+12.1f} dB{flag}")

    print(f"\n  NOTE the synth's own spectral purity is better below its +{SYNTH_MAX_DBM:.0f} dBm")
    print(f"  ceiling, so driving it at +5..+8 dBm and taking more gain in the")
    print(f"  branch amp is preferable to running it flat out (see the SMB100A")
    print(f"  spur measurements in the project notes).")

    print(f"\n  -> SPEC TO SHOP FOR (x2, one per branch):")
    lo_gain = amp_out_needed - (8.0 - SPLITTER_SPLIT_DB - SPLITTER_IL_DB - 2*cable_loss_db)
    hi_gain = amp_out_needed - (5.0 - SPLITTER_SPLIT_DB - SPLITTER_IL_DB - 2*cable_loss_db)
    print(f"     frequency : must cover {lo_hz/1e9:.2f} GHz "
          f"(wider if the LO is ever stepped)")
    print(f"     P1dB      : >= {p1db_needed:+.0f} dBm")
    print(f"     gain      : ~{lo_gain:.0f}-{hi_gain:.0f} dB for a +8..+5 dBm synth drive")
    print(f"\n  The on-hand ZX60-123LN-S+ CANNOT do this job: its P1dB is only")
    print(f"  {AMP_LO_P1DB_12GHZ_DBM:+.1f} dBm at 12 GHz, i.e. it cannot even REACH "
          f"{MIXER_LO_DRIVE_DBM:+.0f} dBm,")
    print(f"  let alone with back-off -- {p1db_needed-AMP_LO_P1DB_12GHZ_DBM:.1f} dB short of "
          f"what's required. Gain is not")
    print(f"  the problem; output power capability is.")
    return p1db_needed


def zx60_04183_as_lo_driver(cable_loss_db=1.0):
    """Can the ON-HAND ZX60-04183LN+ serve as the post-split LO branch amp,
    and if so what synth power does that need?

    The catch: this amp's output P1dB is +15.0 dBm at 12-18 GHz -- exactly
    the mixer's Level-15 requirement, with nothing to spare. So the answer
    depends entirely on the loss between amp and mixer, and on whether
    running an LO amp at full compression is acceptable.
    """
    print("\n" + "=" * 72)
    print("CAN THE ZX60-04183LN+ DRIVE THE MIXER LO AFTER THE SPLIT?")
    print("=" * 72)
    print(f"  ZX60-04183LN+ : 4-18 GHz (covers 11.92 GHz OK)")
    print(f"                  gain {AMP_KU_GAIN_MIN_DB:.0f} min / {AMP_KU_GAIN_TYP_DB:.0f} typ dB "
          f"at 12-18 GHz")
    print(f"                  output P1dB {AMP_KU_P1DB_DBM:+.1f} dBm")
    print(f"  mixer needs     {MIXER_LO_DRIVE_DBM:+.1f} dBm  -> the amp's P1dB is EXACTLY "
          f"the target, 0 dB spare")

    for cl, tag in ((0.0, "amp bolted DIRECTLY to the mixer (no cable)"),
                     (cable_loss_db, f"{cable_loss_db:.1f} dB cable amp->mixer")):
        need_out = MIXER_LO_DRIVE_DBM + cl
        print(f"\n  --- {tag} ---")
        print(f"  amp output required : {need_out:+.1f} dBm   "
              f"(P1dB {AMP_KU_P1DB_DBM:+.1f} dBm)")
        if need_out > AMP_KU_P1DB_DBM:
            print(f"  -> IMPOSSIBLE: {need_out-AMP_KU_P1DB_DBM:.1f} dB ABOVE this amp's P1dB. "
                  f"It physically")
            print(f"     cannot deliver that power at any drive level.")
            continue
        print(f"  -> reachable, but ONLY by running AT P1dB (1 dB compressed).")
        # At P1dB the effective gain is 1 dB below the small-signal figure.
        for g, gtag in ((AMP_KU_GAIN_TYP_DB, "typ"), (AMP_KU_GAIN_MIN_DB, "min")):
            eff = g - 1.0
            amp_in = need_out - eff
            synth = amp_in + cable_loss_db + SPLITTER_SPLIT_DB + SPLITTER_IL_DB + cable_loss_db
            over = "  << EXCEEDS SYNTH MAX" if synth > SYNTH_MAX_DBM else ""
            print(f"     {gtag} gain {g:.0f} dB (eff {eff:.0f} at compression): "
                  f"amp in {amp_in:+.1f} dBm -> SYNTH {synth:+.1f} dBm{over}")

    print(f"\n  VERDICT")
    print(f"  Numerically the synth CAN supply enough drive (needs roughly")
    print(f"  +10.7 dBm typ-gain / +12.7 dBm min-gain, against a "
          f"{SYNTH_MAX_DBM:+.0f} dBm ceiling)")
    print(f"  -- but only with the amp bolted straight to the mixer and running")
    print(f"  fully compressed. Three reasons that's a poor LO driver:")
    print(f"    1. At P1dB the amp generates harmonics; an LO's harmonics mix")
    print(f"       to new spurs, and this mixer only has {MIXER_LR_ISOLATION_DB:.0f} dB L-R isolation.")
    print(f"    2. Zero margin: the MIN-gain unit needs {12.7:+.1f} dBm from the synth,")
    print(f"       essentially its {SYNTH_MAX_DBM:+.0f} dBm ceiling -- where the SMB100A's own")
    print(f"       spur purity is worst (see project notes).")
    print(f"    3. Any real cable between amp and mixer makes it outright")
    print(f"       impossible, not merely marginal.")
    print(f"  AND using both units this way leaves the TX/RX SIGNAL paths with")
    print(f"  no amplifier at all -- the TX path is already {3.1:.1f} dB short of the")
    print(f"  AMC minimum WITH one.")


def rx_chain(resp, lo_hz=11.92e9, if_hz=1.5e9):
    """Receive-side LO generation: AWG CH2 (CW) mixed up to drive the MixAMC."""
    print("\n" + "=" * 72)
    print("RECEIVE PATH  (AWG CH2 CW -> mixer -> isolator -> amp -> MixAMC LO)")
    print("=" * 72)
    awg = float(awg_dbm_at(resp, np.array([if_hz]))[0])
    print(f"  AWG CH2 CW at {if_hz/1e9:.2f} GHz   {awg:+.2f} dBm (measured)")
    out = awg - MIXER_CL_SPEC_MAX_DB - ISOLATOR2_IL_DB + AMP_KU_GAIN_TYP_DB
    print(f"  - mixer (conv loss {MIXER_CL_SPEC_MAX_DB:.1f})  "
          f"{awg-MIXER_CL_SPEC_MAX_DB:+.2f} dBm")
    print(f"  - PE8304 isolator ({ISOLATOR2_IL_DB:.1f} dB) "
          f"{awg-MIXER_CL_SPEC_MAX_DB-ISOLATOR2_IL_DB:+.2f} dBm")
    print(f"  + ZX60-04183LN+ ({AMP_KU_GAIN_TYP_DB:.0f} dB)   {out:+.2f} dBm  "
          f"<- available to drive the MixAMC LO")
    print(f"\n  MixAMC WR2.8 x24 required LO drive: UNKNOWN -- quote A-241209-2")
    print(f"  lists the part (DSB CL 10 dB typ) but carries no electrical spec,")
    print(f"  and there is no MixAMC datasheet in the parts folder.")
    print(f"  ** Cannot close the receive budget until ACST supplies it. **")
    print(f"  If it needs a Level-15-like +15 dBm, this path is ~{15-out:.0f} dB short")
    print(f"  and would need its own booster, exactly like the TX path.")
    return out


def inventory_check():
    print("\n" + "=" * 72)
    print("PARTS / STRUCTURAL CONSTRAINTS")
    print("=" * 72)
    print("  2x ZX60-04183LN+ owned -- but the full chain wants ONE PER PATH")
    print("     (TX to the AMC, RX to the MixAMC LO), so NEITHER path can have")
    print("     the cascaded pair that the TX-only budget showed it needs.")
    print("  1x ZMDB-24H-K+ mixer in the parts folder. A TX chirp mixer AND an")
    print("     RX CW mixer are two separate mixers -- a SECOND ONE IS NEEDED")
    print("     (confirm whether one is already owned).")
    print("  2 isolators: PE83IR1023 (32 dB iso, 1.4 dB loss) and PE8304")
    print("     (16 dB iso, 0.6 dB loss) -- one per path. Put the 32 dB part")
    print("     on the TX/AMC side where reflections matter most.")
    print("  ZC2PD-06263-S+ (6-26.5 GHz) is the right LO splitter.")
    print(f"  ZFRSC-4-842+ is DC-{SPLITTER4_FREQ_HI_HZ/1e9:.1f} GHz -- IF side only,")
    print("     CANNOT be used in the ~12 GHz LO distribution.")
    print("  ZX60-H242+ is 700-2400 MHz -- IF side only, useless at 13-15 GHz.")


def budget_measured(resp, cl_data, if_lo_hz, if_hi_hz, lo_hz, label,
                     extra_gain_db=0.0):
    """Budget using the MEASURED conversion-loss trend rather than datasheet
    numbers -- the version to trust now that run_mixer_check.py has run."""
    _, _, fit = cl_data
    print("\n" + "=" * 72)
    print(f"{label}   [MEASURED conversion loss]")
    print("=" * 72)
    edges = np.array([if_lo_hz, if_hi_hz])
    awg = awg_dbm_at(resp, edges)
    cl = np.polyval(fit, edges / 1e9)
    out = awg - cl - ISOLATOR_IL_DB + AMP_KU_GAIN_TYP_DB + extra_gain_db

    print(f"  IF {if_lo_hz/1e9:.2f}-{if_hi_hz/1e9:.2f} GHz + LO {lo_hz/1e9:.2f} GHz "
          f"-> RF {(lo_hz+if_lo_hz)/1e9:.2f}-{(lo_hz+if_hi_hz)/1e9:.2f} GHz")
    print(f"\n  {'':24s} {'@'+f'{if_lo_hz/1e9:.2f}'+' GHz':>13s} "
          f"{'@'+f'{if_hi_hz/1e9:.2f}'+' GHz':>13s}")
    print(f"  {'AWG output (measured)':24s} {awg[0]:+13.2f} {awg[1]:+13.2f}")
    print(f"  {'- mixer (MEASURED CL)':24s} {-cl[0]:13.2f} {-cl[1]:13.2f}")
    print(f"  {'- isolator':24s} {-ISOLATOR_IL_DB:13.2f} {-ISOLATOR_IL_DB:13.2f}")
    print(f"  {'+ Ku amp(s)':24s} "
          f"{AMP_KU_GAIN_TYP_DB+extra_gain_db:+13.2f} {AMP_KU_GAIN_TYP_DB+extra_gain_db:+13.2f}")
    print(f"  {'= AMC input':24s} {out[0]:+13.2f} {out[1]:+13.2f}")

    tilt = out.max() - out.min()
    awg_tilt, cl_tilt = awg.max() - awg.min(), cl.max() - cl.min()
    print(f"\n  tilt {tilt:.2f} dB  =  AWG droop {awg_tilt:.2f} dB + mixer CL rise {cl_tilt:.2f} dB")
    print(f"  (they COMPOUND -- both worsen with increasing IF)")
    print(f"  AMC window {AMC_IN_MIN_DBM:.0f}..{AMC_IN_MAX_DBM:+.0f} dBm is "
          f"{AMC_IN_MAX_DBM-AMC_IN_MIN_DBM:.0f} dB wide")
    if tilt > AMC_IN_MAX_DBM - AMC_IN_MIN_DBM:
        print(f"  -> TILT ALONE ({tilt:.2f} dB) EXCEEDS THE WHOLE WINDOW: no amount of")
        print(f"     gain or padding can fit this band in. Needs flattening")
        print(f"     (pre-emphasis) or a narrower IF span with a stepped LO.")
    elif out.min() >= AMC_IN_MIN_DBM and out.max() <= AMC_IN_MAX_DBM:
        print(f"  -> fits, {min(out.min()-AMC_IN_MIN_DBM, AMC_IN_MAX_DBM-out.max()):.2f} dB margin")
    else:
        shift = AMC_IN_TYP_DBM - (out.max() + out.min()) / 2
        print(f"  -> would fit if shifted {shift:+.2f} dB "
              f"(tilt {tilt:.2f} dB vs {AMC_IN_MAX_DBM-AMC_IN_MIN_DBM:.0f} dB window, "
              f"{AMC_IN_MAX_DBM-AMC_IN_MIN_DBM-tilt:.2f} dB spare)")
    return out


def budget(resp, if_lo_hz, if_hi_hz, lo_hz, label, extra_gain_db=0.0,
           if_preamp_gain_db=0.0, conv_loss_db=None):
    """One chain scenario, evaluated at both IF band edges."""
    print("\n" + "=" * 72)
    print(f"{label}")
    print("=" * 72)
    rf_lo, rf_hi = lo_hz + if_lo_hz, lo_hz + if_hi_hz
    print(f"  IF {if_lo_hz/1e9:.2f}-{if_hi_hz/1e9:.2f} GHz + LO {lo_hz/1e9:.2f} GHz "
          f"-> RF {rf_lo/1e9:.2f}-{rf_hi/1e9:.2f} GHz")
    covers = rf_lo <= AMC_IN_LO_HZ + 1e6 and rf_hi >= AMC_IN_HI_HZ - 1e6
    print(f"  covers the AMC's {AMC_IN_LO_HZ/1e9:.2f}-{AMC_IN_HI_HZ/1e9:.2f} GHz "
          f"input band? {'yes' if covers else 'NO'}")

    edges = np.array([if_lo_hz, if_hi_hz])
    awg = awg_dbm_at(resp, edges)

    if conv_loss_db is None:
        # Range: datasheet typ as an optimistic floor, and the 30 MHz-IF
        # table value plus a realistic penalty for our GHz-scale IF and
        # LO under-drive as the likely case.
        cl_opt = MIXER_CL_SPEC_TYP_DB
        cl_real = MIXER_CL_SPEC_MAX_DB
    else:
        cl_opt = cl_real = conv_loss_db

    print(f"\n  {'':22s} {'@'+str(if_lo_hz/1e9)+' GHz':>14s} {'@'+str(if_hi_hz/1e9)+' GHz':>14s}")
    print(f"  {'AWG output (measured)':22s} {awg[0]:+14.2f} {awg[1]:+14.2f}")
    lvl = awg.copy()
    if if_preamp_gain_db:
        lvl = lvl + if_preamp_gain_db
        print(f"  {'+ IF preamp':22s} {lvl[0]:+14.2f} {lvl[1]:+14.2f}")
        hdr = MIXER_LO_DRIVE_DBM - IF_BELOW_LO_DB
        if lvl.max() > hdr:
            print(f"    ! IF drive {lvl.max():+.1f} dBm exceeds ~{hdr:+.1f} dBm "
                  f"(LO {IF_BELOW_LO_DB:.0f} dB below) -- pad down to stay linear")

    for cl, tag in ((cl_opt, "optimistic"), (cl_real, "realistic")):
        out = lvl - cl - ISOLATOR_IL_DB + AMP_KU_GAIN_TYP_DB + extra_gain_db
        print(f"\n  [{tag}, conversion loss {cl:.1f} dB]")
        print(f"  {'  - mixer':22s} {lvl[0]-cl:+14.2f} {lvl[1]-cl:+14.2f}")
        print(f"  {'  - isolator':22s} {lvl[0]-cl-ISOLATOR_IL_DB:+14.2f} "
              f"{lvl[1]-cl-ISOLATOR_IL_DB:+14.2f}")
        print(f"  {'  + Ku amp(s)':22s} {out[0]:+14.2f} {out[1]:+14.2f}   <- AMC input")
        lo_e, hi_e = out.min(), out.max()
        tilt = hi_e - lo_e
        if lo_e >= AMC_IN_MIN_DBM and hi_e <= AMC_IN_MAX_DBM:
            verdict = "OK - inside the 0..+7 dBm window"
        elif hi_e < AMC_IN_MIN_DBM:
            verdict = f"TOO LOW by {AMC_IN_MIN_DBM-hi_e:.1f} dB even at the strongest point"
        elif lo_e < AMC_IN_MIN_DBM:
            verdict = f"PARTLY BELOW min (weak end short by {AMC_IN_MIN_DBM-lo_e:.1f} dB)"
        else:
            verdict = f"TOO HIGH by {hi_e-AMC_IN_MAX_DBM:.1f} dB - add a pad"
        print(f"    tilt across band {tilt:.2f} dB "
              f"(AMC window is only {AMC_IN_MAX_DBM-AMC_IN_MIN_DBM:.0f} dB wide) -> {verdict}")

    leak = MIXER_LO_DRIVE_DBM - MIXER_LR_ISOLATION_DB
    print(f"\n  LO leakage at the mixer RF port ~{leak:+.1f} dBm "
          f"(L-R isolation {MIXER_LR_ISOLATION_DB:.1f} dB)")
    print(f"    at LO {lo_hz/1e9:.2f} GHz this sits "
          f"{(AMC_IN_LO_HZ-lo_hz)/1e9:.2f} GHz below the AMC's lower band edge")


def main():
    resp = load_awg_response()
    if resp is None:
        print("No CW level check found -- run chirp_bench/run_specan_band_scan.py's "
              "CW scan first (or the GUI's 'Run CW Level Check').")
        return 1
    print(f"  {resp[0][0]/1e9:.2f}-{resp[0][-1]/1e9:.2f} GHz, "
          f"{resp[1].max():+.2f} to {resp[1].min():+.2f} dBm "
          f"({resp[1].max()-resp[1].min():.2f} dB droop)")

    cl_data = load_measured_conv_loss()
    if cl_data is not None:
        # LO 12.42 GHz was what the mixer was actually measured at, so the
        # AMC's 12.91-15.42 GHz band corresponds to IF 0.49-3.00 GHz here.
        budget_measured(resp, cl_data, 0.49e9, 3.00e9, 12.42e9,
                        "TRANSMIT PATH -- 1 x ZX60-04183LN+")
        budget_measured(resp, cl_data, 0.49e9, 3.00e9, 12.42e9,
                        "TRANSMIT PATH -- BOTH amps cascaded",
                        extra_gain_db=AMP_KU_GAIN_TYP_DB)

    inventory_check()

    # Full chain: one synth feeding BOTH paths through the 2-way splitter.
    # First the topology actually intended (split at the synth, amplify each
    # branch), then the alternative for comparison.
    zx60_04183_as_lo_driver()
    lo_amp_requirement()
    check_lo_chain(split=True)

    # Transmit path, with the ONE amp this path can have if RX needs the other.
    budget(resp, 0.99e9, 3.50e9, 11.92e9,
           "TRANSMIT PATH -- 1 x ZX60-04183LN+ (the other is committed to RX)")

    # What it would take if both amps went to TX instead (leaving RX with none).
    budget(resp, 0.99e9, 3.50e9, 11.92e9,
           "TRANSMIT PATH -- BOTH amps cascaded (starves the receive path)",
           extra_gain_db=AMP_KU_GAIN_TYP_DB)

    rx_chain(resp)

    print("\n" + "=" * 72)
    print("NOTE: every conversion-loss number above is the datasheet spec, which")
    print("is stated AT 30 MHz IF and explicitly 'increases with IF frequency'.")
    print("Our IF is 1.0-3.5 GHz. The true loss is therefore HIGHER than even the")
    print("'realistic' column -- by how much is unknown until measured, which is")
    print("exactly what the mixer bring-up measurement should establish first.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
