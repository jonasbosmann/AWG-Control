"""Single source of truth for the x8 multiplier-chain bench measurements.

WHY THIS FILE EXISTS
--------------------
Three CW level scans were taken on 2026-07-27 and two of them are unusable:
an external pad was changed mid-session and never recorded, so the traces read
19-27 dB low by an offset that VARIES ~20 dB across the band (i.e. it cannot
even be backed out afterwards). The measurement itself was fine; the
bookkeeping killed it.

So: every attenuator, every band, every instrument limit for this chain lives
HERE, and every run script imports it and writes the values it used into the
saved file's metadata. Nothing gets typed into a script twice, and no run can
be orphaned from the pad that was in line when it was taken.

If a pad is physically changed at the bench, change it in this file in the
same motion. That is the whole discipline.

THE ARRANGEMENT
---------------
    AWG CH1 -> [drive pad] -> stage under test -> [PAD_AT_INSTRUMENT_DB]
            -> scope or EXA

The 10 dB pad sits in front of the measuring instrument for EVERY measurement
in this campaign, without exception -- it is the largest pad on hand and it
never moves, which is what makes it safe to treat as a constant.
"""
import os
import sys

import numpy as np

# multiplier_chain.py (the design model) lives one level up, at the project
# root -- import the part constants from there rather than retyping datasheet
# numbers that would then drift out of sync with the design.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiplier_chain as mc


# ── attenuators ───────────────────────────────────────────────────────────
PAD_AT_INSTRUMENT_DB = 10.0
# The FIXED pad between whatever is being measured and the scope/EXA input.
# Largest pad on hand. Present for every measurement in this campaign so that
# it is a constant of the setup rather than a per-run variable -- see the
# module docstring for what happened the last time it was a variable.
#
# It is never subtracted from raw saved data. Raw traces are stored exactly as
# the instrument reported them; the correction is applied when REPORTING, with
# the value written into each file's metadata, so the raw numbers stay
# recoverable if this constant ever turns out to be wrong.
#
# Frequency range MUST cover the stage being measured. A DC-6 GHz pad (VAT
# series) is fine for stages A and B but NOT for the 12.9-15.4 GHz output of
# stage D -- that needs a DC-18 GHz part (BW-S series). Check before trusting
# a high-band number.

PADS_ON_HAND_DB = (3.0, 7.0, 10.0)
# Everything available to build a drive level out of. The chain design assumes
# a ~4 dB pad at stage 16 that has not been bought yet; 3 dB is the nearest
# thing on hand and is close enough that the difference is inside the amp's
# unit-to-unit gain spread.
#
# The 7 dB is a Mini-Circuits 2W-S7W2+ (added 2026-08-03), and it is the only
# spare of that size -- the 10 dB is permanently committed to
# PAD_AT_INSTRUMENT_DB, so it is NOT available to build a drive level from.
# CHECK ITS BANDWIDTH covers the stage it is used in before trusting a
# number: the 2W-S series is specified to 18 GHz, which is ample for stages
# A-C, but read the label rather than assuming.


# ── bands ─────────────────────────────────────────────────────────────────
# The AMC's input window, and what each doubling stage sees on the way there.
AMC_IN_LO_HZ, AMC_IN_HI_HZ = 12.910e9, 15.420e9
AMC_WINDOW_DB = 7.0          # 0 to +7 dBm -- the binding constraint on the
                              # whole design, and why flatness matters more
                              # than absolute level.

BAND_A_HZ = (AMC_IN_LO_HZ / 8, AMC_IN_HI_HZ / 8)     # 1.614 - 1.928 GHz, AWG
BAND_B_HZ = (AMC_IN_LO_HZ / 4, AMC_IN_HI_HZ / 4)     # 3.228 - 3.855 GHz
BAND_C_HZ = (AMC_IN_LO_HZ / 2, AMC_IN_HI_HZ / 2)     # 6.455 - 7.710 GHz
BAND_D_HZ = (AMC_IN_LO_HZ, AMC_IN_HI_HZ)             # 12.910 - 15.420 GHz


# ── AWG ───────────────────────────────────────────────────────────────────
AWG_MAX_VPP = mc.AWG_MAX_AMP_VPP        # 0.55, the AC-coupled module ceiling
AWG_DESIGN_VPP_NO_PAD = 0.311
# The design's drive level with the AWG going STRAIGHT into the stage-A amp.
# With a 3 dB pad added in front of the amp the AWG has to come back up by the
# same 3 dB to land on the same amp output: 0.311 * 10^(3/20) = 0.439 Vpp.
AWG_DESIGN_VPP_WITH_3DB_PAD = AWG_DESIGN_VPP_NO_PAD * 10 ** (3.0 / 20)


# ── instrument limits ─────────────────────────────────────────────────────
SCOPE_MAX_VRMS_50R = 2.3
# MSO64B 50 ohm input, derated figure that applies at the <100 mV/div settings
# this campaign uses. 2.3 Vrms into 50 ohm = +20.24 dBm. Larger V/div settings
# permit more, but there is no reason to lean on that -- treat this as THE
# limit and keep margin against it.
SCOPE_BW_HZ = 8.0e9
# Nominal. NOT verified against this unit's actual option set -- if a stage-C
# (6.5-7.7 GHz) measurement reads low, suspect the scope's own rolloff before
# suspecting the chain. Stage D (12.9-15.4 GHz) is out of reach either way and
# has to go to the EXA.

EXA_DAMAGE_DBM = 30.0        # absolute maximum at the RF input
EXA_MAX_MIXER_DBM = -10.0    # level at the first mixer for spec'd linearity;
                              # exceed it and the analyzer compresses and
                              # invents harmonics that are not in the signal.


def scope_max_dbm():
    """The scope's 50 ohm input limit expressed as dBm."""
    return 10 * np.log10((SCOPE_MAX_VRMS_50R ** 2 / 50) / 1e-3)


def required_exa_atten_db(p_in_dbm):
    """Minimum EXA INTERNAL attenuation for a given level at its connector.

    The external 10 dB pad is already counted by the caller passing the level
    that actually arrives. The analyzer refers readings to its own connector,
    so internal attenuation does not shift measured dBm -- it only trades
    noise floor against mixer headroom. Rounded up to the instrument's 2 dB
    step.
    """
    need = p_in_dbm - EXA_MAX_MIXER_DBM
    return max(0.0, float(np.ceil(need / 2.0) * 2.0))


# ── the drive ladder ──────────────────────────────────────────────────────
TARGET_OUT_DBM = 15.0
# Amplifier OUTPUT power for the experiment, before the 10 dB pad.

OUT_LEVEL_BACKOFF_DB = (10.0, 6.0, 3.0, 0.0)
# Every measurement is taken at each of these backoffs below the target, in
# this order, ending AT the target. Two reasons, and both matter:
#
#  1. Ramping up. The first setting is 10 dB down, where nothing is near any
#     limit, so the things that go wrong when a chain is first connected --
#     wrong pad fitted, amp unpowered, cable on the wrong port, DUT
#     oscillating -- show up as an obviously wrong number instead of as
#     damage. Each step is predicted from the measured level of the step
#     below and checked before it is commanded.
#
#  2. A single level is not a characterisation. Gain, flatness and spurious
#     content all move with drive, and the whole point of measuring is to
#     find out whether the level the experiment runs at is still in the
#     linear region. One curve cannot answer that; a family of them can, and
#     the lowest rung doubles as the small-signal reference.

FREQ_STEP_HZ = 10e6
FREQ_MARGIN_HZ = 50e6   # scan a little outside the band so the edges are
                         # visible rather than being the first/last point
N_SPOT_FREQS = 3         # band low / mid / high, for fast level-setting


def out_levels_dbm(target=None):
    """The output levels to measure at, ascending, ending at the target."""
    t = TARGET_OUT_DBM if target is None else target
    return tuple(sorted(t - b for b in OUT_LEVEL_BACKOFF_DB))


def band_freqs(stage_key, step_hz=FREQ_STEP_HZ, margin_hz=FREQ_MARGIN_HZ):
    lo, hi = STAGES[stage_key]["band_hz"]
    return np.arange(lo - margin_hz, hi + margin_hz + 1, step_hz)


def spot_freqs(stage_key, n=N_SPOT_FREQS):
    lo, hi = STAGES[stage_key]["band_hz"]
    return np.linspace(lo, hi, n)


def level_key(out_dbm):
    """Series name for a measurement taken at a given amp output level."""
    return f"out{out_dbm:+.1f}dBm"


# ── level helpers ─────────────────────────────────────────────────────────
def dbm_from_vpp(vpp):
    """A vpp sine into 50 ohm, in dBm."""
    return 10 * np.log10(((vpp / 2) ** 2 / (2 * 50)) / 1e-3)


def vpp_from_dbm(dbm):
    return 2 * np.sqrt(2 * 50 * 10 ** (dbm / 10) * 1e-3)


def volts_per_div_for(dbm, fill_frac=0.7, n_div=10):
    """Vertical scale that puts a `dbm` sine at `fill_frac` of full screen.

    Used to pin ONE scale per amplitude step instead of letting autoscale run
    inside the frequency sweep, where a rescale would appear as a step in the
    flatness curve that has nothing to do with the hardware.
    """
    return vpp_from_dbm(dbm) / (fill_frac * n_div)


# ── the stages ────────────────────────────────────────────────────────────
# One entry per amplifier in the chain as built. `drive_pad_db` is the pad
# between the source and the amp input for the measurement of THAT stage;
# `p1db_dbm` and the gain figures are datasheet, i.e. exactly the numbers this
# campaign exists to replace with measured ones.
STAGES = {
    "A": dict(
        part="ZX60-2522MA-S+",
        # Generic on purpose -- which part actually sits in the doubler-A slot
        # (HMC188 vs the Pulsar FX-05-410 under test) is per-run, recorded in
        # instrument_settings.doubler/element by whatever script saves the
        # measurement. Naming a specific part HERE would get silently stamped
        # into every stage-A save regardless of what was really in the chain --
        # which is exactly what happened 2026-08-03 (see the
        # 2026-08-03_114841_A_doubler measurements' label_correction field).
        role="stage-A drive amp (AWG -> doubler A)",
        band_hz=BAND_A_HZ,
        source="AWG CH1 DUC CW",
        # REMOVED 2026-07-31. The pad and the AWG level are interchangeable for
        # setting the amp's drive, but the pad was eating headroom that
        # pre-emphasis needs: flattening the band's 2.4 dB of tilt by raising
        # the weak edge needs ~2.4 dB, and with the pad in there was only
        # 2.08 dB left to the 0.55 Vpp ceiling. Without it there is ~5 dB.
        #
        # CONSEQUENCE: every measurement taken before this date used the pad,
        # so at any given AWG amplitude the amp is now driven 3 dB harder.
        # The scope anchor (0.3342 Vpp -> +15.05 dBm) is INVALID; levels must
        # be re-established by measurement, not scaled from it.
        drive_pad_db=0.0,
        gain_typ_db=mc.ZX602522MA_GAIN_TYP_DB,
        gain_min_db=mc.ZX602522MA_GAIN_MIN_DB,
        p1db_dbm=mc.ZX602522MA_P1DB_DBM,
        supply=mc.ZX602522MA_SUPPLY,
        instrument="scope",
    ),
    "B": dict(
        # CHANGED 2026-08-03: was ZX60-123LN-S+. multiplier_chain.py rejected
        # the ZX60-V62+ for the stage-A DRIVE slot ("5 dB short on gain"), but
        # that verdict does not carry here -- stage A needed ~21 dB, stage B
        # needs ~12, and the V62 has a higher P1dB at this band than the
        # 123LN (whose P1dB is only +13.4 dBm at the top of its range).
        part="ZX60-V62+",
        role="stage-B amp (doubler A -> MMD-0415HS)",
        band_hz=BAND_B_HZ,
        source="doubler A output",
        # 7 dB pad (2W-S7W2+) IN FRONT of the amp. Not optional: doubler A now
        # delivers +2.56..+4.60 dBm (measured, FX-05-410 through the VLF-3800+
        # and VHF-2100+), and +4.60 with ~15.4 dB of gain is +20.0 dBm out --
        # at or PAST this part's P1dB. Without the pad the amp is compressed,
        # the gain reading is understated, and the AM-to-PM it adds is
        # multiplied by 96 on the way to the AMC output.
        #
        # 7 rather than 10 dB because 7 is what is spare (the 10 dB pad is
        # committed to the instrument), and it turns out to be the better
        # choice anyway: it leaves 4.2 dB of backoff against a MIN-spec unit
        # and 6.0 dB against a typical one, while landing the MMD-0415HS at
        # +10.96..+13.00 dBm -- nearer its +14 dBm nominal than a 10 dB pad's
        # +7.96..+10.00 would. The 4.2 dB is just under multiplier_chain's
        # 5 dB AMP_BACKOFF_MIN_DB, which is worth knowing but is a worst-case-
        # unit figure; the measurement will say which case this unit is.
        drive_pad_db=7.0,
        # ** BOTH FIGURES BELOW ARE QUOTED AT 2 GHz ** (multiplier_chain.py's
        # candidate notes). Band B is 3.228-3.855 GHz, where gain and P1dB are
        # both expected to be LOWER. They are here as a planning bound only --
        # replacing them with measured band-B numbers is what this stage's
        # measurement is for. Note also that the +-0.2 dB flatness figure that
        # made this part attractive was taken over 0.7-2.6 GHz, which does NOT
        # reach band B, so the reason for choosing it is itself unverified here.
        gain_typ_db=15.4,
        gain_min_db=None,
        # The MIN spec (+17.2), not the typ (+19.0), deliberately: this value
        # feeds backoff/safety arithmetic, where the conservative end is the
        # correct one to design against.
        p1db_dbm=17.2,
        supply=None,
        # EXA, not scope: band B is inside the scope's 8 GHz bandwidth, but the
        # measurement that matters here is harmonic content (F1/F3/F4 through
        # the amp), and the scope's AC-RMS reading cannot separate tones.
        instrument="exa",
    ),
    "C": dict(
        part="ZX60-04183LN+ (6.4-7.8 GHz slot)",
        role="stage-C amp (MMD-0415HS -> ZX90-2-24-S+)",
        band_hz=BAND_C_HZ,
        source="MMD-0415HS output",
        drive_pad_db=0.0,
        gain_typ_db=11.0,
        gain_min_db=None,
        p1db_dbm=None,
        supply=None,
        instrument="scope",   # near the scope's BW limit -- see SCOPE_BW_HZ
    ),
    "D": dict(
        part="ZX60-04183LN+ (12.9-15.4 GHz slot)",
        role="stage-D output amp (-> AMC)",
        band_hz=BAND_D_HZ,
        source="ZX90-2-24-S+ output",
        drive_pad_db=3.0,
        gain_typ_db=11.0,
        gain_min_db=None,
        p1db_dbm=None,
        supply=None,
        instrument="exa",     # above the scope's bandwidth
    ),
}

# Gain/P1dB left as None above are the ones I do not have a datasheet figure
# for in multiplier_chain.py. They are deliberately None rather than guessed,
# so a script that needs them fails loudly instead of quietly modelling a
# number nobody measured.


def arrangement(stage_key, thru=False):
    """Human-readable one-line description of the physical setup, written into
    every saved file so a trace can never be separated from what produced it."""
    st = STAGES[stage_key]
    pad = f"{st['drive_pad_db']:g} dB pad -> " if st["drive_pad_db"] else ""
    dut = "THRU (adapter in place of amp)" if thru else st["part"]
    return (f"{st['source']} -> {pad}{dut} -> "
            f"{PAD_AT_INSTRUMENT_DB:g} dB pad -> {st['instrument']}")


def safety_check(stage_key, p_out_max_dbm, verbose=True):
    """Check a stage's worst-case output against the measuring instrument.

    p_out_max_dbm: the highest level expected AT THE STAGE OUTPUT (before the
    10 dB pad). Returns (ok, list_of_lines).
    """
    st = STAGES[stage_key]
    at_inst = p_out_max_dbm - PAD_AT_INSTRUMENT_DB
    lines = [f"stage {stage_key} worst case {p_out_max_dbm:+.2f} dBm at the DUT "
             f"output, {at_inst:+.2f} dBm at the instrument "
             f"(after the {PAD_AT_INSTRUMENT_DB:g} dB pad)"]
    ok = True

    if st["instrument"] == "scope":
        lim = scope_max_dbm()
        margin = lim - at_inst
        lines.append(f"  scope 50 ohm limit {lim:+.2f} dBm "
                     f"({SCOPE_MAX_VRMS_50R} Vrms)  ->  margin {margin:+.1f} dB")
        if margin < 3.0:
            ok = False
            lines.append("  ** TOO CLOSE -- do not connect. **")
        lines.append(f"  set vertical to at least "
                     f"{volts_per_div_for(at_inst)*1e3:.0f} mV/div")
        hi = max(st["band_hz"])
        if hi > SCOPE_BW_HZ:
            lines.append(f"  ** {hi/1e9:.2f} GHz is above the scope's "
                         f"{SCOPE_BW_HZ/1e9:g} GHz bandwidth -- use the EXA. **")
            ok = False
    else:
        lines.append(f"  EXA damage level {EXA_DAMAGE_DBM:+.0f} dBm -- "
                     f"{EXA_DAMAGE_DBM - at_inst:.0f} dB of margin, no risk")
        need = required_exa_atten_db(at_inst)
        lines.append(f"  set EXA INTERNAL attenuation to >= {need:.0f} dB "
                     f"(keeps the first mixer at or below "
                     f"{EXA_MAX_MIXER_DBM:+.0f} dBm)")
        if need > 10.0:
            lines.append("  the habitual 10 dB internal setting is NOT enough here")

    if verbose:
        for ln in lines:
            print(ln)
    return ok, lines
