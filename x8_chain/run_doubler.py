"""Characterise doubler A in place -- HMC188 by default, or the Pulsar
FX-05-410 swap with DOUBLER=FX-05-410 (see the DOUBLER constant below).

    AWG -> ZX60-2522MA-S+ -> PE87FL1012 -> [HMC188 | FX-05-410] -> 10 dB pad -> EXA

Two measurements, and the second is the one the whole design rests on.

1. BAND SWEEP at the operating drive. At each AWG frequency the EXA measures
   the tones at 1f, 2f, 3f and 4f. 2f is the wanted output; the rest are what
   the 3.2-3.9 GHz filter will have to remove, so this measurement writes that
   filter's specification.

   Conversion loss comes out by subtraction and needs nothing disconnected:
   the drive into the doubler was measured directly in the preceding
   PE87FL1012 session, at the same AWG amplitude, at the same frequencies.

2. DRIVE SWEEP -- the saturation test. The AWG amplitude is stepped down and
   the 2f output followed, giving the TILT TRANSFER RATIO:

       r = d(P_out at 2f) / d(P_in at f)     both in dB

   A square-law multiplier operating small-signal has r = 2: every dB of input
   tilt becomes two dB at its output. A hard-saturated one has r -> 0, and the
   input tilt is suppressed entirely. Across three doublers the drive tilt is
   multiplied by r^3, so with the 2.2 dB measured at stage A:

       r = 2.0  ->  17 dB at the AMC input   (hopeless against a 7 dB window)
       r = 1.5  ->   7 dB                    (fails)
       r = 1.2  ->   3.7 dB                  (works)
       r = 1.0  ->   2.2 dB                  (comfortable)

   multiplier_chain.tilt_budget() assumes tilt passes 1:1, which is neither
   the small-signal nor the saturated bound -- it is a value in between with
   no physical justification. THIS measurement replaces it with a number.

   Pre-emphasis at the AWG can correct whatever tilt results, so r is no
   longer make-or-break, but it sets how much correction is needed and how
   much the AWG's own 0.19 dB scatter gets amplified on the way to the AMC.

The drive level is NOT re-set here: the AWG amplitude from the PE87FL1012
session is reused verbatim, so the doubler sees exactly the drive that was
characterised. Nothing upstream changes when the doubler goes in.

Run:  python x8_chain/run_doubler.py [n_freqs]
      NO_SHOW=1 no windows, LIVE=0 no live window.
"""
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import chainlog
import multiplier_chain as mc
from liveplot import LiveSpectrum
from awg import AWG
from specan import SpecAn


# STAGE stays "A" whichever doubler is under test: it selects the AWG SWEEP
# RANGE (1.614-1.928 GHz), and every tone in this chain is a multiple of that.
STAGE = "A"
# Which multiples of the AWG frequency to measure, and which one is the wanted
# output of the doubler under test. Doubler A outputs 2f; doubler B outputs 4f
# and its interesting products run 2f..8f (see the harmonic map):
#     TONES=2,3,4,5,6,7,8 WANTED=4 python x8_chain/run_doubler.py
TONES = tuple(int(x) for x in os.environ.get("TONES", "1,2,3,4").split(","))
WANTED = int(os.environ.get("WANTED", "2"))

# Which physical part is bolted into the doubler-A slot right now. Overridable
# so a bench swap is recorded at the moment it is made:
#     DOUBLER=FX-05-410 python x8_chain/run_doubler.py
DOUBLER = os.environ.get("DOUBLER", "HMC188")
_ELEMENT_DEFAULT = {
    "HMC188": "HMC188 doubler A, direct off the PE87FL1012",
    "FX-05-410": "Pulsar FX-05-410 doubler A, direct off the PE87FL1012",
    "MMD-0415HS": "MMD-0415HS doubler B, off the stage-B amp",
}[DOUBLER]
ELEMENT = os.environ.get("ELEMENT", _ELEMENT_DEFAULT)


def doubler_ref():
    """(cl_db, pin_min, pin_nom, pin_max, supp_f1_dbc) for whichever part
    DOUBLER selects. pin_min/pin_max are None where the datasheet publishes no
    window at all -- the FX-05-410 gives only a nominal, so there is nothing
    to check a measured drive against."""
    if DOUBLER == "FX-05-410":
        return (mc.FX05410_CL_MAX_DB, None, mc.FX05410_PIN_NOM_DBM, None,
                mc.FX05410_SUPP_F1_DBC)
    if DOUBLER == "MMD-0415HS":
        # CL 11.0 dB is the datasheet TYP, and it is quoted at the +14 dBm
        # nominal. This chain drives it 1.3-4.2 dB under that, so expect the
        # measurement to come in worse -- that is the part being characterised,
        # not an error.
        return (mc.MMD_CL_TYP_DB, mc.MMD_PIN_MIN_DBM, mc.MMD_PIN_NOM_DBM,
                mc.MMD_PIN_MAX_DBM, mc.MMD_SUPP_1F_DBC)
    return (mc.HMC188_CL_TYP_DB, mc.HMC188_PIN_MIN_DBM, mc.HMC188_PIN_NOM_DBM,
            mc.HMC188_PIN_MAX_DBM, mc.HMC188_SUPP_DBC)


SPAN_HZ = 50e6
RBW_HZ = 100e3
N_POINTS = 801
SETTLE_S = 0.25
N_FREQS = 11

EXA_ATTEN_DB = 20        # floor; raised automatically if the level needs it
REF_LEVEL_DBM = 10

# Net gain, dB, of anything inserted AFTER the doubler and before the
# instrument -- a stage-B amplifier, for instance:
#     EXTRA_GAIN_DB=15.4 python x8_chain/run_doubler.py
#
# The EXA safety check below sizes its worst case as "the doubler passes the
# full drive straight through", which is the correct bound for a LOSSY DUT.
# An amplifier inverts that -- its output EXCEEDS its input -- so without this
# the check silently under-attenuates, the analyzer's first mixer compresses,
# and it invents harmonics that are not in the signal. Those are precisely
# what this measurement exists to quantify, so the failure would not announce
# itself; it would just produce plausible wrong numbers.
EXTRA_GAIN_DB = float(os.environ.get("EXTRA_GAIN_DB", "0"))

# The physical signal path, in order, exactly as cabled. Override whenever the
# bench changes, in the same motion as changing the bench:
#     CHAIN="AWG CH1 -> ... -> EXA" python x8_chain/run_doubler.py
#
# This is printed at the top of every run and written into the saved file's
# setup block, replacing config.arrangement()'s single-stage description --
# which cannot represent a cascade and was recording the wrong instrument and
# none of the parts under test. Verify it against the bench BEFORE trusting a
# number: it is the only record of what produced the data.
CHAIN = os.environ.get("CHAIN", "")

# PREEMPH=awg flattens the AWG's OWN frequency response across the band, by
# commanding a different amplitude at every point instead of one fixed level:
#     PREEMPH=awg python x8_chain/run_doubler.py
#
# The correction comes from run_awg_level_scope.py's saved scans, which measured
# the AWG straight into the scope's 50 ohm input with NO pad in the path -- the
# one measurement in this project that cannot be corrupted by unrecorded
# attenuation. Its droop over the x8 band is 0.98 dB.
#
# The table is amplitude-INDEPENDENT and that is what makes it reusable: the
# 0.50 and 0.55 Vpp scans give 0.985 and 0.983 dB of droop with shapes agreeing
# to 0.011 dB rms, so this is a linear frequency response, not compression. One
# table therefore applies at any base amplitude, including the 0.2656 Vpp these
# runs use, which neither scan was taken at.
#
# It corrects the AWG ALONE. It knows nothing about the filters or the stage-B
# amp, which contribute most of the tilt further down the chain -- flattening
# here removes the source's contribution so what remains is attributable to the
# chain.
PREEMPH = os.environ.get("PREEMPH", "")

# Where the drive into THIS doubler was measured -- see known_drive().
DRIVE_FROM = os.environ.get("DRIVE_FROM", "")
DRIVE_SERIES = os.environ.get("DRIVE_SERIES", "2f")
_PREEMPH_NPZ = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "awg_level_scope", "awg_level_scope.npz")


def preemph_db(freqs_hz):
    """Correction in dB to ADD at each frequency, 0 where PREEMPH is off.

    Always flattens UPWARD (raise the weak end) rather than downward: the chain
    is already short of drive at the top of the band -- doubler B sits 0.65 dB
    above its minimum there -- so throwing away level at the strong end to match
    the weak one would make the binding problem worse.
    """
    f = np.asarray(freqs_hz, dtype=float)
    if PREEMPH != "awg":
        return np.zeros_like(f)
    d = np.load(_PREEMPH_NPZ)
    key = "dbm_0p55vpp" if "dbm_0p55vpp" in d else "dbm_0p5vpp"
    sf, sa = d["freqs_hz"], d[key]
    # The 0 dB reference is the STRONGEST point inside the chain's own band,
    # evaluated from the band itself -- NOT from whatever frequencies happen to
    # be passed in. Deriving it from the argument makes a single-frequency call
    # return zero (its own max is itself), which would silently disable
    # pre-emphasis in exactly the per-point loop that applies it, while the
    # whole-band print still showed the correct curve.
    lo, hi = config.BAND_A_HZ
    ref = float(np.interp(np.linspace(lo, hi, 401), sf, sa).max())
    # np.asarray/np.interp preserve shape: scalar in -> scalar out, so float()
    # on the result is valid (NumPy 2 rejects float() on a 1-element 1-D array).
    return ref - np.interp(f, sf, sa)


def preemph_vpp(freqs_hz, base_vpp):
    """Per-frequency AWG amplitude implementing the correction."""
    return float(base_vpp) * 10 ** (preemph_db(freqs_hz) / 20.0)


def chain_description():
    if CHAIN:
        return CHAIN
    return (f"AWG CH1 -> {config.STAGES['A']['part']} -> PE87FL1012 -> "
            f"{DOUBLER} -> {config.PAD_AT_INSTRUMENT_DB:g} dB pad -> EXA")

# Drive sweep: how far below the operating amplitude to go, and in how many
# steps. Downward only -- going above the operating point would push the
# stage-A amp toward its P1dB, and the informative part of the curve (where a
# saturated doubler starts following its input again) is below, not above.
DRIVE_RANGE_DB = 9.0
DRIVE_STEPS = 7
DRIVE_FREQS = 3


def measure_tone(sa, centre_hz):
    """(peak_dbm, floor_dbm, freqs, amps) in a span around centre_hz."""
    sa.set_freq(centre_hz, SPAN_HZ)
    time.sleep(0.05)
    f, a = sa.sweep_once()
    return float(np.max(a)), float(np.median(a)), f, a


def known_drive():
    """The measured level going INTO the doubler under test.

    Returns (vpp, freqs_hz, level_dbm, path). This is what makes conversion
    loss a subtraction rather than a separate measurement: the level at the
    doubler's input plane was measured directly, in the preceding session, at
    the same AWG amplitude and the same frequencies.

    For doubler A that plane is the PE87FL1012 output (the default below). For
    any later doubler it is the previous session's measurement of the tone that
    now feeds it -- point at it explicitly:

        DRIVE_FROM=2026-08-04_162549 DRIVE_SERIES=2f ...

    Whatever is named must have been measured with the SAME cabling upstream of
    that plane, or the subtraction silently attributes the difference to the
    doubler.
    """
    pad = config.PAD_AT_INSTRUMENT_DB
    if DRIVE_FROM:
        hits = sorted(glob.glob(os.path.join(
            chainlog.MEAS_DIR, f"{DRIVE_FROM}*", "*_doubler_band.json")))
        if not hits:
            print(f"DRIVE_FROM={DRIVE_FROM!r} matched no saved band sweep.")
            return None, None, None, None
        d = chainlog.load(hits[-1])
        key = f"dbm_{DRIVE_SERIES}"
        if key not in d:
            print(f"{hits[-1]} has no series {key!r} "
                  f"(has: {[k for k in d if k.startswith('dbm_')]})")
            return None, None, None, None
        s = d.get("instrument_settings", {})
        return (float(s["operating_vpp"]), d["freqs_hz"], d[key] + pad, hits[-1])
    for p in chainlog.find(STAGE, label_contains="exa_harmonics", thru=False):
        d = chainlog.load(p)
        s = d.get("instrument_settings", {})
        if "PE87FL1012" not in str(s.get("element", "")):
            continue
        key = chainlog.series_key("H1")
        if key not in d:
            continue
        return (float(s["operating_vpp"]), d["freqs_hz"], d[key] + pad, p)
    return None, None, None, None


def run(n_freqs=N_FREQS, drive_offset_db=0.0):
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = config.STAGES[STAGE]["band_hz"]
    st = config.STAGES[STAGE]

    vpp, dfreqs, drive, dpath = known_drive()
    if vpp is None:
        print("No PE87FL1012 session found -- run run_exa_harmonics.py first so")
        print("the drive into the doubler is a measured quantity, not a guess.")
        return
    freqs = np.linspace(lo, hi, int(n_freqs))
    drive_at = np.interp(freqs, dfreqs, drive)

    # The drive reference was measured at ONE flat amplitude. Pre-emphasis
    # commands a different amplitude per point, so the level actually arriving
    # at the doubler is higher by exactly the correction -- fold it in here,
    # before anything (safety sizing, conversion loss, the drive sweep) reads
    # drive_at. Forgetting this would understate conversion loss by up to the
    # full correction at the weak end.
    pe_db = preemph_db(freqs)
    if PREEMPH:
        drive_at = drive_at + pe_db
        print(f"\n  PRE-EMPHASIS '{PREEMPH}': {pe_db.min():+.2f} to {pe_db.max():+.2f} dB "
              f"across the band")
        print(f"    AWG {vpp:.4f} Vpp flat  ->  "
              f"{preemph_vpp(freqs, vpp).min():.4f}-{preemph_vpp(freqs, vpp).max():.4f} Vpp "
              f"(ceiling {config.AWG_MAX_VPP})")
        if preemph_vpp(freqs, vpp).max() > config.AWG_MAX_VPP:
            print("    ** correction exceeds the AWG ceiling -- ABORT **")
            return

    # Raising the drive is open-loop here: the doubler now sits where the
    # analyzer used to, so the filter output cannot be measured directly. The
    # AWG is scaled from the amplitude that WAS measured there, which is sound
    # because nothing upstream of the doubler changed. The stage-A amp adds
    # ~0.1 dB of its own compression over a 2 dB step (measured <=0.33 dB over
    # 10 dB), so the assumed drive is high by about that much.
    if drive_offset_db:
        vpp = float(np.clip(vpp * 10 ** (drive_offset_db / 20), 0.001,
                            config.AWG_MAX_VPP))
        drive_at = drive_at + drive_offset_db
        # The stage-A amp's own P1dB is the binding constraint, not the AWG's
        # ceiling and not the doubler's input window. Its output is the drive
        # plus the filter loss that follows it.
        amp_out = drive_at.max() + 0.64
        back = mc.ZX602522MA_P1DB_DBM - amp_out
        print(f"  DRIVE RAISED {drive_offset_db:+.2f} dB -> {vpp:.4f} Vpp")
        print(f"    stage-A amp now {amp_out:+.2f} dBm, {back:.2f} dB below its "
              f"P1dB {mc.ZX602522MA_P1DB_DBM:+.1f}")
        # ALLOW_COMPRESSION=1 downgrades this from an abort to a loud warning.
        # Deliberate, and deliberately narrow: P1dB is a "gets worse" limit for
        # a CW signal into an amp, not a damage threshold, so it is the one
        # guard worth making overridable when the point of the run IS to see
        # what compression does. The doubler's own Pin max below is NOT
        # overridable -- exceeding a diode's rated input is a real overstress
        # risk, not a graceful-degradation one, and this campaign does not get
        # to spend the on-hand HMC188 to find that out.
        allow_compression = os.environ.get("ALLOW_COMPRESSION") == "1"
        if back < 1.0:
            print(f"    ** under 1 dB of backoff -- the amp is at or past P1dB. "
                  f"AM-to-PM here is multiplied by 192 to the AMC output. "
                  + ("PROCEEDING (ALLOW_COMPRESSION=1) **" if allow_compression
                     else "ABORT **"))
            if not allow_compression:
                return
        if back < 2.0:
            print(f"    ** below the 2 dB this campaign has been treating as the "
                  f"floor; AM-to-PM is the thing to watch, not the amp **")
        _, _, _, pin_max, _ = doubler_ref()
        if pin_max is not None and drive_at.max() > pin_max:
            print(f"    ** {drive_at.max():+.2f} dBm exceeds the {DOUBLER}'s "
                  f"{pin_max:+.0f} dBm maximum input. ABORT ** "
                  f"(not overridable -- this is the doubler's own rating, not "
                  f"the amp's)")
            return
        if pin_max is None:
            print(f"    note: {DOUBLER} has no published maximum input to check "
                  f"{drive_at.max():+.2f} dBm against -- no datasheet ceiling, "
                  f"proceed cautiously")

    cl_ref, pin_min, pin_nom, pin_max, supp_ref = doubler_ref()
    print("=" * 78)
    print(f"DOUBLER A -- {ELEMENT}")
    print("=" * 78)
    print("  CHAIN AS CABLED -- check this against the bench before trusting "
          "the numbers:")
    for i, part in enumerate(chain_description().split("->")):
        print(f"    {i+1:2d}. {part.strip()}")
    print("=" * 78)
    print(f"  drive into the doubler, measured in the preceding session:")
    print(f"    {drive_at.max():+.2f} to {drive_at.min():+.2f} dBm "
          f"at {vpp:.4f} Vpp   ({dpath})")
    if pin_min is not None:
        print(f"    {DOUBLER} window {pin_min:+.0f} to "
              f"{pin_max:+.0f} dBm, nominal "
              f"{pin_nom:+.0f} -> margin "
              f"{drive_at.min()-pin_min:.2f} dB at the weak edge")
    else:
        print(f"    {DOUBLER} has no published Pin window, nominal "
              f"{pin_nom:+.0f} dBm -> running {drive_at.min()-pin_nom:+.2f} to "
              f"{drive_at.max()-pin_nom:+.2f} dB over nominal across the band")
    print(f"  wanted output {WANTED*lo/1e9:.3f}-{WANTED*hi/1e9:.3f} GHz; "
          f"measuring tones at {TONES}x the AWG frequency")

    # Safety. Worst case at the analyzer is the drive passing straight through
    # the doubler (not connected, or total F1 feedthrough) AND then being
    # amplified by anything inserted after it -- see EXTRA_GAIN_DB.
    at_exa = drive_at.max() + EXTRA_GAIN_DB - pad
    need = config.required_exa_atten_db(at_exa)
    if EXTRA_GAIN_DB:
        print(f"\n  downstream gain declared: {EXTRA_GAIN_DB:+.1f} dB")
    print(f"\n  worst case at the EXA is the full drive feeding through"
          + (" and being amplified" if EXTRA_GAIN_DB else "") + f": {at_exa:+.1f} dBm")
    # Internal attenuation does not shift measured dBm (it trades noise floor
    # against mixer headroom), so raising it to what the level needs is always
    # safe -- unlike leaving it too low, which silently compresses.
    atten_db = max(EXA_ATTEN_DB, need)
    print(f"  needs >= {need:.0f} dB internal attenuation; using {atten_db:.0f} dB"
          + (f"  (raised from the {EXA_ATTEN_DB:.0f} dB floor)"
             if atten_db > EXA_ATTEN_DB else ""))
    if at_exa > config.EXA_DAMAGE_DBM - 10:
        print(f"  ** {at_exa:+.1f} dBm is within 10 dB of the EXA's "
              f"{config.EXA_DAMAGE_DBM:+.0f} dBm damage level -- ABORT **")
        return
    print(f"  EXA damage {config.EXA_DAMAGE_DBM:+.0f} dBm -> "
          f"{config.EXA_DAMAGE_DBM-at_exa:.0f} dB of margin\n")

    awg = AWG()
    sa = SpecAn()
    lv = {n: [] for n in TONES}
    fl = {n: [] for n in TONES}
    drive_out, drive_in = {}, {}
    try:
        sa.set_trace_mode("NORM")
        sa.set_attenuation(atten_db)
        sa.set_ref_level(REF_LEVEL_DBM)
        sa.set_rbw(RBW_HZ)
        sa.set_vbw(None)
        sa.set_points(N_POINTS)
        print(f"  EXA: RBW {sa.get_rbw()/1e3:.0f} kHz, {N_POINTS} points, "
              f"span {SPAN_HZ/1e6:.0f} MHz\n")

        lp = LiveSpectrum([f"{n}f" for n in TONES],
                          f"{DOUBLER} doubler A   ({pad:g} dB pad added back)")

        # ---- 1. band sweep at the operating drive -----------------------
        print("  BAND SWEEP at the operating drive")
        for i, f in enumerate(freqs):
            v_i = float(preemph_vpp(f, vpp))
            if i == 0:
                awg.duc_cw_setup(float(f), amplitude_vpp=v_i, channel=1)
            else:
                # amplitude_vpp rides along with the NCO retune in one
                # transaction -- no waveform re-upload, so a pre-emphasised
                # sweep costs the same as a flat one.
                awg.duc_cw_step(float(f), channel=1, amplitude_vpp=v_i)
            time.sleep(SETTLE_S)
            row = []
            for n in TONES:
                pk, floor, tf, ta = measure_tone(sa, float(f) * n)
                lv[n].append(pk); fl[n].append(floor)
                row.append((n, pk, floor))
                lp.trace(f"{n}f", tf, ta, offset_db=pad)
                lp.summary(f"{n}f", f / 1e9, pk + pad)
            w = dict((n, pk) for n, pk, _ in row)[WANTED]
            lp.note(f"{DOUBLER} band sweep   AWG {f/1e9:.3f} GHz   "
                    f"({i+1}/{len(freqs)})")
            txt = "  ".join(
                f"{n}f {pk+pad:+7.2f}" +
                (f" ({pk-w:+6.1f} dBc)" if n != WANTED else " (WANTED)") +
                ("*" if pk - floor < 6.0 else " ")
                for n, pk, floor in row)
            print(f"    {f/1e9:.3f} GHz  CL {drive_at[i]-(w+pad):5.2f} dB  {txt}")

        # ---- 2. drive sweep: the saturation test ------------------------
        print(f"\n  DRIVE SWEEP -- {DRIVE_STEPS} levels over "
              f"{DRIVE_RANGE_DB:g} dB, at {DRIVE_FREQS} frequencies")
        print(f"  (r = d(2f out)/d(in); 2.0 = small-signal square law, "
              f"0 = hard saturated)")
        sweep_f = np.linspace(lo, hi, DRIVE_FREQS)
        steps_db = np.linspace(-DRIVE_RANGE_DB, 0.0, DRIVE_STEPS)
        for f in sweep_f:
            outs, ins = [], []
            print(f"\n    {f/1e9:.3f} GHz")
            for s_db in steps_db:
                # Pre-emphasis is a per-FREQUENCY correction, so within one
                # spot-frequency ladder it is a constant offset -- it moves the
                # whole ladder, not its slope, leaving r unaffected. It is
                # applied anyway so the ladder sits at the same drive the band
                # sweep used, and drive_at already carries the same offset.
                v = float(np.clip(preemph_vpp(f, vpp) * 10 ** (s_db / 20),
                                  0.001, config.AWG_MAX_VPP))
                awg.duc_cw_setup(float(f), amplitude_vpp=v, channel=1)
                time.sleep(SETTLE_S)
                pk, floor, tf, ta = measure_tone(sa, float(f) * WANTED)
                lp.trace(f"{WANTED}f", tf, ta, offset_db=pad)
                lp.note(f"drive sweep {f/1e9:.3f} GHz   {s_db:+.1f} dB")
                # Input follows the commanded amplitude. The stage-A amp adds
                # <=0.3 dB of its own compression over this range, which is
                # small against the 1.0-vs-2.0 distinction being made.
                ins.append(np.interp(f, freqs, drive_at) + s_db)
                # RAW, matching chainlog's "store raw, add pad back on read"
                # contract -- the band-sweep save two calls below does the same.
                # (Previously stored pk+pad directly: harmless for the r-value,
                # since a slope is invariant to a constant offset, but it meant
                # any reader adding the pad again on load -- as the contract
                # says to -- got the pad counted twice and saw impossible
                # negative conversion loss. Found 2026-08-03.)
                outs.append(pk)
                print(f"      {s_db:+5.1f} dB  in {ins[-1]:+7.2f}  "
                      f"out {pk+pad:+7.2f} dBm")
            drive_in[f"{f/1e9:.3f}ghz"] = np.array(ins)
            drive_out[f"{f/1e9:.3f}ghz"] = np.array(outs)

        lp.done()
        settings = {"rbw_hz": float(sa.get_rbw()), "span_hz": SPAN_HZ,
                    "n_points": N_POINTS, "atten_db": atten_db,
                    "extra_gain_db": EXTRA_GAIN_DB,
                    "preemph": PREEMPH,
                    "preemph_db": [float(x) for x in pe_db],
                    "settle_s": SETTLE_S, "operating_vpp": float(vpp),
                    "element": ELEMENT, "doubler": DOUBLER, "tones": list(TONES),
                    "wanted": WANTED, "drive_reference": dpath,
                    "drive_range_db": DRIVE_RANGE_DB,
                    "drive_offset_db": float(drive_offset_db)}
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(freqs, lv, fl, drive_at, drive_in, drive_out, vpp, settings)


def report(freqs, lv, fl, drive_at, drive_in, drive_out, vpp, settings):
    pad = config.PAD_AT_INSTRUMENT_DB
    out = np.array(lv[WANTED]) + pad
    cl = drive_at - out
    cl_ref, pin_min, pin_nom, pin_max, supp_ref = doubler_ref()

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  output {out.max():+.2f} to {out.min():+.2f} dBm at "
          f"{WANTED*freqs[0]/1e9:.3f}-{WANTED*freqs[-1]/1e9:.3f} GHz, "
          f"flatness {out.max()-out.min():.2f} dB")
    print(f"  input  {drive_at.max():+.2f} to {drive_at.min():+.2f} dBm, "
          f"tilt {drive_at.max()-drive_at.min():.2f} dB")
    print(f"  CONVERSION LOSS {cl.min():.2f} to {cl.max():.2f} dB "
          f"(datasheet {DOUBLER} {cl_ref:.1f} dB"
          f"{' max' if DOUBLER == 'FX-05-410' else ' typ'})")

    print(f"\n  unwanted tones at the output -- this is the 3.2-3.9 GHz "
          f"filter's specification:")
    for n in TONES:
        if n == WANTED:
            continue
        h = np.array(lv[n]); floor = np.array(fl[n])
        dbc = h - np.array(lv[WANTED])
        det = (h - floor) >= 6.0
        band = f"{n*freqs[0]/1e9:.3f}-{n*freqs[-1]/1e9:.3f} GHz"
        if det.any():
            print(f"    {n}f  {band:>20}  worst {dbc[det].max():6.1f} dBc "
                  f"({det.sum()}/{len(h)} detected)"
                  + ("   <- the doubler's F1 leakage" if n == 1 else ""))
        else:
            print(f"    {n}f  {band:>20}  below the floor everywhere, "
                  f"better than {(floor-np.array(lv[WANTED])).max():.1f} dBc")
        if n == 1:
            print(f"        datasheet F1 suppression {supp_ref:.0f} dBc")

    print(f"\n  SATURATION -- tilt transfer ratio r = d(out)/d(in)")
    rs = []
    for k in drive_in:
        x, y = drive_in[k], drive_out[k]
        r_all = np.polyfit(x, y, 1)[0]
        # The top of the sweep is the operating point; fit the top third
        # separately, since that is the regime the chain actually runs in.
        m = x >= (x.max() - (x.max() - x.min()) / 3)
        r_top = np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 2 else float("nan")
        rs.append(r_top)
        print(f"    {k}  r = {r_all:.2f} over the whole {settings['drive_range_db']:g} dB, "
              f"{r_top:.2f} near the operating point")
    r = float(np.nanmean(rs))
    print(f"\n    mean r at the operating point: {r:.2f}")
    tilt_a = 2.17          # measured stage-A + filter tilt
    print(f"    -> across three doublers, r^3 = {r**3:.2f}, so stage A's "
          f"{tilt_a:.2f} dB of tilt becomes {tilt_a*r**3:.1f} dB at the AMC")
    win = config.AMC_WINDOW_DB
    if tilt_a * r ** 3 > win:
        print(f"    ** that EXCEEDS the AMC's {win:g} dB window on its own -- "
              f"pre-emphasis is mandatory, not optional **")
    else:
        print(f"    fits the AMC's {win:g} dB window with "
              f"{win-tilt_a*r**3:.1f} dB to spare, before the other terms")
    print(f"    AWG per-point scatter 0.19 dB rms -> {0.19*r**3:.2f} dB at the AMC")

    sess = chainlog.new_session(STAGE, "doubler")
    series = {f"{n}f": np.array(lv[n]) for n in TONES}
    series.update({f"{n}f_floor": np.array(fl[n]) for n in TONES})
    p = chainlog.save(sess, "doubler_band", STAGE, freqs, series, settings,
                      x_name="freqs_hz", thru=False,
                      arrangement=chain_description(),
                      notes=f"{ELEMENT}. RAW analyzer levels, add {pad:g} dB. "
                            f"x is the AWG frequency; tone n was measured at "
                            f"n times it. Drive reference: "
                            f"{settings['drive_reference']}")
    print(f"\n  saved {p}")
    if drive_in:
        k0 = list(drive_in)[0]
        p2 = chainlog.save(sess, "doubler_drive", STAGE, drive_in[k0],
                           {k: v for k, v in drive_out.items()}, settings,
                           x_name="drive_in_dbm", thru=False,
                           arrangement=chain_description(),
                           notes=f"2f output vs input level, for the tilt "
                                 f"transfer ratio. RAW analyzer levels, add "
                                 f"{pad:g} dB for the level at the doubler "
                                 f"output. drive_in_dbm is already referred "
                                 f"to the doubler's input plane (via "
                                 f"known_drive()) -- do not add the pad to it.")
        print(f"        {p2}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(WANTED * freqs / 1e9, out, "o-", ms=4)
    axes[0].set_xlabel("output frequency (GHz)"); axes[0].set_ylabel("dBm")
    axes[0].set_title(f"2f output ({out.max()-out.min():.2f} dB flat)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(freqs / 1e9, cl, "o-", ms=4, label="measured")
    axes[1].axhline(cl_ref, ls="--", color="k", alpha=0.6,
                    label=f"datasheet {cl_ref:.1f} dB")
    axes[1].set_xlabel("AWG frequency (GHz)"); axes[1].set_ylabel("dB")
    axes[1].set_title("conversion loss"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    for k in drive_in:
        axes[2].plot(drive_in[k], drive_out[k] + pad, "o-", ms=4, label=k)
    axes[2].set_xlabel("input at f (dBm)"); axes[2].set_ylabel("output at 2f (dBm)")
    axes[2].set_title(f"saturation: r = {r:.2f} (2 = square law, 0 = saturated)")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "doubler.png")
    fig.savefig(png, dpi=130)
    print(f"        {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_FREQS,
        float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
