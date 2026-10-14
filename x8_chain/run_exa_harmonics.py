"""Sweep the AWG across the band and measure, on the EXA, the fundamental AND
its harmonics at every point.

    AWG -> ZX60-2522MA-S+ -> PE87FL1012 -> 10 dB pad -> EXA
    (the 3 dB drive pad was removed 2026-07-31 to free AWG headroom)

WHAT THIS IS FOR
----------------
The scope cannot do this. measure_vpp() is an AC-RMS calculation, so it sums
everything present and cannot separate a carrier from its harmonics. The EXA
resolves them, which answers the question the 1-2 GHz filter exists to answer:

  the drive amp's 2nd harmonic falls at 3.228-3.855 GHz, which is EXACTLY
  doubler A's output band.

That degeneracy is not an accident -- doubling is doubling -- and it means no
filter downstream of the doubler can separate the amp's harmonic from the
wanted signal. They are the same frequency and coherent. Whatever H2 survives
into the HMC188 adds to its output as a frequency-dependent amplitude and
phase error, and the phase part is multiplied by 96 on the way to the AMC
output (x2, x2, then the AMC's x24). So the residual H2 measured here, in
dBc, is a direct input to how good the chain can be.

MEASUREMENT NOTESad we 
-----------------
Attenuation is set for the FUNDAMENTAL, which is the strongest thing present,
and then held fixed while the harmonics are measured. It has to be: the
attenuator is broadband, so backing it off to chase a weak harmonic would put
the full fundamental on the first mixer. That fixes the noise floor, and the
floor is measured and reported alongside every harmonic so a "not detected"
result carries its own limit rather than reading as a zero.

Levels are stored RAW as the analyzer reports them. config.PAD_AT_INSTRUMENT_DB
is added when reporting, never when saving.

Run:  python x8_chain/run_exa_harmonics.py [stage] [target_dBm] [n_freqs]
      target_dBm is the band MAXIMUM at the filter output; the level is
      set closed-loop on the EXA, ramping up from 6 dB below it.
      NO_SHOW=1 to skip the plot window.
"""
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


HARMONICS = (1, 2, 3)
SPAN_HZ = 50e6           # around each harmonic -- wide enough to catch the
                          # tone despite any frequency error, narrow enough
                          # that 801 points still resolve it cleanly
RBW_HZ = 100e3
N_POINTS = 801
SETTLE_S = 0.25
N_FREQS = 11             # coarser than the scope sweeps: this is about
                          # harmonic content, not fine flatness structure

EXA_ATTEN_DB = 20        # set for the fundamental, held for the harmonics
REF_LEVEL_DBM = 10

# The element under test, recorded with the data.
ELEMENT = os.environ.get(
    "ELEMENT", "PE87FL1012 1-2 GHz BPF after the drive amp")
# WHAT IS PHYSICALLY IN THE PATH. Overridable from the environment so a swap
# can be recorded at the moment it is made:
#     ELEMENT="VLF-2500+ DC-2500 LPF" python x8_chain/run_exa_harmonics.py
#
# This was got wrong once (2026-07-31): the filter was swapped back to the
# Pasternack and two runs saved as "VLF-2500+" because the constant was not
# edited. Recoverable only because the two filters leave different signatures
# in the data -- a stopband is flat across the band, a skirt is not. Do not
# rely on that next time.

# Target for the closed-loop level set: the band MAXIMUM at the FILTER output,
# which is what this measurement can actually see. The HMC188 sits one cable
# beyond it and wants +10..+20 dBm, nominal +15, so +15.5 here puts the strong
# band edge at its nominal drive.
DEFAULT_TARGET_DBM = 15.5

CONVERGE_TOL_DB = 0.20
MAX_TRIM_ITERS = 5
START_BACKOFF_DB = 6.0   # first measurement this far below target, so a wrong
                          # assumption shows up somewhere harmless


def measure_tone(sa, centre_hz):
    """Peak level in a span around centre_hz, plus the local noise floor.

    Returns (peak_dbm, floor_dbm, freqs_hz, amps_dbm). The floor is the trace
    median, which for a span containing one narrow tone is the noise, not the
    signal -- it gives every non-detection an explicit limit instead of a bare
    'nothing there'. The full trace comes back too so the live view can show
    what the analyzer actually sees rather than only the extracted number.
    """
    sa.set_freq(centre_hz, SPAN_HZ)
    time.sleep(0.05)
    f, a = sa.sweep_once()
    return float(np.max(a)), float(np.median(a)), f, a


def start_amplitude(stage_key, freq_hz, target_dbm):
    """A first AWG amplitude, from the measured AWG curve and datasheet gain.

    Deliberately NOT scaled from the earlier scope sessions: those were taken
    with a 3 dB pad in front of the amp that has since been removed, so at any
    given amplitude the amp is now driven 3 dB harder and the old anchor would
    overdrive it. Only has to be close -- the loop below corrects from
    measurement -- and it starts START_BACKOFF_DB below target regardless.
    """
    st = config.STAGES[stage_key]
    resp = mc.load_scope_band_response()
    ref_vpp = mc.AWG_OPERATING_AMP_VPP
    awg_ref = (float(np.interp(freq_hz, resp[0], resp[1])) if resp is not None
               else config.dbm_from_vpp(ref_vpp))
    # filter output = AWG - cable + gain - filter IL   (IL guessed at 0.5 dB;
    # a lowpass with its corner at 2.5 GHz should be near-lossless at 1.9 GHz)
    chain_gain = -mc.CABLE_IL_DB + (st["gain_typ_db"] or 0.0) \
        - st["drive_pad_db"] - 0.5
    want_awg = (target_dbm - START_BACKOFF_DB) - chain_gain
    return float(np.clip(ref_vpp * 10 ** ((want_awg - awg_ref) / 20),
                         0.001, config.AWG_MAX_VPP))


def converge_level(sa, awg, f_ref, target_dbm, vpp, lp=None):
    """Trim the AWG amplitude until the filter output at f_ref hits target.

    Runs entirely on the EXA, so it needs no prior calibration of the amp, the
    cable or the filter -- it closes the loop on the quantity being set. Every
    step is predicted from the measurement below it and checked against the
    analyzer's mixer limit before being commanded.
    """
    pad = config.PAD_AT_INSTRUMENT_DB
    want_read = target_dbm - pad
    mixer_ceiling = config.EXA_MAX_MIXER_DBM + EXA_ATTEN_DB
    for it in range(MAX_TRIM_ITERS):
        awg.duc_cw_setup(float(f_ref), amplitude_vpp=float(vpp), channel=1)
        time.sleep(SETTLE_S)
        pk, floor, tf, ta = measure_tone(sa, float(f_ref))
        if lp is not None:
            lp.trace("H1", tf, ta, offset_db=pad)
            lp.note(f"setting level: {vpp:.4f} Vpp -> {pk+pad:+.2f} dBm "
                    f"(target {target_dbm:+.2f})")
        err = want_read - pk
        print(f"    {vpp:.4f} Vpp -> {pk+pad:+7.2f} dBm at the filter output "
              f"({err:+.2f} dB from target)")
        if abs(err) <= CONVERGE_TOL_DB:
            return vpp, True
        new = float(np.clip(vpp * 10 ** (err / 20), 0.001, config.AWG_MAX_VPP))
        if pk + err > mixer_ceiling:
            print(f"    ** the next step would put {pk+err:+.1f} dBm on the "
                  f"analyzer, past the {mixer_ceiling:+.1f} dBm its "
                  f"{EXA_ATTEN_DB:.0f} dB attenuation allows. STOPPING. **")
            return vpp, False
        if new >= config.AWG_MAX_VPP - 1e-9 and err > 0:
            print(f"    ** {err:+.2f} dB short at the AWG's "
                  f"{config.AWG_MAX_VPP} Vpp ceiling -- target unreachable **")
            return new, False
        vpp = new
    print(f"    ** did not converge in {MAX_TRIM_ITERS} trims **")
    return vpp, False


def run(stage_key="A", target_out_dbm=None, n_freqs=N_FREQS):
    st = config.STAGES[stage_key]
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = st["band_hz"]
    freqs = np.linspace(lo, hi, int(n_freqs))

    target = DEFAULT_TARGET_DBM if target_out_dbm is None else target_out_dbm
    expect_amp_out = target      # for the safety sizing below; the filter can
                                  # only take away, so this is conservative
    vpp = start_amplitude(stage_key, float(freqs[0]), target)

    print("=" * 74)
    print(f"EXA HARMONICS -- stage {stage_key}")
    drive = f"{st['drive_pad_db']:g} dB pad -> " if st["drive_pad_db"] else ""
    print(f"  AWG CH1 -> {drive}{st['part']} -> "
          f"{ELEMENT.split(' after')[0]} -> {pad:g} dB pad -> EXA")
    print("=" * 74)
    print(f"  {len(freqs)} points {lo/1e9:.3f}-{hi/1e9:.3f} GHz, "
          f"harmonics {HARMONICS}")
    print(f"  TARGET {target:+.2f} dBm at the filter output (band max), set "
          f"closed-loop on the EXA")
    print(f"  starting {START_BACKOFF_DB:g} dB low at {vpp:.4f} Vpp")
    print(f"  H2 lands at {2*lo/1e9:.3f}-{2*hi/1e9:.3f} GHz -- doubler A's "
          f"OUTPUT band, which is why it matters")

    # Safety: the fundamental is the strongest thing at the EXA. Size it on
    # the AMP output with no allowance for the filter -- the filter can only
    # take away, so this is the conservative direction.
    at_exa = expect_amp_out - pad
    need = config.required_exa_atten_db(at_exa)
    print(f"\n  expect ~{at_exa:+.1f} dBm at the EXA connector "
          f"(after the {pad:g} dB pad)")
    print(f"  EXA damage level {config.EXA_DAMAGE_DBM:+.0f} dBm -> "
          f"{config.EXA_DAMAGE_DBM-at_exa:.0f} dB of margin, no risk")
    print(f"  internal attenuation needs >= {need:.0f} dB to keep the first "
          f"mixer at {config.EXA_MAX_MIXER_DBM:+.0f} dBm; using "
          f"{EXA_ATTEN_DB:.0f} dB")
    if EXA_ATTEN_DB < need:
        print(f"  ** {EXA_ATTEN_DB} dB is NOT enough -- the analyzer would "
              f"compress and invent harmonics that are not in the signal, "
              f"which is exactly what this run is trying to measure. ABORT **")
        return

    awg = AWG()
    sa = SpecAn()
    lv = {n: [] for n in HARMONICS}
    fl = {n: [] for n in HARMONICS}
    try:
        sa.set_trace_mode("NORM")
        sa.set_attenuation(EXA_ATTEN_DB)
        sa.set_ref_level(REF_LEVEL_DBM)
        sa.set_rbw(RBW_HZ)
        sa.set_vbw(None)
        sa.set_points(N_POINTS)
        print(f"  EXA: RBW {sa.get_rbw()/1e3:.0f} kHz, {N_POINTS} points, "
              f"span {SPAN_HZ/1e6:.0f} MHz per tone\n")

        # Live view of the ACTUAL analyzer traces, one panel per harmonic,
        # plus a panel accumulating the extracted peaks. Levels are shown
        # referred to the DUT output (the 10 dB pad added back).
        lp = LiveSpectrum([f"H{n}" for n in HARMONICS],
                          f"stage {stage_key}: {ELEMENT.split(' after')[0]}"
                          f"   ({pad:g} dB pad added back)")

        print(f"  setting the level at {freqs[0]/1e9:.3f} GHz "
              f"(the band's strong edge):")
        vpp, converged = converge_level(sa, awg, float(freqs[0]), target, vpp,
                                        lp=lp)
        if not converged:
            print("  ** level not set -- sweeping anyway at the amplitude "
                  "reached, but the numbers are not at the requested level **")
        print()

        for i, f in enumerate(freqs):
            if i == 0:
                awg.duc_cw_setup(float(f), amplitude_vpp=float(vpp), channel=1)
            else:
                awg.duc_cw_step(float(f), channel=1)
            time.sleep(SETTLE_S)

            row = []
            for n in HARMONICS:
                pk, floor, tf, ta = measure_tone(sa, float(f) * n)
                lv[n].append(pk); fl[n].append(floor)
                row.append((n, pk, floor))
                lp.trace(f"H{n}", tf, ta, offset_db=pad)
                lp.summary(f"H{n}", f / 1e9, pk + pad)
            f0 = row[0][1]
            lp.note(f"stage {stage_key}: {ELEMENT.split(' after')[0]}   "
                    f"AWG {f/1e9:.3f} GHz   ({i+1}/{len(freqs)})")
            txt = "  ".join(
                f"H{n} {pk+pad:+7.2f}" +
                (f" ({pk-f0:+6.1f} dBc)" if n > 1 else "") +
                ("*" if pk - floor < 6.0 else " ")
                for n, pk, floor in row)
            print(f"  {f/1e9:.3f} GHz  {txt}")
        lp.done()
        print("\n  * = within 6 dB of the local noise floor, i.e. a LIMIT "
              "rather than a measurement")

        settings = {"rbw_hz": float(sa.get_rbw()), "span_hz": SPAN_HZ,
                    "n_points": N_POINTS, "atten_db": EXA_ATTEN_DB,
                    "ref_level_dbm": REF_LEVEL_DBM, "settle_s": SETTLE_S,
                    "operating_vpp": float(vpp), "element": ELEMENT,
                    "harmonics": list(HARMONICS),
                    "expected_amp_out_dbm": float(expect_amp_out)}
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", sa.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(stage_key, freqs, lv, fl, vpp, settings)


def report(stage_key, freqs, lv, fl, vpp, settings):
    pad = config.PAD_AT_INSTRUMENT_DB
    st = config.STAGES[stage_key]
    f0 = np.array(lv[1])

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    out = f0 + pad
    print(f"  fundamental {out.max():+.2f} to {out.min():+.2f} dBm at the "
          f"filter output, flatness {out.max()-out.min():.2f} dB")
    exp = settings.get("expected_amp_out_dbm")
    if exp is not None:
        print(f"  target was {exp:+.2f} dBm at the band max -> "
              f"{out.max()-exp:+.2f} dB")
    print(f"  AWG {settings['operating_vpp']:.4f} Vpp, "
          f"{20*np.log10(settings['operating_vpp']/config.AWG_MAX_VPP):+.2f} dB "
          f"from the {config.AWG_MAX_VPP} Vpp ceiling")
    print(f"    -> {-20*np.log10(settings['operating_vpp']/config.AWG_MAX_VPP):.2f} dB "
          f"of headroom for pre-emphasis; the band needs "
          f"{out.max()-out.min():.2f} dB to flatten by raising the weak edge")

    for n in HARMONICS:
        if n == 1:
            continue
        h = np.array(lv[n]); floor = np.array(fl[n])
        dbc = h - f0
        detected = (h - floor) >= 6.0
        print(f"\n  H{n} at {n*st['band_hz'][0]/1e9:.3f}-"
              f"{n*st['band_hz'][1]/1e9:.3f} GHz")
        if detected.any():
            print(f"    worst {dbc[detected].max():.1f} dBc "
                  f"(level {h[detected].max()+pad:+.1f} dBm), "
                  f"detected at {detected.sum()}/{len(h)} points")
        if (~detected).any():
            lim = (floor - f0)[~detected]
            print(f"    below the noise floor at {(~detected).sum()}/{len(h)} "
                  f"points, limit better than {lim.max():.1f} dBc there")
        if n == 2:
            worst = dbc[detected].max() if detected.any() else (floor-f0).max()
            r = 10 ** (worst / 20)
            print(f"    -> as a coherent product on doubler A's output: "
                  f"+-{20*np.log10(1+r):.3f} dB amplitude, "
                  f"{np.degrees(np.arcsin(min(r, 1.0))):.2f} deg phase")
            print(f"       which the remaining x4 and the AMC's x24 turn into "
                  f"{96*np.degrees(np.arcsin(min(r,1.0))):.1f} deg at "
                  f"310-370 GHz")

    sess = chainlog.new_session(stage_key, "exa_harmonics")
    series = {f"H{n}": np.array(lv[n]) for n in HARMONICS}
    series.update({f"H{n}_floor": np.array(fl[n]) for n in HARMONICS})
    p = chainlog.save(sess, "exa_harmonics", stage_key, freqs, series,
                      settings, x_name="freqs_hz", thru=False,
                      notes=f"EXA, {ELEMENT}. Levels RAW at the analyzer "
                            f"connector; add {pad:g} dB. x axis is the AWG "
                            f"(fundamental) frequency -- Hn was measured at "
                            f"n times each value.")
    print(f"\n  saved {p}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for n in HARMONICS:
        axes[0].plot(freqs / 1e9, np.array(lv[n]) + pad, "o-", ms=4,
                     label=f"H{n} (measured at {n}f)")
        axes[0].plot(freqs / 1e9, np.array(fl[n]) + pad, ":", lw=1, alpha=0.5)
    axes[0].set_xlabel("AWG frequency (GHz)")
    axes[0].set_ylabel("level (dBm)")
    axes[0].set_title("fundamental and harmonics (dotted = noise floor)")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    for n in HARMONICS:
        if n == 1:
            continue
        axes[1].plot(freqs / 1e9, np.array(lv[n]) - f0, "o-", ms=4, label=f"H{n}")
        axes[1].plot(freqs / 1e9, np.array(fl[n]) - f0, ":", lw=1, alpha=0.5)
    axes[1].set_xlabel("AWG frequency (GHz)")
    axes[1].set_ylabel("dBc")
    axes[1].set_title("harmonic suppression (dotted = floor limit)")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "harmonics.png")
    fig.savefig(png, dpi=130)
    print(f"  saved {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "A",
        float(sys.argv[2]) if len(sys.argv) > 2 else None,
        int(sys.argv[3]) if len(sys.argv) > 3 else N_FREQS)
