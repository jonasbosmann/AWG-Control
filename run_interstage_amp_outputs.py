"""Bench script: post-interstage-amp survey -- wanted output + spurs, one
amplification stage further than run_tripler_outputs.py.

Current physical setup (2026-10-08, ZX60-06183LN+ now connected after the
HPF; this is a substitute for the not-yet-delivered ZX60-83LN-S+ first
interstage amp -- the real part needed a +12 VDC supply via one other
candidate (ZX60-123LN-S+) that wasn't on hand, so ZX60-06183LN+ (+5 V) was
used instead):

    AWG CH1 -> ZX60-2522MA-S+ (temp stand-in for ZX60-P105LN+)
            -> VBFZ-2340-S+ BPF
            -> ZX90-3-812-S+ tripler
            -> VHF-4600+ HPF
            -> ZX60-06183LN+ (temp stand-in for ZX60-83LN-S+)
            -> 10 dB pad -> Keysight N9010A EXA

VLF-8400+ LPF still not installed (on order) -- F4/F5 still reach this amp
unfiltered, same caveat as the HPF-only run.

Why this amp substitution matters for the F4/F5 problem specifically
(datasheet-checked, not assumed -- project_buy_chain_x6 memory):
  - The REAL part, ZX60-83LN-S+, is spec'd 0.5-8 GHz. F4 (8.40-10.48 GHz)
    and F5 (10.50-13.10 GHz) sit entirely outside its spec'd range, so it
    would have given some INCIDENTAL extra rejection of both, on top of
    whatever the (still missing) LPF provides -- a bonus, not something
    the design depends on, but a bonus nonetheless.
  - ZX60-06183LN+ is spec'd 6-18 GHz, ~20-21 dB gain FLAT across that
    whole range -- F3 (6.30-7.86), F4 (8.40-10.48), AND F5 (10.50-13.10)
    all land inside its passband and get amplified by essentially the
    SAME ~20-21 dB. Unlike the real part, this substitute gives F4/F5 a
    free ride right alongside the wanted signal -- no incidental help at
    all. (It does help on the OTHER side: F1 2.10-2.62 and F2 4.20-5.24
    GHz fall BELOW its 6 GHz corner, so they should come out suppressed
    rather than re-amplified -- a bonus on top of the HPF, but F1/F2 were
    already in good shape after the HPF, so this mostly doesn't matter.)
  - Net prediction: F3/F4/F5 dBc spacing should come out roughly
    UNCHANGED from the tripler+HPF run (same relative levels, just all
    ~20-21 dB hotter in absolute terms); F1/F2 should drop further than
    they already had, possibly into the noise floor.

Level budget note: this run expects levels ~20 dB hotter than the
tripler+HPF run (ZX60-06183LN+'s ~21 dB gain), so ATTEN_DB/REF_LEVEL_DBM
are raised back up from that run's sensitivity-tuned values. Predicted
peak (F3 at the low end of f_in, ~-12.6 dBm pre-amp + ~21 dB gain =
~+8.4 dBm at the amp output) is still well inside this amp's +14/+15 dBm
P1dB and +17 dBm abs-max input rating with headroom to spare -- no
compression/damage concern at these drive levels.

Method identical to run_tripler_outputs.py (wide overview + narrow refine
per predicted harmonic); see that script's docstring for the full
rationale. Overview remains noise-floor-limited per the standing caveat
in project_buy_chain_x6 memory -- trust the refine numbers, not the
overview's "unidentified" count, for anything subtler than ~20+ dB.

Data saved under x6_chain/interstage-amp-outputs/<timestamp>/.

Run with `python run_interstage_amp_outputs.py` from the project root.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awg import AWG
from specan import SpecAn, find_peaks
import specanlog

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
X6_AMP_DIR = os.path.join(PROJECT_ROOT, "x6_chain", "interstage-amp-outputs")

# ── bench config ────────────────────────────────────────────────────
AMP_VPP = 0.124           # unchanged -- still set at the AWG, upstream of
                           # everything in this docstring's chain
FREQ_START_HZ = 2.10e9
FREQ_STOP_HZ  = 2.62e9
FREQ_STEP_HZ  = 0.02e9
FREQS_HZ = [round(f, 9) for f in
            np.arange(FREQ_START_HZ, FREQ_STOP_HZ + FREQ_STEP_HZ / 2, FREQ_STEP_HZ)]

EXTERNAL_ATTEN_DB = 10.0   # same physical pad, now after the interstage amp
ATTEN_DB = 10              # raised from the tripler+HPF run's 0 dB -- this
                           # stage runs ~20 dB hotter (ZX60-06183LN+ gain)
REF_LEVEL_DBM = 0          # raised from -10 dBm for the same reason --
                           # adjust at the bench if F3 clips or sits low

# Tracked harmonics -- same set as the tripler/HPF runs (F5 confirmed real
# and dominant there; still tracked here to see what one more gain stage
# does to it).
HARMONICS = (1, 2, 3, 4, 5)

# Overview window: unchanged from run_tripler_outputs.py -- still needs to
# hold 1x..5x of f_in across the whole swept input range.
OV_CENTER_HZ = 7.55e9
OV_SPAN_HZ   = 11.5e9
OV_POINTS    = 8001
MAX_RBW_HZ   = 3e6
OV_PROMINENCE_DB = 15.0
HARMONIC_MATCH_HZ = 15e6

REFINE_SPAN_HZ = 5e6
REFINE_RBW_HZ  = 100e3
REFINE_POINTS  = 401
SETTLE_S = 0.25

# ZX90-3-812-S+ datasheet spurs, same reference point as before -- these
# describe the TRIPLER's own spec, still the right yardstick for whether
# anything downstream has made the relative spur picture worse.
DATASHEET_DBC = {1: 30.0, 2: 48.0, 4: 48.0}

NOTES = ("Post-interstage-amp survey: AWG -> ZX60-2522MA-S+ -> VBFZ-2340-S+ "
         "-> ZX90-3-812-S+ tripler -> VHF-4600+ HPF -> ZX60-06183LN+ "
         "(substitute for ZX60-83LN-S+, 6-18 GHz passband) -> 10 dB pad -> "
         "EXA, 0.124 Vpp AWG drive. VLF-8400+ LPF still not installed -- "
         "F4/F5 dBc expected roughly unchanged from the tripler+HPF run "
         "since this amp's passband covers F3/F4/F5 with near-equal gain.")


def _settle_and_read_rbw(specan):
    time.sleep(0.3)
    return specan.get_rbw()


def overview_sweep(specan, run_dir, f_in):
    specan.set_freq(OV_CENTER_HZ, OV_SPAN_HZ)
    specan.set_points(OV_POINTS)
    point_spacing = OV_SPAN_HZ / (OV_POINTS - 1)
    specan.set_rbw(min(max(point_spacing * 3, 1e6), MAX_RBW_HZ))
    specan.set_vbw(min(max(point_spacing * 3, 1e6), MAX_RBW_HZ))
    specan.set_trace_mode("NORM")
    actual_rbw = _settle_and_read_rbw(specan)
    freqs, amps = specan.sweep_once()

    specanlog.save_trace(
        f"overview_{f_in/1e9:.3f}GHz".replace(".", "p"), NOTES,
        {"atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM,
         "rbw_hz": actual_rbw, "vbw_hz": actual_rbw, "points": OV_POINTS,
         "trace_mode": "NORM", "cw_input_freq_hz": f_in, "amp_vpp": AMP_VPP,
         "external_atten_db": EXTERNAL_ATTEN_DB, "role": "wide overview"},
        freqs, amps, subdir=run_dir)
    return freqs, amps


def refine_point(specan, run_dir, f_in, n, freq_hz, first):
    specan.set_freq(freq_hz, REFINE_SPAN_HZ)
    specan.set_points(REFINE_POINTS)
    specan.set_rbw(REFINE_RBW_HZ)
    specan.set_vbw(REFINE_RBW_HZ)
    specan.set_trace_mode("NORM")
    time.sleep(SETTLE_S)
    freqs, amps = specan.sweep_once()

    i = int(np.argmax(amps))
    peak_dbm = float(amps[i]) + EXTERNAL_ATTEN_DB
    peak_hz = float(freqs[i])

    specanlog.save_trace(
        f"F{n}_{f_in/1e9:.3f}GHz".replace(".", "p"), NOTES,
        {"atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM,
         "rbw_hz": REFINE_RBW_HZ, "vbw_hz": REFINE_RBW_HZ,
         "points": REFINE_POINTS, "trace_mode": "NORM",
         "cw_input_freq_hz": f_in, "harmonic_n": n,
         "predicted_hz": freq_hz, "amp_vpp": AMP_VPP,
         "external_atten_db": EXTERNAL_ATTEN_DB, "role": f"F{n} refine"},
        freqs, amps, subdir=run_dir)
    return peak_hz, peak_dbm


def run_input_point(awg, specan, f_in, run_dir, first):
    if first:
        awg.duc_cw_setup(f_in, amplitude_vpp=AMP_VPP)
    else:
        awg.duc_cw_step(f_in)
    time.sleep(SETTLE_S)

    ov_freqs, ov_amps = overview_sweep(specan, run_dir, f_in)
    ov_peaks = find_peaks(ov_freqs, ov_amps, min_prominence_db=OV_PROMINENCE_DB,
                           min_spacing_hz=HARMONIC_MATCH_HZ)

    harmonics = {}
    for n in HARMONICS:
        _, dbm = refine_point(specan, run_dir, f_in, n, n * f_in, first=False)
        harmonics[n] = dbm

    unidentified = []
    for pf, pa in ov_peaks:
        if any(abs(pf - n * f_in) <= HARMONIC_MATCH_HZ for n in HARMONICS):
            continue
        unidentified.append((pf, pa + EXTERNAL_ATTEN_DB))

    f3 = harmonics[3]
    print(f"\n  f_in {f_in/1e9:.3f} GHz:")
    print(f"    F3 (wanted, 3x) = {f3:7.2f} dBm @ {3*f_in/1e9:.3f} GHz")
    for n in HARMONICS:
        if n == 3:
            continue
        dbc = f3 - harmonics[n]
        spec = DATASHEET_DBC.get(n)
        if spec is None:
            print(f"    F{n} ({n}x) = {harmonics[n]:7.2f} dBm  "
                  f"({dbc:5.1f} dBc, no datasheet spec)")
        else:
            flag = "  <-- WORSE than datasheet" if dbc < spec else ""
            print(f"    F{n} ({n}x) = {harmonics[n]:7.2f} dBm  "
                  f"({dbc:5.1f} dBc, datasheet {spec:.0f}){flag}")
    if unidentified:
        for pf, pa in unidentified:
            print(f"    UNIDENTIFIED peak: {pa:7.2f} dBm @ {pf/1e9:.4f} GHz "
                  f"(overview precision, not refined)")
    else:
        print("    no unidentified peaks in the overview")

    return harmonics, unidentified


def run():
    awg = AWG()
    specan = SpecAn()
    specan.set_attenuation(ATTEN_DB)
    specan.set_ref_level(REF_LEVEL_DBM)

    run_dir = os.path.join(X6_AMP_DIR, time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== Post-interstage-amp survey: {len(FREQS_HZ)} input points, "
          f"{FREQS_HZ[0]/1e9:.3f}-{FREQS_HZ[-1]/1e9:.3f} GHz, "
          f"drive {AMP_VPP} Vpp ===")

    series = {n: [] for n in HARMONICS}
    all_unidentified = []
    try:
        for i, f_in in enumerate(FREQS_HZ):
            harmonics, unidentified = run_input_point(awg, specan, f_in, run_dir,
                                                        first=(i == 0))
            for n in HARMONICS:
                series[n].append(harmonics[n])
            all_unidentified.extend((f_in, pf, pa) for pf, pa in unidentified)
    finally:
        try:
            awg.stop()
        except Exception as e:
            print(f"cleanup (awg.stop): {e}")

    freqs = np.array(FREQS_HZ)
    series = {n: np.array(v) for n, v in series.items()}
    f3s = series[3]

    fig, (ax_abs, ax_dbc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for n in HARMONICS:
        style = dict(ms=4, lw=1.8, color="k") if n == 3 else dict(ms=3, lw=1.2)
        label = "F3 (3x, wanted)" if n == 3 else f"F{n} ({n}x{', leak' if n==1 else ''})"
        ax_abs.plot(freqs / 1e9, series[n], "o-", label=label, **style)
    ax_abs.set_ylabel("dBm at EXA (pad corrected out)")
    ax_abs.set_title(f"Post-interstage-amp survey -- wanted + spurs "
                     f"({AMP_VPP} Vpp drive, {EXTERNAL_ATTEN_DB:.0f} dB pad corrected)")
    ax_abs.grid(True, alpha=0.3); ax_abs.legend(fontsize=8)

    for n in HARMONICS:
        if n == 3:
            continue
        dbc = f3s - series[n]
        line, = ax_dbc.plot(freqs / 1e9, dbc, "o-", ms=3, lw=1.2, label=f"F{n} dBc")
        spec = DATASHEET_DBC.get(n)
        if spec is not None:
            ax_dbc.axhline(spec, color=line.get_color(), ls="--", lw=1, alpha=0.6,
                           label=f"F{n} datasheet ({spec:.0f})")
    ax_dbc.set_xlabel(f"input GHz (output harmonics at {'/'.join(f'{n}x' for n in HARMONICS)})")
    ax_dbc.set_ylabel("dBc below F3")
    ax_dbc.set_title("Spur rejection vs. datasheet where specified (higher = better)")
    ax_dbc.grid(True, alpha=0.3); ax_dbc.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    png_path = os.path.join(run_dir, "_interstage_amp_overview.png")
    fig.savefig(png_path, dpi=130)
    print(f"\nsaved plot: {png_path}")

    if all_unidentified:
        print(f"\n{len(all_unidentified)} unidentified peak(s) across the scan "
              f"(overview precision -- see console log above / saved overview "
              f"traces for exact values):")
        for f_in, pf, pa in all_unidentified:
            print(f"  f_in {f_in/1e9:.3f} GHz: {pa:.2f} dBm @ {pf/1e9:.4f} GHz")
    else:
        print("\nNo unidentified peaks anywhere in the scan.")
    print("(single N=1 run -- not yet repeated, see feedback_bench_rigor)")

    return dict(freqs=freqs, **{f"f{n}": series[n] for n in HARMONICS},
                unidentified=all_unidentified, png_path=png_path, run_dir=run_dir)


if __name__ == "__main__":
    run()
