"""Bench script: ZX90-2-24-S+ doubler bring-up -- wanted output + spurs,
driven directly from the (underdriven, no pad/2nd-amp stage) interstage amp.

Current physical setup (2026-10-08, doubler connected directly after the
single interstage amp -- the 12.5 dB pad + second interstage amp from the
designed chain are NOT in place yet):

    AWG CH1 -> ZX60-2522MA-S+ (temp stand-in for ZX60-P105LN+)
            -> VBFZ-2340-S+ BPF
            -> ZX90-3-812-S+ tripler
            -> VHF-4600+ HPF
            -> ZX60-06183LN+ (temp stand-in for ZX60-83LN-S+)
            -> ZX90-2-24-S+ doubler
            -> 10 dB pad -> Keysight N9010A EXA

Known going in (see project_buy_chain_x6 memory for the full trail):
  - VLF-8400+ LPF still not installed -- F4/F5 reach the doubler input
    essentially unfiltered.
  - The interstage amp (ZX60-06183LN+) measurably WORSENED F1/F2 (real
    excess out-of-band gain, bench-confirmed, not predicted by its
    datasheet) and left F4 similar/slightly worse, F5 ~unchanged.
  - The doubler is being driven directly by that amp's output, +6.1 to
    +10.2 dBm across the band -- 1-6 dB UNDER the ZX90-2-24-S+'s ~+11 dBm
    drive floor (+12.3 dBm was the design target). Per the project's
    standing rule (diode multipliers show worse spur rejection AND worse
    conversion loss when backed off), expect this run's numbers to be
    worse than final hardware on every axis, not just "not yet filtered".
  This run is a real data point, not a representative one -- useful for
  "does the doubler work at all / what's actually coming out", not for
  judging final spec compliance.

Why harmonics 1-10 of f_in (not just a "wanted + 4 spurs" list like the
tripler scripts): the doubler's input is already a composite of F1-F5
(1x-5x of f_in, all present from the tripler stage). A doubler's dominant
nonlinear terms are ~2nd order, so its output consists of: each input
tone passed through at 1x (leakage), each input tone's own 2nd harmonic
(2x, 4x, 6x, 8x, 10x of f_in), and pairwise sum/difference products
between any two input tones (e.g. F3+F4 = 3+4 = 7x f_in, F1+F5 = 6x,
F4-F3 = 1x, etc.). Every one of these lands EXACTLY on an integer
multiple of f_in, because every input tone already is one -- so scanning
harmonics 1-10 at exact predicted frequencies (DUC NCO is frequency-exact,
no searching needed) catches essentially all 2nd-order content without
having to separately derive which combination produced it. The WANTED
tone is 2xF3 = 6x f_in (the doubler's intended output).

f_in 2.10-2.62 GHz -> harmonics span 2.10 GHz (1x) to 26.2 GHz (10x) --
within this EXA's installed 32 GHz option (confirmed via datasheet,
project_exa_specan memory; the "12.42 GHz highest confirmed" note there
was just the highest frequency previously USED, not an instrument limit).

Method identical to the tripler/amp scripts otherwise: wide overview
(still noise-floor-limited per the standing caveat -- trust the refine
numbers) + narrow refine at each exact predicted harmonic.

Data saved under x6_chain/doubler-outputs/<timestamp>/.

Run with `python run_doubler_outputs.py` from the project root.
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
X6_DOUBLER_DIR = os.path.join(PROJECT_ROOT, "x6_chain", "doubler-outputs")

# ── bench config ────────────────────────────────────────────────────
AMP_VPP = 0.124
FREQ_START_HZ = 2.10e9
FREQ_STOP_HZ  = 2.62e9
FREQ_STEP_HZ  = 0.02e9
FREQS_HZ = [round(f, 9) for f in
            np.arange(FREQ_START_HZ, FREQ_STOP_HZ + FREQ_STEP_HZ / 2, FREQ_STEP_HZ)]

EXTERNAL_ATTEN_DB = 10.0   # same physical pad, now after the doubler
ATTEN_DB = 10              # EXA internal atten -- some tones (e.g. input
                           # leakage) could ride close to the amp's
                           # +6..+10 dBm input level if doubler isolation
                           # is poor, so keep real margin
REF_LEVEL_DBM = 5          # headroom above that same worst case

WANTED_N = 6                # 2x of F3 (3x f_in) = the doubler's intended
                             # output
HARMONICS = tuple(range(1, 11))   # 1x-10x f_in -- see docstring for why
                                   # this covers all dominant 2nd-order
                                   # doubler products

OV_CENTER_HZ = 14.15e9       # (1x min 2.10 + 10x max 26.2) / 2
OV_SPAN_HZ   = 25.0e9
OV_POINTS    = 8001
MAX_RBW_HZ   = 3e6
OV_PROMINENCE_DB = 15.0
HARMONIC_MATCH_HZ = 15e6

REFINE_SPAN_HZ = 5e6
REFINE_RBW_HZ  = 100e3
REFINE_POINTS  = 401
SETTLE_S = 0.25

NOTES = ("Doubler bring-up (underdriven, no pad/2nd-amp stage): AWG -> "
         "ZX60-2522MA-S+ -> VBFZ-2340-S+ -> ZX90-3-812-S+ tripler -> "
         "VHF-4600+ HPF -> ZX60-06183LN+ -> ZX90-2-24-S+ doubler -> 10 dB "
         "pad -> EXA, 0.124 Vpp AWG drive. Doubler driven +6.1..+10.2 dBm, "
         "below its ~+11 dBm floor -- not representative of final "
         "hardware. Harmonics 1-10 of f_in tracked; wanted = 6x (2xF3).")


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
        f"H{n}_{f_in/1e9:.3f}GHz".replace(".", "p"), NOTES,
        {"atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM,
         "rbw_hz": REFINE_RBW_HZ, "vbw_hz": REFINE_RBW_HZ,
         "points": REFINE_POINTS, "trace_mode": "NORM",
         "cw_input_freq_hz": f_in, "harmonic_n": n,
         "predicted_hz": freq_hz, "amp_vpp": AMP_VPP,
         "external_atten_db": EXTERNAL_ATTEN_DB, "role": f"H{n} refine"},
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

    wanted = harmonics[WANTED_N]
    print(f"\n  f_in {f_in/1e9:.3f} GHz:")
    print(f"    H{WANTED_N} (wanted, 2xF3) = {wanted:7.2f} dBm "
          f"@ {WANTED_N*f_in/1e9:.3f} GHz")
    for n in HARMONICS:
        if n == WANTED_N:
            continue
        dbc = wanted - harmonics[n]
        print(f"    H{n} ({n}x f_in) = {harmonics[n]:7.2f} dBm  ({dbc:5.1f} dBc)")
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

    run_dir = os.path.join(X6_DOUBLER_DIR, time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== Doubler output survey: {len(FREQS_HZ)} input points, "
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
    wanted_s = series[WANTED_N]

    fig, (ax_abs, ax_dbc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for n in HARMONICS:
        style = dict(ms=4, lw=1.8, color="k") if n == WANTED_N else dict(ms=3, lw=1.1)
        label = f"H{n} (wanted, 2xF3)" if n == WANTED_N else f"H{n} ({n}x f_in)"
        ax_abs.plot(freqs / 1e9, series[n], "o-", label=label, **style)
    ax_abs.set_ylabel("dBm at EXA (pad corrected out)")
    ax_abs.set_title(f"Doubler output survey -- wanted + harmonics 1-10 "
                     f"({AMP_VPP} Vpp drive, {EXTERNAL_ATTEN_DB:.0f} dB pad corrected)")
    ax_abs.grid(True, alpha=0.3); ax_abs.legend(fontsize=7, ncol=2)

    for n in HARMONICS:
        if n == WANTED_N:
            continue
        dbc = wanted_s - series[n]
        ax_dbc.plot(freqs / 1e9, dbc, "o-", ms=3, lw=1.1, label=f"H{n} dBc")
    ax_dbc.set_xlabel(f"input GHz (harmonics at {'/'.join(f'{n}x' for n in HARMONICS)})")
    ax_dbc.set_ylabel("dBc below H6 (wanted)")
    ax_dbc.set_title("All tracked harmonics relative to the wanted 2xF3 tone")
    ax_dbc.grid(True, alpha=0.3); ax_dbc.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    png_path = os.path.join(run_dir, "_doubler_overview.png")
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

    return dict(freqs=freqs, **{f"h{n}": series[n] for n in HARMONICS},
                unidentified=all_unidentified, png_path=png_path, run_dir=run_dir)


if __name__ == "__main__":
    run()
