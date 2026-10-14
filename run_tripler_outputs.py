"""Bench script: ZX90-3-812-S+ tripler bring-up -- wanted output + spurs.

Current physical setup (2026-10-08, VHF-4600+ HPF now connected after the
tripler; VLF-8400+ LPF still on order):

    AWG CH1 -> ZX60-2522MA-S+ (temp stand-in for ZX60-P105LN+)
            -> VBFZ-2340-S+ BPF
            -> ZX90-3-812-S+ tripler
            -> VHF-4600+ HPF
            -> 10 dB pad -> Keysight N9010A EXA

(Same 10 dB pad as the previous bare-tripler run, just now sitting after
the HPF instead of directly after the tripler -- correct this script's
EXTERNAL_ATTEN_DB if that assumption is wrong.)

Predicted HPF effect, worked out from the memory's measured passband
(VHF-4600+ "pass 5.2-10.5 GHz" -- project_buy_chain_x6) against this run's
harmonic frequencies (f_in 2.10-2.62 GHz -> F1 2.10-2.62, F2 4.20-5.24,
F3 6.30-7.86, F4 8.40-10.48, F5 10.50-13.10 GHz):
  - F1: well below cutoff -> should drop into the noise floor. This is the
    harmonic the HPF is actually for.
  - F2: mostly below cutoff, only the top of the input range (f_in near
    2.62 -> F2 near 5.24) sits in the transition -- expect partial help,
    strongest at the low end of f_in.
  - F3 (wanted): solidly inside the passband -- should pass with only the
    filter's normal in-band insertion loss.
  - F4 and F5: BOTH already sit inside or above the HPF's own passband
    (8.40-10.48 and 10.50-13.10 vs. a HPF with no specified upper rolloff)
    -- the HPF cannot touch either. Rejecting these is the LPF's job
    (VLF-8400+, not yet delivered). Expect F4/F5 dBc to come out basically
    unchanged from the bare-tripler run until the LPF is in the chain --
    if this run shows otherwise, that's the signal something about this
    prediction is wrong, not a free win.

Drive: 0.124 Vpp -- the level dialled in during the amp+BPF bring-up
(run_frontend_cw_scan.py) that lands the tripler's input inside its
0..+5 dBm window across the whole 2.152-2.570 GHz target band.

Method, per input CW frequency f_in:
  1. One WIDE overview sweep (fixed 1.8-10.8 GHz window, covers 1x-4x of
     f_in across the whole scanned input range in one span) to catch
     anything present, expected or not. Point count forced to 8001 BEFORE
     RBW is derived from point spacing (not instrument Auto) -- the same
     undersampling bug class documented in the Auto Spur Hunt history
     (project_cpftmw_spectrometer memory) applies here too: a wide span at
     a coarse/Auto RBW can miss or under-read a narrow CW line.
  2. A NARROW refine (5 MHz span, 100 kHz RBW) at each of the four
     PREDICTED harmonics f_in*1/2/3/4 -- exact frequencies, since the DUC
     NCO is frequency-exact, so no searching is needed, just a precise
     level reading at each.
  3. Any overview peak that does NOT land within 15 MHz of a predicted
     harmonic is reported separately as an unidentified candidate (at
     overview precision -- not refined).

F3 is the wanted tripled output. F1/F2/F4 dBc-below-F3 are reported
alongside the ZX90-3-812-S+ datasheet's own spec (F1 30 typ/21 min, F2 48,
F4 48 dBc) for a direct bring-up comparison (see the "Bring-up watch list"
in project_buy_chain_x6 memory).

Data saved under x6_chain/tripler-outputs/<timestamp>/ (project root,
alongside x6_chain/amp-bpf/ from the previous stage) -- NOT under
specan_traces/, by passing specanlog.save_trace() an absolute subdir path
(os.path.join discards the TRACE_DIR prefix when the second argument is
already absolute).

Run with `python run_tripler_outputs.py` from the project root.
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
X6_TRIPLER_DIR = os.path.join(PROJECT_ROOT, "x6_chain", "tripler-hpf-outputs")

# ── bench config ────────────────────────────────────────────────────
AMP_VPP = 0.124           # established in the amp+BPF bring-up (iteration 3)
FREQ_START_HZ = 2.10e9
FREQ_STOP_HZ  = 2.62e9
FREQ_STEP_HZ  = 0.02e9
FREQS_HZ = [round(f, 9) for f in
            np.arange(FREQ_START_HZ, FREQ_STOP_HZ + FREQ_STEP_HZ / 2, FREQ_STEP_HZ)]

EXTERNAL_ATTEN_DB = 10.0   # the physical pad, now after tripler -> HPF
ATTEN_DB = 0               # EXA internal attenuator -- lowered from the
                           # amp+BPF stage's 10 dB for sensitivity: every
                           # level here is much weaker (tripler CL + the
                           # pad), no damage-margin concern at these powers
REF_LEVEL_DBM = -10        # headroom above the expected ~-20 dBm F3 peak
                           # at the EXA (see module docstring math) --
                           # adjust at the bench if F3 clips or sits low

# Tracked harmonics. The datasheet only specs F1/F2/F4, but run 1's wide
# overview (before this list included 5) turned up a strong, real F5 at
# ~23 dBc -- worse than F1 -- sitting right at 5x, so it's tracked properly
# now rather than left to fall through as "unidentified". F6 not yet added:
# 6x reaches 15.7 GHz at the top of the input sweep, unconfirmed this EXA
# covers that (12.42 GHz is the highest frequency this unit has been
# verified at, project_exa_specan memory) -- worth a one-off spot check
# before adding it here.
HARMONICS = (1, 2, 3, 4, 5)

# Overview window: fixed across every input frequency, wide enough to hold
# 1x..5x for the WHOLE swept input range (1x min 2.10, 5x max 13.10 GHz).
OV_CENTER_HZ = 7.55e9
OV_SPAN_HZ   = 11.5e9
OV_POINTS    = 8001
MAX_RBW_HZ   = 3e6         # this N9010A's measured RBW ceiling (silently
                           # clamps above this -- see project_exa_specan /
                           # run_specan_band_scan.py)
OV_PROMINENCE_DB = 15.0    # raised from run 1's 8 dB: a single un-averaged
                           # NORM sweep over an 11.5 GHz span is noisy
                           # enough that 8 dB caught thousands of noise-
                           # floor wiggles even after min_spacing_hz fixed
                           # the merging bug (run 2, ~340 false positives/
                           # point). A real spur like F5 cleared this by
                           # >15 dB of margin, so 15 dB should hold onto
                           # anything of that scale while dropping noise.
HARMONIC_MATCH_HZ = 15e6   # within this of n*f_in counts as "identified"

# Narrow refine, per predicted harmonic -- same recipe as the amp+BPF CW
# scan (a stationary tone needs no Max Hold, unlike the pulsed chirp case).
REFINE_SPAN_HZ = 5e6
REFINE_RBW_HZ  = 100e3
REFINE_POINTS  = 401
SETTLE_S = 0.25

# ZX90-3-812-S+ datasheet spurs, for direct comparison (project_buy_chain_x6
# memory): F1 30 typ / 21 min, F2 48, F4 48 dBc.
DATASHEET_DBC = {1: 30.0, 2: 48.0, 4: 48.0}

NOTES = ("Tripler+HPF bring-up: AWG -> ZX60-2522MA-S+ -> VBFZ-2340-S+ -> "
         "ZX90-3-812-S+ tripler -> VHF-4600+ HPF -> 10 dB pad -> EXA, "
         "0.124 Vpp drive. VLF-8400+ LPF not yet installed (on order) -- "
         "F4/F5 expected unchanged from the bare-tripler run until it is. "
         "Wanted output F3 (3x) plus F1/F2/F4/F5 spurs, against the "
         "datasheet's F1 30/F2 48/F4 48 dBc (F5 undocumented, see memory).")


def _settle_and_read_rbw(specan):
    time.sleep(0.3)   # stale-read trap: querying RBW/sweep-time immediately
                       # after a span/points change returns the PREVIOUS
                       # setting (see Auto Spur Hunt debugging history)
    return specan.get_rbw()


def overview_sweep(specan, run_dir, f_in):
    specan.set_freq(OV_CENTER_HZ, OV_SPAN_HZ)
    specan.set_points(OV_POINTS)           # forced high BEFORE deriving RBW
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

    run_dir = os.path.join(X6_TRIPLER_DIR, time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== Tripler output survey: {len(FREQS_HZ)} input points, "
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
    ax_abs.set_title(f"Tripler output survey -- wanted + spurs "
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

    png_path = os.path.join(run_dir, "_tripler_overview.png")
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
