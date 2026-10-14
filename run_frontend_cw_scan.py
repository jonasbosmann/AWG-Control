"""Bench script: front-end bring-up CW level check.

Current physical setup (2026-10-08, ZX60-P105LN+ LNA not yet delivered --
ETA ~Nov per project_receiver_chain memory):

    AWG CH1 -> ZX60-2522MA-S+ (temporary stand-in for the missing
               ZX60-P105LN+ first LNA)
            -> VBFZ-2340-S+ BPF (2020-2660 MHz passband, per the x6 chain
               design -- project_buy_chain_x6 memory)
            -> 10 dB pad -> Keysight N9010A EXA

No tripler/doubler/AMC in the path yet -- this measures only what the
substitute amp + filter combination actually delivers, ahead of them.

Method: step a CW tone across the band and read the true level off the EXA
at each point -- same recipe as chirp_bench/run_specan_band_scan.py's
run_cw_point() (narrow RBW, Normal trace, single_sweep). A stationary CW
tone needs none of that script's Max Hold machinery (that's only needed for
the pulsed, fast-swept chirp) -- this just reuses the proven CW method with
this bench's actual frequency range, drive level, and external pad.

Drive level -- 0.139 Vpp (iteration 2, 2026-10-08):
run 1 at 0.311 Vpp (the P1dB-safe backoff for THIS amp, see
project_multiplier_chain memory) measured 12.6/9.9 dBm at the BPF output
across the x6 chain's 2.152-2.570 GHz target band -- 5-7 dB too hot for the
downstream ZX90-3-812-S+ tripler's 0..+5 dBm input window (this amp has
~9-10 dB more gain than the designed ZX60-P105LN+). Rather than add a
physical pad, the user chose to absorb that excess on the AWG side: a flat
7 dB cut from 0.311 Vpp -> 0.139 Vpp, no per-segment correction (the 2.7 dB
in-band tilt already fits inside the tripler's 5 dB window on its own, so a
single flat level shift is enough for now -- see [[project-buy-chain-x6]]
memory for the full number trail and why a post-amp pad costs nothing in
SNR either way, this was just the quicker-to-try option). Should land the
band's hot edge (2.15 GHz) right at ~+5 dBm and the cool edge (2.57 GHz)
around ~+3 dBm -- this run checks that prediction against the real EXA.

Run with `python run_frontend_cw_scan.py` from the project root.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe: this run only needs the saved PNG,
                         # not an interactive window
import matplotlib.pyplot as plt

from awg import AWG
from specan import SpecAn
import specanlog

# ── bench config -- edit here if the wiring/parts change ──────────────
AMP_VPP = 0.124          # see module docstring -- iteration 3, one more
                         # -1 dB from iteration 2's 0.139 Vpp (8 dB total
                         # cut from the 0.311 Vpp P1dB-safe level), to clear
                         # the small ~0.85 dB overshoot iteration 2 left at
                         # the band's hot (low-frequency) edge
FREQ_START_HZ = 1.90e9
FREQ_STOP_HZ  = 2.80e9   # VBFZ-2340-S+ passband (2.02-2.66 GHz) + ~120 MHz
                          # margin each side, so the filter's own rolloff is
                          # visible, not just its flat top
FREQ_STEP_HZ  = 0.02e9
FREQS_HZ = [round(f, 9) for f in
            np.arange(FREQ_START_HZ, FREQ_STOP_HZ + FREQ_STEP_HZ / 2, FREQ_STEP_HZ)]

ATTEN_DB = 10             # EXA's own internal input attenuator
EXTERNAL_ATTEN_DB = 10.0  # the physical 10 dB pad between the BPF and the EXA.
                           # SET TO 0.0 if the pad is ever removed, or every
                           # reading below will be 10 dB high.
REF_LEVEL_DBM = 0          # top-of-screen headroom above the expected
                           # ~+3..+5 dBm at 0.139 Vpp (see docstring) --
                           # adjust at the bench if the trace clips or sits low

CW_SPAN_HZ = 10e6      # narrow: the DUC NCO is frequency-exact, this only
                       # needs to contain the tone
CW_RBW_HZ  = 100e3     # narrow RBW is fine (and more accurate) for a
                       # stationary tone
CW_POINTS  = 401
CW_SETTLE_S = 0.25

NOTES = ("Front-end bring-up, iteration 3: AWG -> ZX60-2522MA-S+ (temporary "
         "stand-in for the not-yet-delivered ZX60-P105LN+ first LNA) -> "
         "VBFZ-2340-S+ BPF -> 10 dB pad -> EXA. No tripler/doubler/AMC in "
         "the path yet. Drive dropped to 0.124 Vpp (one more -1 dB from "
         "iteration 2's 0.139 Vpp, 8 dB total off the 0.311 Vpp P1dB-safe "
         "level) to clear the ~0.85 dB overshoot iteration 2 left at the "
         "band's low-frequency edge -- see module docstring / "
         "project_buy_chain_x6 memory.")


def run_point(awg, specan, freq_hz, run_subdir, first):
    if first:
        awg.duc_cw_setup(freq_hz, amplitude_vpp=AMP_VPP)
    else:
        awg.duc_cw_step(freq_hz)
    time.sleep(CW_SETTLE_S)

    specan.set_freq(freq_hz, CW_SPAN_HZ)
    specan.set_points(CW_POINTS)
    specan.set_rbw(CW_RBW_HZ)
    specan.set_vbw(CW_RBW_HZ)
    specan.set_trace_mode("NORM")
    freqs, amps = specan.sweep_once()

    i = int(np.argmax(amps))
    peak_dbm = float(amps[i]) + EXTERNAL_ATTEN_DB
    peak_hz = float(freqs[i])
    err_hz = peak_hz - freq_hz
    flag = "  <-- PEAK OFF-FREQUENCY" if abs(err_hz) > CW_SPAN_HZ / 4 else ""
    print(f"  {freq_hz/1e9:.3f} GHz -> {peak_dbm:7.2f} dBm "
          f"(peak at {peak_hz/1e9:.4f} GHz, {err_hz/1e6:+.3f} MHz){flag}")

    specanlog.save_trace(
        f"frontend_{freq_hz/1e9:.3f}GHz".replace(".", "p"),
        NOTES,
        {"atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM,
         "rbw_hz": CW_RBW_HZ, "vbw_hz": CW_RBW_HZ, "points": CW_POINTS,
         "trace_mode": "NORM", "cw_freq_hz": freq_hz, "amp_vpp": AMP_VPP,
         "external_atten_db": EXTERNAL_ATTEN_DB},
        freqs, amps, subdir=run_subdir)
    return peak_hz, peak_dbm


def run():
    awg = AWG()
    specan = SpecAn()
    specan.set_attenuation(ATTEN_DB)
    specan.set_ref_level(REF_LEVEL_DBM)

    run_subdir = os.path.join("x6_chain",
                               time.strftime("frontend_cw_scan_%Y-%m-%d_%H%M%S"))
    print(f"\n=== Front-end CW level check: {len(FREQS_HZ)} points, "
          f"{FREQS_HZ[0]/1e9:.3f}-{FREQS_HZ[-1]/1e9:.3f} GHz, "
          f"drive {AMP_VPP} Vpp ===")
    freqs_out, levels_out = [], []
    try:
        for i, f in enumerate(FREQS_HZ):
            _, pdbm = run_point(awg, specan, f, run_subdir, first=(i == 0))
            freqs_out.append(f)
            levels_out.append(pdbm)
    finally:
        try:
            awg.stop()
        except Exception as e:
            print(f"cleanup (awg.stop): {e}")

    freqs_out = np.array(freqs_out)
    levels_out = np.array(levels_out)

    peak_i = int(np.argmax(levels_out))
    peak_dbm = levels_out[peak_i]
    idx = np.where(levels_out >= (peak_dbm - 3.0))[0]
    lo_hz, hi_hz = freqs_out[idx[0]], freqs_out[idx[-1]]
    print(f"\nPeak {peak_dbm:.2f} dBm @ {freqs_out[peak_i]/1e9:.3f} GHz")
    print(f"Measured -3 dB passband (within the scanned window): "
          f"{lo_hz/1e9:.3f}-{hi_hz/1e9:.3f} GHz ({(hi_hz-lo_hz)/1e6:.0f} MHz wide)")
    if idx[0] == 0 or idx[-1] == len(freqs_out) - 1:
        print("NOTE: the -3 dB point touches an edge of the scanned window -- "
              "widen FREQ_START_HZ/FREQ_STOP_HZ to see the true rolloff.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs_out / 1e9, levels_out, "o-", ms=3, lw=1.3, color="C0")
    ax.axhline(peak_dbm, color="gray", ls="--", lw=1, label=f"peak {peak_dbm:.1f} dBm")
    ax.axhline(peak_dbm - 3, color="red", ls=":", lw=1, label="-3 dB")
    ax.axvspan(2.02, 2.66, color="green", alpha=0.08,
               label="VBFZ-2340-S+ nominal passband")
    ax.axvspan(2.152, 2.570, color="blue", alpha=0.06,
               label="x6 chain target band (2.152-2.570 GHz)")
    ax.axhspan(0, 5, color="orange", alpha=0.10,
               label="ZX90-3-812-S+ tripler input window (0..+5 dBm)")
    ax.set_xlabel("GHz")
    ax.set_ylabel("dBm at BPF output (pad corrected out)")
    ax.set_title(f"Front-end CW level check (iter 3) -- AWG + ZX60-2522MA-S+ + BPF "
                 f"({AMP_VPP} Vpp drive, {EXTERNAL_ATTEN_DB:.0f} dB pad corrected)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir = os.path.join(specanlog.TRACE_DIR, run_subdir)
    png_path = os.path.join(out_dir, "_frontend_overview.png")
    fig.savefig(png_path, dpi=130)
    print(f"saved plot: {png_path}")
    print("(single N=1 run -- not yet repeated, see feedback_bench_rigor)")

    return freqs_out, levels_out, png_path


if __name__ == "__main__":
    run()
