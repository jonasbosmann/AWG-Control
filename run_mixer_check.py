"""Mixer bring-up: measure the ZMDB-24H-K+'s REAL conversion loss vs IF
frequency, plus LO leakage and sideband balance.

This is the single number the transmit power budget is missing. The
datasheet quotes conversion loss AT 30 MHz IF with an explicit note that it
"increases with IF frequency" -- our IF is 1-3.5 GHz, three orders of
magnitude beyond that test condition, so the budget cannot be closed from
paper (see power_budget.py).

Wiring assumed (no isolator, no filter -- deliberately, so this measures the
mixer itself and not the chain around it):
    AWG CH1  -> mixer IF port
    LO chain -> mixer LO port   (SMB100A -> splitter -> amp, ~+13.9 dBm)
    mixer RF -> external pad -> EXA

Method: step a CW tone (not a chirp) across the IF band using the DUC NCO,
and at each point read the upper sideband, the lower sideband (image) and
the LO feedthrough. Conversion loss follows directly because the AWG's
absolute output is already calibrated to ~0.03 dB by the CW level check:

    conversion_loss(f) = awg_output(f) - rf_output(LO + f)

CW rather than a chirp on purpose: a stationary tone has no pulse-repetition
comb and no swept-filter settling loss, so the analyzer reports true power
(the chirp scan's levels are power DENSITY and are not comparable). See
chirp_bench/run_specan_band_scan.py for why that distinction matters.

Run:  python run_mixer_check.py
"""
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from awg import AWG
from specan import SpecAn
import specanlog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirp_bench"))
from run_specan_band_scan import EXTERNAL_ATTEN_DB

# ── config — edit at the bench ─────────────────────────────────────
LO_HZ = 12.42e9        # what the synth is actually set to. The LO chain was
                        # characterised at this frequency (delivered
                        # ~+13.9 dBm, 1.07 dB under the mixer's Level-15
                        # spec) -- see project_power_budget memory.
LO_DELIVERED_DBM = 13.93   # measured, for the record in saved metadata

IF_START_HZ, IF_STOP_HZ, IF_STEP_HZ = 0.2e9, 3.5e9, 0.1e9
AMP_VPP = 0.5          # same drive the AWG was calibrated at

SPAN_HZ = 20e6         # both LO and NCO are frequency-exact, so a narrow
                        # span is plenty and keeps the noise floor low
RBW_HZ = 100e3
POINTS = 401
ATTEN_DB = 10
REF_LEVEL_DBM = 0
SETTLE_S = 0.25

MONITOR_CH2 = True
# Mirror CH1's tone onto CH2 so the sweep can be WATCHED rather than
# trusted. CH1 disappears into the mixer's IF port where it can't be
# observed; CH2 carries an identical copy for a scope, so you can confirm
# the AWG really is producing the commanded tone at every step -- and at
# what amplitude -- instead of inferring it from the analyzer alone.
# Costs nothing to the measurement: CH2 is a separate DAC output and is not
# connected to the mixer. Set False to leave CH2 idle.


def awg_reference():
    """The AWG's calibrated absolute output vs frequency, from the most
    recent CW level check -- this is what makes conversion loss a direct
    subtraction rather than another unknown."""
    runs = sorted(glob.glob(os.path.join(specanlog.TRACE_DIR, "cw_level_check_*")))
    if not runs:
        raise SystemExit(
            "No CW level check found. Run the GUI's 'Run CW Level Check' (or "
            "chirp_bench/run_specan_band_scan.py's run_cw_scan) first -- this "
            "measurement needs the AWG's absolute output as its reference.")

    # A valid reference is AWG CH1 cabled STRAIGHT to the EXA, so it should
    # peak near the power a full-scale sine delivers into 50 ohm. Taken with
    # the mixer in line instead, it measures a leakage path tens of dB down --
    # and since conversion loss here is (reference - mixer output), such a
    # reference would silently produce a near-zero or negative answer that
    # still looks like a number. So: walk runs newest-first and use the newest
    # PLAUSIBLE one rather than blindly taking the latest.
    expect = 10 * np.log10(((AMP_VPP / 2) ** 2 / (2 * 50)) / 1e-3)
    rejected = []
    for run in reversed(runs):
        f, a = [], []
        for p in sorted(glob.glob(os.path.join(run, "*.json"))):
            d = specanlog.load_trace(p)
            s = d["settings"]
            if "cw_freq_hz" not in s:
                continue
            f.append(s["cw_freq_hz"])
            a.append(d["amps_dbm"].max() + s.get("external_atten_db", EXTERNAL_ATTEN_DB))
        if not f:
            continue
        f, a = np.array(f), np.array(a)
        if a.max() < expect - 6.0:
            rejected.append((os.path.basename(run), a.max()))
            continue
        o = np.argsort(f)
        if rejected:
            print(f"  SKIPPED {len(rejected)} implausible CW reference run(s) -- "
                  f"peak levels far below the ~{expect:+.2f} dBm a {AMP_VPP} Vpp "
                  f"sine gives into 50 ohm:")
            for name, mx in rejected:
                print(f"    {name}: peaks at {mx:+.2f} dBm ({expect-mx:.1f} dB low) "
                      f"-- looks like it was taken THROUGH the mixer, not AWG->EXA")
            print(f"  Using the newest plausible run instead. To replace it, re-run\n"
                  f"  the CW Level Check with CH1 cabled DIRECTLY to the EXA.")
        print(f"AWG reference: {len(f)} points from {os.path.basename(run)} "
              f"({a.max():+.2f} to {a.min():+.2f} dBm)")
        return f[o], a[o]

    raise SystemExit(
        f"No usable CW reference: all {len(rejected)} run(s) peak far below the "
        f"~{expect:+.2f} dBm expected for {AMP_VPP} Vpp into 50 ohm "
        f"({', '.join(f'{n}={m:+.1f} dBm' for n, m in rejected)}).\n"
        f"Re-run the CW Level Check with AWG CH1 cabled DIRECTLY to the EXA "
        f"(no mixer, no isolator in line) -- that measurement is the absolute "
        f"reference conversion loss is computed against.")


def read_tone(specan, center_hz, run_subdir, label):
    """Peak level at one frequency, corrected for the external pad."""
    specan.set_freq(center_hz, SPAN_HZ)
    specan.set_points(POINTS)
    specan.set_rbw(RBW_HZ)
    specan.set_vbw(RBW_HZ)
    specan.set_trace_mode("NORM")
    freqs, amps = specan.sweep_once()
    i = int(np.argmax(amps))
    peak = float(amps[i]) + EXTERNAL_ATTEN_DB
    specanlog.save_trace(
        label, "mixer bring-up: raw mixer output, no isolator/filter",
        {"atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM, "rbw_hz": RBW_HZ,
         "vbw_hz": RBW_HZ, "points": POINTS, "trace_mode": "NORM",
         "external_atten_db": EXTERNAL_ATTEN_DB, "center_hz": center_hz,
         "lo_hz": LO_HZ, "lo_delivered_dbm": LO_DELIVERED_DBM},
        freqs, amps, subdir=run_subdir)
    return peak, float(freqs[i])


def new_run_subdir():
    """Timestamped at CALL time, not import time -- a long-lived caller (the
    GUI) must get a fresh folder per run rather than overwriting the first."""
    return time.strftime("mixer_check_%Y-%m-%d_%H%M%S")


def measure(awg, specan, run_subdir, on_point=None):
    """Core sweep. Takes ALREADY-CONNECTED instruments so the GUI can drive
    this with its own AWG/SpecAn handles instead of opening a second session.

    on_point(rows_so_far, lo_leak) is called after each IF point so a GUI can
    plot the curve building up. Returns (rows, lo_leak); rows entries are
    (if_hz, awg_in_dbm, usb_dbm, conv_loss_db, lsb_dbm, sideband_rej_db).
    """
    ref = awg_reference()
    if_freqs = np.arange(IF_START_HZ, IF_STOP_HZ + 1, IF_STEP_HZ)
    specan.set_attenuation(ATTEN_DB)
    specan.set_ref_level(REF_LEVEL_DBM)
    print(f"\nLO {LO_HZ/1e9:.3f} GHz (delivered ~{LO_DELIVERED_DBM:+.2f} dBm)")
    print(f"IF sweep {IF_START_HZ/1e9:.2f}-{IF_STOP_HZ/1e9:.2f} GHz "
          f"in {IF_STEP_HZ/1e6:.0f} MHz steps -> USB {(LO_HZ+IF_START_HZ)/1e9:.2f}-"
          f"{(LO_HZ+IF_STOP_HZ)/1e9:.2f} GHz")
    print(f"saving to specan_traces/{run_subdir}/\n")

    rows = []
    if True:
        # LO feedthrough at the RF port, measured once with the IF driven --
        # this is the mixer's raw L-R isolation, the thing the (absent)
        # filter would later have to deal with.
        if MONITOR_CH2:
            awg.duc_cw_setup_dual(if_freqs[0], amplitude_vpp=AMP_VPP)
            print("  CH2 mirrors CH1 -- put a scope on CH2 to watch the sweep "
                  "(CH1 is inside the mixer and can't be observed)\n")
        else:
            awg.duc_cw_setup(if_freqs[0], amplitude_vpp=AMP_VPP)
        time.sleep(SETTLE_S)
        lo_leak, lo_peak_hz = read_tone(specan, LO_HZ, run_subdir, "lo_feedthrough")
        iso = LO_DELIVERED_DBM - lo_leak
        print(f"LO feedthrough at RF port: {lo_leak:+.2f} dBm "
              f"-> L-R isolation {iso:.1f} dB (datasheet ~25.6 dB at 13-15 GHz)\n")

        # Fail fast on a dead signal path. LO feedthrough does NOT depend on
        # the AWG, so its absence isolates the fault to LO->mixer or
        # mixer->analyzer -- and without it every subsequent point would
        # just be noise-floor maxima that LOOK like plausible dBm numbers
        # (a first run produced "68 dB conversion loss / 84 dB isolation"
        # before this check existed). Two independent symptoms are used:
        # an implausibly good isolation figure, and a peak that isn't at
        # the commanded frequency (a real tone lands exactly on it, since
        # both the NCO and the synth are frequency-exact).
        off_hz = abs(lo_peak_hz - LO_HZ)
        if iso > 50.0 or off_hz > SPAN_HZ / 20:
            raise SystemExit(
                f"\nABORTING -- no LO feedthrough detected at the RF port.\n"
                f"  measured {lo_leak:+.2f} dBm => {iso:.1f} dB 'isolation' "
                f"(real mixers give ~25-30 dB; >50 dB means NO SIGNAL)\n"
                f"  peak sits {off_hz/1e6:+.2f} MHz off the commanded frequency "
                f"(a real tone lands within ~0.00 MHz)\n"
                f"This is a wiring fault, not a measurement result. Since LO\n"
                f"feedthrough is independent of the AWG, the fault is on the\n"
                f"LO->mixer or mixer->analyzer side. Check:\n"
                f"  - ZMDB-24H-K+ ports are NUMBERED: 1=RF, 2=LO, 3=IF\n"
                f"    (swapping RF and LO gives exactly this result)\n"
                f"  - the analyzer cable actually moved from the LO amp\n"
                f"    output onto the mixer's RF port\n"
                f"  - the LO amp output is on the mixer's LO port and powered\n")

        print(f"{'IF GHz':>7} {'AWG in':>8} {'USB out':>9} {'CONV LOSS':>10} "
              f"{'LSB(img)':>9} {'sb rej':>7}")
        for f_if in if_freqs:
            awg.duc_cw_step(f_if)
            if MONITOR_CH2:
                # NCO retune is per-channel, so the monitor copy must be
                # stepped too or it would sit at the first frequency and
                # silently stop tracking CH1.
                awg.duc_cw_step(f_if, channel=2)
            time.sleep(SETTLE_S)
            p_in = float(np.interp(f_if, ref[0], ref[1]))
            usb, _ = read_tone(specan, LO_HZ + f_if, run_subdir,
                               f"usb_{f_if/1e9:.2f}GHz".replace(".", "p"))
            lsb, _ = read_tone(specan, LO_HZ - f_if, run_subdir,
                               f"lsb_{f_if/1e9:.2f}GHz".replace(".", "p"))
            cl = p_in - usb
            rows.append((f_if, p_in, usb, cl, lsb, usb - lsb))
            print(f"{f_if/1e9:7.2f} {p_in:+8.2f} {usb:+9.2f} {cl:10.2f} "
                  f"{lsb:+9.2f} {usb-lsb:+7.2f}")
            if on_point is not None:
                try:
                    on_point(np.array(rows), lo_leak)
                except Exception as e:
                    print(f"  (live update skipped: {e})")

    r = np.array(rows)
    print(f"\nconversion loss: {r[:,3].min():.2f} to {r[:,3].max():.2f} dB "
          f"across the IF band (datasheet: 8.5 typ / 10.8 max, AT 30 MHz IF)")
    print(f"tilt across band: {r[:,3].max()-r[:,3].min():.2f} dB")
    return r, lo_leak


def plot_results(r, lo_leak, fig=None):
    """Two-panel summary. fig=None makes a new figure (standalone use);
    passing the GUI's embedded Figure redraws into it instead, so both
    entry points render identically."""
    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    else:
        fig.clear()
        axes = fig.subplots(2, 1, sharex=True)
    ax = axes[0]
    ax.plot(r[:, 0] / 1e9, r[:, 3], "o-", ms=4, color="C3", label="measured")
    ax.axhline(8.5, ls="--", lw=1, color="gray", label="datasheet typ (at 30 MHz IF)")
    ax.axhline(10.8, ls=":", lw=1, color="gray", label="datasheet max (at 30 MHz IF)")
    ax.set_ylabel("conversion loss (dB)")
    ax.set_title(f"ZMDB-24H-K+ conversion loss vs IF frequency  "
                 f"(LO {LO_HZ/1e9:.2f} GHz @ {LO_DELIVERED_DBM:+.1f} dBm, "
                 f"{15.0-LO_DELIVERED_DBM:.1f} dB under Level-15 spec)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); ax.invert_yaxis()

    ax = axes[1]
    ax.plot(r[:, 0] / 1e9, r[:, 2], "o-", ms=4, label="USB (wanted)")
    ax.plot(r[:, 0] / 1e9, r[:, 4], "s--", ms=3, label="LSB (image)")
    ax.plot(r[:, 0] / 1e9, r[:, 1], "-", lw=1, color="gray", label="AWG input")
    ax.axhline(lo_leak, ls=":", color="red", lw=1,
               label=f"LO leakage ({lo_leak:+.1f} dBm)")
    ax.set_xlabel("IF frequency (GHz)"); ax.set_ylabel("dBm")
    ax.set_title("Output levels -- wanted sideband vs image vs LO feedthrough", fontsize=9)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run():
    """Standalone entry point: opens its own instruments, measures, plots."""
    awg = AWG()
    specan = SpecAn()
    run_subdir = new_run_subdir()
    try:
        r, lo_leak = measure(awg, specan, run_subdir)
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", specan.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    fig = plot_results(r, lo_leak)
    out_dir = os.path.join(specanlog.TRACE_DIR, run_subdir)
    png = os.path.join(out_dir, "_mixer_check.png")
    fig.savefig(png, dpi=130)
    print(f"saved {png}")
    plt.show()
    return r


if __name__ == "__main__":
    run()
