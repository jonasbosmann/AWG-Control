"""Spur scan: step the mixer's IF and look for spurious signals close to the
wanted upper sideband, with an RBW narrow enough to actually resolve them.

Complements run_mixer_check.py. That one records only the PEAK at each IF
point (fast, gives conversion loss); this one keeps the whole spectrum
around each wanted USB tone (LO+IF) so anything sitting near it -- LO-derived
spurs, mixer intermodulation, AWG DAC images -- becomes visible.

Wiring (same as run_mixer_check.py):
    AWG CH1  -> mixer IF port
    LO chain -> mixer LO port
    mixer RF -> isolator -> external pad -> EXA

Two sampling rules matter here and are enforced rather than assumed:

1. RBW must be comfortably NARROWER than the span (to resolve close-in
   spurs) but comfortably WIDER than the trace point spacing -- otherwise
   the sweep steps OVER spurs between points and reports a clean spectrum
   that simply wasn't looked at properly. Point count is therefore derived
   from the RBW, not chosen independently. (This is the undersampling bug
   the Auto Spur Hunt hit: RBW far below point spacing looked fine and
   silently missed real spurs.)
2. A narrow RBW makes the sweep slow, so the VISA timeout is raised from
   the instrument's reported sweep time instead of being left at its
   default -- otherwise a legitimately slow sweep looks like a dead
   instrument.

Run:  python run_spur_scan.py
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from awg import AWG
from specan import SpecAn, find_peaks
import specanlog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirp_bench"))
from run_specan_band_scan import EXTERNAL_ATTEN_DB

# ── config ─────────────────────────────────────────────────────────
LO_HZ = 12.42e9
LO_DELIVERED_DBM = 13.93

IF_START_HZ, IF_STOP_HZ, IF_STEP_HZ = 0.2e9, 3.5e9, 0.1e9
AMP_VPP = 0.5
MONITOR_CH2 = True          # mirror the tone on CH2 for scope watching

SPAN_HZ = 100e6             # +-50 MHz around the wanted USB tone -- wide enough
                             # to catch spur families sitting tens of MHz out
                             # (the SMB100A's known ~19 and ~27 MHz offset
                             # families would fall inside this)
RBW_HZ = 10e3               # narrow enough to resolve close-in spurs
VBW_HZ = 1e3                # VBW << RBW: video filtering smooths the noise
                             # floor without touching a coherent tone. With
                             # VBW = RBW (no smoothing) an unaveraged trace's
                             # noise produces hundreds of false "peaks" per
                             # sweep -- measured: 721 per trace, 24645 total.
                             # Costs sweep time, which is fine here.
RBW_OVERSAMPLE = 3.0        # require RBW >= this x point spacing
ATTEN_DB = 10
REF_LEVEL_DBM = 0
SETTLE_S = 0.3

# Spur criterion. Local topographic prominence ALONE is useless on a noisy
# trace: random noise maxima routinely clear 8 dB of prominence. A spur must
# instead stand SPUR_SNR_DB above the LOCAL noise floor, estimated by a
# running median that follows the USB tone's skirt instead of assuming a flat
# floor.
SPUR_SNR_DB = 12.0
FLOOR_WINDOW_PTS = 501      # running-median width for the floor estimate --
                             # must be >> a spur (a few points wide at this
                             # RBW) so real spurs don't pull their own floor up
USB_EXCLUDE_HZ = 2e6    # the USB tone's phase-noise/filter skirt extends
                             # well past 200 kHz; everything inside this window
                             # is the wanted USB tone itself, not a spur


def required_points(span_hz, rbw_hz):
    """Points needed so RBW stays >= RBW_OVERSAMPLE x point spacing."""
    return int(np.ceil(span_hz / (rbw_hz / RBW_OVERSAMPLE))) + 1


def find_spurs(freqs, amps, usb_hz):
    """Spurs = local maxima standing SPUR_SNR_DB above the LOCAL noise floor.

    Deliberately not a topographic-prominence test: on an unaveraged trace,
    random noise maxima clear any reasonable prominence threshold (measured:
    721 false detections per sweep). The floor is a running median, so it
    tracks the USB tone's skirt rather than assuming the floor is flat, and
    is wide enough that a real spur cannot lift its own floor."""
    from scipy.ndimage import median_filter
    floor = median_filter(amps, size=FLOOR_WINDOW_PTS, mode="nearest")
    thresh = floor + SPUR_SNR_DB
    ok = (np.abs(freqs - usb_hz) > USB_EXCLUDE_HZ) & (amps > thresh)
    ok[0] = ok[-1] = False
    ok &= (amps > np.roll(amps, 1)) & (amps > np.roll(amps, -1))
    idx = np.flatnonzero(ok)
    # Merge detections belonging to one spur (a tone is ~RBW wide).
    out, min_gap = [], max(RBW_HZ * 3, (freqs[1] - freqs[0]) * 3)
    for i in idx:
        if out and freqs[i] - out[-1][0] <= min_gap:
            if amps[i] > out[-1][1]:
                out[-1] = (float(freqs[i]), float(amps[i]))
        else:
            out.append((float(freqs[i]), float(amps[i])))
    return out


def measure(awg, specan, run_subdir, on_point=None):
    if_freqs = np.arange(IF_START_HZ, IF_STOP_HZ + 1, IF_STEP_HZ)
    want_pts = required_points(SPAN_HZ, RBW_HZ)

    specan.set_attenuation(ATTEN_DB)
    specan.set_ref_level(REF_LEVEL_DBM)
    specan.set_freq(LO_HZ + if_freqs[0], SPAN_HZ)
    specan.set_points(want_pts)
    specan.set_rbw(RBW_HZ)
    specan.set_vbw(VBW_HZ)
    specan.set_trace_mode("NORM")
    time.sleep(0.3)          # let the instrument recompute before querying

    got_pts = int(specan._dev.query(":SWE:POIN?"))
    rbw = specan.get_rbw()
    spacing = SPAN_HZ / (got_pts - 1)
    sweep_s = specan.get_sweep_time()
    print(f"span {SPAN_HZ/1e6:.0f} MHz, RBW {rbw/1e3:.1f} kHz, "
          f"{got_pts} points -> spacing {spacing/1e3:.2f} kHz "
          f"(RBW/spacing = {rbw/spacing:.1f}x)")
    if rbw < RBW_OVERSAMPLE * spacing:
        print(f"  WARNING: RBW is only {rbw/spacing:.1f}x the point spacing "
              f"(want >={RBW_OVERSAMPLE:.0f}x). The sweep can step OVER narrow "
              f"spurs between points and report a clean spectrum that was never "
              f"properly sampled. Requested {want_pts} points but the instrument "
              f"gave {got_pts} -- widen RBW or narrow the span.")
    print(f"sweep time {sweep_s*1e3:.0f} ms per point, "
          f"{len(if_freqs)} IF points -> ~{len(if_freqs)*(sweep_s+SETTLE_S+0.6)/60:.1f} min\n")

    # A slow sweep must not look like a dead instrument: give VISA room.
    specan._dev.timeout = max(8000, int(sweep_s * 1000 * 4) + 5000)

    if MONITOR_CH2:
        awg.duc_cw_setup_dual(if_freqs[0], amplitude_vpp=AMP_VPP)
        print("CH2 mirrors CH1 (scope monitor)\n")
    else:
        awg.duc_cw_setup(if_freqs[0], amplitude_vpp=AMP_VPP)

    results = []
    print(f"{'IF GHz':>7} {'USB dBm':>12} {'spurs':>6}  worst spur")
    for f_if in if_freqs:
        awg.duc_cw_step(f_if)
        if MONITOR_CH2:
            awg.duc_cw_step(f_if, channel=2)
        time.sleep(SETTLE_S)

        fc = LO_HZ + f_if
        # USBTMC occasionally throws VI_ERROR_INP_PROT_VIOL mid-run and wedges
        # the endpoint (it killed a previous run after 25 of 34 points, losing
        # ~4 minutes of sweeping). viClear recovers it, so retry rather than
        # abandoning the scan -- but only a bounded number of times, so a
        # genuinely dead link still fails instead of looping forever.
        for attempt in range(3):
            try:
                specan.set_freq(fc, SPAN_HZ)
                freqs, amps = specan.sweep_once()
                break
            except Exception as e:
                print(f"  [{f_if/1e9:.2f} GHz] transport error ({type(e).__name__}), "
                      f"clearing and retrying ({attempt+1}/3)")
                try:
                    specan._dev.clear()
                except Exception:
                    pass
                time.sleep(1.0)
        else:
            raise RuntimeError(f"3 consecutive transport failures at IF "
                               f"{f_if/1e9:.2f} GHz -- aborting")
        amps = amps + EXTERNAL_ATTEN_DB

        i = int(np.argmax(amps))
        usb_dbm, usb_hz = float(amps[i]), float(freqs[i])
        spurs = find_spurs(freqs, amps, usb_hz)
        worst = max(spurs, key=lambda s: s[1]) if spurs else None
        results.append({"if_hz": float(f_if), "usb_hz": usb_hz,
                        "usb_dbm": usb_dbm, "freqs": freqs, "amps": amps,
                        "spurs": spurs})
        wtxt = (f"{worst[1]-usb_dbm:+6.1f} dBc @ {(worst[0]-usb_hz)/1e6:+7.3f} MHz"
                if worst else "-")
        print(f"{f_if/1e9:7.2f} {usb_dbm:+12.2f} {len(spurs):6d}  {wtxt}")

        specanlog.save_trace(
            f"spur_if_{f_if/1e9:.2f}GHz".replace(".", "p"),
            "spur scan around the wanted USB (raw trace, pad not applied)",
            {"atten_db": ATTEN_DB, "ref_level_dbm": REF_LEVEL_DBM, "rbw_hz": rbw,
             "vbw_hz": VBW_HZ, "points": got_pts, "trace_mode": "NORM",
             "external_atten_db": EXTERNAL_ATTEN_DB, "center_hz": fc,
             "lo_hz": LO_HZ, "if_hz": float(f_if), "amp_vpp": AMP_VPP},
            freqs, amps - EXTERNAL_ATTEN_DB, subdir=run_subdir)
        if on_point is not None:
            try:
                on_point(results)
            except Exception as e:
                print(f"  (live update skipped: {e})")
    return results


def plot_results(results, fig=None):
    """Every spectrum overlaid.

    Top panel uses OFFSET FROM THE WANTED USB TONE and normalises each trace to its
    own peak (dBc). That is what makes the overlay informative: a spur at a
    fixed offset -- an LO artefact, a reference feedthrough -- lines up
    across every IF step, while something that moves is tied to the IF or
    the output frequency instead. On an absolute axis the traces would sit
    side by side and never overlap."""
    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    else:
        fig.clear()
        axes = fig.subplots(2, 1)

    # Common offset grid, then MAX-BIN down to a displayable width. Plotting
    # 30001 raw points per trace makes the noise a solid band that hides the
    # spurs; taking the MAX of each bin (not the mean) thins the trace while
    # preserving every peak, which is exactly what must survive here.
    n_disp = 1500
    off_full = np.linspace(-SPAN_HZ / 2, SPAN_HZ / 2, min(len(r["freqs"]) for r in results)) / 1e6
    rows = []
    for r in results:
        y = np.interp(off_full, (r["freqs"] - r["usb_hz"]) / 1e6,
                      r["amps"] - r["usb_dbm"])
        k = len(y) // n_disp
        if k > 1:
            y = y[:n_disp * k].reshape(n_disp, k).max(axis=1)
        rows.append(y)
    grid = np.vstack(rows)
    off = np.linspace(off_full[0], off_full[-1], grid.shape[1])
    ifs = np.array([r["if_hz"] / 1e9 for r in results])

    # Panel 1: 2D map. A spur at a FIXED offset draws a vertical stripe; one
    # that tracks the IF draws a diagonal. That distinction is the whole
    # point of scanning rather than measuring one frequency.
    ax = axes[0]
    # Scale tuned to the region spurs actually occupy (-75..-40 dBc). A
    # 0..-90 scale spends most of its range on the USB tone and the noise
    # floor and leaves the spurs almost invisible.
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="inferno",
                   extent=[off[0], off[-1], ifs[0], ifs[-1]],
                   vmin=-75, vmax=-40, interpolation="nearest")
    ax.set_ylabel("IF frequency (GHz)")
    ax.set_title(f"Spur map — {len(results)} IF steps, each normalised to its own "
                 f"USB tone.  Vertical stripe = fixed offset (LO-derived); "
                 f"diagonal = tracks IF", fontsize=9)
    fig.colorbar(im, ax=ax, label="dBc", pad=0.01)

    # Panel 2: the same traces stacked, one per row, so individual spur
    # shapes stay readable instead of being averaged into a colour.
    ax = axes[1]
    # Step must exceed the noise spread or the rows merge into one band.
    step = 25.0
    for k in range(len(results)):
        ax.plot(off, grid[k] + k * step, lw=0.5, color="C0")
    ax.set_yticks([k * step for k in range(0, len(results), 3)])
    ax.set_yticklabels([f"{ifs[k]:.1f}" for k in range(0, len(results), 3)],
                       fontsize=7)
    ax.set_ylabel("IF frequency (GHz)  [traces offset]")
    ax.set_xlabel("offset from USB (LO+IF) (MHz)")
    ax.set_title("Same data stacked — each trace spans "
                 f"{step:.0f} dB per division", fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')
    ax.set_xlim(off[0], off[-1])
    fig.tight_layout()
    return fig


def summarise(results):
    print("\n=== spur summary ===")
    n = sum(len(r["spurs"]) for r in results)
    print(f"{n} spur detection(s) across {len(results)} IF steps "
          f"(>{SPUR_SNR_DB:.0f} dB above the local noise floor, "
          f"outside +-{USB_EXCLUDE_HZ/1e6:.0f} MHz of the USB tone)")
    if not n:
        print("None found. Note this only clears the +-"
              f"{SPAN_HZ/2e6:.0f} MHz window at {RBW_HZ/1e3:.0f} kHz RBW -- "
              f"anything below the noise floor or outside that span is untested.")
        return
    # Group by offset: a family repeating at the same offset across many IF
    # steps is LO-derived; one that appears once is likely noise.
    allsp = [((sf - r["usb_hz"]) / 1e6, sa - r["usb_dbm"], r["if_hz"] / 1e9)
             for r in results for sf, sa in r["spurs"]]
    allsp.sort()
    used = [False] * len(allsp)
    print(f"\n{'offset MHz':>11} {'count':>6} {'worst dBc':>10}  IF steps seen at")
    for i, (off, dbc, iff) in enumerate(allsp):
        if used[i]:
            continue
        grp = [j for j in range(len(allsp))
               if not used[j] and abs(allsp[j][0] - off) < 0.5]
        for j in grp:
            used[j] = True
        offs = [allsp[j][0] for j in grp]
        dbcs = [allsp[j][1] for j in grp]
        ifs = [allsp[j][2] for j in grp]
        tag = "  <- REPEATS: likely real" if len(grp) >= 3 else ""
        print(f"{np.mean(offs):+11.3f} {len(grp):6d} {max(dbcs):10.1f}  "
              f"{', '.join(f'{x:.1f}' for x in ifs[:6])}"
              f"{'...' if len(ifs)>6 else ''}{tag}")


def run():
    awg = AWG()
    specan = SpecAn()
    run_subdir = time.strftime("spur_scan_%Y-%m-%d_%H%M%S")
    print(f"saving to specan_traces/{run_subdir}/\n")
    try:
        results = measure(awg, specan, run_subdir)
    finally:
        for name, fn in (("awg.stop", awg.stop), ("specan.close", specan.close)):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    summarise(results)
    fig = plot_results(results)
    png = os.path.join(specanlog.TRACE_DIR, run_subdir, "_spur_scan.png")
    fig.savefig(png, dpi=130)
    print(f"\nsaved {png}")
    plt.show()
    return results


if __name__ == "__main__":
    run()
