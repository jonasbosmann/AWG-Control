"""Aggregate view across MANY narrow-band captures spanning the whole
tunable range -- unlike pipeline_view.py (one capture at a time), this
stitches a whole band-scan RUN together into two panels:

  1. Amplitude vs frequency, stitched across every capture's own commanded
     band, on ONE shared 0-4.2 GHz axis (synthesizes a wideband amplitude
     sweep from many narrow individual chirp captures, the same idea as
     chirp_quality.swept_amplitude_response() but built for the WHOLE
     tunable range instead of one segment, and NOT normalized per-segment
     so genuine frequency-dependent rolloff across segments stays visible
     instead of being hidden by each capture doing its own local
     normalization).
  2. Every capture's raw time-domain trace, stacked in its own row ordered
     by frequency, so you can visually scan the whole run for anything
     obviously wrong in one view instead of opening 40 files one at a time.

Usage:
    python band_overview.py                    most recent run (auto-detected:
                                                 a contiguous cluster of captures
                                                 with no >5min gap between them)
    python band_overview.py 2026-07-24_1457     substring filter, same convention
                                                 as pipeline_view.py -- e.g. a
                                                 timestamp prefix to pick one run
"""
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

import chirp_quality as cq

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirp_captures")

RUN_GAP_S = 300  # captures more than this far apart are treated as separate runs


def _ts_of(path):
    b = os.path.basename(path)
    parts = b.split("_")
    return datetime.strptime(parts[0] + "_" + parts[1], "%Y-%m-%d_%H%M%S")


def select_run(query):
    files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.json")))
    if not files:
        raise SystemExit(f"no captures found in {CAPTURE_DIR}")

    if query:
        matches = [f for f in files if query in os.path.basename(f)]
        if not matches:
            raise SystemExit(f"no capture matches '{query}'")
        return matches

    # No query: split into contiguous runs (gap > RUN_GAP_S starts a new
    # one), then pick the most recent run that's actually substantial
    # (>= MIN_RUN_SIZE captures) -- NOT just whichever cluster happens to
    # be chronologically last. A blind "most recent cluster" picked up a
    # tiny 3-file ad-hoc test batch instead of the real 40-segment scan
    # that preceded it (confirmed 2026-07-24) -- a small leftover test
    # run shouldn't shadow a real one just for being newer.
    times = [_ts_of(f) for f in files]
    runs = []
    start = 0
    for i in range(1, len(files)):
        if (times[i] - times[i - 1]).total_seconds() > RUN_GAP_S:
            runs.append(files[start:i])
            start = i
    runs.append(files[start:])

    MIN_RUN_SIZE = 10
    substantial = [r for r in runs if len(r) >= MIN_RUN_SIZE]
    run = substantial[-1] if substantial else runs[-1]
    print(f"no query given -- auto-selected most recent substantial run "
          f"({len(runs)} runs found, sizes {[len(r) for r in runs]}): "
          f"{len(run)} captures, {os.path.basename(run[0])} .. {os.path.basename(run[-1])}")
    return run


def load_run(files):
    """Loads each capture, then deduplicates by (f_start_hz, f_stop_hz) --
    keeping only the MOST RECENT capture for any frequency band that
    appears more than once. Needed because a run-selection based on
    timestamp gaps alone can still merge in a short leftover/aborted batch
    that covers some of the same frequencies again (confirmed 2026-07-24:
    a stray 5-capture batch with a different chirp duration got pulled into
    the same 'run' as the real 40-segment scan right after it, producing
    duplicate rows with mismatched pulse lengths in the stacked time-domain
    panel) -- deduplicating by label is robust to that regardless of how
    the run-clustering heuristic draws its boundaries."""
    files = sorted(files, key=_ts_of)  # oldest first, so later dupes win below
    by_label = {}
    for f in files:
        with open(f) as fh:
            doc = json.load(fh)
        key = (doc.get("f_start_designated_hz", doc["f_start_hz"]),
               doc.get("f_stop_designated_hz", doc["f_stop_hz"]))
        by_label[key] = (f, doc)  # later (more recent) file overwrites earlier

    dropped = len(files) - len(by_label)
    if dropped:
        print(f"  dropped {dropped} duplicate-frequency capture(s), kept the most recent of each")

    docs = []
    for f, doc in by_label.values():
        v = np.array(doc["voltage_v"])
        dt = doc["dt_s"]
        t = np.arange(len(v)) * dt
        docs.append((doc, t, v))
    docs.sort(key=lambda d: d[0]["f_start_hz"])
    return docs


def band_amplitude(docs):
    """Peak spectral magnitude within each capture's own DESIGNATED band
    (falls back to the commanded f_start_hz/f_stop_hz for older captures
    that predate widen_for_flat_plateau() and don't have designated fields
    at all), kept in ABSOLUTE (not per-capture-normalized) units so
    amplitude is comparable ACROSS segments -- chirp_quality.spectrum_db()
    normalizes each capture to its own peak, which would hide exactly the
    frequency-dependent rolloff this view exists to show.

    Cropping to the DESIGNATED range (not the wider actual-commanded one,
    when the two differ) is deliberate -- run_chirp_quality.py now commands
    a wider sweep than requested specifically so the tapered edges fall
    OUTSIDE the designated band (see awg.widen_for_flat_plateau()); cropping
    here to the same designated range means this stitched view only ever
    shows the flat plateau part of each segment, not the ramps."""
    freqs_list, amp_list, labels = [], [], []
    for doc, t, v in docs:
        fs = 1.0 / float(np.median(np.diff(t)))
        n = len(v)
        spec = np.fft.rfft(v - np.mean(v))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(spec) * 2.0 / n
        f_start = doc.get("f_start_designated_hz", doc["f_start_hz"])
        f_stop = doc.get("f_stop_designated_hz", doc["f_stop_hz"])
        mask = (freqs >= f_start) & (freqs <= f_stop)
        freqs_list.append(freqs[mask])
        amp_list.append(mag[mask])
        labels.append(doc["label"])
    return freqs_list, amp_list, labels


def plot_band_overview(docs):
    freqs_list, amp_list, labels = band_amplitude(docs)
    nonempty = [a for a in amp_list if len(a)]
    global_max = max(a.max() for a in nonempty) if nonempty else 1.0

    n = len(docs)
    fig, (ax_amp, ax_time) = plt.subplots(
        2, 1, figsize=(12, 4 + 0.35 * n),
        gridspec_kw={"height_ratios": [1, min(3.5, 0.12 * n + 1)]})

    for freqs, amp in zip(freqs_list, amp_list):
        if len(amp) == 0:
            continue
        db = 20 * np.log10(amp / global_max + 1e-300)
        ax_amp.plot(freqs / 1e9, db, lw=1.2, color="C0")
    ax_amp.set_xlabel("GHz"); ax_amp.set_ylabel("dB (norm. to global peak)")
    ax_amp.set_title(f"Amplitude vs frequency, stitched across {n} captures "
                      f"({docs[0][0]['f_start_hz']/1e9:.2f}-{docs[-1][0]['f_stop_hz']/1e9:.2f} GHz)")
    fmax = max(doc["f_stop_hz"] for doc, _, _ in docs)
    fmin = min(doc["f_start_hz"] for doc, _, _ in docs)
    ax_amp.set_xlim(max(fmin / 1e9 - 0.1, 0), fmax / 1e9 + 0.1)

    row_h = 2.2
    for i, (doc, t, v) in enumerate(docs):
        v_n = v / (np.max(np.abs(v)) + 1e-30)
        offset = i * row_h
        xd, yd = cq.sinc_reconstruct(t * 1e6, v_n)
        ax_time.plot(xd, yd + offset, lw=0.35, color="C0")
    ax_time.set_yticks([i * row_h for i in range(n)])
    ax_time.set_yticklabels([doc["label"] for doc, _, _ in docs], fontsize=6)
    ax_time.set_ylim(-row_h, n * row_h)
    ax_time.set_xlabel("time (us)")
    ax_time.set_title("Time-domain traces, stacked by frequency "
                       "(each row independently amplitude-normalized)", fontsize=9)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else None
    files = select_run(query)
    docs = load_run(files)
    fig = plot_band_overview(docs)
    plt.show()
