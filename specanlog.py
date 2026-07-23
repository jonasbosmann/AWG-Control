"""Persist spectrum analyzer traces so spur/spectrum measurements can be
reviewed and compared later.

Each trace is written as one self-contained, human-readable JSON file under
``specan_traces/``:

    specan_traces/2026-07-23_143012_LO_10p42GHz.json

The metadata is pretty-printed; the frequency/amplitude arrays are stored as
compact single-line arrays (rounded) so the file stays readable without
exploding to one line per sample — same trick used in sweeplog.py.
"""

import json
import os
import time

import numpy as np

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specan_traces")


def _safe(name):
    name = (name or "trace").strip() or "trace"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def save_trace(label, notes, settings, freqs_hz, amps_dbm, subdir=None):
    """Write one spectrum trace to disk and return the JSON path.

    settings: dict of the SA acquisition settings in effect (ref level,
    RBW/VBW, attenuation, trace mode) so a saved trace is self-documenting.

    subdir: if given, save under TRACE_DIR/subdir/ instead of directly in
    TRACE_DIR — e.g. one folder per Auto Spur Hunt run, so a run's overview +
    main peak + every refined side peak stay grouped instead of scattering
    across a flat, ever-growing directory.
    """
    out_dir = os.path.join(TRACE_DIR, subdir) if subdir else TRACE_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    base = f"{ts}_{_safe(label)}"

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    amps_dbm = np.asarray(amps_dbm, dtype=float)

    doc = {
        "label":         label,
        "notes":         notes,
        "timestamp":     ts,
        "settings":      settings,
        "freq_start_hz": float(freqs_hz[0]),
        "freq_stop_hz":  float(freqs_hz[-1]),
        "n_points":      int(len(freqs_hz)),
        "freqs_hz":      "@@F@@",
        "amps_dbm":      "@@A@@",
    }

    text = json.dumps(doc, indent=2)
    f_arr = "[" + ",".join(f"{x:.1f}" for x in freqs_hz) + "]"
    a_arr = "[" + ",".join(f"{x:.3f}" for x in amps_dbm) + "]"
    text = text.replace('"@@F@@"', f_arr, 1)
    text = text.replace('"@@A@@"', a_arr, 1)

    path = os.path.join(out_dir, base + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def load_trace(path):
    """Load a saved trace JSON, returning the dict with numpy arrays."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["freqs_hz"] = np.asarray(doc["freqs_hz"], dtype=float)
    doc["amps_dbm"] = np.asarray(doc["amps_dbm"], dtype=float)
    return doc


def list_traces():
    """Return (path, label, timestamp) for every saved trace, newest first —
    recurses into per-hunt subfolders, not just the top-level directory."""
    if not os.path.isdir(TRACE_DIR):
        return []
    out = []
    for root, _dirs, files in os.walk(TRACE_DIR):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                out.append((path, doc.get("label", ""), doc.get("timestamp", "")))
            except Exception:
                continue
    out.sort(key=lambda x: x[2], reverse=True)
    return out
