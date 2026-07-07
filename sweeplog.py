"""Persist frequency-sweep measurements tagged to a microwave-chain setup.

Each sweep is written as ONE self-contained, human-readable JSON file under
``sweeps/`` plus a sidecar copy of the setup photo:

    sweeps/2026-07-07_140312_config_A.json
    sweeps/2026-07-07_140312_config_A.jpg     (copy of the setup photo)

The JSON keeps the metadata + per-frequency summary pretty-printed, while the
raw waveform samples are stored as compact single-line arrays (rounded) so the
file stays readable without exploding to one line per sample.  The time axis is
NOT stored per sample — it is linear, so only the sample step ``dt_s`` is kept
and reconstructed on load as ``t = arange(n) * dt_s``.
"""

import json
import os
import shutil
import time

import numpy as np

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")


def _safe(name):
    name = (name or "setup").strip() or "setup"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def save_sweep(setup_name, description, photo_path, params, records):
    """Write one sweep run to disk and return the JSON path.

    records: list of dicts, one per frequency, each with keys
        target_hz, actual_hz, vpp_v, loss_db, dt_s, voltage (1-D array-like).
    """
    os.makedirs(SWEEP_DIR, exist_ok=True)
    ts   = time.strftime("%Y-%m-%d_%H%M%S")
    base = f"{ts}_{_safe(setup_name)}"

    # Copy the setup photo next to the data so the run is self-documenting.
    photo_saved = None
    if photo_path and os.path.isfile(photo_path):
        ext = os.path.splitext(photo_path)[1] or ".img"
        photo_saved = base + ext
        try:
            shutil.copy2(photo_path, os.path.join(SWEEP_DIR, photo_saved))
        except Exception as e:
            print(f"sweeplog: could not copy photo: {e}\n")
            photo_saved = None

    doc = {
        "setup":       setup_name,
        "description": description,
        "photo":       photo_saved,
        "timestamp":   ts,
        "params":      params,
        "measurements": [
            {
                "target_hz": float(r["target_hz"]),
                "actual_hz": float(r["actual_hz"]),
                "vpp_v":     round(float(r["vpp_v"]), 9),
                "loss_db":   round(float(r["loss_db"]), 4),
                "dt_s":      float(r["dt_s"]),
                "n_points":  int(len(r["voltage"])),
                "voltage_v": f"@@WF{i}@@",   # placeholder → compact array below
            }
            for i, r in enumerate(records)
        ],
    }

    text = json.dumps(doc, indent=2)
    # Swap each placeholder for a compact one-line array of the samples.
    for i, r in enumerate(records):
        arr = "[" + ",".join(f"{float(x):.6g}" for x in r["voltage"]) + "]"
        text = text.replace(f'"@@WF{i}@@"', arr, 1)

    path = os.path.join(SWEEP_DIR, base + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def load_sweep(path):
    """Load a saved sweep JSON, reconstructing time axes and numpy arrays.

    Returns the parsed dict with each measurement gaining ``time_s`` and
    ``voltage_v`` numpy arrays (``voltage_v`` replaces the stored list).
    """
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    for m in doc.get("measurements", []):
        v = np.asarray(m["voltage_v"], dtype=float)
        m["voltage_v"] = v
        m["time_s"] = np.arange(len(v)) * m["dt_s"]
    return doc
