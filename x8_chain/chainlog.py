"""Persist x8-chain bench measurements, with the setup attached.

Same shape as specanlog.py (pretty-printed metadata, compact single-line
arrays) but with one difference that matters: the `setup` block is written by
this module from config.py, not by the caller. A run script cannot save a
measurement without recording which pads were in line, because it never gets
the chance to omit them.

Layout:

    x8_chain/measurements/2026-07-28_A_gain/
        2026-07-28_161204_thru.json
        2026-07-28_161540_amp.json
        gain.png

One folder per measurement session, so the reference run and the run it
references stay together -- a thru trace is worthless on its own, and a gain
curve derived from a thru trace taken on a different day with different cables
is worse than worthless.
"""
import json
import os
import time

import numpy as np

import config

MEAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "measurements")


def _safe(name):
    name = (name or "meas").strip() or "meas"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def series_key(name):
    """JSON field name for a series called `name`.

    Series names are things like "0.439vpp" and "1.614ghz", and the '.' has to
    be sanitized out for the field name -- so a reader that reconstructs the
    key by hand gets "dbm_0.439vpp" and finds nothing. Both save() and every
    reader go through this function so the two cannot drift apart.
    """
    return "dbm_" + _safe(name)


def new_session(stage_key, what):
    """Create and return a fresh session directory.

    Timestamped at CALL time, not import time, so a long-lived caller gets a
    new folder per run instead of silently overwriting the first one's files
    (the same trap run_specan_band_scan.new_run_subdir() exists to avoid).
    """
    name = time.strftime(f"%Y-%m-%d_%H%M%S_{_safe(stage_key)}_{_safe(what)}")
    path = os.path.join(MEAS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def save(session_dir, label, stage_key, x, series, instrument_settings,
         x_name="freqs_hz", thru=False, notes="", arrangement=None):
    """Write one measurement.

    series: {series_name: 1-D array of dBm as the instrument reported it}.
        Keys are free-form -- typically the AWG amplitude, e.g. "0.55vpp"
        for a frequency sweep, or the spot frequency for a drive sweep.
    x/x_name: the swept axis shared by every series -- "freqs_hz" for a
        flatness sweep, "amps_vpp" for a compression sweep.

    Levels are stored RAW. The pad is recorded, not applied; anything that
    reports levels adds it back via config.PAD_AT_INSTRUMENT_DB. That way a
    later correction to the pad value does not require re-taking data.
    """
    st = config.STAGES[stage_key]
    x = np.asarray(x, dtype=float)

    doc = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "stage": stage_key,
        "part": "THRU (adapter in place of amp)" if thru else st["part"],
        "role": st["role"],
        "thru_reference": bool(thru),
        "notes": notes,
        "setup": {
            # Written from config.py unless the caller passes an explicit
            # `arrangement`. The override exists because config.arrangement()
            # can only describe ONE stage -- source, amp, pad, instrument --
            # and says nothing about a multi-element cascade. Once the bench
            # moved past a bare stage it was writing
            # "AWG -> ZX60-2522MA-S+ -> 10 dB pad -> scope" into files that
            # were actually a doubler + two filters + a pad + a second amp,
            # measured on the EXA: the wrong instrument and none of the parts
            # under test. A confidently wrong setup record is worse than none,
            # so a caller that knows the real path MUST be able to state it.
            # The field is still mandatory and still written here -- the
            # discipline is unchanged, only its accuracy is improved.
            "arrangement": arrangement or config.arrangement(stage_key, thru=thru),
            "pad_at_instrument_db": config.PAD_AT_INSTRUMENT_DB,
            # The drive pad stays in line for the thru reference too -- the
            # thru replaces ONLY the amp. That is what makes gain come out as
            # a clean (amp - thru) subtraction with every pad and cable in
            # the path cancelling exactly.
            "drive_pad_db": st["drive_pad_db"],
            "instrument": st["instrument"],
            "levels_are": "RAW as measured at the instrument; "
                          "ADD pad_at_instrument_db to refer to the DUT output",
        },
        "instrument_settings": instrument_settings,
        "band_hz": list(st["band_hz"]),
        "x_name": x_name,
        "n_points": int(len(x)),
        x_name: "@@F@@",
    }
    for name in series:
        doc[series_key(name)] = f"@@S_{_safe(name)}@@"

    text = json.dumps(doc, indent=2)
    fmt = "{:.1f}" if x_name.endswith("_hz") else "{:.4f}"

    def _arr(a, f):
        # Non-finite values MUST be written as JSON null. Python formats a NaN
        # as lowercase "nan", and json.load accepts "NaN" but not "nan" -- so
        # a single skipped point made the ENTIRE file unparseable, and
        # find()/load() skip unparseable files silently. The measurement had
        # been taken, saved, and was gone. Found 2026-07-31 when a run with
        # one skipped point vanished from find(). null reads back as nan via
        # np.asarray(..., dtype=float), so nothing downstream changes.
        return "[" + ",".join("null" if not np.isfinite(v) else f.format(v)
                              for v in a) + "]"

    text = text.replace('"@@F@@"', _arr(x, fmt), 1)
    for name, arr in series.items():
        text = text.replace(f'"@@S_{_safe(name)}@@"',
                            _arr(np.asarray(arr, dtype=float), "{:.3f}"), 1)

    path = os.path.join(session_dir, f"{time.strftime('%H%M%S')}_{_safe(label)}.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def load(path):
    """Load a saved measurement, arrays as numpy."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    x_name = doc.get("x_name", "freqs_hz")
    for k, v in list(doc.items()):
        if k == x_name or k.startswith("dbm_"):
            doc[k] = np.asarray(v, dtype=float)
    return doc


def find(stage_key, label_contains=None, thru=None, session_dir=None):
    """Saved measurements for a stage, newest first.

    Searched across sessions so a reference taken earlier in the day can be
    reused, but the caller is told which file it got -- reusing a reference
    across a cable change is exactly the kind of thing that has to be visible,
    not silent.

    thru: True/False to filter on the thru flag, None for either.
    """
    hits = []
    root = session_dir or MEAS_DIR
    if not os.path.isdir(root):
        return hits
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except Exception:
                continue
            if doc.get("stage") != stage_key:
                continue
            if thru is not None and bool(doc.get("thru_reference")) != thru:
                continue
            if label_contains and label_contains not in doc.get("label", ""):
                continue
            hits.append((doc.get("timestamp", ""), p))
    hits.sort(reverse=True)
    return [p for _ts, p in hits]


def find_thru(stage_key, label_contains=None, session_dir=None):
    """Most recent thru reference for a stage, newest first."""
    return find(stage_key, label_contains=label_contains, thru=True,
                session_dir=session_dir)
