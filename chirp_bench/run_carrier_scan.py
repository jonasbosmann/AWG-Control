"""Diagnostic bench script: fine-carrier-resolution scan across 1.0-2.2 GHz,
to localize the persistent positive freq_err offset found in the standard
0.8 GHz-wide "1.0-1.8 GHz" segment (2026-07-24 dechirp investigation --
confirmed repeatable across two independent sessions, ~1.5x-10x worse than
neighboring segments, NOT explained by the known ~1.5 GHz environmental
spur since the offset is already present before the sweep reaches 1.5 GHz).

Reuses run_chirp_quality.py's actual capture/analysis path (run_segment())
unchanged -- only SEGMENTS differs -- so results are directly comparable to
the standard band scan and any future fixes to the capture pipeline apply
here automatically.

Segments are 0.2 GHz wide (vs. the standard 0.8 GHz), tiling 1.0-2.2 GHz in
0.2 GHz steps -> carriers at 1.1, 1.3, 1.5, 1.7, 1.9, 2.1 GHz. This brackets
the known-bad ~1.4 GHz carrier on both sides and extends into the
better-behaved ~2.2 GHz region for context on where it transitions.

Run this exactly like run_chirp_quality.py (same wiring, same prompts).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from awg import AWG
from scope import Scope
import run_chirp_quality as rcq

SEGMENTS = [
    (1.0e9, 0.2e9),   # carrier 1.1 GHz
    (1.2e9, 0.2e9),   # carrier 1.3 GHz
    (1.4e9, 0.2e9),   # carrier 1.5 GHz -- also the known ~1.5 GHz spur frequency
    (1.6e9, 0.2e9),   # carrier 1.7 GHz
    (1.8e9, 0.2e9),   # carrier 1.9 GHz
    (2.0e9, 0.2e9),   # carrier 2.1 GHz -- approaching the better-behaved 1.8-2.6 GHz region
]


def run():
    awg = AWG()
    awg.set_reference_external(freq_mhz=10)
    scope = None
    summary = []
    try:
        scope = Scope()
        for f_start, span_hz in SEGMENTS:
            label, result = rcq.run_segment(awg, scope, f_start, span_hz)
            summary.append((label, result))
    finally:
        cleanups = [("awg.stop", awg.stop)]
        if scope is not None:
            cleanups.append(("scope.restore", lambda: scope.restore(channel=rcq.SIG_CH)))
        for name, fn in cleanups:
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}\n")

    print("\n=== carrier scan summary (1.0-2.2 GHz, 0.2 GHz segments) ===")
    for label, result in summary:
        lin, mf = result["linearity"], result["matched_filter"]
        dc, evm = result["dechirp"], result["evm"]
        print(f"{label:16s}  dechirp rms={dc['freq_err_rms_hz']/1e6:6.3f} MHz  "
              f"peak={dc['freq_err_peak_hz']/1e6:7.3f} MHz  EVM={evm['evm_rms_pct']:6.2f}%")
    return summary


if __name__ == "__main__":
    run()
