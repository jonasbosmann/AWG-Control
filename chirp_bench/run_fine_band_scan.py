"""Diagnostic bench script: very fine-carrier-resolution scan across the
full 0.2-4.2 GHz band, following up on run_carrier_scan.py's finding that
narrower per-segment span dramatically reduces the dechirp residual/EVM
(e.g. the original 0.8 GHz-wide 1.0-1.8 GHz segment measured ~1-1.5 MHz
rms; the same range split into 0.2 GHz segments dropped to <0.3 MHz rms
everywhere). This asks whether that trend continues at +-0.05 GHz
(0.1 GHz span) half-spans, all the way across the band -- i.e. whether
0.1 GHz segments are uniformly clean, or whether some region still shows
an elevated residual even at this resolution.

40 segments (0.2-4.2 GHz in 0.1 GHz steps) is too many to babysit
one-by-one, so unlike run_chirp_quality.py / run_carrier_scan.py this
runs FULLY UNATTENDED: no per-segment pauses, no popup plots. Progress
prints to the terminal; every segment's capture + diagnostic PNG still
gets saved to chirp_captures/ same as always, for review afterward.

Reuses run_chirp_quality.py's actual run_segment() unchanged -- only
SEGMENTS and the INTERACTIVE/SHOW_PLOTS flags differ.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from awg import AWG
from scope import Scope
import run_chirp_quality as rcq

# 0.2-4.2 GHz in 0.1 GHz steps, 0.1 GHz span each (+-0.05 GHz half-span).
SEGMENTS = [(round(f, 9), 0.1e9) for f in np.arange(0.2e9, 4.2e9, 0.1e9)]

# Runs unattended -- 40 segments is too many to manually confirm each one.
# The scope-side fix (arm_scope() now sets the display timebase to match
# the acquired window automatically) is what makes this safe to run
# without a human watching the live trace every time.
rcq.INTERACTIVE = False
rcq.SHOW_PLOTS = False


def run():
    awg = AWG()
    awg.set_reference_external(freq_mhz=10)
    scope = None
    summary = []
    try:
        scope = Scope()
        for i, (f_start, span_hz) in enumerate(SEGMENTS):
            print(f"\n--- segment {i+1}/{len(SEGMENTS)} ---")
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

    print(f"\n=== fine band scan summary ({len(summary)} segments, 0.1 GHz each) ===")
    for label, result in summary:
        dc, evm = result["dechirp"], result["evm"]
        mf = result["matched_filter"]
        print(f"{label:16s}  dechirp rms={dc['freq_err_rms_hz']/1e6:6.3f} MHz  "
              f"peak={dc['freq_err_peak_hz']/1e6:7.3f} MHz  "
              f"EVM={evm['evm_rms_pct']:6.2f}%  PSLR={mf['psl_db']:6.2f}dB")
    return summary


if __name__ == "__main__":
    run()
