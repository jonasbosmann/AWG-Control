"""Offline check of phasecycle.split_cycles() on a synthetic K-cycle record.

No instruments. Builds four 1.99-2.49 GHz chirps with phases 0/180/0/180 at
the LABBOOK 21 timing (3.1004 us per cycle, which is NOT a whole number of
scope samples), with a sub-sample trigger offset, and splits and recombines
them. There are no artifacts in the record, so whatever the straight sum
reports is the re-sampling's own error.

What it pins:
  * linear interpolation at 12.5 GS/s (5.6 samples per carrier cycle)
    reproduces the false -24 dB artifact floor of LABBOOK 21;
  * the default band-limited re-sampling is clean at 50, 25, 12.5 and
    6.25 GS/s, so the scope no longer has to run at 50 GS/s for this.

Run:  python tests/test_phasecycle_split.py     (or pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phasecycle                                             # noqa: E402

K, T_CYCLE = 4, 3488 / 1.125e9
F0, F1, T_CHIRP = 1.99e9, 2.49e9, 1.0e-6
PHASES = [0.0, np.pi, 0.0, np.pi]
SKEW = 0.5483e-6                      # marker-to-RF, as measured


def record(rate):
    dt = 1.0 / rate
    n = int(round((K * T_CYCLE + 1.2e-6) * rate))
    t = -0.02e-6 + 0.37 * dt + np.arange(n) * dt
    v = np.zeros(n)
    for k in range(K):
        tau = t - (SKEW + k * T_CYCLE)
        m = (tau >= 0) & (tau <= T_CHIRP)
        w = np.sin(np.pi * tau[m] / T_CHIRP) ** 0.25
        v[m] += 0.16 * w * np.cos(
            2 * np.pi * (F0 * tau[m] + (F1 - F0) / (2 * T_CHIRP) * tau[m] ** 2)
            + PHASES[k])
    return t, v


def run(rate, interp):
    t, v = record(rate)
    cut = phasecycle.split_cycles(t, v, K, T_CYCLE, 0.9e-6,
                                  t_first=SKEW + 0.05e-6, carrier_hz=2.24e9,
                                  warn=None, interp=interp)
    return phasecycle.combine(cut["analytic"], PHASES)


def test_linear_reproduces_the_labbook_artifact():
    r = run(12.5e9, "linear")
    assert -27.0 < r["artifact_db"] < -20.0, r["artifact_db"]


def test_band_limited_is_clean_at_every_rate():
    for rate in (50e9, 25e9, 12.5e9, 6.25e9):
        r = run(rate, "fft")
        assert r["artifact_db"] < -80.0, (rate, r["artifact_db"])
        assert abs(r["gain"] - K) < 1e-3, (rate, r["gain"])
        assert max(abs(e) for e in r["error_deg"]) < 0.01, (rate, r["error_deg"])


def test_default_is_band_limited():
    t, v = record(12.5e9)
    cut = phasecycle.split_cycles(t, v, K, T_CYCLE, 0.9e-6,
                                  t_first=SKEW + 0.05e-6, warn=None)
    assert phasecycle.combine(cut["analytic"], PHASES)["artifact_db"] < -80.0


if __name__ == "__main__":
    print(f"{'GS/s':>6} {'interp':>7} {'gain':>7} {'artifact dB':>12}")
    for rate in (50e9, 25e9, 12.5e9, 6.25e9):
        for interp in ("linear", "fft"):
            r = run(rate, interp)
            print(f"{rate/1e9:6.2f} {interp:>7} {r['gain']:7.3f} "
                  f"{r['artifact_db']:12.1f}")
    test_linear_reproduces_the_labbook_artifact()
    test_band_limited_is_clean_at_every_rate()
    test_default_is_band_limited()
    print("ALL PASS")
