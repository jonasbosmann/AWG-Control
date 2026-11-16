"""Offline check of chirp_quality.align_and_average().

No instruments. Sixteen shots of one chirp (1.70-1.80 GHz, 1 us, 12.5 GS/s --
about 7 samples per carrier cycle) with the same carrier phase and random
trigger offsets of up to +-0.3 ns, i.e. about half a carrier period either
way. Aligned and averaged, the result must be the single shot: same
amplitude, and noise down by about sqrt(16).

The version before 2026-11-16 kept 0.37 of the amplitude on this input,
because it aligned a third of the shots half a carrier cycle out.

Run:  python tests/test_chirp_quality_align.py     (or pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chirp_bench"))
import chirp_quality as cq                                    # noqa: E402

DT, N = 1 / 12.5e9, 20000
T = np.arange(N) * DT
NOISE = 0.01


def shot(rng, shift, noise=NOISE):
    tau = T - 0.2e-6 - shift
    m = (tau >= 0) & (tau <= 1e-6)
    v = np.zeros(N)
    v[m] = np.sin(np.pi * tau[m] / 1e-6) ** 0.3 * np.cos(
        2 * np.pi * (1.7e9 * tau[m] + 0.05e9 * tau[m] ** 2 / 1e-6))
    return v + noise * rng.standard_normal(N)


def test_delay_estimate_is_right_to_a_hundredth_of_a_sample():
    rng = np.random.default_rng(1)
    ref = shot(rng, 0.0, noise=0.0)
    for s in (0.0, 0.1e-9, -0.2e-9, 0.3e-9, 0.25e-9, -0.05e-9):
        got = cq._shot_delay(shot(rng, s, noise=0.0), ref)
        assert abs(got - s / DT) < 0.01, (s / DT, got)


def test_average_keeps_the_amplitude_and_drops_the_noise():
    rng = np.random.default_rng(0)
    clean = shot(rng, 0.0, noise=0.0)
    shots = [(T, shot(rng, 0.0))] + [
        (T, shot(rng, s)) for s in rng.uniform(-0.3e-9, 0.3e-9, 15)]
    _, avg = cq.align_and_average(shots)
    # amplitude: projection of the average onto the clean waveform
    kept = float(np.dot(avg, clean) / np.dot(clean, clean))
    assert abs(kept - 1.0) < 0.01, kept
    resid = float(np.std(avg - kept * clean))
    # the first shot's noise is common to every alignment, so not quite 1/4
    assert resid < NOISE / 2.5, resid


def test_t0_and_evm_do_not_depend_on_carrier_phase():
    """A perfect chirp must read ~0 % EVM whatever its carrier phase is.

    Before 2026-11-16 refine_t0() took the peak of |corr|, which follows the
    carrier, and this same input read 7-38 % EVM with t0 up to 280 ps out.
    """
    f0, f1, dur = 1.8e9, 2.6e9, 1.0e-6
    t = np.arange(25000) * DT
    t0_true = 0.3e-6 + 0.31 * DT
    tau = t - t0_true
    tc = np.clip(tau, 0, dur)
    win = cq._gauss_win_continuous(tau, dur, 0.05)
    for deg in (0, 45, 90, 135, 180, 270):
        v = win * np.sin(2 * np.pi * (f0 * tc + (f1 - f0) / (2 * dur) * tc ** 2)
                         + np.radians(deg))
        t0 = cq.refine_t0(t, v, f0, f1, dur)
        assert abs(t0 - t0_true) < 1e-12, (deg, (t0 - t0_true) * 1e12)
        evm = cq.evm_score(t, v, f0, f1, dur, t0=t0, fit_t0=False)
        assert evm["evm_rms_pct"] < 0.5, (deg, evm["evm_rms_pct"])
        # and the timing fit recovers from a start that is 400 ps out
        fit = cq.evm_score(t, v, f0, f1, dur, t0=t0 + 0.4e-9)
        assert fit["evm_rms_pct"] < 0.5, (deg, fit["evm_rms_pct"])
        assert fit["evm_rms_pct_unfitted"] > 5.0
        assert abs(fit["t0_shift_s"] + 0.4e-9) < 5e-12, fit["t0_shift_s"]
        assert not fit["t0_fit_at_edge"]


if __name__ == "__main__":
    test_t0_and_evm_do_not_depend_on_carrier_phase()
    test_delay_estimate_is_right_to_a_hundredth_of_a_sample()
    test_average_keeps_the_amplitude_and_drops_the_noise()
    print("ALL PASS")
