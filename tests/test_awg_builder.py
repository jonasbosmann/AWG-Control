"""Offline checks for AWG's general dual-channel waveform builder.

No instrument is touched: _plan_timing() and _build_channel() need no
connection, so the AWG object is created with __new__ and only its class
attributes are used.

Why this file exists: the CP-FTMW shot (chirp on CH1, gated exact-frequency
LO on CH2, both in one shared buffer) used to be built by its own dedicated
function. It is now built by the general builder that also serves CW, square
and ramp on either channel. The reference arrays below are the ORIGINAL
construction, written out independently (own Gaussian window, own u16
mapping, own phase expression) so a change in the builder cannot quietly
change the waveform that goes on the air.

Result when written (2026-10-14): DUC segments bit-identical; the direct-mode
gated CW differs on 21 of 27,776 samples by one code (-96 dBFS), purely from
floating-point reassociation in the phase expression.

Run:  python tests/test_awg_builder.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awg import AWG, SAMPLE_RATE_DUAL


def builder():
    """An AWG whose builder methods work without a connection."""
    return AWG.__new__(AWG)


# ── independent reference implementations ─────────────────────────────

def ref_gauss(n, frac):
    w = np.ones(n)
    if n < 2:
        return w
    n_w = min(max(int(frac * n), 1), n // 2)
    sigma = n_w / 3.0
    idx = np.arange(n_w, dtype=np.float64)
    w[:n_w] = np.exp(-0.5 * ((idx - n_w) / sigma) ** 2)
    w[-n_w:] = np.exp(-0.5 * (idx / sigma) ** 2)
    return w


def ref_real_u16(wave):
    return np.round((np.asarray(wave) + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)


def ref_iq_u16(I, Q):
    iq = np.empty(2 * len(I), dtype=np.float64)
    iq[0::2] = I
    iq[1::2] = Q
    return (np.round(iq * 32767.0) + 32768.0).clip(1, 65535).astype(np.uint16)


def ref_duc_chirp_lo(f_start_bb, f_stop_bb, chirp_us, dead_us, detect_us, frac=0.05):
    """Original send_chirp_with_lo_duc() sample construction."""
    rate_cx = 9e9 / 8.0
    n_chirp = round(chirp_us * 1e-6 * rate_cx)
    n_dead = round(dead_us * 1e-6 * rate_cx)
    n_detect = round(detect_us * 1e-6 * rate_cx)
    n_total = max(int(np.ceil((n_chirp + n_dead + n_detect) / 32)) * 32, 64)

    t = np.arange(n_chirp, dtype=np.float64) / rate_cx
    T = t[-1] if n_chirp > 1 else 1e-9
    phase = 2 * np.pi * (f_start_bb * t + (f_stop_bb - f_start_bb) / (2 * T) * t ** 2)
    win1 = ref_gauss(n_chirp, frac)
    I1 = np.zeros(n_total); I1[:n_chirp] = np.cos(phase) * win1
    Q1 = np.zeros(n_total); Q1[:n_chirp] = np.sin(phase) * win1

    lo_start = n_chirp + n_dead
    win2 = ref_gauss(n_detect, frac)
    I2 = np.zeros(n_total); I2[lo_start:lo_start + n_detect] = win2
    Q2 = np.zeros(n_total)
    return ref_iq_u16(I1, Q1), ref_iq_u16(I2, Q2)


def ref_direct_chirp_lo(f_start, f_stop, chirp_us, dead_us, f_lo, detect_us, frac=0.05):
    """Original send_chirp_with_lo() sample construction."""
    rate = SAMPLE_RATE_DUAL
    n_chirp = round(chirp_us * 1e-6 * rate)
    n_dead = round(dead_us * 1e-6 * rate)
    n_detect = round(detect_us * 1e-6 * rate)
    n_total = max(int(np.ceil((n_chirp + n_dead + n_detect) / 64)) * 64, 128)

    t = np.arange(n_chirp, dtype=np.float64)
    phase = 2 * np.pi * (f_start / rate * t +
                         (f_stop - f_start) / (2 * rate * max(n_chirp - 1, 1)) * t ** 2)
    sig = np.sin(phase) * ref_gauss(n_chirp, frac)
    ch1 = np.zeros(n_total); ch1[:n_chirp] = sig

    t_lo = np.arange(n_detect, dtype=np.float64)
    lo = np.sin(2 * np.pi * f_lo / rate * t_lo) * ref_gauss(n_detect, frac)
    ch2 = np.zeros(n_total)
    ch2[n_chirp + n_dead:n_chirp + n_dead + n_detect] = lo
    return ref_real_u16(ch1), ref_real_u16(ch2)


# ── checks ────────────────────────────────────────────────────────────

def _diff(new, ref):
    assert len(new) == len(ref), f"length {len(new)} vs {len(ref)}"
    d = np.abs(new.astype(np.int64) - ref.astype(np.int64))
    return int(d.max()), int((d > 0).sum()), len(new)


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return cond


def test_duc_chirp_lo():
    print("DUC chirp + gated LO matches the original construction")
    ok = True
    for carrier, f0, f1, lo, ch_us, dd_us, dt_us in [
        (2000e6, 0.0, 500e6, 2300e6, 1.0, 0.1, 10.0),
        (2250e6, -400e6, 400e6, 1875e6, 2.5, 0.25, 4.0),
        (1200e6, 50e6, 60e6, 4400e6, 0.2, 0.05, 1.0),
    ]:
        a = builder()
        timing = {"chirp_us": ch_us, "dead_us": dd_us, "detect_us": dt_us}
        plan = a._plan_timing("duc", timing, a.DUC_DAC_RATE / a.DUC_INTERP)
        got1, _ = a._build_channel("duc", {
            "kind": "chirp", "carrier_hz": carrier,
            "f_start_hz": f0, "f_stop_hz": f1, "gate": "chirp"}, plan, 0.05)
        got2, _ = a._build_channel("duc", {
            "kind": "cw", "carrier_hz": lo, "freq_hz": 0.0,
            "gate": "detect"}, plan, 0.05)
        ref1, ref2 = ref_duc_chirp_lo(f0, f1, ch_us, dd_us, dt_us)
        m1, n1, ln = _diff(got1, ref1)
        m2, n2, _ = _diff(got2, ref2)
        ok &= check(f"chirp {f0/1e6:+.0f}..{f1/1e6:+.0f} MHz, {ch_us} us "
                    f"({ln//2:,} cx samp)", m1 == 0, f"max|d|={m1}")
        ok &= check(f"LO gate {dt_us} us", m2 == 0, f"max|d|={m2}")
    return ok


def test_direct_chirp_lo():
    print("Direct chirp + gated LO matches the original construction (<=1 code)")
    ok = True
    for f0, f1, lo, ch_us, dd_us, dt_us in [
        (10e6, 500e6, 100e6, 1.0, 0.1, 10.0),
        (200e6, 1200e6, 333.333e6, 3.0, 0.2, 6.0),
    ]:
        a = builder()
        timing = {"chirp_us": ch_us, "dead_us": dd_us, "detect_us": dt_us}
        plan = a._plan_timing("direct", timing, SAMPLE_RATE_DUAL)
        got1, _ = a._build_channel("direct", {
            "kind": "chirp", "f_start_hz": f0, "f_stop_hz": f1,
            "gate": "chirp"}, plan, 0.05)
        got2, _ = a._build_channel("direct", {
            "kind": "cw", "freq_hz": lo, "gate": "detect"}, plan, 0.05)
        ref1, ref2 = ref_direct_chirp_lo(f0, f1, ch_us, dd_us, lo, dt_us)
        m1, n1, ln = _diff(got1, ref1)
        m2, n2, _ = _diff(got2, ref2)
        # One code is -96 dBFS on a 16-bit DAC: the phase expression is
        # algebraically identical, only the order of the divisions differs.
        ok &= check(f"chirp {f0/1e6:.0f}..{f1/1e6:.0f} MHz", m1 <= 1,
                    f"max|d|={m1}, {n1}/{ln} samples differ")
        ok &= check(f"LO {lo/1e6:.3f} MHz exact (gated)", m2 <= 1,
                    f"max|d|={m2}, {n2}/{ln} samples differ")
    return ok


def test_gated_lo_is_exact_and_continuous_is_snapped():
    """A gated burst keeps the exact frequency; a continuously repeating
    buffer is snapped to a whole number of cycles so it closes on itself."""
    print("Integer-cycle snapping applies to continuous signals only")
    a = builder()
    plan = a._plan_timing("direct", {"chirp_us": 1.0, "dead_us": 0.1,
                                     "detect_us": 10.0}, SAMPLE_RATE_DUAL)
    _, gated = a._build_channel("direct", {
        "kind": "cw", "freq_hz": 333.333e6, "gate": "detect"}, plan, 0.05)
    _, cont = a._build_channel("direct", {
        "kind": "cw", "freq_hz": 333.333e6, "gate": "continuous"}, plan, 0.05)
    n = plan["n_total"]
    cycles = cont["freq_actual_hz"] * n / SAMPLE_RATE_DUAL
    ok = check("gated LO frequency is exact", gated["freq_actual_hz"] == 333.333e6)
    ok &= check("continuous frequency lands on a whole cycle count",
                abs(cycles - round(cycles)) < 1e-9,
                f"{cont['freq_actual_hz']/1e6:.6f} MHz -> {cycles:.6f} cycles")
    ok &= check("snapping error below one frequency step",
                abs(cont["freq_actual_hz"] - 333.333e6) <= SAMPLE_RATE_DUAL / n / 2,
                f"err {(cont['freq_actual_hz']-333.333e6)/1e3:+.1f} kHz, "
                f"step {SAMPLE_RATE_DUAL/n/1e3:.1f} kHz")
    return ok


def test_duc_sidebands():
    """Upper/lower/both must put the baseband where they claim: a DUC square
    wave is a real baseband made analytic, so the sideband choice is the only
    thing that decides which side of the carrier it lands on."""
    print("DUC sideband selection")
    a = builder()
    rate_cx = a.DUC_DAC_RATE / a.DUC_INTERP
    plan = a._plan_timing("duc", {"chirp_us": 1.0, "dead_us": 0.0,
                                  "detect_us": 9.0}, rate_cx)
    n = plan["n_total"]
    f_bb = 20e6
    out = {}
    for sb in ("upper", "lower", "both"):
        u16, _ = a._build_channel("duc", {
            "kind": "square", "carrier_hz": 2000e6, "freq_hz": f_bb,
            "sideband": sb, "gate": "continuous"}, plan, 0.05)
        iq = (u16.astype(np.float64) - 32768.0) / 32767.0
        z = iq[0::2] + 1j * iq[1::2]
        spec = np.abs(np.fft.fft(z))
        freqs = np.fft.fftfreq(n, 1 / rate_cx)
        pos = spec[(freqs > 0.5 * f_bb) & (freqs < 1.5 * f_bb)].max()
        neg = spec[(freqs < -0.5 * f_bb) & (freqs > -1.5 * f_bb)].max()
        out[sb] = (pos, neg)
    ok = check("upper puts the fundamental above the carrier only",
               out["upper"][0] > 100 * out["upper"][1],
               f"ratio {out['upper'][0]/max(out['upper'][1],1e-12):.0f}x")
    ok &= check("lower puts it below the carrier only",
                out["lower"][1] > 100 * out["lower"][0],
                f"ratio {out['lower'][1]/max(out['lower'][0],1e-12):.0f}x")
    ok &= check("both gives a symmetric pair (DSB / two-tone)",
                abs(out["both"][0] - out["both"][1]) < 0.01 * out["both"][0],
                f"{out['both'][0]:.1f} vs {out['both'][1]:.1f}")
    return ok


def ref_analytic(x):
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def test_no_clipping():
    """The analytic square/ramp overshoots +-1 (the Hilbert transform of a step
    is unbounded), so it must be SCALED to fit the DAC range. Clipping it
    instead would flat-top the peaks and add harmonics that are not in the
    requested signal. _iq_to_u16 clips, so the scaling has to happen first:
    the test is that the I/Q read back out of the uint16 words never exceeds
    full scale, while the unscaled baseband would have.
    """
    print("Analytic square/ramp are scaled to fit rather than clipped")
    a = builder()
    rate_cx = a.DUC_DAC_RATE / a.DUC_INTERP
    plan = a._plan_timing("duc", {"chirp_us": 1.0, "dead_us": 0.0,
                                  "detect_us": 9.0}, rate_cx)
    n = plan["n_total"]
    f_bb = 20e6
    t = np.arange(n, dtype=np.float64) / rate_cx
    cycles = max(round(f_bb * n / rate_cx), 1)
    phase = 2 * np.pi * (cycles * rate_cx / n) * t
    ok = True
    for kind, real in (("square", np.where(np.sin(phase) >= 0, 1.0, -1.0)),
                       ("ramp", 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0)):
        u16, info = a._build_channel("duc", {
            "kind": kind, "carrier_hz": 2000e6, "freq_hz": f_bb,
            "sideband": "upper", "gate": "continuous"}, plan, 0.05)
        iq = (u16.astype(np.float64) - 32768.0) / 32767.0
        peak = max(np.abs(iq[0::2]).max(), np.abs(iq[1::2]).max())
        z = ref_analytic(real)
        raw_peak = max(np.abs(np.real(z)).max(), np.abs(np.imag(z)).max())
        ok &= check(f"{kind}: the overshoot is real and needs handling",
                    raw_peak > 1.0, f"unscaled peak {raw_peak:.3f}")
        ok &= check(f"{kind}: nothing exceeds full scale after scaling",
                    peak <= 1.0 + 0.5 / 32767, f"peak {peak:.6f} of full scale")
        ok &= check(f"{kind}: reported scale factor matches the overshoot",
                    abs(info.get("scaled_by", 1.0) - 1.0 / raw_peak) < 1e-9,
                    f"scaled_by {info.get('scaled_by', 1.0):.4f} "
                    f"({20*np.log10(info.get('scaled_by', 1.0)):.1f} dB of range used)")
    return ok


def test_limits_rejected():
    """Aliasing and out-of-range carriers must raise, not silently produce a
    plausible-looking wrong frequency."""
    print("Out-of-range requests raise")
    a = builder()
    rate_cx = a.DUC_DAC_RATE / a.DUC_INTERP
    cases = [
        ("duc", {1: {"kind": "cw", "carrier_hz": 5e9, "freq_hz": 0.0}}, rate_cx,
         "carrier above the NCO range"),
        ("duc", {1: {"kind": "cw", "carrier_hz": 2e9, "freq_hz": 700e6}}, rate_cx,
         "baseband above complex Nyquist"),
        ("duc", {1: {"kind": "chirp", "carrier_hz": 4.4e9,
                     "f_start_hz": 0.0, "f_stop_hz": 300e6}}, rate_cx,
         "chirp top above the NCO range"),
        ("direct", {1: {"kind": "cw", "freq_hz": 2e9}}, SAMPLE_RATE_DUAL,
         "direct tone above Nyquist"),
    ]
    ok = True
    for mode, specs, rate, label in cases:
        try:
            a._check_limits(mode, specs, rate)
            ok &= check(label, False, "no exception raised")
        except ValueError as e:
            ok &= check(label, True, f"-> {e}")
    return ok


def test_plan_timing():
    print("Timing plan"); ok = True
    a = builder()
    rate_cx = a.DUC_DAC_RATE / a.DUC_INTERP
    plan = a._plan_timing("duc", {"chirp_us": 1.0, "dead_us": 0.1,
                                  "detect_us": 10.0}, rate_cx)
    ok &= check("DUC period is a multiple of 32 complex samples",
                plan["n_total"] % 32 == 0, f"n_total={plan['n_total']:,}")
    ok &= check("segment length (2 u16 per sample) is a multiple of 64",
                (2 * plan["n_total"]) % 64 == 0)
    ok &= check("windows do not overlap and fit the period",
                plan["windows"]["chirp"][1] <= plan["windows"]["detect"][0]
                and sum(plan["windows"]["detect"]) <= plan["n_total"])
    ok &= check("granularity pad lands after the detect window, not before it",
                plan["pad_s"] >= 0 and
                plan["windows"]["detect"][0] == plan["n_chirp"] + plan["n_dead"],
                f"pad {plan['pad_s']*1e9:.1f} ns")
    direct = a._plan_timing("direct", {"chirp_us": 1.0, "dead_us": 0.1,
                                       "detect_us": 10.0}, SAMPLE_RATE_DUAL)
    ok &= check("direct period is a multiple of 64 samples",
                direct["n_total"] % 64 == 0, f"n_total={direct['n_total']:,}")
    return ok


if __name__ == "__main__":
    results = [t() for t in (
        test_duc_chirp_lo,
        test_direct_chirp_lo,
        test_gated_lo_is_exact_and_continuous_is_snapped,
        test_duc_sidebands,
        test_no_clipping,
        test_limits_rejected,
        test_plan_timing,
    )]
    print(f"\n{sum(results)}/{len(results)} groups passed")
    sys.exit(0 if all(results) else 1)
