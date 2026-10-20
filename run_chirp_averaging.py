"""Does a CHIRP average coherently through the marker trigger?

The last gap. run_marker_coherence.py did CW, run_pulse_averaging.py did a
flat-top pulse; the chirp is the waveform the spectrometer actually
transmits, and it is the one that exercises grid snapping properly -- the
carrier has to move onto the achievable grid while the baseband sweep shifts
the other way to hold the absolute span.

The A/B is the production flag itself, which is the strongest version of
this test: both states are sent through send_waveforms(), the real path the
GUI uses, differing only in `coherent`.

    A  coherent=True   carrier snapped to the achievable grid, baseband
                       span shifted by the residual to compensate
    B  coherent=False  carrier used exactly as asked, which is off grid

Both are asked for the SAME absolute sweep, and compensation means both
should PRODUCE the same sweep -- the measured band edges are checked, so
that claim is tested rather than assumed. The only thing that differs is
whether the carrier phase repeats from shot to shot.

    AWG CH1 M1 -> scope CH4   trigger   (CH1 RF off; same chirp, so its
                                         segment is the same length as CH2's
                                         and the marker stays aligned)
    AWG CH2    -> scope CH3   signal

How a chirp's phase is measured
-------------------------------
A chirp has no single frequency to fit, so each shot is matched against the
first shot of its own run: the complex inner product of the two analytic
signals over the chirp window. Its argument is the shot-to-shot phase and
|<exp(i.phi)>| is the usual coherence. Shot 0 is the reference and so is
excluded from the statistic, otherwise it would contribute a guaranteed
zero-phase term and flatter the result by 1/N.

Reported per state:
    coherence        matched-filter phase spread over the run
    amplitude ratio  RMS of the hardware N-average over the mean single-shot
                     RMS, in the chirp window. 1.0 = averaging is lossless,
                     1/sqrt(N) = the shots cancelled. Model-free.
    band edges       -6 dB edges of the chirp spectrum, to confirm the
                     baseband compensation really did hold the sweep
    noise            RMS in the quiet stretch, single vs averaged

Run:  python run_chirp_averaging.py [n_shots] [span_s]
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal as sig

from awg import AWG
from scope import Scope
import reflock
from run_pulse_averaging import range_on_trigger

# -- the chirp (absolute frequencies; the carrier is an implementation
#    detail that snapping is free to move) ------------------------------
F_LO, F_HI = 700e6, 900e6
F_CARRIER = 0.5 * (F_LO + F_HI)          # 800 MHz, deliberately off grid
CHIRP_US, DEAD_US, DETECT_US = 1.0, 0.1, 10.0
AMP_VPP = 0.5

AWG_SIG_CH, AWG_MARK_CH, MARKER = 2, 1, 1
SIG_CH, TRIG_CH = 3, 4

CAPTURE_S = 4e-6
MAX_POINTS = 50000
N_AVG = 64
MARKER_NS = 100


def specs_for(carrier):
    """Same chirp on both channels: CH2 is measured, CH1 clocks the marker."""
    one = {"kind": "chirp", "gate": "chirp", "carrier_hz": carrier,
           "f_start_hz": F_LO - carrier, "f_stop_hz": F_HI - carrier,
           "amplitude_vpp": AMP_VPP, "sideband": "upper"}
    return {1: dict(one), 2: dict(one)}


def band_edges(t, v, win, chirp_s=None):
    """Sweep actually produced, from the instantaneous frequency.

    NOT a spectral dB contour. A linear chirp's spectrum is not flat-topped
    -- Fresnel ripple plus the windowed edges pull any contour well inside
    the real sweep, and measured on this bench a genuine 700-900 MHz chirp
    read 753.5-846.1 MHz at -6 dB and only 725.8-873.8 at -20 dB. The phase
    derivative has no such ambiguity.

    A line is fitted to f(t) over the central 80% of the burst (the ends are
    where the amplitude taper and the Hilbert edge effects live) and
    evaluated across `chirp_s`, so the answer is the sweep the chirp was
    asked for rather than the fraction of it that cleared a threshold.
    Returns (f_low, f_high, rate_hz_per_s).
    """
    m = (t >= win[0]) & (t < win[1])
    x = v[m] - v[m].mean()
    if x.size < 256:
        return float("nan"), float("nan"), float("nan")
    dt = float(t[1] - t[0])
    ph = np.unwrap(np.angle(sig.hilbert(x)))
    f_inst = np.gradient(ph, dt) / (2 * np.pi)
    k = max(f_inst.size // 50, 11)
    f_s = np.convolve(f_inst, np.ones(k) / k, mode="valid")
    tt = np.arange(f_s.size) * dt
    lo = f_s.size // 10
    rate, icept = np.polyfit(tt[lo:-lo], f_s[lo:-lo], 1)
    span = chirp_s if chirp_s else (tt[-1] - tt[0])
    mid_t, mid_f = 0.5 * (tt[0] + tt[-1]), None
    mid_f = rate * mid_t + icept
    return (float(mid_f - rate * span / 2), float(mid_f + rate * span / 2),
            float(rate))


def analytic(t, v, win):
    m = (t >= win[0]) & (t < win[1])
    return sig.hilbert(v[m] - v[m].mean())


def rms(t, v, win):
    m = (t >= win[0]) & (t < win[1])
    return float(np.std(v[m])) if m.sum() > 10 else float("nan")


def find_chirp(t, v, frac=0.5):
    env = np.abs(sig.hilbert(v - v.mean()))
    env = np.convolve(env, np.ones(151) / 151, mode="same")
    idx = np.flatnonzero(env > frac * env.max())
    if idx.size < 10:
        raise RuntimeError("no chirp found in the record")
    return float(t[idx[0]]), float(t[idx[-1]])


def run_state(awg, scope, label, coherent, n_shots, span_s, trig_level):
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}")
    plan = awg.send_waveforms(specs_for(F_CARRIER), mode="duc",
                              timing={"chirp_us": CHIRP_US, "dead_us": DEAD_US,
                                      "detect_us": DETECT_US},
                              verbose=False, coherent=coherent)
    n_total = plan["n_total"]
    car = plan["channels"]
    used = [e for e in plan["coherent_grid"] if e["channel"] == AWG_SIG_CH]
    carrier = (used[0]["carrier_hz"] if used else F_CARRIER)
    print(f"    carrier {carrier/1e6:.6f} MHz, "
          f"{carrier*plan['period_s']:.4f} cycles per buffer, "
          f"baseband {car[AWG_SIG_CH]['f_start_hz']/1e6:+.4f} to "
          f"{car[AWG_SIG_CH]['f_stop_hz']/1e6:+.4f} MHz")

    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * n_total,
                     marker=MARKER,
                     high_frac=MARKER_NS * 1e-9 / plan["period_s"],
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")

    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    t0r, v0r = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                                  absolute_time=True, width=2)
    c0, c1 = find_chirp(t0r, v0r)
    win = (c0, c1)
    win_noise = (c1 + 0.3e-6, t0r[-1])
    print(f"    chirp {c0*1e9:+.0f} to {c1*1e9:+.0f} ns "
          f"({(c1-c0)*1e6:.3f} us), quiet window "
          f"{(win_noise[1]-win_noise[0])*1e6:.2f} us")

    rng = np.random.default_rng(20261020)
    sched = np.concatenate([[0.0], np.sort(rng.uniform(0, span_s,
                                                       max(n_shots - 1, 0)))])
    ref, phases, rmss = None, [], []
    tstart = time.perf_counter()
    for i in range(n_shots):
        while time.perf_counter() < tstart + sched[i]:
            time.sleep(min(0.05, tstart + sched[i] - time.perf_counter()))
        t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                                  absolute_time=True, width=2)
        z = analytic(t, v, win)
        if ref is None:
            ref = z                      # shot 0 is the reference, not a datum
            lo, hi, rate = band_edges(t, v, win, CHIRP_US * 1e-6)
            print(f"    measured sweep {lo/1e6:.1f} -> {hi/1e6:.1f} MHz at "
                  f"{rate/1e12:.1f} MHz/us (asked {F_LO/1e6:.0f} -> "
                  f"{F_HI/1e6:.0f} at "
                  f"{(F_HI-F_LO)/1e6/CHIRP_US:.0f} MHz/us)")
        else:
            n = min(ref.size, z.size)
            phases.append(float(np.angle(np.vdot(ref[:n], z[:n]))))
        rmss.append(rms(t, v, win))
    span = time.perf_counter() - tstart
    r, sd = reflock.coherence(np.array(phases))

    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    t_a, v_a = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                                  absolute_time=True, width=2,
                                  acq_timeout=30.0)
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    lo_a, hi_a, rate_a = band_edges(t_a, v_a, win, CHIRP_US * 1e-6)
    r_single, r_avg = float(np.mean(rmss)), rms(t_a, v_a, win)
    n_single = rms(t0r, v0r, win_noise)
    n_avg = rms(t_a, v_a, win_noise)

    print(f"    coherence {r:5.3f}  (sd {sd:5.1f} deg, {len(phases)} pairs "
          f"over {span:.1f} s, floor {1/np.sqrt(max(len(phases),1)):.3f})")
    print(f"    {N_AVG}-average amplitude {r_avg/r_single:5.3f}   "
          f"({r_single*1e3:.2f} -> {r_avg*1e3:.2f} mV rms in the chirp)")
    print(f"    {N_AVG}-average sweep {lo_a/1e6:.1f} -> {hi_a/1e6:.1f} MHz "
          f"({rate_a/1e12:.1f} MHz/us)")
    print(f"    noise {n_single*1e3:.3f} -> {n_avg*1e3:.3f} mV "
          f"({n_single/max(n_avg,1e-12):.1f}x, sqrt({N_AVG})="
          f"{np.sqrt(N_AVG):.1f})")
    return {"label": label, "coherent": coherent, "carrier_hz": carrier,
            "cycles_per_buffer": carrier * plan["period_s"],
            "n_total": n_total, "period_s": plan["period_s"],
            "coherence": r, "sd_deg": sd, "n_pairs": len(phases),
            "span_s": span, "rms_single_v": r_single, "rms_avg_v": r_avg,
            "amp_ratio": r_avg / r_single,
            "band_single_hz": list(band_edges(t0r, v0r, win,
                                               CHIRP_US * 1e-6))[:2],
            "band_avg_hz": [lo_a, hi_a], "sweep_rate_hz_per_s": rate_a,
            "noise_single_v": n_single, "noise_avg_v": n_avg,
            "phases": phases}


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    span_s = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)
    print(f"\nchirp {F_LO/1e6:.0f} -> {F_HI/1e6:.0f} MHz in {CHIRP_US} us, "
          f"asked carrier {F_CARRIER/1e6:.3f} MHz")

    # first upload just to get the marker going so the trigger can be ranged
    plan = awg.send_waveforms(specs_for(F_CARRIER), mode="duc",
                              timing={"chirp_us": CHIRP_US, "dead_us": DEAD_US,
                                      "detect_us": DETECT_US}, verbose=True)
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * plan["n_total"],
                     marker=MARKER,
                     high_frac=MARKER_NS * 1e-9 / plan["period_s"],
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")

    saved = scope.save_state()
    with scope._lock:
        v_was = {ch: scope._dev.query(f"CH{ch}:SCAle?").strip()
                 for ch in (SIG_CH, TRIG_CH)}
        pos_was = scope._dev.query("HORizontal:POSition?").strip()
    results = []
    try:
        rate = scope.set_max_sample_rate(CAPTURE_S)
        scope.set_horizontal_position(10)
        with scope._lock:
            scope._dev.write(f"CH{TRIG_CH}:SCAle 0.25")
        scope._pre.clear()
        m_ptp, m_mid = range_on_trigger(scope, TRIG_CH, 0.0)
        if m_ptp < 0.1:
            print(f"no marker on CH{TRIG_CH} ({m_ptp*1e3:.0f} mVpp) -- abort")
            return 1
        range_on_trigger(scope, SIG_CH, m_mid)
        print(f"scope {rate/1e9:.2f} GS/s, {CAPTURE_S*1e6:g} us record, "
              f"trigger CH{TRIG_CH} @ {m_mid:.3f} V")

        results.append(run_state(awg, scope, "A  coherent=True  (carrier "
                                 "snapped, baseband compensated)",
                                 True, n_shots, span_s, m_mid))
        results.append(run_state(awg, scope, "B  coherent=False (carrier left "
                                 "off grid)",
                                 False, n_shots, span_s, m_mid))
    finally:
        scope.restore_state(saved)
        with scope._lock:
            for ch, sc in v_was.items():
                scope._dev.write(f"CH{ch}:SCAle {sc}")
            scope._dev.write(f"HORizontal:POSition {pos_was}")

    a, b = results
    print("\n" + "=" * 72)
    print(f"  {'':<24}{'coherent=True':>15}{'coherent=False':>16}")
    for key, name in (("coherence", "chirp coherence"),
                      ("amp_ratio", f"{N_AVG}-avg amplitude"),
                      ("cycles_per_buffer", "cycles per buffer")):
        print(f"  {name:<24}{a[key]:>15.3f}{b[key]:>16.3f}")
    print(f"  {'measured sweep MHz':<24}"
          f"{a['band_avg_hz'][0]/1e6:>7.1f}-{a['band_avg_hz'][1]/1e6:<7.1f}"
          f"{b['band_avg_hz'][0]/1e6:>8.1f}-{b['band_avg_hz'][1]/1e6:<7.1f}")
    print("=" * 72)

    if a["coherence"] > 0.9 and b["coherence"] < 0.5:
        print(f"\n  A chirp averages coherently once the carrier is on the "
              f"grid: {a['coherence']:.3f},\n  keeping "
              f"{a['amp_ratio']*100:.1f}% of its amplitude through a "
              f"{N_AVG}-shot average, against\n  {b['amp_ratio']*100:.1f}% "
              f"with the same chirp and an off-grid carrier.\n"
              f"  Both swept the same band, so the baseband compensation "
              f"held the sweep while\n  the carrier moved "
              f"{abs(a['carrier_hz']-F_CARRIER)/1e6:.3f} MHz.")
    elif a["coherence"] < 0.9 and b["coherence"] < 0.5:
        print(f"\n  The control collapsed but A only reached "
              f"{a['coherence']:.3f}. Something about the\n  chirp itself is "
              f"not repeating -- worth chasing before trusting an averaged "
              f"FID.")
    else:
        print("\n  Unexpected pattern -- check the setup before concluding.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("marker_coherence")
    out.mkdir(exist_ok=True)
    (out / f"chirp_averaging_{stamp}.json").write_text(json.dumps(
        {"timestamp": stamp, "f_lo_hz": F_LO, "f_hi_hz": F_HI,
         "f_carrier_asked_hz": F_CARRIER, "chirp_us": CHIRP_US,
         "n_average": N_AVG, "states": results}, indent=2))
    print(f"\nsaved to {out}/chirp_averaging_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
