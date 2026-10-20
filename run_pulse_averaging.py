"""Average a PULSE on the marker trigger, the way an FID acquisition will.

What this adds over run_marker_coherence.py
-------------------------------------------
That test used a continuous carrier, which is the easy case and not the case
the spectrometer runs in. This one plays what the acquisition will actually
see: a gated burst at the start of the buffer followed by a long quiet
stretch, one marker pulse per buffer, scope triggered on the marker, CH2
averaged.

    buffer   16384 complex samples at 1.125 GS/s = 14.5636 us
    pulse    2 us flat top with raised-cosine edges, then 12.5 us of nothing
    marker   ~100 ns at the start of the buffer, AWG CH1 M1

A flat-top pulse, deliberately not a decaying FID: constant amplitude makes
the averaged-against-single comparison unambiguous, with no envelope to
disentangle from the thing being measured. Set DECAY_TAU_US to add a decay
once the flat case is understood.

The point of doing it with a transient
--------------------------------------
A CW carrier has to close the buffer exactly or its phase does not repeat --
measured 2026-10-20, 0.997 on the grid against 0.21-0.37 off it. The grid is
coarse: 68664.55 Hz here.

A pulse should not be bound by that, and the reason is worth stating
precisely. The constraint comes from the baseband having to JOIN UP across
the buffer boundary. A pulse has decayed to zero well before the boundary, so
there is nothing to join. Only the NCO carrier, which runs continuously
underneath, still has to land on the grid. If that is right, the baseband can
place the line at any offset at all and averaging still works -- which is the
only way to reach an arbitrary absolute frequency with a coarse carrier grid.

So the baseband offset here is deliberately NOT a grid multiple:

    f_baseband = 1.234567 MHz = 17.98 grid steps

    state A   carrier ON grid  (808.593750 MHz, k = 11776)   -> expect coherent
    state B   carrier 25 kHz off grid (0.364 of a step)      -> expect collapse

If A is coherent with that baseband offset, a pulse can be put anywhere.

What is measured
----------------
Per state, N single shots plus one hardware N-average:

  coherence        |<exp(i phi)>| of the pulse phase at the trigger
  amplitude ratio  averaged pulse / mean single pulse -- 1.0 = lossless
  noise ratio      RMS in the quiet stretch, single / averaged. Should be
                   sqrt(N) if the noise is random. If it comes out well below
                   that, what is in the quiet stretch is not noise but
                   something coherent -- carrier feedthrough or a DAC spur --
                   and averaging will never remove it.
  SNR gain         the two combined: the number the experiment actually buys.

Run:  python run_pulse_averaging.py [n_shots] [span_s]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal as sig

from awg import AWG
from scope import Scope
import reflock

# -- waveform ----------------------------------------------------------
DAC_RATE, INTERP = 9e9, 8
RATE_CX = DAC_RATE / INTERP                 # 1.125 GS/s complex
N_TOTAL = 16384                             # power of two -> exact on the NCO
T_BUF = N_TOTAL / RATE_CX                   # 14.5636 us
GRID_HZ = RATE_CX / N_TOTAL                 # 68664.55078125 Hz
K_ON = 11776
F_NCO_ON = K_ON * GRID_HZ                   # 808 593 750 Hz exactly
F_NCO_OFF = F_NCO_ON + 25e3                 # 0.364 of a grid step
F_BB = 1.234567e6                           # 17.98 grid steps -- NOT on grid
PULSE_US = 2.0
EDGE_FRAC = 0.05                            # raised-cosine edges
DECAY_TAU_US = None                         # set a number to make it decay
AMP_VPP = 0.5

# -- wiring ------------------------------------------------------------
AWG_SIG_CH, AWG_MARK_CH, MARKER = 2, 1, 1
SIG_CH, TRIG_CH = 3, 4

# -- acquisition -------------------------------------------------------
CAPTURE_S = 5e-6
MAX_POINTS = 62500
PRETRIG_PCT = 10
N_AVG = 64
MARKER_NS = 100


def build_pulse():
    """Complex baseband: a gated tone at F_BB, then silence. Returns (I, Q)."""
    n_pulse = int(round(PULSE_US * 1e-6 * RATE_CX / 32)) * 32
    t = np.arange(n_pulse) / RATE_CX
    env = AWG._gauss_win(n_pulse, EDGE_FRAC)
    if DECAY_TAU_US:
        env = env * np.exp(-t / (DECAY_TAU_US * 1e-6))
    z = env * np.exp(2j * np.pi * F_BB * t)
    z = z / max(np.abs(z).max(), 1e-12)     # exactly full scale, never over
    i_arr = np.zeros(N_TOTAL)
    q_arr = np.zeros(N_TOTAL)
    i_arr[:n_pulse] = z.real
    q_arr[:n_pulse] = z.imag
    return i_arr, q_arr, n_pulse


def range_on_trigger(scope, ch, trig_level, divisions=6.0, lo=0.002, hi=1.0):
    """Range a channel from a MARKER-TRIGGERED capture.

    A free-running look is useless here: the marker is 100 ns in a 14.56 us
    buffer (0.7% duty) and the pulse is 14%, so a short untriggered read
    almost always lands in the quiet stretch and reports 0 mVpp. Triggering
    guarantees the feature is in the record.
    """
    scope.setup(channel=ch, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    t, v = scope.get_waveform(ch, max_points=MAX_POINTS, fresh=True,
                              width=2)
    ptp = float(np.ptp(v))
    if ptp > 0:
        with scope._lock:
            scope._dev.write(
                f"CH{ch}:SCAle {min(max(ptp / divisions, lo), hi):.4f}")
        scope._pre.clear()
        t, v = scope.get_waveform(ch, max_points=MAX_POINTS, fresh=True,
                              width=2)
        ptp = float(np.ptp(v))
    return ptp, float((v.max() + v.min()) / 2)


def find_pulse(t, v, frac=0.5):
    """(t_start, t_end) of the burst, from the analytic envelope."""
    env = np.abs(sig.hilbert(v - v.mean()))
    env = np.convolve(env, np.ones(101) / 101, mode="same")
    above = env > frac * env.max()
    idx = np.flatnonzero(above)
    if idx.size < 10:
        raise RuntimeError("no burst found in the record")
    return float(t[idx[0]]), float(t[idx[-1]])


def measure(t, v, f_abs, win_fit, win_noise):
    """(amplitude, phase at trigger, noise RMS) for one record."""
    m = (t >= win_fit[0]) & (t < win_fit[1])
    a, p = reflock.fit_tone(t[m], v[m], f_abs, t_ref=0.0)
    n = (t >= win_noise[0]) & (t < win_noise[1])
    rms = float(np.std(v[n])) if n.sum() > 10 else float("nan")
    return a, p, rms


def run_state(awg, scope, label, f_nco, n_shots, trig_level, span_s,
              win_fit, win_noise, keep_traces=False):
    import time
    f_abs = f_nco + F_BB
    for ch in (AWG_SIG_CH, AWG_MARK_CH):
        awg.duc_cw_step(f_nco, channel=ch)
    print(f"\n  {label}")
    print(f"    carrier {f_nco/1e6:.6f} MHz "
          f"({(f_nco/GRID_HZ):.4f} grid steps), baseband +{F_BB/1e6:.6f} MHz"
          f"  -> {f_abs/1e6:.6f} MHz")

    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")

    rng = np.random.default_rng(20261020)
    sched = np.concatenate([[0.0], np.sort(rng.uniform(0, span_s,
                                                       max(n_shots - 1, 0)))])
    amps, phases, noises, traces = [], [], [], []
    t0 = time.perf_counter()
    for i in range(n_shots):
        while time.perf_counter() < t0 + sched[i]:
            time.sleep(min(0.05, t0 + sched[i] - time.perf_counter()))
        t, v = scope.get_waveform(SIG_CH, max_points=MAX_POINTS,
                                  fresh=True, absolute_time=True, width=2)
        a, p, rms = measure(t, v, f_abs, win_fit, win_noise)
        amps.append(a)
        phases.append(p)
        noises.append(rms)
        if keep_traces and i < 10:
            traces.append((t, v))
        if i == 0:
            m = (t >= win_fit[0]) & (t < win_fit[1])
            f_meas, cyc = reflock.estimate_frequency(t[m], v[m])
            print(f"    first shot: pulse at {f_meas/1e6:.4f} MHz "
                  f"({(f_meas-f_abs)/1e3:+.1f} kHz from predicted, "
                  f"{cyc:.0f} cycles), {a*2*1000:.1f} mVpp")
    span = time.perf_counter() - t0

    r, sd = reflock.coherence(np.array(phases))
    a_single = float(np.mean(amps))
    n_single = float(np.mean(noises))

    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")
    t_av, v_av = scope.get_waveform(SIG_CH, max_points=MAX_POINTS, fresh=True,
                                    absolute_time=True, acq_timeout=30.0, width=2)
    a_avg, _, n_avg = measure(t_av, v_av, f_abs, win_fit, win_noise)
    scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                trigger_level=trig_level, trigger_mode="NORMal")

    amp_ratio = a_avg / a_single
    noise_ratio = n_single / n_avg if n_avg > 0 else float("inf")
    snr_gain = amp_ratio * noise_ratio
    print(f"    coherence {r:5.3f}  (sd {sd:5.1f} deg, {n_shots} shots "
          f"over {span:.1f} s, floor {1/np.sqrt(n_shots):.3f})")
    print(f"    {N_AVG}-average: amplitude {amp_ratio:5.3f}  "
          f"({a_single*2*1000:.1f} -> {a_avg*2*1000:.1f} mVpp)")
    print(f"    {N_AVG}-average: noise RMS {n_single*1e3:.3f} -> "
          f"{n_avg*1e3:.3f} mV, down {noise_ratio:4.1f}x "
          f"(sqrt({N_AVG}) = {np.sqrt(N_AVG):.1f})")
    print(f"    -> SNR gain {snr_gain:4.1f}x")

    return ({"label": label, "f_nco_hz": f_nco, "f_abs_hz": f_abs,
             "coherence": r, "sd_deg": sd, "span_s": span, "n_shots": n_shots,
             "amp_single_vpp": a_single * 2, "amp_avg_vpp": a_avg * 2,
             "amp_ratio": amp_ratio, "noise_single_v": n_single,
             "noise_avg_v": n_avg, "noise_ratio": noise_ratio,
             "snr_gain": snr_gain, "n_average": N_AVG,
             "phases": list(map(float, phases)),
             "amps_vpp": [float(a) * 2 for a in amps]},
            traces, (t_av, v_av))


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    span_s = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    i_arr, q_arr, n_pulse = build_pulse()
    print(f"buffer {N_TOTAL} complex @ {RATE_CX/1e6:.1f} MS/s = "
          f"{T_BUF*1e6:.4f} us, grid {GRID_HZ:.6f} Hz")
    print(f"pulse  {n_pulse} samples = {n_pulse/RATE_CX*1e6:.3f} us "
          f"({n_pulse/N_TOTAL*100:.1f}% duty), baseband {F_BB/1e6:.6f} MHz "
          f"= {F_BB/GRID_HZ:.2f} grid steps (NOT on grid, deliberately)")
    print(f"carrier on grid {F_NCO_ON/1e6:.6f} MHz (k={K_ON}), "
          f"off grid {F_NCO_OFF/1e6:.6f} MHz")

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)
    awg.duc_iq_setup_dual(i_arr, q_arr, F_NCO_ON, AMP_VPP)

    marker_frac = MARKER_NS * 1e-9 / T_BUF
    awg.marker_pulse(AWG_MARK_CH, segnum=1, n_wave_points=2 * N_TOTAL,
                     marker=MARKER, high_frac=marker_frac, level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")
    print(f"  AWG CH{AWG_MARK_CH} RF OFF, M{MARKER} playing "
          f"{MARKER_NS} ns per {T_BUF*1e6:.2f} us buffer")

    saved = scope.save_state()
    with scope._lock:
        v_was = {ch: scope._dev.query(f"CH{ch}:SCAle?").strip()
                 for ch in (SIG_CH, TRIG_CH)}
        pos_was = scope._dev.query("HORizontal:POSition?").strip()
    results, traces, avtrace = [], {}, {}
    try:
        rate = scope.set_max_sample_rate(CAPTURE_S)
        scope.set_horizontal_position(PRETRIG_PCT)
        # The marker swings about +-0.5 V (:MARK:VOLT:PTOP 1.0, OFFS 0), so
        # 0 V is a safe level to trigger on before anything is known about it.
        with scope._lock:
            scope._dev.write(f"CH{TRIG_CH}:SCAle 0.25")
        scope._pre.clear()
        m_ptp, m_mid = range_on_trigger(scope, TRIG_CH, 0.0)
        if m_ptp < 0.1:
            print(f"no marker on CH{TRIG_CH} ({m_ptp*1000:.0f} mVpp) -- abort")
            return 1
        s_ptp, _ = range_on_trigger(scope, SIG_CH, m_mid)
        print(f"marker {m_ptp*1000:.0f} mVpp (mid {m_mid:.3f} V), "
              f"pulse {s_ptp*1000:.0f} mVpp")
        print(f"\nscope {rate/1e9:.2f} GS/s, {CAPTURE_S*1e6:g} us record, "
              f"{PRETRIG_PCT}% pretrigger, trigger CH{TRIG_CH} @ {m_mid:.3f} V")

        # Locate the burst ONCE, from a reference capture, and reuse the same
        # windows for every shot and both states -- a window that moved
        # between shots would itself shift the phase reference.
        scope.setup(channel=SIG_CH, n_averages=1, trigger_channel=TRIG_CH,
                    trigger_level=m_mid, trigger_mode="NORMal")
        t_r, v_r = scope.get_waveform(SIG_CH, max_points=MAX_POINTS,
                                      fresh=True, absolute_time=True, width=2)
        p0, p1 = find_pulse(t_r, v_r)
        span = p1 - p0
        win_fit = (p0 + 0.15 * span, p1 - 0.15 * span)
        win_noise = (p1 + 0.4e-6, t_r[-1])
        print(f"burst {p0*1e9:+.0f} to {p1*1e9:+.0f} ns after the trigger "
              f"({span*1e6:.3f} us wide; marker-to-RF skew {p0*1e9:+.0f} ns)")
        print(f"  fit window  {win_fit[0]*1e9:+.0f} to {win_fit[1]*1e9:+.0f} ns")
        print(f"  quiet window {win_noise[0]*1e9:+.0f} to "
              f"{win_noise[1]*1e9:+.0f} ns ({(win_noise[1]-win_noise[0])*1e6:.2f} us)")
        if win_noise[1] - win_noise[0] < 0.5e-6:
            print("  !! quiet window shorter than 0.5 us -- noise figures weak")

        for label, f_nco in (("ON GRID  carrier (baseband off grid)", F_NCO_ON),
                             ("OFF GRID carrier (control, +25 kHz)", F_NCO_OFF)):
            res, tr, av = run_state(awg, scope, label, f_nco, n_shots, m_mid,
                                    span_s, win_fit, win_noise,
                                    keep_traces=True)
            results.append(res)
            traces[label] = tr
            avtrace[label] = av
    finally:
        scope.restore_state(saved)
        with scope._lock:
            for ch, sc in v_was.items():
                scope._dev.write(f"CH{ch}:SCAle {sc}")
            scope._dev.write(f"HORizontal:POSition {pos_was}")
        for ch in (AWG_SIG_CH, AWG_MARK_CH):
            awg.duc_cw_step(F_NCO_ON, channel=ch)

    on, off = results
    print("\n" + "=" * 72)
    print(f"  {'':<22}{'ON GRID':>12}{'OFF GRID':>12}")
    for key, name in (("coherence", "pulse coherence"),
                      ("amp_ratio", f"{N_AVG}-avg amplitude"),
                      ("noise_ratio", "noise reduction"),
                      ("snr_gain", "SNR gain")):
        print(f"  {name:<22}{on[key]:>12.3f}{off[key]:>12.3f}")
    print("=" * 72)

    if off["coherence"] > 0.5:
        print("\n  Control did not collapse -- nothing here is established.")
    elif on["coherence"] > 0.95:
        print(f"\n  A pulse averages coherently with the baseband "
              f"{F_BB/GRID_HZ:.2f} grid steps off the\n  grid: coherence "
              f"{on['coherence']:.3f}, amplitude {on['amp_ratio']:.3f} through "
              f"a {N_AVG}-shot average.\n  Only the CARRIER has to be on the "
              f"grid. A transient never joins up across the\n  buffer "
              f"boundary, so the baseband can place the line anywhere.")
    else:
        print(f"\n  On-grid pulse coherence is only {on['coherence']:.3f} "
              f"while the control works --\n  real, and worth chasing before "
              f"trusting an averaged FID.")

    nr = on["noise_ratio"]
    ideal = np.sqrt(N_AVG)
    if nr < 0.6 * ideal:
        print(f"\n  Noise fell only {nr:.1f}x against sqrt({N_AVG}) = "
              f"{ideal:.1f}. Part of what is in the quiet\n  stretch is "
              f"COHERENT -- carrier feedthrough or a DAC spur -- and no amount "
              f"of\n  averaging will remove it. Worth a spectrum of that "
              f"window before building the\n  FID acquisition.")
    else:
        print(f"\n  Noise fell {nr:.1f}x against an ideal sqrt({N_AVG}) = "
              f"{ideal:.1f}, so the quiet stretch is\n  essentially random "
              f"and averaging buys the full SNR there.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("marker_coherence")
    out.mkdir(exist_ok=True)
    (out / f"pulse_averaging_{stamp}.json").write_text(json.dumps(
        {"timestamp": stamp, "n_total": N_TOTAL, "t_buffer_s": T_BUF,
         "grid_hz": GRID_HZ, "f_baseband_hz": F_BB, "pulse_us": PULSE_US,
         "decay_tau_us": DECAY_TAU_US, "capture_s": CAPTURE_S,
         "fit_window_s": win_fit, "noise_window_s": win_noise,
         "states": results}, indent=2))
    try:
        _plot(out / f"pulse_averaging_{stamp}.png", results, traces, avtrace,
              win_fit, win_noise)
    except Exception as e:
        print(f"  (plot skipped: {e})")
    print(f"\nsaved to {out}/pulse_averaging_{stamp}.*")
    return 0


def _plot(path, results, traces, avtrace, win_fit, win_noise):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on = results[0]
    label = on["label"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    t_s, v_s = traces[label][0]
    t_a, v_a = avtrace[label]
    ax[0][0].plot(t_s * 1e6, v_s * 1e3, lw=0.5, color="0.6", label="single shot")
    ax[0][0].plot(t_a * 1e6, v_a * 1e3, lw=0.5, color="crimson",
                  label=f"{on['n_average']}-average")
    ax[0][0].axvspan(win_noise[0] * 1e6, win_noise[1] * 1e6, color="tab:blue",
                     alpha=0.12, label="quiet window")
    ax[0][0].set_xlabel("time from trigger [us]")
    ax[0][0].set_ylabel("mV")
    ax[0][0].legend(fontsize=8)
    ax[0][0].set_title("pulse, single vs averaged", fontsize=10)

    mid = 0.5 * (win_fit[0] + win_fit[1])
    for t, v in traces[label]:
        m = (t > mid) & (t < mid + 8e-9)
        ax[0][1].plot(t[m] * 1e9, v[m] * 1e3, lw=0.8, alpha=0.7)
    ax[0][1].set_xlabel("time from trigger [ns]")
    ax[0][1].set_ylabel("mV")
    ax[0][1].set_title(f"10 shots overlaid, coherence {on['coherence']:.3f}",
                       fontsize=10)

    for key, c in ((0, "0.6"), (1, "crimson")):
        pass
    m_s = (t_s >= win_noise[0]) & (t_s < win_noise[1])
    m_a = (t_a >= win_noise[0]) & (t_a < win_noise[1])
    ax[1][0].plot(t_s[m_s] * 1e6, v_s[m_s] * 1e3, lw=0.5, color="0.6",
                  label=f"single, {on['noise_single_v']*1e3:.3f} mV rms")
    ax[1][0].plot(t_a[m_a] * 1e6, v_a[m_a] * 1e3, lw=0.5, color="crimson",
                  label=f"averaged, {on['noise_avg_v']*1e3:.3f} mV rms")
    ax[1][0].set_xlabel("time from trigger [us]")
    ax[1][0].set_ylabel("mV")
    ax[1][0].legend(fontsize=8)
    ax[1][0].set_title(f"quiet window, noise down "
                       f"{on['noise_ratio']:.1f}x", fontsize=10)

    for st, c in zip(results, ("tab:green", "tab:red")):
        ph = np.array(st["phases"])
        ax[1][1].plot(np.cos(ph), np.sin(ph), "o", ms=5, alpha=0.7, color=c,
                      label=f"{st['label'].split()[0]} {st['coherence']:.3f}")
    ax[1][1].add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7"))
    ax[1][1].set_aspect("equal")
    ax[1][1].set_xlim(-1.2, 1.2)
    ax[1][1].set_ylim(-1.2, 1.2)
    ax[1][1].legend(fontsize=8)
    ax[1][1].set_title("pulse phase at trigger", fontsize=10)

    fig.suptitle(f"Marker-triggered pulse averaging, "
                 f"{on['f_abs_hz']/1e6:.4f} MHz")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
