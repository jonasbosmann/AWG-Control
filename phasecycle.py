"""Split a K-cycle capture back into its cycles and recombine them.

The transmit half of this lives in `AWG.send_waveforms()`: with
`timing["n_cycles"] = K` and a per-channel `phases` list it writes K
chirp/dead/detect cycles into one buffer, each chirp carrying a programmed
phase, with the free-running DUC NCO's advance removed so the phase actually
emitted is the phase asked for.

This is the receive half. One marker fires per buffer, so the scope returns
ONE hardware-averaged record holding all K cycles; this cuts it back into K
and combines them two ways:

    conjugate-phase sum   sum_k exp(-i.phi_k) z_k    -> the signal
    straight sum          sum_k z_k                  -> the artifacts

Anything that follows the chirp phase (the molecular FID, and with no sample
present the AWG's own chirp) adds in the first. Anything that ignores it --
chirp breakthrough into the receiver, detector ringing, DC offset, digital
pickup -- adds in the second and cancels in the first. Computing both means
the artifact level is MEASURED rather than assumed.

Two traps, both of which cost a measurement on 2026-10-20 (LABBOOK 21):

  CUT AT EXACT TIMES, NOT ROUNDED SAMPLES. One cycle is rarely a whole
    number of scope samples. Stepping by round(t_cycle/dt) left 0.44 samples
    of slip per cycle = 35 ps = 28 deg of phase at 2.24 GHz, and showed up
    as a dead-straight phase ramp that dragged a 4-cycle sum from 4.00x to
    3.40x. split_cycles() interpolates onto the exact grid instead.

  SAMPLE FAST ENOUGH TO INTERPOLATE. np.interp is linear, and at 5.6
    samples per carrier cycle (2.24 GHz at 12.5 GS/s) its own error is
    percent-level -- enough to fake a -23.9 dB artifact floor where the
    truth was -52 dB. Pass carrier_hz and this warns below ~20 samples per
    cycle.
"""
import numpy as np
from scipy import signal as sig

MIN_SAMPLES_PER_CARRIER_CYCLE = 20.0


def find_first(t, v, smooth=151, thresh=0.5):
    """Index of the first sample where the analytic envelope crosses `thresh`
    of its maximum. Returns None if nothing stands out of the noise."""
    env = np.abs(sig.hilbert(v - v.mean()))
    k = max(int(smooth) | 1, 3)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    idx = np.flatnonzero(env > thresh * env.max())
    return int(idx[0]) if idx.size else None


def split_cycles(t, v, n_cycles, t_cycle, win_s, t_first=None,
                 carrier_hz=None, warn=print):
    """Cut a K-cycle record into K segments of `win_s` seconds each.

    t, v        one captured record (absolute_time=True, so t is referenced
                to the interpolated trigger)
    n_cycles    K, from plan["n_cycles"]
    t_cycle     plan["t_cycle"] -- the EXACT cycle period, not a rounded one
    win_s       how much of each cycle to analyse, from its start
    t_first     where cycle 0 begins; found from the envelope if None
    carrier_hz  only used to check the sample rate is high enough

    Returns a dict: raw (K x n, mean-removed real), analytic (K x n complex),
    t0, dt, n_win.  Raises ValueError if the cycles do not fit the record.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    dt = float(t[1] - t[0])
    n_win = int(round(win_s / dt))
    if n_win < 16:
        raise ValueError(f"analysis window {win_s*1e9:.1f} ns is only "
                         f"{n_win} samples at {1/dt/1e9:.1f} GS/s")

    if carrier_hz and warn:
        per_cycle = 1.0 / (carrier_hz * dt)
        if per_cycle < MIN_SAMPLES_PER_CARRIER_CYCLE:
            warn(f"  WARNING {per_cycle:.1f} samples per carrier cycle at "
                 f"{carrier_hz/1e9:.3f} GHz and {1/dt/1e9:.1f} GS/s. Linear "
                 f"interpolation needs >= {MIN_SAMPLES_PER_CARRIER_CYCLE:.0f}; "
                 f"below that its own error shows up as a phase and amplitude "
                 f"spread that looks like a real artifact floor.")

    if t_first is None:
        first = find_first(t, v)
        if first is None:
            raise ValueError("no burst found in the record")
        t_first = float(t[first])

    raw, analytic = [], []
    for k in range(n_cycles):
        grid = t_first + k * t_cycle + np.arange(n_win) * dt
        if grid[-1] > t[-1]:
            raise ValueError(
                f"cycle {k} ends at {grid[-1]*1e6:.3f} us but the record "
                f"stops at {t[-1]*1e6:.3f} us -- lengthen the record or "
                f"shorten the detect window")
        seg = np.interp(grid, t, v)
        seg = seg - seg.mean()
        raw.append(seg)
        analytic.append(sig.hilbert(seg))
    return {"raw": np.array(raw), "analytic": np.array(analytic),
            "t0": float(t_first), "dt": dt, "n_win": n_win}


def combine(segs, phases):
    """Conjugate-phase sum and straight sum of K analytic segments.

    `phases` are the phases actually EMITTED (radians), which for a lower
    sideband is the negative of what was programmed -- see
    _build_channel's info["phases_emitted_rad"].

    Returns a dict of waveforms and scalar metrics. The scalars are the ones
    worth quoting: `gain` should equal K, `artifact_db` is how far below the
    combined signal everything that ignored the chirp phase ended up.
    """
    segs = np.asarray(segs)
    n = segs.shape[0]
    phases = [float(phases[k % len(phases)]) for k in range(n)]
    signs = np.exp(-1j * np.array(phases))

    cycled_w = (segs * signs[:, None]).sum(0)
    straight_w = segs.sum(0)
    cycled = float(np.abs(cycled_w).mean())
    straight = float(np.abs(straight_w).mean())
    single = float(np.abs(segs).mean())

    ref = segs[0]
    meas_deg, err_deg, corr, amp_mv = [], [], [], []
    for k in range(n):
        rel = float(np.degrees(np.angle(np.vdot(ref, segs[k]))))
        want = float(np.degrees(phases[k] - phases[0]))
        meas_deg.append(rel)
        err_deg.append((rel - want + 180.0) % 360.0 - 180.0)
        a, b = np.real(ref), np.real(segs[k])
        corr.append(float(np.dot(a, b) /
                          np.sqrt(np.dot(a, a) * np.dot(b, b))))
        amp_mv.append(float(np.abs(segs[k]).mean() * 1e3))

    return {
        "cycled_w": cycled_w, "straight_w": straight_w,
        "cycled": cycled, "straight": straight, "single": single,
        "gain": cycled / single if single else float("nan"),
        "ideal_gain": float(n),
        "artifact_db": 20 * np.log10(max(straight / cycled, 1e-12))
                       if cycled else float("nan"),
        "phases_rad": phases,
        "programmed_deg": [float(np.degrees(p - phases[0])) for p in phases],
        "measured_deg": meas_deg, "error_deg": err_deg,
        "correlation": corr, "amplitude_mv": amp_mv,
    }


def summary_lines(res):
    """The same text for the CLI and the GUI log, so they cannot disagree."""
    out = [f"  {'cycle':>6}{'programmed':>12}{'measured':>11}{'error':>9}"
           f"{'amplitude':>11}{'corr':>8}"]
    for k in range(len(res["phases_rad"])):
        out.append(f"  {k:>6}{res['programmed_deg'][k]:>11.1f}d"
                   f"{res['measured_deg'][k]:>10.1f}d"
                   f"{res['error_deg'][k]:>8.1f}d"
                   f"{res['amplitude_mv'][k]:>10.2f}mV"
                   f"{res['correlation'][k]:>+8.3f}")
    out.append("")
    out.append(f"  phase-cycled sum : {res['cycled']*1e3:8.2f} mV   "
               f"({res['gain']:.2f}x a single cycle, "
               f"ideal {res['ideal_gain']:.2f})")
    out.append(f"  straight sum     : {res['straight']*1e3:8.2f} mV   "
               f"({res['straight']/res['cycled']*100:.2f}% of it)"
               if res["cycled"] else "  straight sum     :   n/a")
    out.append(f"  -> everything not following the chirp phase is "
               f"{res['artifact_db']:.1f} dB down")
    return out
