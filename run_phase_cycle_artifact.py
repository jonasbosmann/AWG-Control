"""Phase cycling against DELIBERATELY INJECTED artifacts.

run_phase_cycling.py measures the artifacts that happen to be there (52 dB
down on a bare AWG-to-scope path, i.e. almost nothing). That proves the
cycling does no harm, but it cannot prove the cycling WORKS -- there was
nothing to reject.

So this injects two artifacts of known size and known coherence order into
the same buffer as the chirp, and measures how much each pattern removes.

Write a component's response to a chirp of phase phi as exp(i.m.phi). An
N-step cycle passes order m only when (m-1) is a multiple of N:

    m = +1   the FID itself                        kept by every pattern
    m =  0   breakthrough, ringing, DC, pickup     killed by 2-step and up
    m = -1   the IMAGE: I/Q imbalance, a folded     killed by 4-step,
             mirror sideband                        PASSED BY 2-STEP

That last row is the reason to run 4 steps rather than 2 once there is a
receiver: an image term adds coherently with the signal and does not look
like an artifact.

Both artifacts are put OUTSIDE the chirp's sweep (chirp 1.99-2.49 GHz,
artifacts at 2.64 and 1.84 GHz) so each can be measured by an exact complex
projection with no contribution from the chirp. They are built at the same
amplitude, differing only in how their phase tracks the cycle:

    m =  0   phase 0        in every cycle
    m = -1   phase -phi_k   in cycle k

Everything is referenced to the cycle start and the free-running NCO's
advance is removed, exactly as _build_channel does for the chirp, so the
phases actually emitted are the ones in that table.

Wiring: AWG CH1 M1 -> scope CH4 (trigger), AWG CH2 -> scope CH3 (signal).

Run:  python run_phase_cycle_artifact.py [n_cycles] [artifact_frac]
"""
import sys

import numpy as np

import phasecycle
from awg import AWG
from scope import Scope
from run_pulse_averaging import range_on_trigger

RATE_DAC, INTERP = 9e9, 8
RATE_CX = RATE_DAC / INTERP
F_LO, F_HI = 1.99e9, 2.49e9
F_CARRIER = 0.5 * (F_LO + F_HI)
BB_M0, BB_M1 = +400e6, -400e6          # artifact baseband offsets
CHIRP_US, DEAD_US, DETECT_US = 1.0, 0.1, 2.0
AMP_VPP = 0.5
EDGE_FRAC = 0.05
AWG_SIG_CH, AWG_MARK_CH, MARKER = 2, 1, 1
MARKER_NS = 100
SIG_CH, TRIG_CH = 3, 4
SCOPE_RATE = 50e9
N_AVG = 16

PATTERNS = [("no cycling", [0.0]),
            ("2-step", [0.0, 180.0]),
            ("4-step", [0.0, 90.0, 180.0, 270.0])]


def build(n_cycles, pattern_deg, frac):
    """K cycles of (phase-cycled chirp + two fixed-order artifacts).

    Returns (I, Q, meta). The production path cannot express this -- one
    channel carries one `kind` -- which is the point: the artifacts are a
    diagnostic fixture, not something the spectrometer should ever emit.
    """
    n_chirp = int(round(CHIRP_US * 1e-6 * RATE_CX))
    n_cycle = int(np.ceil(((CHIRP_US + DEAD_US + DETECT_US) * 1e-6 * RATE_CX)
                          / 32)) * 32
    n_total = n_cycle * n_cycles
    t_cycle = n_cycle / RATE_CX

    carrier = AWG.snap_duc_carrier(F_CARRIER, n_total, RATE_DAC, INTERP)[0]
    bb_lo, bb_hi = F_LO - carrier, F_HI - carrier
    phases = [np.radians(pattern_deg[k % len(pattern_deg)])
              for k in range(n_cycles)]

    t = np.arange(n_chirp) / RATE_CX
    T = t[-1] if n_chirp > 1 else 1e-9
    chirp_ph = 2 * np.pi * (bb_lo * t + (bb_hi - bb_lo) / (2 * T) * t ** 2)
    env = AWG._gauss_win(n_chirp, EDGE_FRAC)
    tone0 = 2 * np.pi * BB_M0 * t
    tone1 = 2 * np.pi * BB_M1 * t

    i_arr, q_arr = np.zeros(n_total), np.zeros(n_total)
    for k in range(n_cycles):
        # the NCO free-runs across the buffer; remove its advance so the
        # emitted phases are the programmed ones (reduced to a fraction of
        # a turn first -- it reaches ~1e5 turns)
        nco = 2 * np.pi * ((carrier * (k * t_cycle)) % 1.0)
        phi = phases[k]
        z = env * (np.exp(1j * (chirp_ph + phi - nco))            # m = +1
                   + frac * np.exp(1j * (tone0 + 0.0 - nco))      # m =  0
                   + frac * np.exp(1j * (tone1 - phi - nco)))     # m = -1
        s = k * n_cycle
        i_arr[s:s + n_chirp] = z.real
        q_arr[s:s + n_chirp] = z.imag

    peak = max(np.abs(i_arr + 1j * q_arr).max(), 1e-12)
    return i_arr / peak, q_arr / peak, {
        "n_total": n_total, "n_cycle": n_cycle, "t_cycle": t_cycle,
        "carrier": carrier, "phases": phases, "n_chirp": n_chirp}


def project(z, f_bb, dt):
    """Complex amplitude of a tone at baseband offset f_bb in an analytic
    segment. Exact and narrowband, so the chirp (which never visits this
    frequency) contributes essentially nothing."""
    n = z.shape[-1]
    ref = np.exp(-1j * 2 * np.pi * f_bb * np.arange(n) * dt)
    return np.abs(np.mean(z * ref, axis=-1)) * 2.0


def band_limit(z, dt, f_lo, f_hi):
    """Keep only f_lo..f_hi of an analytic segment, by zeroing FFT bins."""
    n = z.shape[-1]
    f = np.fft.fftfreq(n, dt)
    keep = (f >= f_lo) & (f <= f_hi)
    return np.fft.ifft(np.fft.fft(z, axis=-1) * keep, axis=-1)


def run_one(awg, scope, label, pattern_deg, n_cycles, frac, trig_lev):
    i_arr, q_arr, m = build(n_cycles, pattern_deg, frac)
    awg.duc_iq_setup_dual(i_arr, q_arr, m["carrier"], AMP_VPP)
    awg.marker_pulse(AWG_MARK_CH, segnum=AWG_MARK_CH,
                     n_wave_points=2 * m["n_total"], marker=MARKER,
                     high_frac=MARKER_NS * 1e-9 / (m["n_total"] / RATE_CX),
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")

    npts = int(round(m["n_total"] / RATE_CX * SCOPE_RATE))
    with scope._lock:
        scope._dev.write("HORizontal:MODE MANual")
        scope._dev.write(f"HORizontal:MODE:SAMPLERate {SCOPE_RATE:.0f}")
        scope._dev.write(f"HORizontal:MODE:RECOrdlength {npts}")
    scope._pre.clear()
    scope.set_horizontal_position(2)
    scope.setup(channel=SIG_CH, n_averages=N_AVG, trigger_channel=TRIG_CH,
                trigger_level=trig_lev, trigger_mode="NORMal")
    t, v = scope.get_waveform(SIG_CH, max_points=npts, fresh=True, width=2,
                              absolute_time=True, acq_timeout=30.0)

    cut = phasecycle.split_cycles(t, v, n_cycles, m["t_cycle"],
                                  CHIRP_US * 1e-6, carrier_hz=m["carrier"])
    res = phasecycle.combine(cut["analytic"], m["phases"])
    dt = cut["dt"]

    # each artifact's amplitude, per cycle and after combining. The scope
    # sees real RF, so a baseband offset f_bb appears at carrier + f_bb; the
    # analytic segments are still at RF, so project at the absolute
    # frequency, taken relative to the cycle's own start.
    # res["gain"] divides by a SINGLE cycle that still contains both
    # artifacts, so it drops as more of them are cancelled out of the
    # numerator -- a property of the denominator, not a loss of chirp. Redo
    # it on segments masked to the chirp band, where the artifacts are not.
    band = band_limit(cut["analytic"], dt, F_LO - 50e6, F_HI + 50e6)
    signs = np.exp(-1j * np.array(m["phases"]))
    gain_chirp = float(np.abs((band * signs[:, None]).sum(0)).mean()
                       / np.abs(band).mean())

    out = {"label": label, "n_steps": len(pattern_deg), "gain": res["gain"],
           "gain_chirp": gain_chirp}
    for name, bb in (("m=0", BB_M0), ("m=-1", BB_M1)):
        single = float(np.mean(project(cut["analytic"], m["carrier"] + bb, dt)))
        comb = float(project(res["cycled_w"], m["carrier"] + bb, dt))
        ideal = n_cycles * single
        out[name] = {"single_mv": single * 1e3, "combined_mv": comb * 1e3,
                     "frac": comb / ideal if ideal else float("nan")}
    return out


def main():
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    print(f"\nchirp {F_LO/1e9:.2f}-{F_HI/1e9:.2f} GHz, phase-cycled (m=+1)")
    print(f"injected artifacts at {frac:.0%} of the chirp:")
    print(f"   m= 0  {(F_CARRIER+BB_M0)/1e9:.2f} GHz, phase 0 every cycle")
    print(f"   m=-1  {(F_CARRIER+BB_M1)/1e9:.2f} GHz, phase -phi_k "
          f"(an image)")
    print(f"{n_cycles} cycles per buffer, {N_AVG}-average, "
          f"{SCOPE_RATE/1e9:g} GS/s\n")

    awg, scope = AWG(), Scope()
    scope.require_external_reference()
    awg.set_reference_external(10)

    # range once, on the first pattern, and keep it fixed for all three
    i0, q0, m0 = build(n_cycles, [0.0], frac)
    awg.duc_iq_setup_dual(i0, q0, m0["carrier"], AMP_VPP)
    awg.marker_pulse(AWG_MARK_CH, segnum=AWG_MARK_CH,
                     n_wave_points=2 * m0["n_total"], marker=MARKER,
                     high_frac=MARKER_NS * 1e-9 / (m0["n_total"] / RATE_CX),
                     level_vpp=1.0)
    with awg._lock:
        awg._dev.write(f":INST:CHAN {AWG_MARK_CH}")
        awg._dev.write(":OUTP OFF")
    with scope._lock:
        scope._dev.write(f"CH{TRIG_CH}:SCAle 0.25")
    scope._pre.clear()
    scope.set_horizontal_position(2)
    m_ptp, m_mid = range_on_trigger(scope, TRIG_CH, 0.0)
    if m_ptp < 0.1:
        print(f"no marker on CH{TRIG_CH} -- abort")
        return 1
    range_on_trigger(scope, SIG_CH, m_mid)

    rows = []
    for label, pat in PATTERNS:
        if len(pat) > 1 and n_cycles % len(pat):
            print(f"  skipping {label}: {n_cycles} is not a whole number of "
                  f"{len(pat)}-step patterns")
            continue
        rows.append(run_one(awg, scope, label, pat, n_cycles, frac, m_mid))

    print("\n" + "=" * 72)
    print("  fraction of ideal coherent addition that survives the sum")
    print("  (1.00 = passed untouched,  0.00 = cancelled)")
    print("=" * 72)
    print(f"  {'pattern':<12}{'chirp m=+1':>13}{'m=0 artifact':>16}"
          f"{'m=-1 image':>14}{'raw gain':>15}")
    print(f"  {'':<12}{'(band-limited)':>13}{'':>16}{'':>14}"
          f"{'(all three)':>15}")
    for r in rows:
        print(f"  {r['label']:<12}{r['gain_chirp']:>12.2f}x"
              f"{r['m=0']['frac']:>16.3f}{r['m=-1']['frac']:>14.3f}"
              f"{r['gain']:>14.2f}x")
    print("=" * 72)
    for r in rows:
        print(f"  {r['label']:<12} m=0  {r['m=0']['single_mv']:7.2f} mV/cycle"
              f" -> {r['m=0']['combined_mv']:7.2f} mV combined"
              f"   ({20*np.log10(max(r['m=0']['frac'],1e-6)):+6.1f} dB)")
        print(f"  {'':<12} m=-1 {r['m=-1']['single_mv']:7.2f} mV/cycle"
              f" -> {r['m=-1']['combined_mv']:7.2f} mV combined"
              f"   ({20*np.log10(max(r['m=-1']['frac'],1e-6)):+6.1f} dB)")
    print()
    print("  Expected: no cycling passes everything; 2-step kills m=0 and")
    print("  PASSES the m=-1 image; 4-step kills both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
