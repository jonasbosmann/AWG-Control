"""Measure ONE newly inserted element by difference against the measurement
taken just before it went in.

This is the tool for building the chain up part by part. Take a reference
sweep, insert the next component, sweep again with EVERYTHING else identical,
and subtract:

    inserted element = new sweep - reference sweep

The AWG amplitude, the frequency list and the scope's vertical scale are all
taken FROM THE REFERENCE rather than recomputed, so none of them can drift
between the two runs. Every pad, cable, connector, amplifier and scope
calibration error upstream appears in both and cancels exactly -- the result
depends on none of their absolute values.

WHY MEASURE IN PLACE RATHER THAN STANDALONE
-------------------------------------------
A part measured on its own, driven by a synthesiser into a 50 ohm load, is not
necessarily the part as it behaves in the chain: the source impedance it sees
is different, and past a doubler so is the spectral content of its input. This
keeps every measurement in the context the part actually runs in.

LIMIT: SINGLE TONE ONLY
-----------------------
The scope reading is an AC-RMS calculation, so it sums EVERYTHING present. That
is a valid level measurement while the signal is one tone -- i.e. anywhere
before the first doubler. Past a doubler the reading is F1 + F2 + F3 together
and cannot be read as the wanted signal's level. Use the EXA there instead;
this script refuses to run if the build point is marked as post-doubler.

Run:  python x8_chain/run_insert.py "<what was inserted>" [stage]
      e.g. python x8_chain/run_insert.py "PE87FL1012 1-2 GHz BPF"
      NO_SHOW=1 to skip the plot window.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import chainlog
from awg import AWG
from scope import Scope
from run_awg_level_scope import phase_coverage_warning, crest_warning


SIG_CH = 1
SETTLE_S = 0.20
CAPTURE_S = 2e-6
N_REPEATS = 2

MAX_FILL_FRAC = 0.90     # above this the RMS is distorted by clipping
MIN_FILL_FRAC = 0.10     # below this quantisation starts to matter


def latest_reference(stage_key):
    """Most recent scope sweep for this stage that can serve as a reference."""
    for p in chainlog.find(stage_key, thru=False):
        d = chainlog.load(p)
        s = d.get("instrument_settings", {})
        if s.get("fixed_vscale_v_per_div") and s.get("operating_vpp"):
            return p, d
    return None, None


def measure(scope, fs, freq_hz, repeats=N_REPEATS):
    vals, notes = [], []
    for _ in range(repeats):
        vpp, _t, v = scope.measure_vpp(channel=SIG_CH, settle=0.05)
        vals.append(config.dbm_from_vpp(vpp) if vpp > 0 else np.nan)
        w = crest_warning(v)
        if w and w not in notes:
            notes.append(w)
    w = phase_coverage_warning(float(freq_hz), fs)
    if w:
        notes.append(w)
    return float(np.mean(vals)), float(np.std(vals)), notes


def run(element, stage_key="A"):
    st = config.STAGES[stage_key]
    pad = config.PAD_AT_INSTRUMENT_DB

    ref_path, ref = latest_reference(stage_key)
    if ref is None:
        print(f"No usable reference sweep for stage {stage_key}.")
        print("Run `python x8_chain/run_operating_point.py` first -- an insertion")
        print("is measured by difference, so it needs something to differ from.")
        return

    freqs = ref["freqs_hz"]
    rs = ref["instrument_settings"]
    vpp = float(rs["operating_vpp"])
    vscale = float(rs["fixed_vscale_v_per_div"])
    rkey = [k for k in ref if k.startswith("dbm_")][0]
    ref_lv = ref[rkey]

    print("=" * 74)
    print(f"INSERT -- stage {stage_key}: {element}")
    print("=" * 74)
    print(f"  reference: {ref_path}")
    print(f"             {ref.get('label','?')}, taken {ref.get('timestamp','?')}")
    print(f"  reusing the reference's AWG {vpp:.4f} Vpp, "
          f"{vscale*1e3:.4g} mV/div, and its {len(freqs)} frequencies")
    print(f"  ** the ONLY thing that may have changed since that sweep is the "
          f"insertion of:")
    print(f"     {element}")
    print(f"     If anything else moved -- a cable remade, a pad swapped, the "
          f"AWG touched -- stop now; the subtraction will attribute it to the "
          f"element. **\n")

    awg = AWG()
    scope = Scope()
    try:
        scope.setup(channel=SIG_CH, n_averages=1,
                    trigger_channel=SIG_CH, trigger_level=0.0)
        scope.set_active_channels([SIG_CH])
        fs = scope.set_max_sample_rate(CAPTURE_S)
        scope.set_vertical(SIG_CH, vscale)
        print(f"scope: {fs/1e9:.2f} GS/s, vertical pinned at "
              f"{vscale*1e3:.4g} mV/div (the reference's)")

        awg.duc_cw_setup(float(freqs[0]), amplitude_vpp=vpp, channel=1)
        time.sleep(SETTLE_S)
        m, s, notes = measure(scope, fs, float(freqs[0]))
        d0 = m - ref_lv[0]
        fill = config.vpp_from_dbm(m) / (vscale * 10)
        print(f"\n  FIRST POINT  {freqs[0]/1e9:.3f} GHz: {m:+.2f} dBm, "
              f"{d0:+.2f} dB vs the reference, {fill*100:.0f}% of screen"
              + ("   " + "; ".join(notes) if notes else ""))

        if not np.isfinite(m):
            print("\n  ** no signal -- check the insertion and the bias. "
                  "Nothing recorded. **")
            return
        if fill > MAX_FILL_FRAC:
            print(f"\n  ** {fill*100:.0f}% of screen -- the level ROSE enough to "
                  f"risk clipping at the reference's vertical scale, so the RMS "
                  f"reading is no longer trustworthy.")
            print("     A difference measurement needs both sweeps on the same "
                  "scale, so this cannot be fixed by rescaling. Take a fresh")
            print("     run_operating_point.py sweep as the new reference "
                  "instead. Nothing recorded. **")
            return
        if fill < MIN_FILL_FRAC:
            print(f"  ** only {fill*100:.0f}% of screen -- the element lost more "
                  f"than expected. Continuing, but treat the numbers as noisy; "
                  f"a fresh reference at a suitable scale would be better. **")
        if m - pad * 0 > config.scope_max_dbm() - 3.0:
            print("\n  ** too close to the scope's input limit -- ABORT **")
            return
        print("     proceeding\n")

        lv, sd, flagged = [m], [s], []
        for f in freqs[1:]:
            awg.duc_cw_step(float(f), channel=1)
            time.sleep(SETTLE_S)
            mi, si, ni = measure(scope, fs, float(f))
            lv.append(mi); sd.append(si)
            if ni:
                flagged.append((f, ni))
        lv, sd = np.array(lv), np.array(sd)

        settings = {"sample_rate_hz": float(fs), "capture_s": CAPTURE_S,
                    "n_repeats": N_REPEATS, "settle_s": SETTLE_S,
                    "operating_vpp": vpp,
                    "fixed_vscale_v_per_div": vscale,
                    "inserted_element": element,
                    "reference_file": ref_path,
                    "reference_label": ref.get("label", "")}
    finally:
        for name, fn in (("awg.stop", awg.stop),
                         ("scope.restore", lambda: scope.restore(channel=SIG_CH))):
            try:
                fn()
            except Exception as e:
                print(f"cleanup ({name}): {e}")

    report(stage_key, element, freqs, lv, sd, ref_lv, vpp, settings, flagged)
    return lv


def report(stage_key, element, freqs, lv, sd, ref_lv, vpp, settings, flagged):
    st = config.STAGES[stage_key]
    pad = config.PAD_AT_INSTRUMENT_DB
    lo, hi = st["band_hz"]
    ib = (freqs >= lo) & (freqs <= hi)
    delta = lv - ref_lv
    d = delta[ib]

    print("\n" + "=" * 74)
    print(f"RESULT -- {element}")
    print("=" * 74)
    sign = "insertion loss" if np.mean(d) < 0 else "gain"
    print(f"  over {lo/1e9:.3f}-{hi/1e9:.3f} GHz:")
    print(f"    mean {sign:<15} {abs(np.mean(d)):.2f} dB")
    print(f"    range                 {-d.max():.2f} to {-d.min():.2f} dB"
          if np.mean(d) < 0 else
          f"    range                 {d.min():.2f} to {d.max():.2f} dB")
    print(f"    RIPPLE                {d.max()-d.min():.2f} dB peak-to-peak")
    print(f"    slope across the band {d[0]-d[-1]:+.2f} dB")

    print(f"\n  new level {(lv[ib]+pad).max():+.2f} to {(lv[ib]+pad).min():+.2f} dBm "
          f"at the output, flatness {(lv[ib]).max()-(lv[ib]).min():.2f} dB")
    print(f"  (was {(ref_lv[ib]+pad).max():+.2f} to {(ref_lv[ib]+pad).min():+.2f} dBm, "
          f"flatness {(ref_lv[ib]).max()-(ref_lv[ib]).min():.2f} dB)")

    print(f"\n  repeat sd {np.median(sd):.3f} dB median -- acquisition noise only, "
          f"NOT the uncertainty.")
    print(f"  Run-to-run scatter on this bench is ~0.19 dB rms, and a difference "
          f"of two sweeps")
    print(f"  carries it twice, so treat anything under ~0.3 dB as not resolved.")
    if flagged:
        print(f"\n  {len(flagged)} point(s) flagged:")
        for f, ni in flagged[:8]:
            print(f"    {f/1e9:.3f} GHz: {'; '.join(ni)}")

    sess = chainlog.new_session(stage_key, "insert")
    p = chainlog.save(sess, "insert", stage_key, freqs,
                      {f"{vpp:.4f}vpp": lv}, settings, x_name="freqs_hz",
                      thru=False,
                      notes=f"after inserting {element}; RAW scope readings, "
                            f"add {pad:g} dB. Reference: "
                            f"{settings['reference_file']}")
    print(f"\n  saved {p}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(freqs / 1e9, ref_lv + pad, "o-", ms=3, label="before")
    axes[0].plot(freqs / 1e9, lv + pad, "o-", ms=3, label="after")
    axes[0].axvspan(lo / 1e9, hi / 1e9, alpha=0.12, color="tab:green")
    axes[0].set_xlabel("frequency (GHz)"); axes[0].set_ylabel("level (dBm)")
    axes[0].set_title("before / after insertion")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].plot(freqs / 1e9, delta, "o-", ms=4, color="tab:red")
    axes[1].axvspan(lo / 1e9, hi / 1e9, alpha=0.12, color="tab:green",
                    label="chain band")
    axes[1].axhline(np.mean(d), ls="--", lw=1, color="k", alpha=0.6,
                    label=f"in-band mean {np.mean(d):+.2f} dB")
    axes[1].set_xlabel("frequency (GHz)"); axes[1].set_ylabel("difference (dB)")
    axes[1].set_title(element)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(sess, "insert.png")
    fig.savefig(png, dpi=130)
    print(f"  saved {png}")
    if os.environ.get("NO_SHOW") != "1":
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "A")
