"""Build the phase-cycling technical note as a PDF.

Uses the shared layout in notestyle.py. Numbers in the selection table are
COMPUTED here rather than quoted; the bench column is measured data from
run_phase_cycle_artifact.py on 2026-10-26.

Set NOTE_PNG_DIR to also dump each page as a PNG for proofing.
"""
import datetime as dt

import numpy as np

from notestyle import (Note, INK, INK2, INK3, RULE, ACCENT, SERIES, SANS)

OUT = (r"C:\Users\Admin\Documents\Claude Programming"
       r"\phase_cycling_2026-10-26.pdf")


# ── the selection rule, computed ────────────────────────────────────
def gain(phases, m):
    """What fraction of a coherence-order-m signal survives the sum."""
    phases = np.asarray(phases, dtype=float)
    return abs(np.sum(np.exp(1j * m * phases) * np.exp(-1j * phases))) \
        / len(phases)


def echo_gain(phases):
    """Two-pulse (Hahn) echo from consecutive chirps: phase 2.phi_k - phi_k-1."""
    phases = np.asarray(phases, dtype=float)
    prev = np.roll(phases, 1)
    return abs(np.sum(np.exp(1j * (2 * phases - prev)) * np.exp(-1j * phases))) \
        / len(phases)


PATTERNS = [("no cycling", np.zeros(4)),
            ("2-step", np.radians([0, 180] * 2)),
            ("4-step", np.radians([0, 90, 180, 270])),
            ("6-step", np.radians(np.arange(6) * 60.0))]
ORDERS = [-1, 0, 1, 2, 3]
ORDER_MEANING = {
    -1: "image: I/Q imbalance, folded mirror sideband",
    0:  "chirp breakthrough, detector ringing, DC offset, digital pickup",
    1:  "THE FID -- the molecular signal",
    2:  "second-order response",
    3:  "third-order response",
    -2: "second-order image",
}

# measured 2026-10-26, run_phase_cycle_artifact.py, 4 cycles, 50 GS/s,
# 16-average, artifacts injected at 30% of the chirp
BENCH = [("no cycling", "4.00x", "-0.0 dB", "-0.0 dB"),
         ("2-step",     "4.00x", "-51.8 dB", "-0.0 dB"),
         ("4-step",     "4.00x", "-48.3 dB", "-52.7 dB")]


def fig_selection(ax):
    w = 0.26
    x = np.arange(len(ORDERS))
    for i, (name, ph) in enumerate([p for p in PATTERNS if p[0] != "no cycling"]):
        g = [gain(ph, m) for m in ORDERS]
        ax.bar(x + (i - 1) * w, g, w, color=SERIES[i], label=name,
               edgecolor="none")
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"m = {m:+d}" if m else "m = 0" for m in ORDERS],
                       fontsize=8.5)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("fraction surviving the sum", fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    # legend sits in the band above the bars (which top out at 1.0) so it
    # cannot sit on top of one
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, 1.27))
    ax.annotate("the image survives\nthe 2-step", xy=(-w, 0.92),
                xytext=(0.95, 0.62), fontsize=7.8, color=SERIES[0],
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=0.9,
                                shrinkA=2, shrinkB=2))
    ax.annotate("the FID", xy=(2, 1.0), xytext=(2, 1.06),
                fontsize=8, color=INK, ha="center")


def main():
    n = Note(OUT, footer="Phase cycling \u2014 CP-FTMW bench note")

    n.h1("Phase cycling in CP-FTMW")
    n.para("What it removes, what it does not, and how to choose the pattern.",
           size=10.8, color=INK2, space_after=0.10)
    n.para(f"Jonas Bosmann \u00b7 {dt.date(2026, 10, 26):%d %B %Y} \u00b7 "
           f"310\u2013370 GHz spectrometer, Tabor P9482D + Tektronix MSO64B",
           size=8.6, color=INK3, space_after=0.05)
    n.rule()

    n.h2("Why there are several chirps in one buffer", 1)
    n.para(
        "A gas pulse lasts around a millisecond but a chirp/FID cycle only a "
        "few microseconds, so many cycles fit inside one pulse. The scope, "
        "not the AWG, sets the throughput \u2014 about 32 acquisitions per "
        "second whatever you do, some 21 ms of re-arm each. Putting K cycles "
        "in one buffer and firing ONE marker means that re-arm is paid once "
        "per gas pulse instead of once per FID.")
    n.para(
        "Once the cycles are there anyway, giving each one a different chirp "
        "phase costs nothing and buys a filter. That is phase cycling.")

    n.h2("How it works", 2)
    n.para(
        "Cycle k transmits its chirp with a programmed phase phi_k. Anything "
        "that responds to the chirp carries that phase back. The K sub-FIDs "
        "are then recombined after de-rotating each one by the phase it was "
        "sent with:")
    n.eq(r"$S \;=\; \frac{1}{K}\,\sum_{k=0}^{K-1} e^{-i\varphi_k}\, z_k$")
    n.para(
        "The molecular signal picks up exactly phi_k, so every term lands at "
        "zero phase and the K add coherently. Anything that ignores the "
        "chirp has the same phase in every cycle, so de-rotating spreads it "
        "around the circle instead and it sums to")
    n.eq(r"$\sum_k e^{-i\varphi_k}$")
    n.para(
        "which is the whole condition. The pattern works if and only if "
        "those phasors close. 0/180 closes. 0/90/180/270 closes. 0/120/240 "
        "closes. What does NOT close is a partial repetition \u2014 K = 4 "
        "running a 3-step pattern gives 0, 120, 240, 0, whose phasors sum to "
        "1, not 0, and nothing cancels at all.")
    n.callout("K must be a whole number of repetitions of the pattern. "
              "Not approximately \u2014 a partial cycle gives no rejection "
              "whatsoever, not reduced rejection.")

    n.h2("Which signals it filters out", 3)
    n.para(
        "Write a component's response to a chirp of phase phi as "
        "exp(i\u00b7m\u00b7phi). The integer m is its coherence order: the "
        "FID is m = +1, and anything indifferent to the chirp is m = 0. "
        "Substituting into the sum above, an N-step cycle passes order m only "
        "when m \u2212 1 is a multiple of N.")
    n.chart(2.20, fig_selection, pad_bottom=0.50)

    rows = ["  order   meaning                                 " +
            "".join(f"{p[0]:>9}" for p in PATTERNS[1:])]
    for m in (1, 0, -1, 2, 3):
        rows.append(f"  m = {m:+d}   {ORDER_MEANING[m][:36]:<36}   " +
                    "".join(f"{gain(p[1], m):>9.2f}" for p in PATTERNS[1:]))
    n.mono(rows, size=7.6)

    n.para(
        "Both patterns kill m = 0, which is the large one in practice: chirp "
        "breakthrough into the receiver, detector ringing, DC offset and "
        "digital pickup all sit there. Two steps are enough for that.")
    n.para(
        "The difference that matters is m = \u22121, the image \u2014 a "
        "receiver I/Q imbalance, or a mirror sideband folding into the IF. "
        "With an even number of steps m = \u22121 satisfies m \u2212 1 = "
        "\u22122, a multiple of 2, so the 2-step passes it at full strength. "
        "It adds coherently with the signal and does not look like an "
        "artifact. The 4-step rejects it because \u22122 is not a multiple "
        "of 4.")
    n.callout("Use the 2-step on a bare bench, where only breakthrough and "
              "ringing exist. Use the 4-step once there is a receiver, "
              "because that is when an image term becomes possible.")

    n.h2("Measured, not just derived", 4)
    n.para(
        "Measuring the artifacts that happen to be present proves only that "
        "the cycling does no harm \u2014 on a bare AWG-to-scope path they are "
        "already 52 dB down. So two artifacts of known size and known order "
        "were injected into the same buffer as the chirp, placed outside its "
        "sweep so each could be measured by an exact complex projection: a "
        "constant-phase tone at 2.64 GHz (m = 0) and a tone carrying "
        "\u2212phi_k at 1.84 GHz (m = \u22121), both at 30% of the chirp.")
    n.mono(
        ["  pattern        chirp m=+1     m=0 artifact     m=-1 image"] +
        [f"  {a:<14}{b:>10}{c:>17}{d:>15}" for a, b, c, d in BENCH],
        size=8.0)
    n.para(
        "4 cycles, 50 GS/s, 16 hardware averages, 2026-10-26. The chirp adds "
        "4.00x in every case. The 2-step row is the one to read: its "
        "m = \u22121 column is identical to no cycling at all. The floor "
        "around \u221250 dB is the bench, not the method \u2014 it matches "
        "the residual measured with nothing injected.",
        size=8.8, color=INK2)

    n.h2("What it does NOT remove", 5)
    n.para(
        "If the same molecules are hit by more than one chirp \u2014 which "
        "happens whenever the cycle period is shorter than a molecule's "
        "transit across the beam \u2014 consecutive chirps produce a "
        "Hahn echo carrying phase 2\u00b7phi_k \u2212 phi_(k\u22121). "
        "De-rotating by phi_k leaves phi_k \u2212 phi_(k\u22121), which for a "
        "uniform N-step pattern is the constant 2\u03c0/N. It therefore adds "
        "coherently in every cycle, and no number of steps changes that:")
    n.mono(
        ["  pattern        echo surviving"] +
        [f"  {name:<14}{echo_gain(ph):>10.2f}"
         for name, ph in PATTERNS[1:]] +
        ["  4-step, order scrambled      0.53"],
        size=8.0)
    n.para(
        "Scrambling the order within a balanced set holds m = 0 and the image "
        "at exactly zero while roughly halving the echo, which is a cheap "
        "option if it turns out to matter.")
    n.para(
        "There is a structural protection, though. With chirps at uniform "
        "spacing T the echo from chirps k\u22121 and k "
        "refocuses at 2t_k \u2212 t_(k\u22121) = t_(k+1), exactly when the "
        "next chirp fires, and the same holds for any pair. In a rigidly "
        "uniform train every two-pulse echo lands under a later chirp and "
        "never inside a detect window.")
    n.callout("Keep the cycle spacing exactly uniform. It is what keeps "
              "echoes out of the detect windows, and phase cycling will not "
              "clean them up if they get in.")

    n.h2("Choosing K", 6)
    n.para(
        "K is the window the sample is present for, over the cycle period, "
        "rounded DOWN to a whole number of patterns. A 1 ms gas pulse is no "
        "constraint \u2014 some 88 cycles fit at a 11.1 \u00b5s period and "
        "you want all of them, a cycle being ~2000x cheaper than a new "
        "pulse. For laser ablation the floor is what matters: 4 cycles for "
        "the 4-step, 2 for the 2-step.")
    w = 30.0
    rows = ["  30 us window, chirp 1 us:   usable K (multiple of 4)",
            "  dead     FID 0.5us   FID 1us   FID 2us   FID 4us   FID 6us"]
    for d in (0.2, 0.5, 1.0, 2.0, 5.0):
        row = f"  {d:<4g}us "
        for f in (0.5, 1.0, 2.0, 4.0, 6.0):
            k = (int(w // (1.0 + d + f)) // 4) * 4
            row += f"{(str(k) if k else '--'):>10}"
        rows.append(row)
    n.mono(rows, size=8.0)
    n.para(
        "Collecting past the FID decay adds noise and no signal, so the best "
        "detect window IS the decay time, not the longest that fits. Doppler "
        "width scales with frequency, so a 350 GHz FID decays far faster "
        "than the same transition at 10 GHz and the cycle is dominated by "
        "chirp plus dead time \u2014 on an ablation window the dead time "
        "alone can swing K by five, a factor of two in sensitivity.")

    n.h2("Where this lives in the code", 7)
    n.mono([
        "awg.py        send_waveforms(timing={'n_cycles': K}) plus a per-",
        "              channel 'phases' list; _build_channel lays the K",
        "              copies and removes the free-running NCO advance.",
        "phasecycle.py split_cycles() cuts at exact times, combine()",
        "              returns BOTH sums so artifacts stay measured.",
        "gui.py        the Phase cycling block; tests/ the offline check.",
    ], size=8.0)
    n.para(
        "Two traps live there: the NCO advance must be removed per cycle, "
        "and cycles cut at exact times, not rounded samples \u2014 a 0.44-sample "
        "slip is 28 degrees at 2.24 GHz and cost a 4-cycle sum 15%.",
        size=8.8, color=INK2)

    n.close({
        "Title": "Phase cycling in CP-FTMW",
        "Author": "Jonas Bosmann",
        "Subject": "CP-FTMW: phase cycling, coherence-order selection",
        "Keywords": "CP-FTMW, phase cycling, coherence order, chirped pulse",
    })
    print(f"wrote {OUT}  ({n.page} pages)")


if __name__ == "__main__":
    main()
