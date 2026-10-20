"""Build the NCO grid-snapping technical note as a PDF.

Typeset with matplotlib (mathtext for the equations, a vector axes for the
figure) so nothing has to be installed into the lab's conda env.

Set NOTE_PNG_DIR to also dump each page as a PNG for proofing.
"""
import datetime as dt
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = r"C:\Users\Admin\Documents\Claude Programming\nco_grid_snapping_2026-10-14.pdf"

# ── ink + palette (dataviz reference palette, light mode, fixed slot order) ──
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
INK3    = "#7a7975"
RULE    = "#d9d8d3"
BOXBG   = "#f1f0ec"
ACCENT  = "#2a78d6"
SERIES  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

# ── the physics, computed not quoted ────────────────────────────────
RATE_CX = 9e9 / 8
N_CHIRP, N_DEAD, N_DETECT = (round(1.0e-6 * RATE_CX), round(0.1e-6 * RATE_CX),
                             round(10.0e-6 * RATE_CX))
N_SUM = N_CHIRP + N_DEAD + N_DETECT
N_TOT = max(int(np.ceil(N_SUM / 32)) * 32, 64)
T = N_TOT / RATE_CX
FREP = 1.0 / T


def G(N, eps):
    N = np.asarray(N, dtype=float)
    if np.isclose(np.sin(np.pi * eps), 0.0):
        return np.ones_like(N)
    return np.abs(np.sin(np.pi * eps * N) / (N * np.sin(np.pi * eps)))


def dB(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


class Note:
    """Minimal flowing-text layout on A4 matplotlib pages."""

    def __init__(self, path):
        self.pdf = PdfPages(path)
        self.W, self.H = 8.27, 11.69
        self.ml, self.mr, self.mt, self.mb = 0.95, 0.95, 0.95, 0.80
        self.tw = self.W - self.ml - self.mr
        self.page = 0
        self._new_page()

    # -- page machinery --
    def _new_page(self):
        self.fig = plt.figure(figsize=(self.W, self.H))
        self.fig.patch.set_facecolor(SURFACE)
        self.renderer = self.fig.canvas.get_renderer()
        self.y = self.H - self.mt
        self.page += 1

    def _flush(self):
        self.fig.text(self.ml / self.W, (self.mb - 0.35) / self.H,
                      "NCO grid snapping — CP-FTMW bench note",
                      fontsize=7.5, color=INK3, family=SANS, va="center")
        self.fig.text(1 - self.mr / self.W, (self.mb - 0.35) / self.H,
                      f"{self.page}", fontsize=7.5, color=INK3, family=SANS,
                      va="center", ha="right")
        self.pdf.savefig(self.fig, facecolor=SURFACE)
        png_dir = os.environ.get("NOTE_PNG_DIR")
        if png_dir:
            self.fig.savefig(os.path.join(png_dir, f"page{self.page}.png"),
                             dpi=110, facecolor=SURFACE)
        plt.close(self.fig)

    def _need(self, h):
        if self.y - h < self.mb:
            self._flush()
            self._new_page()

    # -- measurement --
    def _measure(self, s, size, weight, family):
        t = self.fig.text(0, 0, s, fontsize=size, fontweight=weight, family=family)
        bb = t.get_window_extent(self.renderer)
        t.remove()
        return bb.width / self.fig.dpi, bb.height / self.fig.dpi

    def _wrap(self, s, size, weight, family, width):
        out = []
        for para in s.split("\n"):
            words, cur = para.split(), ""
            for w in words:
                trial = (cur + " " + w).strip()
                if not cur or self._measure(trial, size, weight, family)[0] <= width:
                    cur = trial
                else:
                    out.append(cur)
                    cur = w
            out.append(cur)
        return out

    # -- blocks --
    def para(self, s, size=9.6, weight="normal", color=INK, indent=0.0,
             space_before=0.0, space_after=0.12, family=SANS, lead=1.42):
        self.y -= space_before
        lh = size / 72.0 * lead
        for line in self._wrap(s, size, weight, family, self.tw - indent):
            self._need(lh)
            self.fig.text((self.ml + indent) / self.W, self.y / self.H, line,
                          fontsize=size, fontweight=weight, color=color,
                          family=family, va="top")
            self.y -= lh
        self.y -= space_after

    def h1(self, s):
        self.para(s, size=15.5, weight="bold", space_after=0.07)

    def h2(self, s, n=None):
        self.y -= 0.20
        self._need(0.55)
        self.para(f"{n}.  {s}" if n else s, size=11.6, weight="bold",
                  space_after=0.09)

    def rule(self, space_before=0.04, space_after=0.14):
        self.y -= space_before
        self._need(0.05)
        self.fig.add_artist(plt.Line2D(
            [self.ml / self.W, (self.ml + self.tw) / self.W],
            [self.y / self.H, self.y / self.H],
            color=RULE, linewidth=0.8, transform=self.fig.transFigure))
        self.y -= space_after

    def eq(self, s, size=12.5, space_before=0.10, space_after=0.16):
        self.y -= space_before
        _, h = self._measure(s, size, "normal", SANS)
        self._need(h + 0.08)
        self.fig.text(0.5, self.y / self.H, s, fontsize=size, color=INK,
                      family=SANS, ha="center", va="top")
        self.y -= h + space_after

    def mono(self, lines, size=8.3, space_before=0.06, space_after=0.16):
        self.y -= space_before
        lh = size / 72.0 * 1.45
        block_h = lh * len(lines) + 0.16
        self._need(block_h)
        top = self.y + 0.07
        self.fig.patches.append(plt.Rectangle(
            ((self.ml - 0.10) / self.W, (top - block_h) / self.H),
            (self.tw + 0.20) / self.W, block_h / self.H,
            transform=self.fig.transFigure, facecolor=BOXBG,
            edgecolor="none", zorder=0))
        for line in lines:
            self.fig.text(self.ml / self.W, self.y / self.H, line, fontsize=size,
                          color=INK, family=MONO, va="top")
            self.y -= lh
        self.y -= space_after

    def callout(self, s, size=9.8):
        """A result worth not missing: accent bar + bold text."""
        self.y -= 0.10
        lines = self._wrap(s, size, "bold", SANS, self.tw - 0.28)
        lh = size / 72.0 * 1.45
        h = lh * len(lines) + 0.14
        self._need(h)
        top = self.y + 0.06
        for w, col, z in (((self.tw + 0.20), "#eaf2fc", 0), (0.035, ACCENT, 1)):
            self.fig.patches.append(plt.Rectangle(
                ((self.ml - 0.10) / self.W, (top - h) / self.H),
                w / self.W, h / self.H, transform=self.fig.transFigure,
                facecolor=col, edgecolor="none", zorder=z))
        for line in lines:
            self.fig.text((self.ml + 0.10) / self.W, self.y / self.H, line,
                          fontsize=size, fontweight="bold", color=INK,
                          family=SANS, va="top")
            self.y -= lh
        self.y -= 0.20

    def chart(self, height, draw, pad_bottom=0.62):
        """Reserve `height`; the axes box sits pad_bottom above the block floor
        so the tick labels and x-label have their own room."""
        self.y -= 0.06
        self._need(height + 0.10)
        ax = self.fig.add_axes([self.ml / self.W,
                                (self.y - height + pad_bottom) / self.H,
                                self.tw / self.W,
                                (height - pad_bottom) / self.H])
        draw(ax)
        self.y -= height + 0.08

    def close(self, meta):
        self._flush()
        self.pdf.infodict().update(meta)
        self.pdf.close()


# ── the figure ──────────────────────────────────────────────────────
def draw_snr(ax):
    """SNR gain vs N for two realistic offsets, against the ideal sqrt(N).

    Linear N up to 300 on purpose: the decision range is N <= 256, and a log
    axis turns the post-null ringing into unreadable grass. Larger offsets
    (10 kHz, 45 kHz) null within the first few shots and live in the table.
    """
    N = np.arange(1, 301)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=RULE, lw=0.8, zorder=1)
    ax.plot(N, 10 * np.log10(N), ls=(0, (5, 3)), lw=1.4, color=INK3, zorder=2,
            label="on grid (ideal, $\\sqrt{N}$)")
    for (df, lab), c in zip([(100.0, "100 Hz off grid"),
                             (1000.0, "1 kHz off grid")], SERIES):
        snr = 10 * np.log10(N) + dB(G(N, df * T))
        ax.plot(N, np.maximum(snr, -45), lw=1.8, color=c, label=lab,
                solid_capstyle="round", zorder=3)
    ax.set_xlim(0, 300)
    ax.set_ylim(-45, 30)
    ax.set_xlabel("Number of shots averaged, N", fontsize=9, color=INK2)
    ax.set_ylabel("SNR gain (dB)", fontsize=9, color=INK2)
    ax.grid(True, color=RULE, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    ax.annotate("null at $N=f_{rep}/\\delta f=90$", xy=(90, -41), xytext=(112, -31),
                fontsize=8.5, color=INK2, family=SANS,
                arrowprops=dict(arrowstyle="->", color=INK3, lw=0.9,
                                shrinkA=0, shrinkB=3))
    # The N=64 peak is named in the caption rather than with a second arrow:
    # any text box near it has to cross one of the two curves to point at it.
    leg = ax.legend(loc="lower left", fontsize=8.3, frameon=True, framealpha=1.0,
                    edgecolor=RULE, facecolor=SURFACE, borderpad=0.6,
                    labelcolor=INK2, handlelength=2.0)
    leg.get_frame().set_linewidth(0.7)


# ── build ───────────────────────────────────────────────────────────
n = Note(OUT)

n.para("CP-FTMW spectrometer — bench note", size=9, color=ACCENT,
       weight="bold", space_after=0.06)
n.h1("Why the DUC carrier must sit on the k/T grid")
n.para("Carrier phase coherence, coherent averaging, and the one condition that "
       "decides whether averaging helps or destroys the FID.",
       size=10.5, color=INK2, space_after=0.10)
n.rule()
n.para(f"Jonas Bosmann · {dt.date(2026, 10, 14):%d %B %Y} · "
       f"Tabor P9482D + Tektronix MSO64B", size=8.6, color=INK3, space_after=0.22)

n.callout("One condition, and everything follows from it:  f_NCO x T must be a "
          "whole number, where T is the buffer period. On grid, every shot is "
          "identical. Off grid, the carrier phase walks and averaging cancels "
          "the signal.")

n.h2("The signal model", 1)
n.para("In DUC mode the instrument plays a stored complex baseband against a "
       "hardware NCO:")
n.eq(r"$s(t)\;=\;\mathrm{Re}\left\{\,z(t)\;e^{\,j\left(2\pi f_{NCO}\,t\,+\,\phi_0\right)}\right\},"
     r"\qquad z(t)=I(t)+jQ(t)$")
n.para("Two very different kinds of behaviour hide in that one line.")
n.para("$z(t)$ lives in segment memory and loops with period "
       "$T=n_{total}/rate_{cx}$. It satisfies $z(t+kT)=z(t)$ exactly — the same "
       "samples, the same DAC codes, on every repeat, forever.", indent=0.22)
n.para("The NCO is a free-running phase accumulator. It is not reset at the "
       "segment wrap and not reset by a trigger. Its phase simply keeps "
       "counting, $\\phi(t)=2\\pi f_{NCO}t+\\phi_0$.", indent=0.22)

n.h2("What the scope captures on shot k", 2)
n.para("The marker fires at the same sample of the buffer every period, so shot "
       "$k$ begins at $t_k=t_0+kT$. Writing the captured record in local time "
       "$\\tau$ (time since the trigger) and using $z(t_k+\\tau)=z(\\tau)$:")
n.eq(r"$s_k(\tau)\;=\;\mathrm{Re}\left\{\,z(\tau)\,e^{\,j2\pi f_{NCO}\tau}\;"
     r"e^{\,j\theta_k}\right\},\qquad \theta_k=\theta_0+2\pi f_{NCO}T\,k$")
n.para("That is the whole problem in one line. Every shot is the identical "
       "complex waveform multiplied by a scalar phasor. The marker gives a rigid "
       "TIME reference; it gives no PHASE reference, because the carrier never "
       "restarted with the buffer.")
n.para("The per-shot phase step is therefore")
n.eq(r"$\Delta\theta=2\pi\varepsilon,\qquad "
     r"\varepsilon\;\equiv\;\mathrm{frac}\left(f_{NCO}T\right)=\delta\!f\cdot T$")
n.para("where $\\delta\\!f$ is how far the carrier sits from the nearest grid "
       "point $k/T$. If $f_{NCO}T$ is an integer then $\\varepsilon=0$, "
       "$\\theta_k=\\theta_0$ for every $k$, and all shots are bit-for-bit "
       "identical.")

n.h2("What averaging does to it", 3)
n.para("Averaging N shots sums those phasors — a geometric series, i.e. the "
       "Dirichlet kernel:")
n.eq(r"$\frac{1}{N}\sum_{k=0}^{N-1}e^{\,j2\pi\varepsilon k}"
     r"=e^{\,j\pi\varepsilon(N-1)}\,\frac{\sin(\pi\varepsilon N)}{N\sin(\pi\varepsilon)}"
     r"\quad\Longrightarrow\quad"
     r"\left|G(N,\varepsilon)\right|=\frac{\left|\sin(\pi\varepsilon N)\right|}"
     r"{N\left|\sin(\pi\varepsilon)\right|}$")
n.para("Noise is uncorrelated shot to shot, so its amplitude still falls as "
       "$\\sqrt{N}$ while the signal is multiplied by $|G|$. Hence:")
n.eq(r"$\mathrm{SNR\ gain}=10\log_{10}N+20\log_{10}\left|G\right|,"
     r"\qquad N_{eff}=N\left|G\right|^{2}$")
n.para("$|G|=1$ only at $\\varepsilon=0$. For small $\\varepsilon$ it "
       "degenerates to $|G|\\approx\\mathrm{sinc}(\\varepsilon\\,N)$: the loss "
       "depends on the phase walk accumulated over the WHOLE average, not on "
       "$\\varepsilon$ alone. That is why the requirement gets harsher the more "
       "you average.")

n.h2("Numbers for the default shot", 4)
n.para(f"$n_{{total}}$ = {N_TOT:,} complex samples, so T = {T*1e6:.4f} us and the "
       f"grid spacing is $f_{{rep}}=1/T$ = {FREP/1e3:.3f} kHz. "
       f"Coherent gain $20\\log_{{10}}|G|$, in dB:")
hdr = (f"{'delta-f (Hz)':>13} {'epsilon':>11} |"
       + "".join(f"{'N='+str(k):>10}" for k in (2, 8, 64, 1000)))
rows = [hdr, "-" * len(hdr)]
for df in (45000.0, 10000.0, 1000.0, 100.0, 1.0):
    eps = df * T
    rows.append(f"{df:>13,.0f} {eps:>11.6f} |" +
                "".join(f"{dB(G(k, eps)):>10.1f}" for k in (2, 8, 64, 1000)))
n.mono(rows)

n.chart(3.30, draw_snr)
n.para("SNR gain against the number of shots averaged, for two carrier offsets "
       f"from the nearest grid point (T = {T*1e6:.3f} us). Larger offsets null "
       "even sooner — 10 kHz at N = 9 — and are left to the table above. The "
       "1 kHz curve is the one to look at twice: it peaks near N = 64 and then "
       "collapses. Averaging harder is not a remedy.",
       size=8.5, color=INK2, space_after=0.16)

n.para("Three things to take from it.", weight="bold", space_after=0.08)
n.para("Half a grid step off (45 kHz) is total loss at any N. Successive shots "
       "are ~180 degrees apart, so two of them already cancel.", indent=0.22)
n.para("Averaging is non-monotonic off grid. More shots can make the FID "
       "SMALLER — on the bench that looks exactly like a hardware fault, which "
       "is why it is worth being able to name.", indent=0.22)
n.para("The budget tightens as 1/N. For 0.5 dB of loss or less you need "
       "$\\varepsilon\\,N\\leq0.186$, i.e. "
       "$\\delta\\!f\\leq0.186\\,f_{rep}/N$.", indent=0.22)

n.mono([
    "delta-f = 1 kHz, the practical case:",
    "",
    "   N =   16    ideal  12.0 dB      actual   11.6 dB",
    "   N =   64    ideal  18.1 dB      actual    9.0 dB",
    "   N =   90    ideal  19.5 dB      actual  -40.8 dB   <- null at f_rep/delta-f",
    "   N =  256    ideal  24.1 dB      actual   -1.7 dB",
    "",
    "-0.5 dB budget   delta-f <= 0.186 * f_rep / N:",
    "",
    f"   N =     64   ->  delta-f <= {0.186*FREP/64:>7.1f} Hz",
    f"   N =   1000   ->  delta-f <= {0.186*FREP/1000:>7.1f} Hz",
    f"   N =  10000   ->  delta-f <= {0.186*FREP/10000:>7.1f} Hz",
])
n.callout("At 1000 averages the carrier must be within ~17 Hz of a 90 kHz grid "
          "point. 'Close enough' is not a strategy — you snap.")

n.h2("Why snapping is exact, and one detail in the driver", 5)
n.para("Set the carrier to an exact multiple of the repetition rate:")
n.eq(r"$f_{NCO}=k\,\frac{rate_{cx}}{n_{total}}\qquad\Longrightarrow\qquad"
     r"f_{NCO}\,T=k\,\frac{rate_{cx}}{n_{total}}\cdot\frac{n_{total}}{rate_{cx}}=k"
     r"\quad\mathrm{exactly}$")
n.para("The sample rate cancels. The coherence condition is a RATIO OF INTEGERS, "
       "so it stays satisfied even if the 9 GS/s clock is off by ppm or drifts "
       "with temperature. That is why this belongs in software rather than in a "
       "hunt for clock accuracy.")
n.para("One caveat from our own driver: _duc_upload_ch writes ':NCO:CFR1 "
       "{ncof:.0f}' — rounded to 1 Hz. At $n_{total}$ = 12,512 the grid is "
       "89,913.6829 Hz, not an integer, so the write leaves up to 0.5 Hz of "
       "residual. Harmless at realistic N (-0.5 dB only around N = 33,000, first "
       "null near N = 180,000), but it does cap how far averaging can go.")
n.para("It is free to remove. Choose $n_{total}$ so that it DIVIDES "
       "$rate_{cx}$ = 1.125e9 exactly: the grid then falls on a whole number of "
       "Hz and the 1 Hz write is exact. The multiples of 32 that qualify are")
good = [m for m in range(64, 60001, 32) if 1.125e9 % m == 0]
half = (len(good) + 1) // 2
n.mono(["  " + ", ".join(f"{m:,}" for m in good[:half]) + ",",
        "  " + ", ".join(f"{m:,}" for m in good[half:]),
        "",
        f"  smallest that fits the 1 / 0.1 / 10 us shot (needs >= {N_SUM:,}):  14,400",
        f"    ->  T = {14400/RATE_CX*1e6:.4f} us,   grid = {RATE_CX/14400:,.0f} Hz exactly,"
        f"   detect = {(14400-N_CHIRP-N_DEAD)/RATE_CX*1e6:.5f} us"])

n.h2("Absorbing the remainder — exact frequency AND coherence", 6)
n.para("Snapping the carrier does not have to move the output frequency. Snap "
       "$f_{NCO}$ to the nearest $k/T$, then put the leftover into the BASEBAND:")
n.eq(r"$z'(\tau)=z(\tau)\,e^{\,j2\pi\delta\tau},\qquad "
     r"\delta=f_{RF}-f'_{NCO}$")
n.para("The output stays centred at $f'_{NCO}+\\delta=f_{RF}$, while the "
       "per-shot phasor becomes $e^{\\,j2\\pi f'_{NCO}Tk}=e^{\\,j2\\pi k}=1$ for "
       "every $k$. It works because the baseband is memory-played: whatever "
       "$\\delta$ is baked into it replays identically every period by "
       "construction.")
n.para("This is also where the two different 'snap to the grid' rules meet, and "
       "they must not be confused:")
n.para("GATED signals (the CP-FTMW chirp and its LO) are windowed to zero at "
       "both ends, so they never reach the buffer wrap and $\\delta$ can be "
       "anything. Result: any exact RF frequency, fully coherent.", indent=0.22)
n.para("CONTINUOUS signals must additionally hold a whole number of cycles in "
       "T, or the phase steps at the wrap and sprays spurs — the rule the "
       "waveform builder already enforces (AWG.snap_continuous). Both conditions "
       "together force $f_{RF}T$ to be an integer, so a continuous DUC CW is "
       "restricted to the grid.", indent=0.22)
n.callout("Gated: exact frequency and full coherence. Continuous: pick from the "
          "78-90 kHz grid. The CP-FTMW shot is gated, so it loses nothing.")

n.h2("What this does not cover, and what is still unproven", 7)
n.para("The derivation assumes the trigger is rigid relative to the buffer. That "
       "is true for an AWG marker, and false for an external pulse: the P908xD "
       "external trigger jitters by 32 SCLK periods (~3.6 ns at 9 GS/s), which "
       "is many carrier cycles. Chain the timing (master pulse -> AWG TRIG 1 -> "
       "AWG marker -> scope); never fan it out.")
n.para("It also assumes the scope timebase is referenced to the AWG. Independent "
       "oscillators drift the sampled time axis from shot to shot — a separate "
       "decoherence mechanism, which is what set_reference_external() exists to "
       "remove.")
n.para("And the premise itself is inferred, not proven: that the Proteus NCO "
       "genuinely free-runs across the wrap. The -176 to +155 degree "
       "per-acquisition phase scatter observed earlier is consistent with it, "
       "which is not the same as evidence for it.")
n.para("The clean test: one shot on grid, one deliberately off grid, N captures "
       "on the marker trigger, per-shot carrier phase plotted. Off grid the "
       "phase should walk by exactly $2\\pi\\varepsilon$ per shot — and the "
       "measured walk gives $\\varepsilon$ directly.", weight="bold")

n.rule(space_before=0.18)
n.para("Status: the waveform builder implements the continuous-signal rule "
       "(AWG.snap_continuous, previewed in the GUI). NCO snapping is NOT "
       "implemented — it is deliberately parked with the marker-trigger work, "
       "since neither is useful without the other.",
       size=8.6, color=INK2, space_after=0.05)
n.para("Numbers in this note were computed, not quoted; see "
       "tests/test_awg_builder.py for the builder's own checks.",
       size=8.6, color=INK3)

n.close({
    "Title": "Why the DUC carrier must sit on the k/T grid",
    "Author": "Jonas Bosmann",
    "Subject": "CP-FTMW: carrier phase coherence and coherent averaging",
    "Keywords": "CP-FTMW, DUC, NCO, coherent averaging, Proteus P9482D",
})
print(f"wrote {OUT}  ({n.page} pages)")
