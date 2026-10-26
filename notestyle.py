"""Shared look for the CP-FTMW bench notes (PDF).

Typeset with matplotlib -- mathtext for the equations, vector axes for the
figures -- so nothing has to be installed into the lab's conda env.

Extracted from make_nco_note.py 2026-10-26 when a second note needed the
same layout. `Note(path, footer)` gives flowing text on A4 with para/h1/h2/
rule/eq/mono/callout/chart blocks; see make_phase_cycling_note.py for a
worked use.

Set NOTE_PNG_DIR to also dump each page as a PNG for proofing.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


class Note:
    """Minimal flowing-text layout on A4 matplotlib pages."""

    def __init__(self, path, footer="CP-FTMW bench note"):
        self.pdf = PdfPages(path)
        self.footer = footer
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
                      self.footer,
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

