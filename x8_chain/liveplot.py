"""A small live-updating plot for bench sweeps.

Bench runs take minutes and the interesting failures -- a level collapsing
halfway across the band, a filter skirt appearing where it should not, a stage
dropping out of compression -- are obvious in a curve as it forms and invisible
in a wall of numbers. This puts the trace on screen point by point so a run
that has clearly gone wrong can be stopped rather than waited out.

Deliberately defensive: if the backend cannot open a window (no display,
headless run, backend not installed) it disables itself and the measurement
continues unaffected. A plotting failure must never take down a measurement
that is driving instruments.

Controlled by environment:
    NO_SHOW=1   no windows at all (for unattended runs)
    LIVE=0      no LIVE window, but the final saved figures still happen

Usage:
    lp = LivePlot("stage A", "frequency (GHz)", "level (dBm)")
    for f, v in ...:
        lp.add("H1", f/1e9, v)
    lp.done()          # leaves the window up, stops updating
"""
import os

import numpy as np

_DISABLED = os.environ.get("NO_SHOW") == "1" or os.environ.get("LIVE") == "0"


class LivePlot:
    def __init__(self, title, xlabel, ylabel, enabled=None, spans=()):
        """spans: iterable of (x_lo, x_hi, label) shaded regions, e.g. the
        chain band, drawn once at creation."""
        self.ok = False
        self._lines = {}
        self._data = {}
        if enabled is False or (enabled is None and _DISABLED):
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(9, 5))
            for lo, hi, lab in spans:
                self.ax.axvspan(lo, hi, alpha=0.12, color="tab:green", label=lab)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.set_title(title)
            self.ax.grid(alpha=0.3)
            self.fig.tight_layout()
            self.fig.show()
            self._pump()
            self.ok = True
        except Exception as e:                       # noqa: BLE001
            print(f"  (live plot unavailable: {e} -- measurement continues)")
            self.ok = False

    def _pump(self):
        """Give the GUI a moment to redraw without blocking the sweep."""
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def add(self, series, x, y):
        """Append one point to a named series and redraw."""
        if not self.ok:
            return
        try:
            if series not in self._lines:
                self._data[series] = ([], [])
                (ln,) = self.ax.plot([], [], "o-", ms=4, label=series)
                self._lines[series] = ln
                self.ax.legend(fontsize=8)
            xs, ys = self._data[series]
            xs.append(x); ys.append(y)
            self._lines[series].set_data(xs, ys)
            self.ax.relim()
            self.ax.autoscale_view()
            self._pump()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def hline(self, y, label=None, **kw):
        if not self.ok:
            return
        try:
            self.ax.axhline(y, ls="--", lw=1, alpha=0.6, label=label, **kw)
            self.ax.legend(fontsize=8)
            self._pump()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def note(self, text):
        """Put a status line in the title -- useful for showing which rung or
        which convergence step is in progress."""
        if not self.ok:
            return
        try:
            self.ax.set_title(text)
            self._pump()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def clear_series(self):
        """Drop the plotted lines but keep the window, for the next sweep."""
        if not self.ok:
            return
        try:
            for ln in self._lines.values():
                ln.remove()
            self._lines.clear(); self._data.clear()
            self._pump()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def done(self):
        """Stop updating; leave the window on screen."""
        if not self.ok:
            return
        try:
            self.plt.ioff()
            self._pump()
        except Exception:                            # noqa: BLE001
            pass


class LiveSpectrum:
    """Live view of the ACTUAL analyzer traces during a sweep.

    One panel per tone being measured, each showing the real 801-point trace
    as the EXA returns it, plus a summary panel accumulating the extracted
    peaks. This shows what the analyzer is really seeing -- the tone, the
    noise floor, and anything else in the span -- rather than only the number
    the script chose to pull out of it. A spur sitting next to the carrier, a
    tone that has drifted out of the span, or a trace that is all noise
    because the path is dead are all obvious here and invisible in a peak
    value.

    Same defensive contract as LivePlot: any failure disables the display and
    the measurement continues.
    """

    def __init__(self, panels, title, enabled=None):
        self.ok = False
        self._tr = {}
        self._pk = {}
        self._sum = {}
        if enabled is False or (enabled is None and _DISABLED):
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            n = len(panels) + 1
            cols = 2 if n <= 4 else 3
            rows = int(np.ceil(n / cols))
            self.fig, axes = plt.subplots(rows, cols, figsize=(13, 8))
            axes = axes.ravel()
            self.axes = {}
            for ax, name in zip(axes, panels):
                ax.set_title(name, fontsize=9)
                ax.set_xlabel("MHz", fontsize=8)
                ax.set_ylabel("dBm", fontsize=8)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)
                self.axes[name] = ax
            self.sum_ax = axes[len(panels)]
            self.sum_ax.set_title("extracted peaks", fontsize=9)
            self.sum_ax.set_xlabel("AWG frequency (GHz)", fontsize=8)
            self.sum_ax.set_ylabel("dBm", fontsize=8)
            self.sum_ax.grid(alpha=0.3)
            self.sum_ax.tick_params(labelsize=7)
            for ax in axes[len(panels) + 1:]:
                ax.axis("off")
            self.fig.suptitle(title, fontsize=10)
            self.fig.tight_layout()
            self.fig.show()
            self._pump()
            self.ok = True
        except Exception as e:                       # noqa: BLE001
            print(f"  (live spectrum unavailable: {e} -- measurement continues)")
            self.ok = False

    def _pump(self):
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def trace(self, panel, freqs_hz, amps_dbm, offset_db=0.0):
        """Push one analyzer trace into a panel. offset_db is added for
        display only (e.g. the external pad), so the axis reads at the DUT."""
        if not self.ok or panel not in self.axes:
            return
        try:
            ax = self.axes[panel]
            x = np.asarray(freqs_hz) / 1e6
            y = np.asarray(amps_dbm) + offset_db
            if panel not in self._tr:
                (ln,) = ax.plot(x, y, lw=0.8)
                self._tr[panel] = ln
                (mk,) = ax.plot([], [], "v", ms=7, color="tab:red")
                self._pk[panel] = mk
            else:
                self._tr[panel].set_data(x, y)
            i = int(np.argmax(y))
            self._pk[panel].set_data([x[i]], [y[i]])
            ax.set_xlim(x.min(), x.max())
            lo_, hi_ = float(np.min(y)), float(np.max(y))
            ax.set_ylim(lo_ - 5, hi_ + 8)
            ax.set_title(f"{panel}   peak {y[i]:+.2f} dBm @ {x[i]:.1f} MHz",
                         fontsize=9)
            self._pump()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def summary(self, series, x, y):
        if not self.ok:
            return
        try:
            if series not in self._sum:
                (ln,) = self.sum_ax.plot([], [], "o-", ms=3, label=series)
                self._sum[series] = (ln, [], [])
                self.sum_ax.legend(fontsize=7)
            ln, xs, ys = self._sum[series]
            xs.append(x); ys.append(y)
            ln.set_data(xs, ys)
            self.sum_ax.relim(); self.sum_ax.autoscale_view()
            self._pump()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def note(self, text):
        if not self.ok:
            return
        try:
            self.fig.suptitle(text, fontsize=10)
            self._pump()
        except Exception:                            # noqa: BLE001
            self.ok = False

    def done(self):
        if not self.ok:
            return
        try:
            self.plt.ioff()
            self._pump()
        except Exception:                            # noqa: BLE001
            pass
