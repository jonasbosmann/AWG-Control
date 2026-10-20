"""AWG + scope control GUI for the CP-FTMW spectrometer bench.

Reworked 2026-10-14. What changed and why:

  * ONE waveform panel instead of one panel per experiment. Pick the AWG
    mode at the top — DUC (internal upconversion, the mode this spectrometer
    actually runs in) or Direct — then pick a signal per channel. The old
    "Standard Waveform", "Chirp + CW", "DUC CW" and "DUC Chirp + CW" panels
    were four special cases of that single choice, and standard waveforms
    were reachable in direct mode only. Every signal is now available in
    either mode, on either channel, and awg.send_waveforms() builds all of
    them through one verified code path.

  * Blocks side by side rather than one tall scrolling column.

  * An Analysis block that owns the FFT settings — window, zero-padding, DC
    removal, magnitude units, time gate — shared by every spectrum this GUI
    draws, so the transform is something you chose rather than something
    hard-coded.

The Marker box in the Timing panel puts a start-of-buffer pulse on an AWG
marker output, which is what the scope should trigger on for anything that
will be averaged -- it is played from the same segment memory, on the same
sample clock, as the waveform. It is re-sent automatically after every Send,
because the *RST inside a waveform upload wipes the segment it lives in.

Still to come (see the GUI backlog): FID capture with N-shot software
averaging, external trigger.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import queue
import sys
import os
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from awg import AWG, SAMPLE_RATE_SINGLE, SAMPLE_RATE_DUAL
from scope import Scope
import reflock
from measure import DEFAULT_FREQS
import sweeplog


class _LogRedirect:
    """Forwards stdout to a ScrolledText widget via a thread-safe queue."""
    def __init__(self, widget, root):
        self._widget = widget
        self._queue = queue.Queue()
        root.after(100, self._poll)

    def write(self, text):
        self._queue.put(text)

    def flush(self):
        pass

    def _poll(self):
        while not self._queue.empty():
            text = self._queue.get_nowait()
            self._widget.configure(state='normal')
            self._widget.insert(tk.END, text)
            self._widget.see(tk.END)
            self._widget.configure(state='disabled')
        self._widget.after(100, self._poll)


class LabGUI:
    # ── vocabulary shared between the widgets and awg.send_waveforms ──
    MODES  = {"DUC  (upconverted, IQM ONE)": "duc", "Direct  (ARB samples)": "direct"}
    KINDS  = {"Off": "off", "CW": "cw", "Chirp": "chirp",
              "Square": "square", "Ramp": "ramp"}
    GATES  = {"Continuous": "continuous", "Chirp window": "chirp",
              "Detect window": "detect"}
    SIDEBANDS = {"Upper": "upper", "Lower": "lower", "Both (DSB)": "both"}
    CLOCKS = {"2.5 GS/s  (CH1+CH2)": SAMPLE_RATE_DUAL,
              "9 GS/s  (CH1 only)":  SAMPLE_RATE_SINGLE}
    WINDOWS = ("Rectangular", "Hann", "Hamming", "Blackman", "Kaiser")
    MAGNITUDES = ("dBV", "dBm (50 Ω)", "Linear (V)")

    def __init__(self, root):
        self.root = root
        self.root.title("AWG / Scope Control — CP-FTMW bench")
        w = min(1500, root.winfo_screenwidth() - 60)
        h = min(960, root.winfo_screenheight() - 100)
        self.root.geometry(f"{w}x{h}")
        self.awg   = None
        self.scope = None
        self._live_running = False
        self._live_gen     = 0
        self._sweep_stop   = threading.Event()
        self._action_btns  = []   # disabled while busy (STOP ALL stays live)
        self._live_win     = None
        self._setup_photo  = None
        self._sent_baseband = {}  # ch -> baseband signature actually uploaded
        # Builder methods need no connection, so an unconnected AWG can plan
        # and snap frequencies for the live preview labels.
        self._planner = AWG.__new__(AWG)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self._live_running = False
        self._sweep_stop.set()
        if self._live_win is not None:
            try: self._live_win.destroy()
            except Exception: pass
        if self.scope:
            try: self.scope.close()
            except Exception: pass
        if self.awg:
            try: self.awg.close()
            except Exception: pass
        self.root.destroy()

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        self._build_instrument_bar()

        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=8, pady=(0, 6))
        main.columnconfigure(0, weight=0, minsize=700)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        left.columnconfigure((0, 1), weight=1, uniform='blocks')

        self._build_awg_block(left,    row=0, col=0, span=2)
        self._build_scope_block(left,  row=1, col=0)
        self._build_analysis_block(left, row=1, col=1)
        self._build_sweep_block(left,  row=2, col=0)
        self._build_setup_block(left,  row=2, col=1)

        self._build_right(main)

        self._on_mode_change()

    def _build_instrument_bar(self):
        conn = ttk.LabelFrame(self.root, text="Instruments", padding=5)
        conn.pack(fill='x', padx=8, pady=(6, 4))
        ttk.Button(conn, text="Connect AWG", command=self._connect_awg).pack(side='left', padx=4)
        self._awg_lbl = ttk.Label(conn, text="AWG: disconnected", foreground='red')
        self._awg_lbl.pack(side='left', padx=(0, 16))
        ttk.Button(conn, text="Connect Scope", command=self._connect_scope).pack(side='left', padx=4)
        self._scope_lbl = ttk.Label(conn, text="Scope: disconnected", foreground='red')
        self._scope_lbl.pack(side='left', padx=(0, 16))

        ttk.Separator(conn, orient='vertical').pack(side='left', fill='y', padx=(0, 10))
        self._ref_var = tk.BooleanVar(value=False)
        self._ref_chk = ttk.Checkbutton(conn, text="Ext 10 MHz",
                                        variable=self._ref_var,
                                        command=self._apply_reference)
        self._ref_chk.pack(side='left', padx=(0, 6))
        self._ref_btn = ttk.Button(conn, text="Check lock",
                                   command=self._check_lock)
        self._ref_btn.pack(side='left', padx=(0, 2))
        # Which AWG channel is cabled to the scope. The scope end comes from
        # the Scope block's Channel field; the GUI cannot know the wiring, so
        # both ends are stated explicitly in the log when the check runs.
        ttk.Label(conn, text="AWG").pack(side='left')
        self._ref_awgch = tk.StringVar(value="1")
        ttk.Combobox(conn, textvariable=self._ref_awgch, values=["1", "2"],
                     width=2, state='readonly').pack(side='left', padx=(2, 8))
        self._ref_lbl = ttk.Label(conn, text="Ref: —", foreground='gray')
        self._ref_lbl.pack(side='left', padx=(0, 16))

        tk.Button(conn, text="STOP ALL", command=self._stop,
                  bg='red', fg='white', font=('', 10, 'bold'),
                  relief='raised', padx=8).pack(side='right', padx=8)

    # ── 10 MHz reference ──────────────────────────────────────────

    def _apply_reference(self):
        """Toggle the AWG between the external 10 MHz and its internal TCXO.

        Wiring reminder: the 10 MHz goes to **REF IN**, the SMP connector
        directly ABOVE the SMA CLK IN. Feeding CLK IN instead does nothing
        (that input is only live once an external sample clock is selected).
        """
        want_ext = self._ref_var.get()
        if not self.awg:
            self._status(self._ref_lbl, "Ref: connect the AWG first", "red")
            self._ref_var.set(not want_ext)
            return

        def work():
            try:
                if want_ext:
                    self.awg.set_reference_external(10)
                else:
                    self.awg.set_reference_internal()
            except Exception as e:
                # set_reference_external raises when the firmware accepted the
                # command but the readback disagrees -- usually nothing on
                # REF IN.
                print(f"reference: {e}\n")
                self.root.after(0, lambda: self._ref_var.set(not want_ext))
            self._refresh_reference()
        self._thread(work)

    def _refresh_reference(self):
        """Read back what each instrument is actually using.

        The AWG has NO lock query -- `:ROSC:SOUR?` reports what was accepted,
        so EXT here means 'configured', not 'locked'; confirm a real lock with
        check_reference.py. The scope does have one (`ROSc:STATe?`), and the
        disagreement is the interesting case: an AWG on the OCXO while the
        scope runs on its own clock is exactly the timebase-stretch problem
        the shared reference exists to remove, and it is invisible otherwise.
        """
        def work():
            awg_txt, scope_txt, colour = "—", "", "gray"
            ext = locked = None
            if self.awg:
                try:
                    st = self.awg.reference_status()
                    ext = "EXT" in st["source"]
                    awg_txt = f"AWG {st['source']}"
                    if ext and st["freq_mhz"]:
                        awg_txt += f" {st['freq_mhz']:g}M"
                except Exception as e:
                    awg_txt, colour = "AWG ?", "red"
                    print(f"reference readback failed: {e}\n")
            if self.scope:
                try:
                    sst = self.scope.reference_status()
                    locked = sst["external"] and sst["locked"]
                    scope_txt = f"  |  scope {sst['state'] or sst['source']}"
                except Exception:
                    scope_txt = "  |  scope ?"
            if ext is True:
                # orange = AWG on the OCXO but the scope is not; green = both.
                colour = "orange" if locked is False else "green"
            elif ext is False:
                colour = "gray"
            if ext is not None:
                self.root.after(0, lambda v=ext: self._ref_var.set(v))
            self._status(self._ref_lbl, f"Ref: {awg_txt}{scope_txt}", colour)
        self._thread(work)

    def _check_lock(self):
        """Measure the AWG's clock error against the reference-locked scope.

        This is the real lock test, and it needs no extra cabling: the scope's
        timebase is on the OCXO, so it is a frequency standard, and whatever
        AWG->scope connection you already use for measurements is enough. The
        `Ext 10 MHz` checkbox only shows what the AWG ACCEPTED -- `:ROSC:SOUR?`
        reads EXT whether or not a reference is actually present on REF IN.

        Reconfigures both instruments (CW tone on the AWG, timebase and record
        length on the scope), so it is a deliberate action, not a background
        poll.
        """
        if not (self.awg and self.scope):
            self._status(self._ref_lbl, "Ref: need AWG + scope", "red")
            return

        def work():
            self._set_busy(True)
            try:
                awg_ch = int(self._ref_awgch.get())
                sc_ch = int(self._chan_var.get())
                print(f"lock check: AWG CH{awg_ch} -> scope CH{sc_ch} "
                      f"(set the Scope block's Channel field if that is wrong)\n")
                res = reflock.check_lock(
                    self.awg, self.scope,
                    awg_channel=awg_ch, scope_channel=sc_ch)
                print(reflock.verdict_text(res) + "\n")
                if not res["conclusive"]:
                    colour = "orange"
                else:
                    colour = "green" if res["locked"] else "red"
                self._status(
                    self._ref_lbl,
                    f"Ref: {res['ppm']:+.2f} ± {res['ppm_sd']:.2f} ppm",
                    colour)
            except Exception as e:
                print(f"lock check failed: {e}\n")
                self._status(self._ref_lbl, "Ref: check failed", "red")
            finally:
                self._set_busy(False)
        self._thread(work)

    # ── AWG block ─────────────────────────────────────────────────

    def _build_awg_block(self, parent, row, col, span=1):
        f = ttk.LabelFrame(parent, text="AWG", padding=8)
        f.grid(row=row, column=col, columnspan=span, sticky='nsew', pady=(0, 6))
        f.columnconfigure((0, 1, 2), weight=1, uniform='awgcols')

        head = ttk.Frame(f)
        head.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 6))
        ttk.Label(head, text="Mode", font=('', 9, 'bold')).pack(side='left')
        self._mode_var = tk.StringVar(value=list(self.MODES)[0])
        ttk.Combobox(head, textvariable=self._mode_var, values=list(self.MODES),
                     width=26, state='readonly').pack(side='left', padx=(6, 14))
        self._mode_var.trace_add('write', lambda *_: self._on_mode_change())

        self._clock_var = tk.StringVar(value=list(self.CLOCKS)[0])
        self._clock_lbl = ttk.Label(head, text="Clock")
        self._clock_box = ttk.Combobox(head, textvariable=self._clock_var,
                                       values=list(self.CLOCKS), width=20,
                                       state='readonly')
        self._clock_var.trace_add('write', lambda *_: self._update_awg_info())
        self._duc_clock_lbl = ttk.Label(
            head, text="9 GS/s DAC · X8 interpolation → 1.125 GS/s complex",
            foreground='gray')

        self._ch = {}
        self._build_channel_panel(f, 1, 0)
        self._build_channel_panel(f, 2, 1)
        self._build_timing_panel(f, 2)

        btns = ttk.Frame(f)
        btns.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        btns.columnconfigure((0, 1, 2, 3), weight=1)
        send = ttk.Button(btns, text="Send to AWG", command=self._send_waveforms)
        send.grid(row=0, column=0, sticky='ew', padx=(0, 3))
        self._retune_btn = ttk.Button(btns, text="Retune carriers (fast)",
                                      command=self._retune)
        self._retune_btn.grid(row=0, column=1, sticky='ew', padx=3)
        copy_btn = ttk.Button(btns, text="Copy CH1 → CH2", command=self._copy_ch1_to_ch2)
        copy_btn.grid(row=0, column=2, sticky='ew', padx=3)
        ttk.Button(btns, text="Stop output", command=self._stop).grid(
            row=0, column=3, sticky='ew', padx=(3, 0))
        self._action_btns.extend([send, self._retune_btn, copy_btn])

        self._awg_warn = ttk.Label(f, text="", foreground='red',
                                   font=('', 8), wraplength=680, justify='left')
        self._awg_warn.grid(row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))

    def _build_channel_panel(self, parent, ch, col):
        f = ttk.LabelFrame(parent, text=f"CH{ch}", padding=6)
        f.grid(row=1, column=col, sticky='nsew', padx=(0, 6) if col == 0 else (3, 3))
        f.columnconfigure(1, weight=1)

        d = {"rows": {}}
        self._ch[ch] = d

        def combo(name, label, values, default, width=13):
            lbl = ttk.Label(f, text=label)
            var = tk.StringVar(value=default)
            box = ttk.Combobox(f, textvariable=var, values=values,
                               width=width, state='readonly')
            d[name] = var
            d["rows"][name] = (lbl, box)
            return var

        def entry(name, label, default):
            lbl = ttk.Label(f, text=label)
            var = tk.StringVar(value=default)
            e = ttk.Entry(f, textvariable=var, width=11)
            var.trace_add('write', lambda *_: self._update_awg_info())
            d[name] = var
            d["rows"][name] = (lbl, e)
            return var

        kind = combo("kind", "Signal", list(self.KINDS),
                     "Chirp" if ch == 1 else "CW")
        kind.trace_add('write', lambda *_: self._on_mode_change())
        entry("carrier", "Carrier (MHz)", "2000" if ch == 1 else "2300")
        entry("freq",    "Freq (MHz)",    "0" if ch == 2 else "100")
        entry("f0",      "Start (MHz)",   "0")
        entry("f1",      "Stop (MHz)",    "500")
        sb = combo("sideband", "Sideband", list(self.SIDEBANDS), "Upper")
        sb.trace_add('write', lambda *_: self._update_awg_info())
        gate = combo("gate", "Gate", list(self.GATES),
                     "Chirp window" if ch == 1 else "Detect window")
        gate.trace_add('write', lambda *_: self._update_awg_info())
        entry("amp", "Amplitude (Vpp)", "0.5")

        d["info"] = ttk.Label(f, text="", foreground='#0a0', font=('Courier', 8),
                              justify='left', wraplength=210)
        d["info"].grid(row=20, column=0, columnspan=2, sticky='w', pady=(6, 0))

    def _build_timing_panel(self, parent, col):
        f = ttk.LabelFrame(parent, text="Timing  (one shared buffer)", padding=6)
        f.grid(row=1, column=col, sticky='nsew', padx=(3, 0))
        f.columnconfigure(1, weight=1)

        self._timing_vars = {}
        for r, (name, label, default) in enumerate([
                ("chirp_us",  "Chirp dur (µs)", "1.0"),
                ("dead_us",   "Dead time (µs)", "0.1"),
                ("detect_us", "Detect win (µs)", "10.0"),
                ("window_pct", "Edge taper (%)", "5")]):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky='w', pady=1)
            var = tk.StringVar(value=default)
            var.trace_add('write', lambda *_: self._update_awg_info())
            ttk.Entry(f, textvariable=var, width=9).grid(
                row=r, column=1, sticky='ew', padx=(4, 0), pady=1)
            self._timing_vars[name] = var

        ttk.Separator(f, orient='horizontal').grid(row=5, column=0, columnspan=2,
                                                   sticky='ew', pady=4)
        # Marker output: the trigger anything averaged should use. Rigid to
        # the waveform because it comes out of the same segment on the same
        # clock, unlike the AWG's external trigger (3.6 ns of jitter).
        self._mark_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Marker out (trigger)", variable=self._mark_on,
                        command=self._update_awg_info).grid(
            row=6, column=0, columnspan=2, sticky='w')
        row = ttk.Frame(f)
        row.grid(row=7, column=0, columnspan=2, sticky='ew', pady=1)
        ttk.Label(row, text="CH").pack(side='left')
        self._mark_ch = tk.StringVar(value="1")
        ttk.Combobox(row, textvariable=self._mark_ch, values=["1", "2"],
                     width=3, state='readonly').pack(side='left', padx=(2, 6))
        ttk.Label(row, text="M").pack(side='left')
        self._mark_num = tk.StringVar(value="1")
        ttk.Combobox(row, textvariable=self._mark_num,
                     values=["1", "2", "3", "4"], width=3,
                     state='readonly').pack(side='left', padx=(2, 6))
        ttk.Label(row, text="width ns").pack(side='left')
        self._mark_ns = tk.StringVar(value="100")
        ttk.Entry(row, textvariable=self._mark_ns, width=6).pack(
            side='left', padx=(2, 0))

        ttk.Separator(f, orient='horizontal').grid(row=10, column=0, columnspan=2,
                                                   sticky='ew', pady=5)
        self._timing_info = ttk.Label(f, text="", foreground='gray',
                                      font=('Courier', 8), justify='left',
                                      wraplength=210)
        self._timing_info.grid(row=11, column=0, columnspan=2, sticky='w')

    # ── AWG panel plumbing ────────────────────────────────────────

    def _mode(self):
        return self.MODES.get(self._mode_var.get(), "duc")

    def _clock_rate(self):
        return self.CLOCKS.get(self._clock_var.get(), SAMPLE_RATE_DUAL)

    def _kind(self, ch):
        return self.KINDS.get(self._ch[ch]["kind"].get(), "off")

    def _on_mode_change(self):
        """Re-label and show/hide the fields each mode and signal actually uses."""
        duc = self._mode() == "duc"
        self._clock_lbl.pack_forget(); self._clock_box.pack_forget()
        self._duc_clock_lbl.pack_forget()
        if duc:
            self._duc_clock_lbl.pack(side='left')
        else:
            self._clock_lbl.pack(side='left')
            self._clock_box.pack(side='left', padx=(6, 0))
        self._retune_btn.config(state='normal' if duc else 'disabled')

        for ch, d in self._ch.items():
            kind = self._kind(ch)
            chirping = kind == "chirp"
            tone = kind in ("cw", "square", "ramp")
            show = {
                "kind": True,
                "carrier": duc and kind != "off",
                "freq": tone,
                "f0": chirping,
                "f1": chirping,
                "sideband": duc and kind != "off",
                "gate": kind != "off",
                "amp": kind != "off",
            }
            d["rows"]["freq"][0].config(
                text="BB offset (MHz)" if duc else "Freq (MHz)")
            d["rows"]["f0"][0].config(text="BB start (MHz)" if duc else "Start (MHz)")
            d["rows"]["f1"][0].config(text="BB stop (MHz)" if duc else "Stop (MHz)")
            for r, name in enumerate(["kind", "carrier", "freq", "f0", "f1",
                                      "sideband", "gate", "amp"]):
                lbl, widget = d["rows"][name]
                if show[name]:
                    lbl.grid(row=r, column=0, sticky='w', pady=1)
                    widget.grid(row=r, column=1, sticky='ew', padx=(4, 0), pady=1)
                else:
                    lbl.grid_remove(); widget.grid_remove()
        self._update_awg_info()

    def _fnum(self, var, scale=1.0, default=None):
        """Entry text → float, scaled. Returns default (or raises) if unusable."""
        try:
            return float(var.get()) * scale
        except (ValueError, AttributeError):
            if default is None:
                raise
            return default

    def _marker_cfg(self):
        """Marker settings. `enabled` is the tick box; the rest stays valid
        either way, so unticking can switch off the marker the user actually
        selected instead of a hardcoded CH1 M1."""
        return {"enabled": bool(self._mark_on.get()),
                "channel": int(self._mark_ch.get()),
                "marker": int(self._mark_num.get()),
                "width_ns": max(self._fnum(self._mark_ns, 1.0, 100.0), 1.0)}

    def _apply_marker(self, plan, mode, specs, cfg):
        """Put the marker back after an upload, or switch it off.

        Has to run after every Send: send_waveforms() starts with a *RST,
        which deletes the segments the marker data lives in. The array length
        follows the segment as :TRAC:DEF saw it -- 2*n_total uint16 for a DUC
        (interleaved I/Q), n_total real samples in direct mode.
        """
        ch, num = cfg["channel"], cfg["marker"]
        if not cfg["enabled"]:
            try:
                self.awg.marker_off(ch, num)
            except Exception:
                pass
            return
        if specs.get(ch, {}).get("kind", "off") == "off":
            print(f"Marker: CH{ch} is switched off, so it has no segment to "
                  f"carry a marker. Set CH{ch} to something, or move the "
                  f"marker to the other channel.\n")
            return
        n_pts = 2 * plan["n_total"] if mode == "duc" else plan["n_total"]
        frac = min(cfg["width_ns"] * 1e-9 / plan["period_s"], 0.95)
        self.awg.marker_pulse(ch, segnum=ch, n_wave_points=n_pts,
                              marker=num, high_frac=frac, level_vpp=1.0)

    def _timing(self):
        return {
            "chirp_us":  self._fnum(self._timing_vars["chirp_us"], 1.0, 1.0),
            "dead_us":   self._fnum(self._timing_vars["dead_us"], 1.0, 0.1),
            "detect_us": self._fnum(self._timing_vars["detect_us"], 1.0, 10.0),
        }

    def _window_frac(self):
        return max(min(self._fnum(self._timing_vars["window_pct"], 0.01, 0.05), 0.5), 0.0)

    def _channel_spec(self, ch, strict=True):
        """Current CH widgets → an awg.send_waveforms channel spec."""
        d = self._ch[ch]
        kind = self._kind(ch)
        if kind == "off":
            return {"kind": "off"}
        g = lambda name, dflt: self._fnum(d[name], 1e6, None if strict else dflt)
        spec = {
            "kind": kind,
            "gate": self.GATES.get(d["gate"].get(), "continuous"),
            "amplitude_vpp": self._fnum(d["amp"], 1.0, None if strict else 0.5),
        }
        if kind == "chirp":
            spec["f_start_hz"] = g("f0", 0.0)
            spec["f_stop_hz"] = g("f1", 0.0)
        else:
            spec["freq_hz"] = g("freq", 0.0)
        if self._mode() == "duc":
            spec["carrier_hz"] = g("carrier", 0.0)
            spec["sideband"] = self.SIDEBANDS.get(d["sideband"].get(), "upper")
        return spec

    def _specs(self, strict=True):
        return {ch: self._channel_spec(ch, strict) for ch in (1, 2)}

    def _baseband_signature(self, ch, spec):
        """What a retune CANNOT change: everything except carrier and level.

        Retuning writes :NCO:CFR1 only, so the uploaded baseband stays as it
        was. If any of these changed since the last Send, the fast path would
        silently play the old shape at the new carrier.
        """
        return (self._mode(), spec.get("kind"), spec.get("gate"),
                spec.get("sideband"), spec.get("freq_hz"),
                spec.get("f_start_hz"), spec.get("f_stop_hz"),
                tuple(sorted(self._timing().items())), self._window_frac(),
                None if self._mode() == "duc" else self._clock_rate())

    def _update_awg_info(self):
        """Live preview: what the hardware will be asked to produce."""
        try:
            mode = self._mode()
            specs = self._specs(strict=False)
            active = {ch: s for ch, s in specs.items() if s["kind"] != "off"}
            rate_eff = (AWG.DUC_DAC_RATE / AWG.DUC_INTERP if mode == "duc"
                        else self._clock_rate())
            plan = self._planner._plan_timing(mode, self._timing(), rate_eff)
            n = plan["n_total"]

            self._timing_info.config(
                text=f"period  {plan['period_s']*1e6:8.3f} µs\n"
                     f"rep     {1e-3/plan['period_s']:8.3f} kHz\n"
                     f"samples {n:8,}\n"
                     f"tail pad{plan['pad_s']*1e9:8.1f} ns")

            warn = []
            if mode == "direct" and len(active) > 1 and self._clock_rate() > SAMPLE_RATE_DUAL:
                warn.append("Direct mode at 9 GS/s is single-channel (both DACs "
                            "interleaved) — set CH2 to Off or pick the 2.5 GS/s clock.")

            for ch in (1, 2):
                d, s = self._ch[ch], specs[ch]
                if s["kind"] == "off":
                    d["info"].config(text="off", foreground='gray')
                    continue
                start, n_win = plan["windows"][s["gate"]]
                gated = s["gate"] != "continuous"
                carrier = s.get("carrier_hz", 0.0) if mode == "duc" else 0.0
                lines = []
                if s["kind"] == "chirp":
                    lo, hi = carrier + s["f_start_hz"], carrier + s["f_stop_hz"]
                    lines.append(f"{lo/1e6:.3f} → {hi/1e6:.3f} MHz")
                    if n_win > 1:
                        lines.append(f"{abs(hi-lo)/1e6:.1f} MHz in "
                                     f"{n_win/rate_eff*1e6:.3f} µs")
                else:
                    f_req = s["freq_hz"]
                    f_act = f_req if gated else AWG.snap_continuous(f_req, n_win, rate_eff)
                    lines.append(f"{(carrier+f_act)/1e6:.6f} MHz")
                    if not gated and f_act != f_req:
                        lines.append(f"snapped {(f_act-f_req)/1e3:+.1f} kHz "
                                     f"(step {rate_eff/n_win/1e3:.1f} kHz)")
                    elif gated and f_req:
                        lines.append("exact (gated burst)")
                if mode == "duc" and s["kind"] in ("square", "ramp") \
                        and s.get("sideband") != "both":
                    lines.append("SSB shaping costs ~7-9 dB of DAC range")
                lines.append(f"{s['gate']}: {n_win/rate_eff*1e6:.3f} µs "
                             f"@ +{start/rate_eff*1e6:.3f} µs")
                d["info"].config(text="\n".join(lines), foreground='#0a0')

                # Nyquist / NCO range, reported where it is entered rather than
                # only when Send fails.
                try:
                    self._planner._check_limits(mode, {ch: s}, rate_eff)
                except ValueError as e:
                    warn.append(str(e))
                    d["info"].config(foreground='red')

            if not active:
                warn.append("Both channels are Off — nothing to send.")
            self._awg_warn.config(text="  •  ".join(warn))
        except (ValueError, TypeError, KeyError, ZeroDivisionError,
                AttributeError, tk.TclError):
            # Half-typed entry ("2", "-", ""): keep the last good preview rather
            # than flashing an error at every keystroke. Narrow on purpose — a
            # broad except here would swallow real bugs in the builder.
            pass

    def _copy_ch1_to_ch2(self):
        """CH2 := CH1, the 'monitor copy' case: drive a DUT from CH1 while CH2
        carries an identical signal to a scope."""
        src, dst = self._ch[1], self._ch[2]
        for name in ("kind", "carrier", "freq", "f0", "f1", "sideband", "gate", "amp"):
            dst[name].set(src[name].get())
        self._on_mode_change()
        print("CH2 set to an identical copy of CH1.\n")

    def _send_waveforms(self):
        if not self._need_awg(): return
        try:
            mode = self._mode()
            specs = self._specs(strict=True)
            timing = self._timing()
            frac = self._window_frac()
            rate = None if mode == "duc" else self._clock_rate()
            marker = self._marker_cfg()
        except Exception as e:
            print(f"Invalid AWG parameter: {e}\n")
            return

        self._set_busy(True)
        def run():
            try:
                if rate:
                    self.awg.sample_rate = rate
                plan = self.awg.send_waveforms(specs, mode=mode,
                                               timing=timing, rate=rate,
                                               window_frac=frac)
                self._apply_marker(plan, mode, specs, marker)
                self._sent_baseband = {
                    ch: self._baseband_signature(ch, s) for ch, s in specs.items()}
                print("Sent.\n")
            except Exception as e:
                self._sent_baseband = {}
                print(f"Send error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(run)

    def _retune(self):
        """Carrier-only update: :NCO:CFR1 (+ :VOLT) per channel, milliseconds.

        Refuses when the baseband changed, because the fast path does not
        re-upload: it would play the previous shape at the new carrier and
        look like a successful change.
        """
        if not self._need_awg(): return
        if self._mode() != "duc":
            print("Retune is a DUC-mode NCO write — in direct mode use Send.\n")
            return
        try:
            specs = self._specs(strict=True)
        except Exception as e:
            print(f"Invalid AWG parameter: {e}\n")
            return
        if not self._sent_baseband:
            print("Nothing uploaded yet — press Send to AWG first.\n")
            return
        carriers, amps, stale = {}, {}, []
        for ch, s in specs.items():
            if s["kind"] == "off":
                continue
            if self._baseband_signature(ch, s) != self._sent_baseband.get(ch):
                stale.append(ch)
            carriers[ch] = s["carrier_hz"]
            amps[ch] = s["amplitude_vpp"]
        if stale:
            print(f"CH{', CH'.join(str(c) for c in stale)}: baseband parameters "
                  f"changed since the last Send — press Send to AWG (a retune "
                  f"only moves the carrier and would play the old shape).\n")
            return
        if not carriers:
            print("Both channels are Off — nothing to retune.\n")
            return

        self._set_busy(True)
        def run():
            try:
                t0 = time.perf_counter()
                self.awg.retune(carriers, amps)
                print(f"Retuned "
                      f"{', '.join(f'CH{c}={f/1e6:.6f} MHz' for c, f in carriers.items())}"
                      f"  ({(time.perf_counter()-t0)*1e3:.0f} ms)\n")
            except Exception as e:
                print(f"Retune error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(run)

    # ── Scope block ───────────────────────────────────────────────

    def _build_scope_block(self, parent, row, col):
        f = ttk.LabelFrame(parent, text="Scope", padding=8)
        f.grid(row=row, column=col, sticky='nsew', padx=(0, 4), pady=(0, 6))
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        def field(r, c, label, var, values=None, width=8):
            ttk.Label(f, text=label).grid(row=r, column=c, sticky='w', pady=1)
            if values:
                w = ttk.Combobox(f, textvariable=var, values=values,
                                 width=width, state='readonly')
            else:
                w = ttk.Entry(f, textvariable=var, width=width)
            w.grid(row=r, column=c + 1, sticky='ew', padx=(4, 10), pady=1)
            return w

        self._chan_var      = tk.StringVar(value="1")
        self._trigsrc_var   = tk.StringVar(value="same as CH")
        self._triglev_var   = tk.StringVar(value="0.0")
        self._trigmode_var  = tk.StringVar(value="AUTO")
        self._trigslope_var = tk.StringVar(value="RISe")
        self._avg_var       = tk.StringVar(value="1")
        self._nsdiv_var     = tk.StringVar(value="")
        self._reclen_var    = tk.StringVar(value="")
        self._maxpts_var    = tk.StringVar(value="10000")

        field(0, 0, "Channel",    self._chan_var, ["1", "2", "3", "4"], 5)
        field(0, 2, "Averages",   self._avg_var,
              ["1", "8", "16", "64", "256", "512"], 6)
        field(1, 0, "Trigger on", self._trigsrc_var,
              ["same as CH", "1", "2", "3", "4"], 10)
        field(1, 2, "Level (V)",  self._triglev_var, None, 6)
        field(2, 0, "Trig mode",  self._trigmode_var, ["AUTO", "NORMal"], 8)
        field(2, 2, "Slope",      self._trigslope_var, ["RISe", "FALL"], 6)
        field(3, 0, "ns/div",     self._nsdiv_var, None, 8)
        field(3, 2, "Rec len",    self._reclen_var, None, 8)
        field(4, 0, "Max pts",    self._maxpts_var, None, 8)

        ttk.Label(f, text="NORMal + averaging = coherent averaging; AUTO free-runs\n"
                          "and washes it out. Blank ns/div or Rec len = leave as is.",
                  foreground='gray', font=('', 8), justify='left').grid(
            row=5, column=0, columnspan=4, sticky='w', pady=(4, 2))

        b = ttk.Frame(f)
        b.grid(row=6, column=0, columnspan=4, sticky='ew', pady=(2, 0))
        b.columnconfigure((0, 1, 2, 3), weight=1)
        setup = ttk.Button(b, text="Apply setup", command=self._setup_scope)
        setup.grid(row=0, column=0, sticky='ew', padx=(0, 2))
        plot = ttk.Button(b, text="Plot waveform", command=self._plot_waveform)
        plot.grid(row=0, column=1, sticky='ew', padx=2)
        live = ttk.Button(b, text="Live view…", command=self._open_live_view)
        live.grid(row=0, column=2, sticky='ew', padx=2)
        restore = ttk.Button(b, text="Restore", command=self._restore_scope)
        restore.grid(row=0, column=3, sticky='ew', padx=(2, 0))
        self._action_btns.extend([setup, plot, live, restore])

    def _scope_setup_args(self):
        src = self._trigsrc_var.get()
        return dict(
            channel=int(self._chan_var.get()),
            n_averages=int(self._fnum(self._avg_var, 1.0, 1)),
            trigger_channel=None if src.startswith("same") else int(src),
            trigger_level=self._fnum(self._triglev_var, 1.0, 0.0),
            trigger_mode=self._trigmode_var.get(),
            trigger_slope=self._trigslope_var.get())

    def _setup_scope(self):
        if not self._need_scope(): return
        args = self._scope_setup_args()
        reclen = self._reclen_var.get().strip()
        def do_setup():
            self._set_busy(True)
            try:
                self.scope.setup(**args)
                if reclen:
                    self.scope.set_record_length(int(float(reclen)))
                self._apply_timebase()
            except Exception as e:
                print(f"Setup error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(do_setup)

    def _restore_scope(self):
        """Undo leftover measurement state (NORMal trigger, averaging, armed
        single-sequence) — e.g. after an automated run."""
        if not self._need_scope(): return
        ch = int(self._chan_var.get())
        def do_restore():
            self._set_busy(True)
            try:
                self.scope.restore(channel=ch)
            except Exception as e:
                print(f"Restore error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(do_restore)

    # ── Analysis block ────────────────────────────────────────────

    def _build_analysis_block(self, parent, row, col):
        f = ttk.LabelFrame(parent, text="Analysis  (FFT of every spectrum shown)",
                           padding=8)
        f.grid(row=row, column=col, sticky='nsew', padx=(4, 0), pady=(0, 6))
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._win_var    = tk.StringVar(value="Hann")
        self._kaiser_var = tk.StringVar(value="8.6")
        self._pad_var    = tk.StringVar(value="×1")
        self._dc_var     = tk.BooleanVar(value=True)
        self._mag_var    = tk.StringVar(value="dBV")
        self._fmin_var   = tk.StringVar(value="")
        self._fmax_var   = tk.StringVar(value="")
        self._tmin_var   = tk.StringVar(value="")
        self._tmax_var   = tk.StringVar(value="")

        ttk.Label(f, text="Window").grid(row=0, column=0, sticky='w', pady=1)
        ttk.Combobox(f, textvariable=self._win_var, values=list(self.WINDOWS),
                     width=11, state='readonly').grid(row=0, column=1, sticky='ew',
                                                      padx=(4, 10), pady=1)
        self._kaiser_lbl = ttk.Label(f, text="Kaiser β")
        self._kaiser_box = ttk.Entry(f, textvariable=self._kaiser_var, width=6)
        self._win_var.trace_add('write', lambda *_: self._sync_kaiser())

        ttk.Label(f, text="Zero-pad").grid(row=1, column=0, sticky='w', pady=1)
        ttk.Combobox(f, textvariable=self._pad_var,
                     values=["×1", "×2", "×4", "×8"], width=11,
                     state='readonly').grid(row=1, column=1, sticky='ew',
                                            padx=(4, 10), pady=1)
        ttk.Checkbutton(f, text="Remove DC", variable=self._dc_var).grid(
            row=1, column=2, columnspan=2, sticky='w', pady=1)

        ttk.Label(f, text="Magnitude").grid(row=2, column=0, sticky='w', pady=1)
        ttk.Combobox(f, textvariable=self._mag_var, values=list(self.MAGNITUDES),
                     width=11, state='readonly').grid(row=2, column=1, sticky='ew',
                                                      padx=(4, 10), pady=1)

        ttk.Label(f, text="Freq (MHz)").grid(row=3, column=0, sticky='w', pady=1)
        fr = ttk.Frame(f); fr.grid(row=3, column=1, columnspan=3, sticky='ew', padx=(4, 0))
        ttk.Entry(fr, textvariable=self._fmin_var, width=8).pack(side='left')
        ttk.Label(fr, text=" … ").pack(side='left')
        ttk.Entry(fr, textvariable=self._fmax_var, width=8).pack(side='left')

        ttk.Label(f, text="Gate (µs)").grid(row=4, column=0, sticky='w', pady=1)
        gt = ttk.Frame(f); gt.grid(row=4, column=1, columnspan=3, sticky='ew', padx=(4, 0))
        ttk.Entry(gt, textvariable=self._tmin_var, width=8).pack(side='left')
        ttk.Label(gt, text=" … ").pack(side='left')
        ttk.Entry(gt, textvariable=self._tmax_var, width=8).pack(side='left')

        ttk.Label(f, text="Blank = full range. The time gate sets the real\n"
                          "resolution bandwidth; zero-padding only interpolates\n"
                          "between bins, it does not resolve anything new.",
                  foreground='gray', font=('', 8), justify='left').grid(
            row=5, column=0, columnspan=4, sticky='w', pady=(6, 0))
        self._sync_kaiser()

    def _sync_kaiser(self):
        if self._win_var.get() == "Kaiser":
            self._kaiser_lbl.grid(row=0, column=2, sticky='w', pady=1)
            self._kaiser_box.grid(row=0, column=3, sticky='ew', pady=1)
        else:
            self._kaiser_lbl.grid_remove(); self._kaiser_box.grid_remove()

    def _window_array(self, n):
        name = self._win_var.get()
        if name == "Hann":      return np.hanning(n)
        if name == "Hamming":   return np.hamming(n)
        if name == "Blackman":  return np.blackman(n)
        if name == "Kaiser":    return np.kaiser(n, self._fnum(self._kaiser_var, 1.0, 8.6))
        return np.ones(n)

    def _spectrum(self, t, v):
        """Apply the Analysis settings to one capture.

        Returns (freq_mhz, magnitude, y_label, caption). The amplitude is
        corrected for the window's coherent gain, so switching windows changes
        the leakage and the resolution — not the height of a tone.
        """
        t = np.asarray(t, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        t0 = self._fnum(self._tmin_var, 1e-6, None) if self._tmin_var.get().strip() else None
        t1 = self._fnum(self._tmax_var, 1e-6, None) if self._tmax_var.get().strip() else None
        if t0 is not None or t1 is not None:
            m = np.ones(len(t), dtype=bool)
            if t0 is not None: m &= t >= t0
            if t1 is not None: m &= t <= t1
            if m.sum() < 8:
                raise ValueError(f"time gate keeps only {int(m.sum())} samples")
            t, v = t[m], v[m]

        if self._dc_var.get():
            v = v - v.mean()
        n = len(v)
        dt = float(t[1] - t[0]) if n > 1 else 1.0
        w = self._window_array(n)
        cg = w.mean() or 1.0            # coherent gain → amplitude stays true
        pad = int(self._pad_var.get().lstrip("×") or 1)
        n_fft = n * pad

        spec = np.fft.rfft(v * w, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, dt)
        amp = np.abs(spec) * 2.0 / (n * cg)      # peak volts per bin

        units = self._mag_var.get()
        if units == "dBV":
            mag, ylabel = 20 * np.log10(amp + 1e-12), "Amplitude (dBV)"
        elif units.startswith("dBm"):
            p_mw = (amp / np.sqrt(2.0)) ** 2 / 50.0 * 1e3
            mag, ylabel = 10 * np.log10(p_mw + 1e-12), "Power (dBm, 50 Ω)"
        else:
            mag, ylabel = amp, "Amplitude (V)"

        fmin = self._fnum(self._fmin_var, 1e6, None) if self._fmin_var.get().strip() else None
        fmax = self._fnum(self._fmax_var, 1e6, None) if self._fmax_var.get().strip() else None
        keep = np.ones(len(freqs), dtype=bool)
        if fmin is not None: keep &= freqs >= fmin
        if fmax is not None: keep &= freqs <= fmax
        if keep.sum() < 2:
            raise ValueError("frequency range keeps fewer than 2 bins")

        caption = (f"{self._win_var.get()}, pad {self._pad_var.get()}, "
                   f"{n:,} pts over {n*dt*1e6:.3f} µs → RBW {1/(n*dt)/1e3:.1f} kHz"
                   f"{', DC removed' if self._dc_var.get() else ''}")
        return freqs[keep] * 1e-6, mag[keep], ylabel, caption

    # ── Sweep block ───────────────────────────────────────────────

    def _build_sweep_block(self, parent, row, col):
        f = ttk.LabelFrame(parent, text="Frequency sweep  (CW level vs frequency)",
                           padding=8)
        f.grid(row=row, column=col, sticky='nsew', padx=(0, 4))
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._sweep_awgch_var = tk.StringVar(value="1")
        self._sweep_chan_var  = tk.StringVar(value="1")
        self._sweep_avg_var   = tk.StringVar(value="1")
        self._settle_var      = tk.StringVar(value="200")
        self._cycles_var      = tk.StringVar(value="8")
        self._sweep_freqs_var = tk.StringVar(value="100:4400:100")

        def field(r, c, label, var, values=None, width=8):
            ttk.Label(f, text=label).grid(row=r, column=c, sticky='w', pady=1)
            if values:
                w = ttk.Combobox(f, textvariable=var, values=values, width=width,
                                 state='readonly')
            else:
                w = ttk.Entry(f, textvariable=var, width=width)
            w.grid(row=r, column=c + 1, sticky='ew', padx=(4, 10), pady=1)

        field(0, 0, "AWG CH",   self._sweep_awgch_var, ["1", "2"], 5)
        field(0, 2, "Scope CH", self._sweep_chan_var, ["1", "2", "3", "4"], 5)
        field(1, 0, "Averages", self._sweep_avg_var,
              ["1", "8", "16", "64", "256", "512"], 6)
        field(1, 2, "Settle ms", self._settle_var, None, 6)
        field(2, 0, "Cycles",   self._cycles_var, None, 6)
        field(2, 2, "Freqs MHz", self._sweep_freqs_var, None, 10)

        ttk.Label(f, text="Freqs: list \"100, 500\" or range \"start:stop:step\".\n"
                          "Uses the AWG mode and that channel's amplitude above:\n"
                          "DUC steps the NCO (exact, ms); Direct pre-loads segments.",
                  foreground='gray', font=('', 8), justify='left').grid(
            row=3, column=0, columnspan=4, sticky='w', pady=(4, 2))

        btn = ttk.Button(f, text="Run sweep", command=self._run_sweep)
        btn.grid(row=4, column=0, columnspan=4, sticky='ew', pady=(2, 0))
        self._action_btns.append(btn)

    # ── Setup (microwave-chain configuration) block ───────────────

    def _build_setup_block(self, parent, row, col):
        f = ttk.LabelFrame(parent, text="Setup  (saved with every sweep)", padding=8)
        f.grid(row=row, column=col, sticky='nsew', padx=(4, 0))
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Name").grid(row=0, column=0, sticky='w', pady=1)
        self._setup_name_var = tk.StringVar(value="config_A")
        ttk.Entry(f, textvariable=self._setup_name_var, width=10).grid(
            row=0, column=1, sticky='ew', padx=(4, 0), pady=1)

        ttk.Label(f, text="Description").grid(row=1, column=0, sticky='nw', pady=1)
        self._setup_desc = tk.Text(f, width=10, height=3, wrap='word')
        self._setup_desc.grid(row=1, column=1, sticky='ew', padx=(4, 0), pady=1)
        self._setup_desc.insert('1.0', "AWG CH1 -> x24 AMC -> ...")

        self._photo_lbl = ttk.Label(f, text="Photo: none", foreground='gray',
                                    font=('Courier', 8), wraplength=250)
        self._photo_lbl.grid(row=2, column=0, columnspan=2, sticky='w', pady=(4, 1))

        pb = ttk.Frame(f)
        pb.grid(row=3, column=0, columnspan=2, sticky='ew')
        pb.columnconfigure((0, 1), weight=1)
        ttk.Button(pb, text="Browse photo…", command=self._browse_photo).grid(
            row=0, column=0, sticky='ew', padx=(0, 2))
        ttk.Button(pb, text="Show setup", command=self._show_setup_photo).grid(
            row=0, column=1, sticky='ew', padx=(2, 0))

    def _browse_photo(self):
        path = filedialog.askopenfilename(
            title="Select setup photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")])
        if path:
            self._setup_photo = path
            self._photo_lbl.config(text=f"Photo: {os.path.basename(path)}",
                                   foreground='black')

    def _show_setup_photo(self):
        if not self._setup_photo or not os.path.isfile(self._setup_photo):
            print("No setup photo selected.\n")
            return
        try:
            from PIL import Image
            img = np.asarray(Image.open(self._setup_photo))
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.imshow(img)
            ax.set_title(self._setup_name_var.get())
            ax.axis('off')
            self._fig.tight_layout()
            self._redraw()
        except Exception as e:
            print(f"Could not show setup photo: {e}\n")

    def _setup_meta(self):
        """Current setup fields as (name, description, photo_path)."""
        return (self._setup_name_var.get(),
                self._setup_desc.get('1.0', 'end').strip(),
                self._setup_photo)

    # ── Right-hand side: log + plot ───────────────────────────────

    def _build_right(self, parent):
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(right, text="Log", padding=4)
        log_frame.grid(row=0, column=0, sticky='ew', pady=(0, 4))
        self._log = scrolledtext.ScrolledText(log_frame, height=9, state='disabled',
                                              font=('Courier', 9))
        self._log.pack(fill='both', expand=True)
        sys.stdout = _LogRedirect(self._log, self.root)

        plot_frame = ttk.LabelFrame(right, text="Plot", padding=4)
        plot_frame.grid(row=1, column=0, sticky='nsew')
        self._fig = Figure(figsize=(6, 5), dpi=90)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── Helpers ───────────────────────────────────────────────────

    def _thread(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _set_busy(self, busy):
        if busy and self._live_running:
            self._live_running = False
            self._live_gen += 1
        state = 'disabled' if busy else 'normal'
        for btn in self._action_btns:
            try:
                self.root.after(0, lambda b=btn: b.config(state=state))
            except Exception:
                pass
        if not busy:
            self.root.after(0, lambda: self._retune_btn.config(
                state='normal' if self._mode() == "duc" else 'disabled'))

    def _redraw(self):
        self.root.after(0, self._canvas.draw)

    def _status(self, label, text, color):
        self.root.after(0, lambda: label.config(text=text, foreground=color))

    @staticmethod
    def _parse_sweep_freqs(text):
        """Sweep frequency field (MHz) → list of Hz, or None for DEFAULT_FREQS.

        Accepts a comma list ("100, 500, 2000") or an inclusive range
        "start:stop:step" ("100:4400:100").
        """
        text = text.strip()
        if not text:
            return None
        if ':' in text:
            parts = text.split(':')
            if len(parts) != 3:
                raise ValueError("range must be start:stop:step (MHz)")
            a, b, s = (float(x) for x in parts)
            if s <= 0 or b < a:
                raise ValueError("range needs step > 0 and stop >= start")
            n = int(np.floor((b - a) / s + 0.5)) + 1   # inclusive, fp-robust
            freqs = [(a + i * s) * 1e6 for i in range(n)]
        else:
            freqs = [float(x) * 1e6 for x in text.split(',') if x.strip()]
        if not freqs or any(f <= 0 for f in freqs):
            raise ValueError("frequencies must be positive (MHz)")
        return freqs

    def _need_awg(self):
        if self.awg is None:
            print("AWG not connected.\n")
            return False
        return True

    def _need_scope(self):
        if self.scope is None:
            print("Scope not connected.\n")
            return False
        return True

    # ── Instrument connections ────────────────────────────────────

    def _connect_awg(self):
        def connect():
            try:
                self.awg = AWG(sample_rate=self._clock_rate())
                self._sent_baseband = {}
                self._status(self._awg_lbl, "AWG: connected", "green")
                self._refresh_reference()
            except Exception as e:
                print(f"AWG connection failed: {e}\n")
                self._status(self._awg_lbl, "AWG: error", "red")
        self._thread(connect)

    def _connect_scope(self):
        def connect():
            try:
                self.scope = Scope()
                self.scope.setup(channel=int(self._chan_var.get()), n_averages=1)
                self._status(self._scope_lbl, "Scope: connected", "green")
                self._refresh_reference()
            except Exception as e:
                print(f"Scope connection failed: {e}\n")
                self._status(self._scope_lbl, "Scope: error", "red")
        self._thread(connect)

    def _stop(self):
        self._sweep_stop.set()
        self._live_running = False
        self._live_gen += 1
        self._set_busy(False)
        self._sent_baseband = {}   # next Send does a full setup, not a retune
        if self.awg is None:
            return
        self._thread(self.awg.stop)

    # ── Sweep ─────────────────────────────────────────────────────

    def _run_sweep(self):
        if not self._need_awg(): return
        self._set_busy(True)   # immediately — prevents a double-click second thread
        try:
            awg_ch = int(self._sweep_awgch_var.get())
            amp = self._fnum(self._ch[awg_ch]["amp"], 1.0)
            settle_s = self._fnum(self._settle_var, 1e-3, 0.2)
            n_cycles = int(self._fnum(self._cycles_var, 1.0, 8))
            use_duc = self._mode() == "duc"
            custom_freqs = self._parse_sweep_freqs(self._sweep_freqs_var.get())
        except Exception as e:
            print(f"Sweep param error: {e}\n")
            self._set_busy(False)
            return

        name, desc, photo = self._setup_meta()
        clock_rate = self._clock_rate()

        def sweep():
            self._sweep_stop.clear()
            records = []
            n_avg = int(self._fnum(self._sweep_avg_var, 1.0, 1))
            scope_ch = int(self._sweep_chan_var.get())
            try:
                base_freqs = custom_freqs if custom_freqs is not None else DEFAULT_FREQS
                # Drop frequencies above Nyquist — they would silently alias
                # (the buffer wraps) and log a bogus "actual" frequency.
                # DUC: the NCO reaches the 9 GS/s DAC Nyquist (4.5 GHz)
                # regardless of the direct-mode clock choice.
                dac_rate = AWG.DUC_DAC_RATE if use_duc else clock_rate
                nyq = dac_rate / 2
                freqs = [f for f in base_freqs if f <= nyq]
                skipped = [f for f in base_freqs if f > nyq]
                if skipped:
                    print(f"Skipping {len(skipped)} freqs above Nyquist "
                          f"({nyq/1e6:.0f} MHz at {dac_rate/1e9:g} GS/s): "
                          f"{', '.join(f'{f/1e6:.0f}' for f in skipped)} MHz\n")
                if not freqs:
                    print("No sweep frequencies at or below Nyquist — nothing to do.\n")
                    return

                if use_duc:
                    # One-time DUC setup; each step only retunes the NCO, so
                    # frequencies are exact (actual = target).
                    print("Setting up DUC CW (NCO-stepped)…\n")
                    t_pre0 = time.perf_counter()
                    self.awg.duc_cw_setup(freqs[0], amp, channel=awg_ch)
                    print(f"  duc_cw_setup: {(time.perf_counter()-t_pre0):.1f} s\n")
                    segments = [(f, None) for f in freqs]
                else:
                    # Pre-load all segments once; the loop then just switches.
                    print("Pre-loading waveform segments…\n")
                    self.awg.sample_rate = clock_rate
                    t_pre0 = time.perf_counter()
                    self.awg.send_sine(freqs[0], amp, channel=awg_ch, reset=True)
                    t_pre1 = time.perf_counter()
                    print(f"  send_sine: {(t_pre1-t_pre0):.1f} s\n")
                    segments = self.awg.sweep_preload(freqs, channel=awg_ch)
                    print(f"  sweep_preload: {(time.perf_counter()-t_pre1):.1f} s\n")

                if self.scope is not None:
                    self.scope.setup(channel=scope_ch, n_averages=n_avg)
                    # Start with the vertical scale matched to the commanded
                    # amplitude (~75% of the 10-div screen); measure_vpp_auto
                    # then tracks the signal as it rolls off.
                    self.scope.set_vertical(scope_ch, amp / 7.5)
                    print(f"  scope: CH{scope_ch}, {n_avg}× avg\n")

                ref_vpp = None
                freqs_plot, losses_plot = [], []

                print(f"\n{'Target (MHz)':>14}  {'Actual (MHz)':>14}  "
                      f"{'Vpp (mV)':>10}  {'Loss (dB)':>10}")
                print("-" * 56)

                for (actual, segnum), f in zip(segments, freqs):
                    if self._sweep_stop.is_set():
                        print("Sweep aborted.\n")
                        break

                    if use_duc:
                        self.awg.duc_cw_step(f, channel=awg_ch)
                    else:
                        self.awg.sweep_step(segnum, amp, channel=awg_ch)

                    if self.scope is None:
                        print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {'(no scope)':>10}")
                        continue

                    self.scope.set_timebase(actual, n_cycles=n_cycles)
                    vpp, wt, wv = self.scope.measure_vpp_auto(
                        channel=scope_ch, settle=settle_s)
                    if ref_vpp is None:
                        ref_vpp = vpp
                    loss = 20 * np.log10(vpp / ref_vpp) if vpp > 0 else float('-inf')
                    print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  "
                          f"{vpp*1e3:>10.1f}  {loss:>10.2f}")
                    freqs_plot.append(actual / 1e6)
                    losses_plot.append(loss)

                    dt_s = float(wt[1] - wt[0]) if len(wt) > 1 else 0.0
                    records.append({
                        "target_hz": f, "actual_hz": actual,
                        "vpp_v": vpp, "loss_db": loss,
                        "dt_s": dt_s, "time": wt, "voltage": wv,
                    })

                    # Live per-step plot — drawn on the Tk main thread only.
                    # Mutating self._fig from this worker races with the
                    # previous iteration's canvas.draw and aborts the sweep
                    # with a matplotlib error on the second step. Pass list
                    # copies: the worker keeps appending.
                    self.root.after(0, self._draw_sweep_step,
                                    wt, wv, actual, vpp,
                                    list(freqs_plot), list(losses_plot),
                                    dac_rate)

                print("Sweep done.\n")
            except Exception as e:
                print(f"Sweep error: {e}\n")
            finally:
                if records:
                    try:
                        params = {
                            "amplitude_vpp": amp, "awg_ch": awg_ch,
                            "awg_mode": "duc_nco" if use_duc else "direct_segments",
                            "dac_rate_hz": AWG.DUC_DAC_RATE if use_duc else clock_rate,
                            "scope_ch": scope_ch, "n_averages": n_avg,
                            "settle_ms": settle_s * 1000.0, "cycles_shown": n_cycles,
                            "sample_rate_hz": self.awg.sample_rate,
                            "n_samples": self.awg.n_samples,
                        }
                        if use_duc:
                            params["interpolation"] = f"X{AWG.DUC_INTERP}"
                            params["baseband_rate_hz"] = AWG.DUC_DAC_RATE / AWG.DUC_INTERP
                        path = sweeplog.save_sweep(name, desc, photo, params, records)
                        print(f"Saved {len(records)} points -> {path}\n")
                    except Exception as e:
                        print(f"Could not save sweep: {e}\n")
                self._set_busy(False)

        self._thread(sweep)

    def _draw_sweep_step(self, wt, wv, actual, vpp, freqs_plot, losses_plot,
                         dac_rate=None):
        """Main-thread only: per-step sweep plot (waveform + loss curve)."""
        self._fig.clear()
        ax1 = self._fig.add_subplot(211)
        ax1.plot(wt * 1e9, wv * 1e3)
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_title(f"{actual/1e6:.3f} MHz — Vpp {vpp*1e3:.1f} mV")
        ax1.grid(True)
        ax2 = self._fig.add_subplot(212)
        ax2.plot(freqs_plot, losses_plot, 'o-', label="measured")
        if dac_rate and len(freqs_plot) > 1:
            # Ideal zero-order-hold DAC roll-off, sinc(f/Fs), normalized to the
            # same reference frequency as the measured loss (the first point).
            fgrid = np.linspace(min(freqs_plot), max(freqs_plot), 200)
            theory = (20 * np.log10(np.sinc(fgrid * 1e6 / dac_rate))
                      - 20 * np.log10(np.sinc(freqs_plot[0] * 1e6 / dac_rate)))
            ax2.plot(fgrid, theory, '--', color='gray',
                     label=f"DAC sinc @ {dac_rate/1e9:g} GS/s")
            ax2.legend(fontsize=8)
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Loss (dB)")
        ax2.grid(True)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ── Waveform capture ──────────────────────────────────────────

    def _apply_timebase(self):
        """Send ns/div to the scope if the field is filled in."""
        val = self._nsdiv_var.get().strip()
        if val and self.scope:
            try:
                self.scope.set_timebase_direct(float(val) * 1e-9)
            except Exception as e:
                print(f"Timebase error: {e}\n")

    def _scope_max_pts(self):
        return int(self._fnum(self._maxpts_var, 1.0, 10000))

    def _draw_capture(self, t, v, spec):
        """Main thread only — see _draw_sweep_step."""
        self._fig.clear()
        ax1 = self._fig.add_subplot(211)
        ax1.plot(t * 1e6, v * 1e3)
        ax1.set_xlabel("Time (µs)")
        ax1.set_ylabel("Voltage (mV)")
        ax1.grid(True)
        ax2 = self._fig.add_subplot(212)
        if spec is None:
            ax2.text(0.5, 0.5, "no spectrum", ha='center', va='center',
                     transform=ax2.transAxes, color='red')
        else:
            freqs, mag, ylabel, caption = spec
            ax2.plot(freqs, mag)
            ax2.set_xlabel("Frequency (MHz)")
            ax2.set_ylabel(ylabel)
            ax2.set_title(caption, fontsize=8)
            ax2.grid(True)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _plot_waveform(self):
        if not self._need_scope(): return
        ch = int(self._chan_var.get())

        def capture():
            self._set_busy(True)
            try:
                self._apply_timebase()
                t, v = self.scope.get_waveform(channel=ch,
                                               max_points=self._scope_max_pts())
                try:
                    spec = self._spectrum(t, v)
                except Exception as e:
                    print(f"Spectrum error: {e}\n")
                    spec = None
                self.root.after(0, self._draw_capture, t, v, spec)
            except Exception as e:
                print(f"Capture error: {e}\n")
            finally:
                self._set_busy(False)

        self._thread(capture)

    # ── Live view ─────────────────────────────────────────────────

    def _open_live_view(self):
        if not self._need_scope(): return
        if self._live_win is not None and self._live_win.winfo_exists():
            self._live_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Scope — Live View")
        win.geometry("900x620")
        self._live_win = win

        ctrl = ttk.Frame(win, padding=4)
        ctrl.pack(fill='x', side='bottom')
        self._live_start_btn = ttk.Button(ctrl, text="Start", command=self._toggle_live)
        self._live_start_btn.pack(side='left', padx=4)
        ttk.Label(ctrl, text="Uses the Analysis settings; change them and the next "
                             "frame follows.", foreground='gray').pack(side='left', padx=8)
        self._action_btns.append(self._live_start_btn)

        self._live_fig = Figure(figsize=(8, 5), dpi=90)
        self._live_canvas = FigureCanvasTkAgg(self._live_fig, master=win)
        self._live_canvas.get_tk_widget().pack(fill='both', expand=True)

        def on_close():
            self._live_running = False
            self._live_gen += 1
            if self._live_start_btn in self._action_btns:
                self._action_btns.remove(self._live_start_btn)
            win.destroy()
            self._live_win = None
        win.protocol("WM_DELETE_WINDOW", on_close)

    def _toggle_live(self):
        if not self._need_scope(): return
        if self._live_running:
            self._live_running = False
            self._live_gen += 1
            self.root.after(0, lambda: self._live_start_btn.config(text="Start"))
        else:
            self._live_running = True
            self._live_gen += 1
            self.root.after(0, lambda: self._live_start_btn.config(text="Stop"))
            gen = self._live_gen
            self._thread(lambda: self._live_loop(gen))

    def _live_draw(self, gen, t, v, spec):
        """Runs on the Tk main thread — the only place the live figure is
        touched. Mutating it from the worker races with canvas draws."""
        if not (self._live_running and self._live_gen == gen):
            return
        if self._live_win is None or not self._live_win.winfo_exists():
            return
        self._live_fig.clear()
        ax1 = self._live_fig.add_subplot(211)
        ax1.plot(t * 1e6, v * 1e3)
        ax1.set_xlabel("Time (µs)"); ax1.set_ylabel("Voltage (mV)"); ax1.grid(True)

        ax2 = self._live_fig.add_subplot(212)
        if spec is not None:
            freqs, mag, ylabel, caption = spec
            ax2.plot(freqs, mag)
            ax2.set_xlabel("Frequency (MHz)"); ax2.set_ylabel(ylabel)
            ax2.set_title(caption, fontsize=8)
        ax2.grid(True)

        self._live_fig.tight_layout()
        self._live_canvas.draw_idle()

    def _live_loop(self, gen):
        ch = int(self._chan_var.get())
        self._apply_timebase()
        max_pts = self._scope_max_pts()
        warned = False
        while self._live_running and self._live_gen == gen:
            try:
                t, v = self.scope.get_waveform(channel=ch, max_points=max_pts)
                try:
                    spec = self._spectrum(t, v)
                except Exception as e:
                    spec = None
                    if not warned:
                        print(f"Spectrum error: {e}\n")
                        warned = True
                self.root.after(0, self._live_draw, gen, t, v, spec)
            except Exception as e:
                print(f"Live view error: {e}\n")

            time.sleep(0.05)

        if self._live_gen == gen:
            self._live_running = False
            if self._live_win is not None and self._live_win.winfo_exists():
                self.root.after(0, lambda: self._live_start_btn.config(text="Start"))


if __name__ == "__main__":
    root = tk.Tk()
    LabGUI(root)
    root.mainloop()
