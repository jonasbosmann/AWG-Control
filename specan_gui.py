import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import queue
import time

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from specan import SpecAn, find_peaks
import specanlog


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


class SpecAnGUI:
    TRACE_MODES = {"Normal": "NORM", "Max Hold": "MAXH",
                   "Min Hold": "MINH", "Average": "AVER"}

    def __init__(self, root):
        self.root = root
        self.root.title("Spectrum Analyzer Control")
        self.root.geometry("950x700")
        self.sa = None
        self._live_running = False
        self._live_gen = 0
        self._action_btns = []
        self._last_freqs = None
        self._last_amps = None
        self._overlays = []  # list of (label, freqs_hz, amps_dbm)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self._live_running = False
        if self.sa:
            try: self.sa.close()
            except Exception: pass
        self.root.destroy()

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        conn = ttk.LabelFrame(self.root, text="Instrument", padding=5)
        conn.pack(fill='x', padx=8, pady=4)
        b = ttk.Button(conn, text="Connect", command=self._connect)
        b.pack(side='left', padx=4)
        self._action_btns.append(b)
        self._status_lbl = ttk.Label(conn, text="disconnected", foreground='red')
        self._status_lbl.pack(side='left', padx=(0, 16))

        params = ttk.LabelFrame(self.root, text="Sweep Setup", padding=5)
        params.pack(fill='x', padx=8, pady=4)

        self._center_var = tk.StringVar(value="13.0")
        self._span_var   = tk.StringVar(value="200")
        self._rlev_var   = tk.StringVar(value="0")
        self._rbw_var    = tk.StringVar(value="1000")
        self._vbw_var    = tk.StringVar(value="1000")
        self._att_var    = tk.StringVar(value="10")
        self._rbw_auto   = tk.BooleanVar(value=True)
        self._vbw_auto   = tk.BooleanVar(value=True)
        self._att_auto   = tk.BooleanVar(value=False)
        self._trace_var  = tk.StringVar(value="Normal")
        self._navg_var   = tk.StringVar(value="10")

        row = 0
        ttk.Label(params, text="Center (GHz)").grid(row=row, column=0, sticky='w', padx=4, pady=2)
        ttk.Entry(params, textvariable=self._center_var, width=10).grid(row=row, column=1, padx=4)
        ttk.Label(params, text="Span (MHz)").grid(row=row, column=2, sticky='w', padx=4)
        ttk.Entry(params, textvariable=self._span_var, width=10).grid(row=row, column=3, padx=4)
        ttk.Label(params, text="Ref Level (dBm)").grid(row=row, column=4, sticky='w', padx=4)
        ttk.Entry(params, textvariable=self._rlev_var, width=8).grid(row=row, column=5, padx=4)

        row += 1
        ttk.Label(params, text="RBW (kHz)").grid(row=row, column=0, sticky='w', padx=4, pady=2)
        self._rbw_entry = ttk.Entry(params, textvariable=self._rbw_var, width=10)
        self._rbw_entry.grid(row=row, column=1, padx=4)
        ttk.Checkbutton(params, text="Auto", variable=self._rbw_auto,
                         command=lambda: self._toggle_entry(self._rbw_entry, self._rbw_auto)
                         ).grid(row=row, column=1, sticky='e')

        ttk.Label(params, text="VBW (kHz)").grid(row=row, column=2, sticky='w', padx=4)
        self._vbw_entry = ttk.Entry(params, textvariable=self._vbw_var, width=10)
        self._vbw_entry.grid(row=row, column=3, padx=4)
        ttk.Checkbutton(params, text="Auto", variable=self._vbw_auto,
                         command=lambda: self._toggle_entry(self._vbw_entry, self._vbw_auto)
                         ).grid(row=row, column=3, sticky='e')

        ttk.Label(params, text="Attenuation (dB)").grid(row=row, column=4, sticky='w', padx=4)
        self._att_entry = ttk.Entry(params, textvariable=self._att_var, width=8)
        self._att_entry.grid(row=row, column=5, padx=4)
        ttk.Checkbutton(params, text="Auto", variable=self._att_auto,
                         command=lambda: self._toggle_entry(self._att_entry, self._att_auto)
                         ).grid(row=row, column=5, sticky='e')
        self._toggle_entry(self._rbw_entry, self._rbw_auto)
        self._toggle_entry(self._vbw_entry, self._vbw_auto)
        self._toggle_entry(self._att_entry, self._att_auto)

        row += 1
        ttk.Label(params, text="Trace Mode").grid(row=row, column=0, sticky='w', padx=4, pady=2)
        trace_combo = ttk.Combobox(params, textvariable=self._trace_var, values=list(self.TRACE_MODES),
                                    state='readonly', width=10)
        trace_combo.grid(row=row, column=1, padx=4, sticky='w')
        trace_combo.bind('<<ComboboxSelected>>', lambda e: self._update_sweep_mode_state())
        ttk.Label(params, text="Averages").grid(row=row, column=2, sticky='w', padx=4)
        self._navg_entry = ttk.Entry(params, textvariable=self._navg_var, width=6)
        self._navg_entry.grid(row=row, column=3, padx=4, sticky='w')
        b = ttk.Button(params, text="Apply Setup", command=self._apply_setup)
        b.grid(row=row, column=4, padx=4)
        self._action_btns.append(b)
        b = ttk.Button(params, text="Clear Trace", command=self._clear_trace)
        b.grid(row=row, column=5, padx=4)
        self._action_btns.append(b)

        sweep = ttk.LabelFrame(self.root, text="Acquisition", padding=5)
        sweep.pack(fill='x', padx=8, pady=4)
        self._single_btn = ttk.Button(sweep, text="Single Sweep", command=self._single_sweep)
        self._single_btn.pack(side='left', padx=4)
        self._action_btns.append(self._single_btn)
        self._live_btn = ttk.Button(sweep, text="Live View: OFF", command=self._toggle_live)
        self._live_btn.pack(side='left', padx=4)
        self._action_btns.append(self._live_btn)
        b = ttk.Button(sweep, text="Auto Spur Hunt", command=self._auto_spur_hunt)
        b.pack(side='left', padx=4)
        self._action_btns.append(b)
        ttk.Label(sweep, text="(Single Sweep only works in Normal trace mode — "
                               "use Live View for Max/Min Hold and Average. "
                               "Spur Hunt uses current Center/Span as the overview "
                               "(RBW/VBW always Auto), Averages as refine depth, "
                               "Label as the save prefix.)"
                  ).pack(side='left', padx=8)

        save = ttk.LabelFrame(self.root, text="Save / Compare", padding=5)
        save.pack(fill='x', padx=8, pady=4)
        self._label_var = tk.StringVar(value="")
        self._notes_var = tk.StringVar(value="")
        ttk.Label(save, text="Label").grid(row=0, column=0, sticky='w', padx=4, pady=2)
        ttk.Entry(save, textvariable=self._label_var, width=20).grid(row=0, column=1, padx=4)
        ttk.Label(save, text="Notes").grid(row=0, column=2, sticky='w', padx=4)
        ttk.Entry(save, textvariable=self._notes_var, width=40).grid(
            row=0, column=3, padx=4, sticky='we')
        save.columnconfigure(3, weight=1)
        b = ttk.Button(save, text="Save Trace", command=self._save_trace)
        b.grid(row=0, column=4, padx=4)
        self._action_btns.append(b)
        b = ttk.Button(save, text="Load & Overlay...", command=self._load_overlay)
        b.grid(row=0, column=5, padx=4)
        self._action_btns.append(b)
        b = ttk.Button(save, text="Clear Overlays", command=self._clear_overlays)
        b.grid(row=0, column=6, padx=4)
        self._action_btns.append(b)

        # Plot
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill='both', expand=True, padx=8, pady=4)
        self._fig = Figure(figsize=(8, 4), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

        # Log
        log = ttk.LabelFrame(self.root, text="Log", padding=5)
        log.pack(fill='both', expand=False, padx=8, pady=4)
        log_text = scrolledtext.ScrolledText(log, height=6, state='disabled')
        log_text.pack(fill='both', expand=True)
        import sys
        sys.stdout = _LogRedirect(log_text, self.root)

        self._update_sweep_mode_state()

    def _toggle_entry(self, entry, auto_var):
        entry.configure(state='disabled' if auto_var.get() else 'normal')

    def _update_sweep_mode_state(self):
        """Single Sweep hangs the instrument in MAXH/MINH/AVER trace mode
        (verified against the hardware) — grey it out outside Normal.
        Averages only affects AVER trace mode — grey it out otherwise."""
        mode = self.TRACE_MODES[self._trace_var.get()]
        self._single_btn.configure(state='normal' if mode == "WRIT" else 'disabled')
        self._navg_entry.configure(state='normal' if mode == "AVER" else 'disabled')

    # ── Actions ────────────────────────────────────────────────────

    def _set_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        for b in self._action_btns:
            if b is not self._live_btn:
                b.configure(state=state)

    def _thread(self, target):
        threading.Thread(target=target, daemon=True).start()

    def _connect(self):
        def work():
            self.root.after(0, lambda: self._set_busy(True))
            try:
                self.sa = SpecAn()
                self.root.after(0, lambda: self._status_lbl.configure(
                    text="connected", foreground='green'))
                self._apply_setup()
            except Exception as e:
                print(f"Connect error: {e}\n")
            finally:
                self.root.after(0, lambda: self._set_busy(False))
        self._thread(work)

    def _need_sa(self):
        if self.sa is None:
            print("Connect the instrument first.\n")
            return False
        return True

    def _apply_setup(self):
        if not self._need_sa(): return
        try:
            center_hz = float(self._center_var.get()) * 1e9
            span_hz   = float(self._span_var.get()) * 1e6
            self.sa.set_freq(center_hz, span_hz)
            self.sa.set_ref_level(float(self._rlev_var.get()))
            self.sa.set_rbw(None if self._rbw_auto.get() else float(self._rbw_var.get()) * 1e3)
            self.sa.set_vbw(None if self._vbw_auto.get() else float(self._vbw_var.get()) * 1e3)
            self.sa.set_attenuation(None if self._att_auto.get() else float(self._att_var.get()))
            self.sa.set_trace_mode(self.TRACE_MODES[self._trace_var.get()])
            self.sa.set_average_count(int(self._navg_var.get()))
            self.root.after(0, self._update_sweep_mode_state)
            print("Setup applied.\n")
        except Exception as e:
            print(f"Apply setup error: {e}\n")

    def _clear_trace(self):
        if not self._need_sa(): return
        self.sa.clear_trace()
        print("Trace cleared.\n")

    def _draw(self, freqs, amps):
        self._last_freqs = freqs
        self._last_amps = amps
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.plot(freqs * 1e-6, amps, label="live")
        for label, f, a in self._overlays:
            ax.plot(f * 1e-6, a, '--', alpha=0.7, label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (dBm)")
        ax.grid(True)
        if self._overlays:
            ax.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

    def _current_settings(self):
        return {
            "ref_level_dbm":   float(self._rlev_var.get()) if self._rlev_var.get() else None,
            "rbw_hz":          "AUTO" if self._rbw_auto.get() else float(self._rbw_var.get()) * 1e3,
            "vbw_hz":          "AUTO" if self._vbw_auto.get() else float(self._vbw_var.get()) * 1e3,
            "attenuation_db":  "AUTO" if self._att_auto.get() else float(self._att_var.get()),
            "trace_mode":      self.TRACE_MODES[self._trace_var.get()],
            "avg_count":       int(self._navg_var.get()),
        }

    def _save_trace(self):
        if self._last_freqs is None:
            print("No trace to save yet — run a sweep or Live View first.\n")
            return
        label = self._label_var.get().strip() or "trace"
        notes = self._notes_var.get().strip()
        try:
            path = specanlog.save_trace(label, notes, self._current_settings(),
                                         self._last_freqs, self._last_amps)
            print(f"Saved trace: {path}\n")
        except Exception as e:
            print(f"Save trace error: {e}\n")

    def _load_overlay(self):
        os.makedirs(specanlog.TRACE_DIR, exist_ok=True)
        path = filedialog.askopenfilename(initialdir=specanlog.TRACE_DIR,
                                           filetypes=[("Spectrum trace", "*.json")])
        if not path:
            return
        try:
            doc = specanlog.load_trace(path)
            self._overlays.append((doc.get("label") or os.path.basename(path),
                                    doc["freqs_hz"], doc["amps_dbm"]))
            base_freqs = self._last_freqs if self._last_freqs is not None else doc["freqs_hz"]
            base_amps = self._last_amps if self._last_amps is not None else doc["amps_dbm"]
            self._draw(base_freqs, base_amps)
            print(f"Loaded overlay: {path}\n")
        except Exception as e:
            print(f"Load trace error: {e}\n")

    def _clear_overlays(self):
        self._overlays = []
        if self._last_freqs is not None:
            self._draw(self._last_freqs, self._last_amps)

    def _single_sweep(self):
        if not self._need_sa(): return
        def work():
            self.root.after(0, lambda: self._set_busy(True))
            try:
                freqs, amps = self.sa.sweep_once()
                self.root.after(0, self._draw, freqs, amps)
            except Exception as e:
                print(f"Sweep error: {e}\n")
            finally:
                self.root.after(0, lambda: self._set_busy(False))
        self._thread(work)

    def _toggle_live(self):
        if not self._live_running:
            if not self._need_sa(): return
            self._live_running = True
            self._live_gen += 1
            gen = self._live_gen
            self.sa.start_continuous()
            self._live_btn.configure(text="Live View: ON")
            self._thread(lambda: self._live_loop(gen))
        else:
            self._live_running = False
            self._live_btn.configure(text="Live View: OFF")

    def _live_loop(self, gen):
        while self._live_running and self._live_gen == gen:
            try:
                freqs, amps = self.sa.get_trace()
                self.root.after(0, self._live_draw, gen, freqs, amps)
            except Exception as e:
                print(f"Live view error: {e}\n")
                break
            time.sleep(0.2)

    def _live_draw(self, gen, freqs, amps):
        if not (self._live_running and self._live_gen == gen):
            return
        self._draw(freqs, amps)

    # ── Auto spur hunt ────────────────────────────────────────────
    # Overview sweep (current Center/Span/RBW) -> find_peaks() -> for each
    # candidate, zoom in with a narrower RBW + Average trace and save it.
    # Only catches spurs sitting above the OVERVIEW's own displayed noise
    # floor — a spur below that floor (e.g. found earlier by hand with a much
    # narrower RBW than the overview uses) needs a second run at a narrower
    # overview RBW to become visible in the first place.

    def _auto_spur_hunt(self):
        if not self._need_sa(): return
        self._thread(self._run_spur_hunt)

    def _run_spur_hunt(self):
        self.root.after(0, lambda: self._set_busy(True))
        if self._live_running:
            print("Stopping Live View so the spur hunt has exclusive use of the instrument.\n")
            self._live_running = False
            self.root.after(0, lambda: self._live_btn.configure(text="Live View: OFF"))
            time.sleep(0.3)  # let the live-view loop's current get_trace() finish
        try:
            carrier_hz = float(self._center_var.get()) * 1e9
            overview_span_hz = float(self._span_var.get()) * 1e6
            atten = None if self._att_auto.get() else float(self._att_var.get())
            label_prefix = self._label_var.get().strip() or "spur_hunt"
            avg_count = int(self._navg_var.get())
            run_dir = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{label_prefix}"
            print(f"Saving this run's traces under specan_traces/{run_dir}/\n")

            # Neither a manually-typed RBW nor the instrument's own Auto coupling
            # is right here: manual RBW can be narrower than span/points (the
            # earlier undersampling bug), while Auto is tuned for "look sane
            # quickly," not for pulling weak spurs out from under a strong
            # carrier's noise floor. So: fix a high point count first, then
            # derive a deliberately narrow RBW from THAT span/point spacing —
            # narrow enough to lower the noise floor for sensitivity, provably
            # wide enough relative to point spacing to never undersample.
            overview_points = 8001
            self.sa.set_freq(carrier_hz, overview_span_hz)
            self.sa.set_points(overview_points)
            target_spacing = overview_span_hz / (overview_points - 1)
            target_rbw = max(target_spacing * 3, 1e3)
            print(f"Spur hunt: overview center={carrier_hz/1e9:.6f} GHz "
                  f"span={overview_span_hz/1e6:.1f} MHz, {overview_points} points "
                  f"(spacing {target_spacing/1e3:.2f} kHz)\n")
            print(f"  requesting RBW/VBW={target_rbw/1e3:.1f} kHz — narrower than "
                  f"Auto coupling would pick, for sensitivity to weak spurs, while "
                  f"staying oversampled relative to point spacing\n")
            self.sa.set_rbw(target_rbw)
            self.sa.set_vbw(target_rbw)
            self.sa.set_attenuation(atten)
            self.sa.set_trace_mode("MAXH")
            self.sa.clear_trace()
            self.sa.start_continuous()
            time.sleep(0.3)  # let the instrument settle before reading back the
                              # applied RBW / sweep time — querying immediately
                              # can return a stale value from the prior setting
            sweep_s = self.sa.get_sweep_time()
            actual_rbw = self.sa.get_rbw()
            wait_s = max(1.5, sweep_s * 3 + 1.0)
            print(f"  instrument applied RBW {actual_rbw/1e3:.1f} kHz, "
                  f"sweep time {sweep_s:.3f} s, waiting {wait_s:.1f} s for MaxHold to build up\n")
            if wait_s > 20:
                print("  (that's slow because span is wide relative to this RBW — "
                      "expected, not stuck)\n")
            time.sleep(wait_s)
            freqs, amps = self.sa.get_trace()
            self.root.after(0, self._draw, freqs, amps)

            overview_settings = {
                "ref_level_dbm":  float(self._rlev_var.get()) if self._rlev_var.get() else None,
                "rbw_hz":         actual_rbw,
                "vbw_hz":         actual_rbw,
                "attenuation_db": atten if atten is not None else "AUTO",
                "trace_mode":     "MAXH",
                "span_hz":        overview_span_hz,
                "center_hz":      carrier_hz,
                "points":         overview_points,
            }
            try:
                path = specanlog.save_trace(f"{label_prefix}_overview", "auto spur hunt overview",
                                             overview_settings, freqs, amps, subdir=run_dir)
                print(f"  saved overview: {path}\n")
            except Exception as e:
                print(f"  save overview error: {e}\n")

            point_spacing_hz = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
            peaks = find_peaks(freqs, amps, exclude_hz=carrier_hz,
                                exclude_width_hz=max(actual_rbw * 3, 2e6),
                                min_prominence_db=10.0,
                                min_spacing_hz=max(actual_rbw, point_spacing_hz * 3))
            print(f"Overview found {len(peaks)} candidate peak(s) above the "
                  f"{actual_rbw/1e3:.0f} kHz-RBW noise floor "
                  f"(point spacing {point_spacing_hz/1e3:.1f} kHz).\n")

            carrier_f, carrier_a = self._refine_peak(
                carrier_hz, carrier_hz, point_spacing_hz, avg_count, label_prefix, "main_peak",
                run_dir)

            results = []
            for i, (pf, pa) in enumerate(peaks, 1):
                print(f"[{i}/{len(peaks)}] refining candidate at {pf/1e9:.6f} GHz "
                      f"(overview level {pa:.1f} dBm)\n")
                rf, ra = self._refine_peak(
                    pf, carrier_hz, point_spacing_hz, avg_count, label_prefix, "side_peak",
                    run_dir)
                results.append((rf, ra))

            print(f"\nSpur hunt summary (relative to main peak "
                  f"{carrier_f/1e9:.6f} GHz @ {carrier_a:.2f} dBm):\n")
            for rf, ra in sorted(results, key=lambda r: r[0]):
                offset_hz = rf - carrier_f
                dbc = ra - carrier_a
                print(f"  {rf/1e9:.6f} GHz  offset {offset_hz/1e6:+9.4f} MHz  "
                      f"{ra:7.2f} dBm  {dbc:7.2f} dBc\n")
        except Exception as e:
            print(f"Spur hunt error: {e}\n")
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _refine_peak(self, coarse_freq_hz, carrier_hz, point_spacing_hz, avg_count,
                      label_prefix, kind, run_dir):
        """Zoom into one candidate with a narrower RBW + Average trace, save
        it via specanlog, and return the refined (freq_hz, amp_dbm).

        Span is sized from the OVERVIEW's point spacing, not its RBW: if RBW
        was set narrower than span/points (easy to do, and what bit us the
        first time), the overview is undersampled and its coarse peak
        frequency can be off by roughly half a point spacing — the refine
        window has to be wide enough to still contain the true peak despite
        that, regardless of how narrow the overview's RBW happened to be.
        """
        offset_hz = abs(coarse_freq_hz - carrier_hz)
        span = max(point_spacing_hz * 8, 300e3)
        rbw = max(offset_hz / 8, 1e3) if offset_hz > 0 else span / 40
        rbw = min(rbw, span / 20)

        self.sa.set_freq(coarse_freq_hz, span)
        self.sa.set_rbw(rbw)
        self.sa.set_vbw(rbw)
        self.sa.set_trace_mode("AVER")
        self.sa.set_average_count(avg_count)
        self.sa.clear_trace()
        self.sa.start_continuous()
        time.sleep(0.3)  # let the instrument recompute sweep time for the new RBW/span
                          # before we ask — see identical note in _run_spur_hunt
        sweep_s = self.sa.get_sweep_time()
        wait_s = max(1.5, sweep_s * (avg_count + 3) + 1.0)
        print(f"  span={span/1e6:.3f} MHz rbw={rbw/1e3:.1f} kHz sweep time {sweep_s:.3f} s, "
              f"waiting {wait_s:.1f} s for {avg_count}-sweep average\n")
        if wait_s > 20:
            print("  (that's slow because RBW is narrow relative to span — "
                  "this is expected, not stuck)\n")
        time.sleep(wait_s)
        freqs, amps = self.sa.get_trace()
        self.root.after(0, self._draw, freqs, amps)

        i_max = int(np.argmax(amps))
        peak_f, peak_a = float(freqs[i_max]), float(amps[i_max])
        label = f"{label_prefix}_{kind}_{peak_f/1e9:.5f}".replace('.', 'p')
        settings = {
            "ref_level_dbm":  float(self._rlev_var.get()) if self._rlev_var.get() else None,
            "rbw_hz":         rbw,
            "vbw_hz":         rbw,
            "attenuation_db": None if self._att_auto.get() else float(self._att_var.get()),
            "trace_mode":     "AVER",
            "avg_count":      avg_count,
            "span_hz":        span,
            "center_hz":      coarse_freq_hz,
        }
        try:
            path = specanlog.save_trace(label, "auto spur hunt", settings, freqs, amps,
                                         subdir=run_dir)
            print(f"  saved: {path}\n")
        except Exception as e:
            print(f"  save error: {e}\n")
        return peak_f, peak_a


if __name__ == "__main__":
    root = tk.Tk()
    app = SpecAnGUI(root)
    root.mainloop()
