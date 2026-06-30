import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import sys
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from awg import AWG, SAMPLE_RATE_SINGLE, SAMPLE_RATE_DUAL
from scope import Scope
from measure import DEFAULT_FREQS


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
    def __init__(self, root):
        self.root = root
        self.root.title("AWG / Scope Control")
        self.root.geometry("1100x780")
        self.awg   = None
        self.scope = None
        self._live_running = False
        self._live_gen     = 0
        self._sweep_stop   = threading.Event()
        self._action_btns  = []   # all buttons disabled while busy (except Stop)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self._live_running = False
        self._sweep_stop.set()
        if self.scope:
            try: self.scope.close()
            except Exception: pass
        if self.awg:
            try: self.awg.close()
            except Exception: pass
        self.root.destroy()

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        conn = ttk.LabelFrame(self.root, text="Instruments", padding=5)
        conn.pack(fill='x', padx=8, pady=4)
        ttk.Button(conn, text="Connect AWG",   command=self._connect_awg).pack(side='left', padx=4)
        self._awg_lbl = ttk.Label(conn, text="AWG: disconnected", foreground='red')
        self._awg_lbl.pack(side='left', padx=(0, 16))
        ttk.Button(conn, text="Connect Scope", command=self._connect_scope).pack(side='left', padx=4)
        self._scope_lbl = ttk.Label(conn, text="Scope: disconnected", foreground='red')
        self._scope_lbl.pack(side='left', padx=(0, 16))

        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=8, pady=4)
        self._build_controls(main)
        self._build_right(main)

    def _build_controls(self, parent):
        outer = ttk.Frame(parent, width=310)
        outer.pack(side='left', fill='y', padx=(0, 8))
        outer.pack_propagate(False)

        vsb = ttk.Scrollbar(outer, orient='vertical')
        vsb.pack(side='right', fill='y')
        canvas = tk.Canvas(outer, yscrollcommand=vsb.set, highlightthickness=0)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.config(command=canvas.yview)

        ctrl = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=ctrl, anchor='nw')

        ctrl.bind('<Configure>', lambda _: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), 'units'))

        self._build_std_waveform(ctrl)
        self._build_chirp_panel(ctrl)
        self._build_scope_panel(ctrl)
        self._build_sweep_panel(ctrl)

    # ── Standard Waveform panel ───────────────────────────────────

    def _build_std_waveform(self, parent):
        f = ttk.LabelFrame(parent, text="Standard Waveform", padding=8)
        f.pack(fill='x', pady=(0, 6))
        f.columnconfigure(1, weight=1)
        f.columnconfigure(2, weight=1)

        ttk.Label(f, text="Sample rate").grid(row=0, column=0, columnspan=3, sticky='w')
        self._rate_var = tk.StringVar(value="9 GS/s  (CH1)")
        ttk.Combobox(f, textvariable=self._rate_var,
                     values=["9 GS/s  (CH1)", "2.5 GS/s  (CH1+2)"],
                     width=14, state='readonly').grid(row=1, column=0, columnspan=3,
                                                       sticky='ew', pady=(0, 6))
        self._rate_var.trace_add('write', lambda *_: self._on_rate_change())

        ttk.Label(f, text="CH1", font=('', 9, 'bold')).grid(row=2, column=1, pady=1)
        ch2_hdr = ttk.Label(f, text="CH2", font=('', 9, 'bold'))
        ch2_hdr.grid(row=2, column=2, pady=1)
        self._ch2_std_widgets = [ch2_hdr]

        rows = [
            ("Waveform",   "wave1", "wave2", "Sine", "Sine", "combo"),
            ("Freq (MHz)", "freq1", "freq2", "100",  "100",  "entry"),
            ("Amp  (Vpp)", "amp1",  "amp2",  "0.5",  "0.5",  "entry"),
        ]
        for gr, (lbl, a1, a2, d1, d2, kind) in enumerate(rows, start=3):
            ttk.Label(f, text=lbl).grid(row=gr, column=0, sticky='w', pady=1)
            v1, v2 = tk.StringVar(value=d1), tk.StringVar(value=d2)
            setattr(self, f'_{a1}_var', v1)
            setattr(self, f'_{a2}_var', v2)
            if kind == "combo":
                w1 = ttk.Combobox(f, textvariable=v1, values=["Sine", "Square", "Ramp"],
                                   width=6, state='readonly')
                w2 = ttk.Combobox(f, textvariable=v2, values=["Sine", "Square", "Ramp"],
                                   width=6, state='readonly')
            else:
                w1 = ttk.Entry(f, textvariable=v1, width=7)
                w2 = ttk.Entry(f, textvariable=v2, width=7)
            w1.grid(row=gr, column=1, sticky='ew', padx=2, pady=1)
            w2.grid(row=gr, column=2, sticky='ew', padx=2, pady=1)
            self._ch2_std_widgets.append(w2)

        btn1 = ttk.Button(f, text="Send CH1", command=lambda: self._run_std_wave(1))
        btn1.grid(row=gr+1, column=0, columnspan=2, sticky='ew', pady=(6, 2), padx=(0, 2))
        btn2 = ttk.Button(f, text="Send CH2", command=lambda: self._run_std_wave(2))
        btn2.grid(row=gr+1, column=2, sticky='ew', pady=(6, 2))
        self._action_btns.extend([btn1, btn2])
        self._ch2_std_widgets.append(btn2)

        self._on_rate_change()

    def _on_rate_change(self):
        dual = "2.5" in self._rate_var.get()
        for w in self._ch2_std_widgets:
            new_state = ('readonly' if isinstance(w, ttk.Combobox) else 'normal') if dual else 'disabled'
            try:
                w.config(state=new_state)
            except tk.TclError:
                pass

    # ── Chirp + CW panel ──────────────────────────────────────────

    def _build_chirp_panel(self, parent):
        f = ttk.LabelFrame(parent, text="Chirp + CW  (CP-FTMW)", padding=8)
        f.pack(fill='x', pady=(0, 6))

        chirp_fields = [
            ("Chirp start (MHz)", "c_start",   "10"),
            ("Chirp stop  (MHz)", "c_stop",    "500"),
            ("Chirp dur   (µs)",  "c_dur",     "1.0"),
            ("Dead time   (µs)",  "c_dead",    "0.1"),
            ("CH2 LO      (MHz)", "c_lo",      "100"),
            ("Detect win  (µs)",  "c_detect",  "10.0"),
            ("Amplitude   (Vpp)", "c_amp",     "0.5"),
        ]
        f.columnconfigure(1, weight=1)
        for row, (lbl, attr, default) in enumerate(chirp_fields):
            ttk.Label(f, text=lbl).grid(row=row, column=0, sticky='w', pady=1)
            var = tk.StringVar(value=default)
            setattr(self, f"_{attr}_var", var)
            e = ttk.Entry(f, textvariable=var, width=8)
            e.grid(row=row, column=1, sticky='ew', padx=(4, 0), pady=1)
            var.trace_add('write', lambda *_: self._update_chirp_info())

        r = len(chirp_fields)
        ttk.Separator(f, orient='horizontal').grid(row=r, column=0, columnspan=2,
                                                    sticky='ew', pady=4)
        self._ch1_info = ttk.Label(f, text="", foreground='gray', font=('Courier', 8))
        self._ch1_info.grid(row=r+1, column=0, columnspan=2, sticky='w')
        self._ch2_info = ttk.Label(f, text="", foreground='gray', font=('Courier', 8))
        self._ch2_info.grid(row=r+2, column=0, columnspan=2, sticky='w')

        btn = ttk.Button(f, text="Generate Chirp+CW", command=self._run_chirp)
        btn.grid(row=r+3, column=0, columnspan=2, sticky='ew', pady=(6, 2))
        self._action_btns.append(btn)

        self._update_chirp_info()

    def _update_chirp_info(self):
        try:
            rate      = SAMPLE_RATE_DUAL
            chirp_us  = float(self._c_dur_var.get())
            dead_us   = float(self._c_dead_var.get())
            detect_us = float(self._c_detect_var.get())

            n_chirp  = round(chirp_us  * 1e-6 * rate)
            n_dead   = round(dead_us   * 1e-6 * rate)
            n_detect = round(detect_us * 1e-6 * rate)
            n_total  = max(int(np.ceil((n_chirp + n_dead + n_detect) / 64)) * 64, 128)
            t_dead_actual = (n_total - n_chirp - n_detect) / rate * 1e6

            self._ch1_info.config(
                text=f"CH1  {n_chirp:,} samp  active {n_chirp/rate*1e6:.3f} µs  "
                     f"period {n_total/rate*1e6:.3f} µs")
            self._ch2_info.config(
                text=f"CH2  {n_detect:,} samp detect  dead {t_dead_actual:.3f} µs  "
                     f"LO freq exact")
        except (ValueError, AttributeError, ZeroDivisionError):
            pass

    # ── Scope Monitor panel ────────────────────────────────────────

    def _build_scope_panel(self, parent):
        s = ttk.LabelFrame(parent, text="Scope Monitor", padding=8)
        s.pack(fill='x', pady=(0, 6))

        ttk.Label(s, text="Scope CH").grid(row=0, column=0, sticky='w', pady=2)
        self._chan_var = tk.StringVar(value="1")
        ttk.Combobox(s, textvariable=self._chan_var, values=["1", "2"],
                     width=4, state='readonly').grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(s, text="Averages").grid(row=1, column=0, sticky='w', pady=2)
        self._avg_var = tk.StringVar(value="1")
        ttk.Combobox(s, textvariable=self._avg_var,
                     values=["1", "8", "16", "64", "256", "512"],
                     width=6, state='readonly').grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(s, text="ns/div").grid(row=2, column=0, sticky='w', pady=2)
        self._nsdiv_var = tk.StringVar(value="")
        ttk.Entry(s, textvariable=self._nsdiv_var, width=7).grid(
            row=2, column=1, sticky='ew', padx=4, pady=2)

        ttk.Label(s, text="Max pts").grid(row=3, column=0, sticky='w', pady=2)
        self._maxpts_var = tk.StringVar(value="10000")
        ttk.Entry(s, textvariable=self._maxpts_var, width=7).grid(
            row=3, column=1, sticky='ew', padx=4, pady=2)

        for row, (lbl, cmd) in enumerate([
                ("Plot Waveform",  self._plot_waveform),
                ("Live View: OFF", self._toggle_live)], start=4):
            btn = ttk.Button(s, text=lbl, command=cmd)
            btn.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2)
            if lbl == "Live View: OFF":
                self._live_btn = btn
            self._action_btns.append(btn)

        ttk.Button(s, text="Stop", command=self._stop).grid(
            row=6, column=0, columnspan=2, sticky='ew', pady=2)

    # ── Sweep panel ────────────────────────────────────────────────

    def _build_sweep_panel(self, parent):
        w = ttk.LabelFrame(parent, text="Sweep", padding=8)
        w.pack(fill='x', pady=(0, 6))
        w.columnconfigure(1, weight=1)

        ttk.Label(w, text="Settle (ms)").grid(row=0, column=0, sticky='w', pady=2)
        self._settle_var = tk.StringVar(value="50")
        ttk.Entry(w, textvariable=self._settle_var, width=6).grid(
            row=0, column=1, sticky='ew', padx=(4, 0), pady=2)

        ttk.Label(w, text="Cycles shown").grid(row=1, column=0, sticky='w', pady=2)
        self._cycles_var = tk.StringVar(value="8")
        ttk.Entry(w, textvariable=self._cycles_var, width=6).grid(
            row=1, column=1, sticky='ew', padx=(4, 0), pady=2)

        btn = ttk.Button(w, text="Frequency Sweep", command=self._run_sweep)
        btn.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(6, 2))
        self._sweep_btn = btn
        self._action_btns.append(btn)

    def _build_right(self, parent):
        right = ttk.Frame(parent)
        right.pack(side='left', fill='both', expand=True)

        log_frame = ttk.LabelFrame(right, text="Log", padding=4)
        log_frame.pack(fill='both', expand=False, pady=(0, 4))
        self._log = scrolledtext.ScrolledText(log_frame, height=10, state='disabled',
                                              font=('Courier', 9))
        self._log.pack(fill='both', expand=True)
        sys.stdout = _LogRedirect(self._log, self.root)

        plot_frame = ttk.LabelFrame(right, text="Plot", padding=4)
        plot_frame.pack(fill='both', expand=True)
        self._fig = Figure(figsize=(6, 4), dpi=90)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── Helpers ───────────────────────────────────────────────────

    def _thread(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    _MODE_TO_RATE = {
        "9 GS/s  (CH1)":    SAMPLE_RATE_SINGLE,
        "2.5 GS/s  (CH1+2)": SAMPLE_RATE_DUAL,
    }

    def _rate(self):
        return self._MODE_TO_RATE.get(self._rate_var.get(), SAMPLE_RATE_SINGLE)

    def _set_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        for btn in self._action_btns:
            self.root.after(0, lambda b=btn: b.config(state=state))

    def _redraw(self):
        self.root.after(0, self._canvas.draw)

    def _status(self, label, text, color):
        self.root.after(0, lambda: label.config(text=text, foreground=color))

    def _std_params(self):
        return (float(self._freq1_var.get()) * 1e6,
                float(self._amp1_var.get()),
                1)

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

    # ── Instrument connections ─────────────────────────────────────

    def _connect_awg(self):
        def connect():
            try:
                self.awg = AWG(sample_rate=self._rate())
                self._status(self._awg_lbl, "AWG: connected", "green")
            except Exception as e:
                print(f"AWG connection failed: {e}\n")
                self._status(self._awg_lbl, "AWG: error", "red")
        self._thread(connect)

    def _connect_scope(self):
        def connect():
            try:
                self.scope = Scope()
                n_avg = int(self._avg_var.get())
                self.scope.setup(channel=int(self._chan_var.get()), n_averages=n_avg)
                self._status(self._scope_lbl, "Scope: connected", "green")
            except Exception as e:
                print(f"Scope connection failed: {e}\n")
                self._status(self._scope_lbl, "Scope: error", "red")
        self._thread(connect)

    # ── Standard waveform commands ────────────────────────────────

    def _run_std_wave(self, channel=1):
        if not self._need_awg(): return
        try:
            freq = float(getattr(self, f'_freq{channel}_var').get()) * 1e6
            amp  = float(getattr(self, f'_amp{channel}_var').get())
            wave = getattr(self, f'_wave{channel}_var').get()
        except ValueError as e:
            print(f"Invalid parameter: {e}\n")
            return
        rate = self._rate()
        reset = (channel == 1)
        self._set_busy(True)
        def run():
            try:
                self.awg.sample_rate = rate
                if wave == "Sine":
                    self.awg.send_sine(freq, amp, channel=channel, reset=reset)
                elif wave == "Square":
                    self.awg.send_square(freq, amp, channel=channel, reset=reset)
                elif wave == "Ramp":
                    self.awg.send_ramp(freq, amp, channel=channel, reset=reset)
            except Exception as e:
                print(f"Waveform error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(run)

    # ── Chirp + CW command ────────────────────────────────────────

    def _run_chirp(self):
        if not self._need_awg(): return
        try:
            f_start  = float(self._c_start_var.get())  * 1e6
            f_stop   = float(self._c_stop_var.get())   * 1e6
            chirp_us = float(self._c_dur_var.get())
            dead_us  = float(self._c_dead_var.get())
            f_lo      = float(self._c_lo_var.get())     * 1e6
            detect_us = float(self._c_detect_var.get())
            amp       = float(self._c_amp_var.get())
        except ValueError as e:
            print(f"Invalid chirp parameter: {e}\n")
            return

        self._set_busy(True)
        def run():
            try:
                self.awg.send_chirp_with_lo(f_start, f_stop, chirp_us, dead_us,
                                             f_lo, detect_us, amplitude_vpp=amp)
            except Exception as e:
                print(f"Chirp error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(run)

    def _stop(self):
        self._sweep_stop.set()
        self._live_running = False
        self._live_gen += 1
        self._set_busy(False)
        self.root.after(0, lambda: self._live_btn.config(text="Live View: OFF"))
        if not self._need_awg(): return
        self._thread(self.awg.stop)

    # ── Sweep ──────────────────────────────────────────────────────

    def _run_sweep(self):
        print("Sweep clicked\n")
        if not self._need_awg(): return
        self._set_busy(True)   # disable buttons immediately — prevents double-click second thread
        try:
            _, amp, awg_ch = self._std_params()
            settle_s = float(self._settle_var.get()) / 1000.0
            n_cycles = int(self._cycles_var.get())
        except Exception as e:
            print(f"Sweep param error: {e}\n")
            self._set_busy(False)
            return

        def sweep():
            self._sweep_stop.clear()
            try:
                # Pre-load all segments once; the sweep loop then just switches
                import time as _time
                print("Pre-loading waveform segments…\n")
                t_pre0 = _time.perf_counter()
                self.awg.send_sine(DEFAULT_FREQS[0], amp, channel=awg_ch, reset=True)
                t_pre1 = _time.perf_counter()
                print(f"  send_sine: {(t_pre1-t_pre0):.1f} s\n")
                segments = self.awg.sweep_preload(DEFAULT_FREQS, channel=awg_ch)
                t_pre2 = _time.perf_counter()
                print(f"  sweep_preload: {(t_pre2-t_pre1):.1f} s\n")

                if self.scope is not None:
                    n_avg = int(self._avg_var.get())
                    ch    = int(self._chan_var.get())
                    self.scope.setup(channel=ch, n_averages=n_avg)
                    print(f"  scope: CH{ch}, {n_avg}× avg\n")

                ref_vpp = None
                freqs_plot, losses_plot = [], []

                print(f"\n{'Target (MHz)':>14}  {'Actual (MHz)':>14}  {'Vpp (mV)':>10}  {'Loss (dB)':>10}")
                print("-" * 56)

                for (actual, segnum), f in zip(segments, DEFAULT_FREQS):
                    if self._sweep_stop.is_set():
                        print("Sweep aborted.\n")
                        break

                    import time as _time
                    t0 = _time.perf_counter()
                    self.awg.sweep_step(segnum, amp, channel=awg_ch)
                    t1 = _time.perf_counter()
                    print(f"  sweep_step: {(t1-t0)*1000:.0f} ms\n")

                    if self.scope is not None:
                        t2 = _time.perf_counter()
                        self.scope.set_timebase(actual, n_cycles=n_cycles)
                        t3 = _time.perf_counter()
                        print(f"  set_timebase: {(t3-t2)*1000:.0f} ms\n")
                        vpp, wt, wv = self.scope.measure_vpp(
                            channel=int(self._chan_var.get()), settle=settle_s)
                        t4 = _time.perf_counter()
                        print(f"  measure_vpp: {(t4-t3)*1000:.0f} ms\n")
                        if ref_vpp is None:
                            ref_vpp = vpp
                        loss = 20 * np.log10(vpp / ref_vpp) if vpp > 0 else float('-inf')
                        print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {vpp*1e3:>10.1f}  {loss:>10.2f}")
                        freqs_plot.append(actual / 1e6)
                        losses_plot.append(loss)

                        # Live per-step plot: waveform on top, loss curve on bottom
                        self._fig.clear()
                        ax1 = self._fig.add_subplot(211)
                        ax1.plot(wt * 1e9, wv * 1e3)
                        ax1.set_xlabel("Time (ns)")
                        ax1.set_ylabel("Voltage (mV)")
                        ax1.set_title(f"{actual/1e6:.3f} MHz — Vpp {vpp*1e3:.1f} mV")
                        ax1.grid(True)
                        ax2 = self._fig.add_subplot(212)
                        ax2.plot(freqs_plot, losses_plot, 'o-')
                        ax2.set_xlabel("Frequency (MHz)")
                        ax2.set_ylabel("Loss (dB)")
                        ax2.grid(True)
                        self._fig.tight_layout()
                        self._redraw()
                    else:
                        print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {'(no scope)':>10}")

                print("Sweep done.\n")
            except Exception as e:
                print(f"Sweep error: {e}\n")
            finally:
                self._set_busy(False)

        self._thread(sweep)

    # ── Waveform capture ───────────────────────────────────────────

    def _apply_timebase(self):
        """Send ns/div to scope if the field is filled in."""
        val = self._nsdiv_var.get().strip()
        if val and self.scope:
            try:
                self.scope.set_timebase_direct(float(val) * 1e-9)
            except Exception as e:
                print(f"Timebase error: {e}\n")

    def _scope_max_pts(self):
        try:
            return int(self._maxpts_var.get())
        except ValueError:
            return 10000

    def _plot_waveform(self):
        print("Plot waveform clicked\n")
        if not self._need_scope(): return
        ch = int(self._chan_var.get())

        def capture():
            self._set_busy(True)
            try:
                self._apply_timebase()
                t, v = self.scope.get_waveform(channel=ch, max_points=self._scope_max_pts())

                n, dt = len(v), t[1] - t[0]
                freqs    = np.fft.rfftfreq(n, dt)
                spectrum = 20 * np.log10(np.abs(np.fft.rfft(v)) * 2 / n + 1e-12)

                self._fig.clear()
                ax1 = self._fig.add_subplot(211)
                ax1.plot(t * 1e9, v * 1e3)
                ax1.set_xlabel("Time (ns)")
                ax1.set_ylabel("Voltage (mV)")
                ax1.grid(True)

                ax2 = self._fig.add_subplot(212)
                ax2.plot(freqs * 1e-6, spectrum)
                ax2.set_xlabel("Frequency (MHz)")
                ax2.set_ylabel("Amplitude (dBV)")
                ax2.grid(True)

                self._fig.tight_layout()
                self._redraw()
            except Exception as e:
                print(f"Capture error: {e}\n")
            finally:
                self._set_busy(False)

        self._thread(capture)

    # ── Live view ──────────────────────────────────────────────────

    def _toggle_live(self):
        print("Live view clicked\n")
        if not self._need_scope(): return
        if self._live_running:
            self._live_running = False
            self._live_gen += 1
            self.root.after(0, lambda: self._live_btn.config(text="Live View: OFF"))
        else:
            self._live_running = True
            self._live_gen += 1
            self.root.after(0, lambda: self._live_btn.config(text="Live View: ON"))
            gen = self._live_gen
            self._thread(lambda: self._live_loop(gen))

    def _live_loop(self, gen):
        ch = int(self._chan_var.get())
        self._apply_timebase()
        max_pts = self._scope_max_pts()
        while self._live_running and self._live_gen == gen:
            try:
                t, v = self.scope.get_waveform(channel=ch, max_points=max_pts)

                n, dt = len(v), t[1] - t[0]
                freqs    = np.fft.rfftfreq(n, dt)
                spectrum = 20 * np.log10(np.abs(np.fft.rfft(v)) * 2 / n + 1e-12)

                self._fig.clear()
                ax1 = self._fig.add_subplot(211)
                ax1.plot(t * 1e9, v * 1e3)
                ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("Voltage (mV)"); ax1.grid(True)

                ax2 = self._fig.add_subplot(212)
                ax2.plot(freqs * 1e-6, spectrum)
                ax2.set_xlabel("Frequency (MHz)"); ax2.set_ylabel("Amplitude (dBV)")
                ax2.grid(True)

                self._fig.tight_layout()
                self._redraw()

            except Exception as e:
                print(f"Live view error: {e}\n")
                self._fig.clear()
                ax = self._fig.add_subplot(111)
                ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                        ha='center', va='center', color='red', fontsize=9,
                        wrap=True)
                self._redraw()

            threading.Event().wait(0.05)

        if self._live_gen == gen:
            self._live_running = False
            self.root.after(0, lambda: self._live_btn.config(text="Live View: OFF"))


if __name__ == "__main__":
    root = tk.Tk()
    LabGUI(root)
    root.mainloop()