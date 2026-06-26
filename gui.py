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
        self.root.geometry("1100x720")
        self.awg = None
        self.scope = None
        self._live_running = False
        self._live_gen = 0       # increments each start; loop exits when gen mismatches
        self._sweep_stop = threading.Event()
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        # Connection bar
        conn = ttk.LabelFrame(self.root, text="Instruments", padding=5)
        conn.pack(fill='x', padx=8, pady=4)

        ttk.Button(conn, text="Connect AWG", command=self._connect_awg).pack(side='left', padx=4)
        self._awg_lbl = ttk.Label(conn, text="AWG: disconnected", foreground='red')
        self._awg_lbl.pack(side='left', padx=(0, 16))

        ttk.Button(conn, text="Connect Scope", command=self._connect_scope).pack(side='left', padx=4)
        self._scope_lbl = ttk.Label(conn, text="Scope: disconnected", foreground='red')
        self._scope_lbl.pack(side='left', padx=(0, 16))

        # Main layout
        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=8, pady=4)

        self._build_controls(main)
        self._build_right(main)

    def _build_controls(self, parent):
        ctrl = ttk.Frame(parent, width=230)
        ctrl.pack(side='left', fill='y', padx=(0, 8))
        ctrl.pack_propagate(False)

        # Parameters
        p = ttk.LabelFrame(ctrl, text="Parameters", padding=8)
        p.pack(fill='x', pady=(0, 6))

        fields = [
            ("Frequency (MHz)", "freq", "100"),
            ("Amplitude (Vpp)", "amp",  "0.5"),
        ]
        for row, (label, attr, default) in enumerate(fields):
            ttk.Label(p, text=label).grid(row=row, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=default)
            setattr(self, f"_{attr}_var", var)
            ttk.Entry(p, textvariable=var, width=10).grid(row=row, column=1, padx=4, pady=2)

        ttk.Label(p, text="Channel").grid(row=2, column=0, sticky='w', pady=2)
        self._chan_var = tk.StringVar(value="1")
        ttk.Combobox(p, textvariable=self._chan_var, values=["1", "2"],
                     width=8, state='readonly').grid(row=2, column=1, padx=4, pady=2)

        ttk.Label(p, text="Sample rate").grid(row=3, column=0, sticky='w', pady=2)
        self._rate_var = tk.StringVar(value="Single (9 GS/s)")
        ttk.Combobox(p, textvariable=self._rate_var,
                     values=["Single (9 GS/s)", "Dual (2.5 GS/s)"],
                     width=14, state='readonly').grid(row=3, column=1, padx=4, pady=2)

        ttk.Label(p, text="Averages").grid(row=4, column=0, sticky='w', pady=2)
        self._avg_var = tk.StringVar(value="1")
        ttk.Combobox(p, textvariable=self._avg_var,
                     values=["1", "8", "16", "64", "256", "512"],
                     width=8, state='readonly').grid(row=4, column=1, padx=4, pady=2)

        ttk.Label(p, text="Freq mode").grid(row=5, column=0, sticky='w', pady=2)
        self._acc_var = tk.StringVar(value="Exact (adjust rate)")
        acc_box = ttk.Combobox(p, textvariable=self._acc_var,
                               values=["Exact (adjust rate)",
                                       "±2 MHz  (fixed rate, fast)",
                                       "±550 kHz (fixed rate)",
                                       "±70 kHz  (fixed rate)",
                                       "±35 kHz  (fixed rate, slow)"],
                               width=22, state='readonly')
        acc_box.grid(row=5, column=1, padx=4, pady=2)
        acc_box.bind("<<ComboboxSelected>>", self._on_buf_change)

        # Waveform
        w = ttk.LabelFrame(ctrl, text="Waveform", padding=8)
        w.pack(fill='x', pady=(0, 6))
        self._waveform_btns = []
        for label, cmd in [("Sine", self._run_sine),
                            ("IQ Sine (CH1=I, CH2=Q)", self._run_iq_sine),
                            ("Square", self._run_square),
                            ("Ramp", self._run_ramp)]:
            btn = ttk.Button(w, text=label, command=cmd)
            btn.pack(fill='x', pady=2)
            self._waveform_btns.append(btn)

        # Actions
        a = ttk.LabelFrame(ctrl, text="Actions", padding=8)
        a.pack(fill='x', pady=(0, 6))
        self._sweep_btn = ttk.Button(a, text="Frequency Sweep", command=self._run_sweep)
        self._sweep_btn.pack(fill='x', pady=2)
        self._plot_btn = ttk.Button(a, text="Plot Waveform", command=self._plot_waveform)
        self._plot_btn.pack(fill='x', pady=2)
        self._live_btn = ttk.Button(a, text="Live View: OFF", command=self._toggle_live)
        self._live_btn.pack(fill='x', pady=2)
        ttk.Button(a, text="Stop", command=self._stop).pack(fill='x', pady=2)

    def _build_right(self, parent):
        right = ttk.Frame(parent)
        right.pack(side='left', fill='both', expand=True)

        # Log
        log_frame = ttk.LabelFrame(right, text="Log", padding=4)
        log_frame.pack(fill='both', expand=False, pady=(0, 4))

        self._log = scrolledtext.ScrolledText(log_frame, height=10, state='disabled',
                                              font=('Courier', 9))
        self._log.pack(fill='both', expand=True)
        sys.stdout = _LogRedirect(self._log, self.root)

        # Plot
        plot_frame = ttk.LabelFrame(right, text="Plot", padding=4)
        plot_frame.pack(fill='both', expand=True)

        self._fig = Figure(figsize=(6, 4), dpi=90)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── Helpers ───────────────────────────────────────────────────

    def _thread(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    # Maps freq mode label → buffer size (multiples of 64 as required by Proteus)
    _ACC_TO_BUF = {
        "Exact (adjust rate)":          2_048,   # small buffer; sample rate adjusted instead
        "±2 MHz  (fixed rate, fast)":   2_048,
        "±550 kHz (fixed rate)":        8_192,
        "±70 kHz  (fixed rate)":       65_536,
        "±35 kHz  (fixed rate, slow)": 131_072,
    }

    def _exact_freq(self):
        return self._acc_var.get() == "Exact (adjust rate)"

    def _on_buf_change(self, _=None):
        n = self._ACC_TO_BUF.get(self._acc_var.get(), 2048)
        if self.awg is not None:
            self.awg.n_samples = n

    def _sync_buf(self):
        """Push the selected buffer size to the AWG before waveform generation."""
        if self.awg is not None:
            self.awg.n_samples = self._ACC_TO_BUF.get(self._acc_var.get(), 2048)

    def _set_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        btns = [self._sweep_btn, self._plot_btn, self._live_btn] + self._waveform_btns
        for btn in btns:
            self.root.after(0, lambda b=btn: b.config(state=state))

    def _redraw(self):
        self.root.after(0, self._canvas.draw)

    def _status(self, label, text, color):
        self.root.after(0, lambda: label.config(text=text, foreground=color))

    def _params(self):
        return (
            float(self._freq_var.get()) * 1e6,
            float(self._amp_var.get()),
            int(self._chan_var.get()),
        )

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
                rate = SAMPLE_RATE_SINGLE if "Single" in self._rate_var.get() else SAMPLE_RATE_DUAL
                self.awg = AWG(sample_rate=rate)
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

    # ── Waveform commands ──────────────────────────────────────────

    def _run_sine(self):
        if not self._need_awg(): return
        freq, amp, ch = self._params()
        self._sync_buf()
        exact = self._exact_freq()
        self._thread(lambda: self.awg.send_sine(freq, amp, channel=ch, exact=exact))

    def _run_iq_sine(self):
        if not self._need_awg(): return
        freq, amp, _ = self._params()   # channel ignored — always CH1=I, CH2=Q
        self._sync_buf()
        exact = self._exact_freq()
        print(f"IQ sine: {freq/1e6:.3f} MHz, {amp:.3f} Vpp — ensure dual-channel mode (2.5 GS/s)\n")
        self._thread(lambda: self.awg.send_iq_sine(freq, amp, exact=exact))

    def _run_square(self):
        if not self._need_awg(): return
        freq, amp, ch = self._params()
        self._sync_buf()
        exact = self._exact_freq()
        self._thread(lambda: self.awg.send_square(freq, amp, channel=ch, exact=exact))

    def _run_ramp(self):
        if not self._need_awg(): return
        freq, amp, ch = self._params()
        self._sync_buf()
        exact = self._exact_freq()
        self._thread(lambda: self.awg.send_ramp(freq, amp, channel=ch, exact=exact))

    def _stop(self):
        self._sweep_stop.set()
        self._live_running = False      # signals live loop to exit
        self._live_gen += 1             # invalidates any running loop generation
        self._set_busy(False)
        self.root.after(0, lambda: self._live_btn.config(text="Live View: OFF"))
        if not self._need_awg(): return
        self._thread(self.awg.stop)

    # ── Sweep ──────────────────────────────────────────────────────

    def _run_sweep(self):
        if not self._need_awg(): return
        _, amp, ch = self._params()
        n_avg = int(self._avg_var.get())

        def sweep():
            self._sweep_stop.clear()
            self._set_busy(True)
            try:
                if self.scope is not None:
                    self.scope.setup(channel=ch, n_averages=1)

                ref_vpp = None
                freqs_plot, losses_plot = [], []

                print(f"\n{'Target (MHz)':>14}  {'Actual (MHz)':>14}  {'Vpp (mV)':>10}  {'Loss (dB)':>10}")
                print("-" * 56)

                self.awg.send_sine(DEFAULT_FREQS[0], amp, channel=ch, reset=True)

                for f in DEFAULT_FREQS:
                    if self._sweep_stop.is_set():
                        print("Sweep aborted.\n")
                        break

                    actual = self.awg.update_sine(f, amp, channel=ch)

                    if self.scope is not None:
                        vpp = self.scope.measure_vpp(settle=0.3, n_readings=n_avg)
                        if ref_vpp is None:
                            ref_vpp = vpp
                        loss = 20 * np.log10(vpp / ref_vpp) if vpp > 0 else float('-inf')
                        print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {vpp*1e3:>10.1f}  {loss:>10.2f}")
                        freqs_plot.append(actual / 1e6)
                        losses_plot.append(loss)
                    else:
                        print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {'(no scope)':>10}")

                if freqs_plot:
                    self._fig.clear()
                    ax = self._fig.add_subplot(111)
                    ax.plot(freqs_plot, losses_plot, 'o-')
                    ax.set_xlabel("Frequency (MHz)")
                    ax.set_ylabel("Loss (dB)")
                    ax.set_title("Amplitude vs Frequency")
                    ax.grid(True)
                    self._fig.tight_layout()
                    self._redraw()

                print("Sweep done.\n")
            except Exception as e:
                print(f"Sweep error: {e}\n")
            finally:
                self._set_busy(False)

        self._thread(sweep)

    # ── Waveform capture ───────────────────────────────────────────

    def _plot_waveform(self):
        if not self._need_scope(): return
        ch = int(self._chan_var.get())

        def capture():
            self._set_busy(True)
            try:
                t, v = self.scope.get_waveform(channel=ch)

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
        while self._live_running and self._live_gen == gen:
            try:
                t, v = self.scope.get_waveform(channel=ch)
            except Exception as e:
                print(f"Live view: {e}\n")
                # pause then retry — don't exit on a transient error
                for _ in range(5):
                    if not self._live_running or self._live_gen != gen:
                        break
                    threading.Event().wait(0.1)
                continue

            n, dt = len(v), t[1] - t[0]
            freqs    = np.fft.rfftfreq(n, dt)
            spectrum = 20 * np.log10(np.abs(np.fft.rfft(v)) * 2 / n + 1e-12)

            self._fig.clear()
            ax1 = self._fig.add_subplot(211)
            ax1.plot(t * 1e9, v * 1e3)
            ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("Voltage (mV)"); ax1.grid(True)

            ax2 = self._fig.add_subplot(212)
            ax2.plot(freqs * 1e-6, spectrum)
            ax2.set_xlabel("Frequency (MHz)"); ax2.set_ylabel("Amplitude (dBV)"); ax2.grid(True)

            self._fig.tight_layout()
            self._redraw()

        if self._live_gen == gen:
            self._live_running = False
            self.root.after(0, lambda: self._live_btn.config(text="Live View: OFF"))


if __name__ == "__main__":
    root = tk.Tk()
    LabGUI(root)
    root.mainloop()