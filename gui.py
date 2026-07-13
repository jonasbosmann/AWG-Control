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
    def __init__(self, root):
        self.root = root
        self.root.title("AWG / Scope Control")
        self.root.geometry("1100x780")
        self.awg   = None
        self.scope = None
        self._awg_needs_reset = True   # *RST once after connect / rate change
        self._live_running = False
        self._live_gen     = 0
        self._sweep_stop   = threading.Event()
        self._action_btns  = []   # all buttons disabled while busy (except Stop)
        self._live_win     = None
        self._setup_photo  = None  # path to current microwave-chain setup photo
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
        conn = ttk.LabelFrame(self.root, text="Instruments", padding=5)
        conn.pack(fill='x', padx=8, pady=4)
        ttk.Button(conn, text="Connect AWG",   command=self._connect_awg).pack(side='left', padx=4)
        self._awg_lbl = ttk.Label(conn, text="AWG: disconnected", foreground='red')
        self._awg_lbl.pack(side='left', padx=(0, 16))
        ttk.Button(conn, text="Connect Scope", command=self._connect_scope).pack(side='left', padx=4)
        self._scope_lbl = ttk.Label(conn, text="Scope: disconnected", foreground='red')
        self._scope_lbl.pack(side='left', padx=(0, 16))

        stop_btn = tk.Button(conn, text="STOP ALL", command=self._stop,
                             bg='red', fg='white', font=('', 10, 'bold'),
                             relief='raised', padx=8)
        stop_btn.pack(side='right', padx=8)

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
        # Scroll only while the pointer is over the control panel — a permanent
        # bind_all would hijack the wheel everywhere (log, plot, live view).
        canvas.bind('<Enter>', lambda _: canvas.bind_all(
            '<MouseWheel>', lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), 'units')))
        canvas.bind('<Leave>', lambda _: canvas.unbind_all('<MouseWheel>'))

        self._build_std_waveform(ctrl)
        self._build_chirp_panel(ctrl)
        self._build_duc_panel(ctrl)
        self._build_scope_panel(ctrl)
        self._build_setup_panel(ctrl)
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
        self._awg_needs_reset = True   # clock change → re-init segments on next send
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

    # ── DUC Chirp + CW panel (upconverted, IQM ONE mode) ─────────

    def _build_duc_panel(self, parent):
        f = ttk.LabelFrame(parent, text="DUC Chirp + CW  (upconverted, IQM ONE)", padding=8)
        f.pack(fill='x', pady=(0, 6))
        f.columnconfigure(1, weight=1)

        # Carrier = NCO frequency for CH1.  BB start/stop = baseband offsets from carrier.
        # RF output sweeps (carrier + BB_start) → (carrier + BB_stop).
        # DAC clock 9 GS/s (x8 interp): max carrier ~4.5 GHz (Nyquist); BB_stop ≤ ~560 MHz
        # (Nyquist of the 1.125 GS/s complex baseband rate).
        duc_fields = [
            ("Carrier     (MHz)", "duc_carrier",   "2000"),
            ("BB start    (MHz)", "duc_bb_start",  "0"),
            ("BB stop     (MHz)", "duc_bb_stop",   "500"),
            ("Chirp dur   (µs)",  "duc_chirp_dur", "1.0"),
            ("Dead time   (µs)",  "duc_dead",      "0.1"),
            ("LO freq     (MHz)", "duc_lo",        "2300"),
            ("Detect win  (µs)",  "duc_detect",    "10.0"),
            ("Amplitude   (Vpp)", "duc_amp",       "0.5"),
        ]
        for row, (lbl, attr, default) in enumerate(duc_fields):
            ttk.Label(f, text=lbl).grid(row=row, column=0, sticky='w', pady=1)
            var = tk.StringVar(value=default)
            setattr(self, f"_{attr}_var", var)
            e = ttk.Entry(f, textvariable=var, width=8)
            e.grid(row=row, column=1, sticky='ew', padx=(4, 0), pady=1)
            var.trace_add('write', lambda *_: self._update_duc_info())

        r = len(duc_fields)
        ttk.Separator(f, orient='horizontal').grid(row=r, column=0, columnspan=2,
                                                    sticky='ew', pady=4)
        self._duc_info = ttk.Label(f, text="", foreground='gray', font=('Courier', 8))
        self._duc_info.grid(row=r+1, column=0, columnspan=2, sticky='w')

        duc_btn = ttk.Button(f, text="DUC Chirp+CW",
                             command=self._run_chirp_duc)
        duc_btn.grid(row=r+2, column=0, columnspan=2, sticky='ew', pady=(6, 2))
        self._action_btns.append(duc_btn)

        self._update_duc_info()

    def _update_duc_info(self):
        try:
            carrier  = float(self._duc_carrier_var.get())
            bb_start = float(self._duc_bb_start_var.get())
            bb_stop  = float(self._duc_bb_stop_var.get())
            f_lo     = float(self._duc_lo_var.get())
            self._duc_info.config(
                text=f"CH1 RF: {carrier+bb_start:.1f} → {carrier+bb_stop:.1f} MHz"
                     f"   CH2 LO: {f_lo:.3f} MHz (NCO exact)")
        except (ValueError, AttributeError):
            pass

    # ── Scope Monitor panel ────────────────────────────────────────

    def _build_scope_panel(self, parent):
        s = ttk.LabelFrame(parent, text="Scope Monitor", padding=8)
        s.pack(fill='x', pady=(0, 6))

        ttk.Label(s, text="Scope CH").grid(row=0, column=0, sticky='w', pady=2)
        self._chan_var = tk.StringVar(value="1")
        ttk.Combobox(s, textvariable=self._chan_var, values=["1", "2"],
                     width=4, state='readonly').grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(s, text="ns/div").grid(row=1, column=0, sticky='w', pady=2)
        self._nsdiv_var = tk.StringVar(value="")
        ttk.Entry(s, textvariable=self._nsdiv_var, width=7).grid(
            row=1, column=1, sticky='ew', padx=4, pady=2)

        ttk.Label(s, text="Max pts").grid(row=2, column=0, sticky='w', pady=2)
        self._maxpts_var = tk.StringVar(value="10000")
        ttk.Entry(s, textvariable=self._maxpts_var, width=7).grid(
            row=2, column=1, sticky='ew', padx=4, pady=2)

        ttk.Button(s, text="Setup Scope", command=self._setup_scope).grid(
            row=3, column=0, columnspan=2, sticky='ew', pady=2)

        plot_btn = ttk.Button(s, text="Plot Waveform", command=self._plot_waveform)
        plot_btn.grid(row=4, column=0, columnspan=2, sticky='ew', pady=2)
        self._action_btns.append(plot_btn)

        live_btn = ttk.Button(s, text="Live View…", command=self._open_live_view)
        live_btn.grid(row=5, column=0, columnspan=2, sticky='ew', pady=2)
        self._action_btns.append(live_btn)

        restore_btn = ttk.Button(s, text="Restore Scope", command=self._restore_scope)
        restore_btn.grid(row=6, column=0, columnspan=2, sticky='ew', pady=2)
        self._action_btns.append(restore_btn)

    # ── Setup (microwave-chain configuration) panel ────────────────

    def _build_setup_panel(self, parent):
        f = ttk.LabelFrame(parent, text="Setup  (microwave chain)", padding=8)
        f.pack(fill='x', pady=(0, 6))
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Name").grid(row=0, column=0, sticky='w', pady=1)
        self._setup_name_var = tk.StringVar(value="config_A")
        ttk.Entry(f, textvariable=self._setup_name_var, width=8).grid(
            row=0, column=1, sticky='ew', padx=(4, 0), pady=1)

        ttk.Label(f, text="Description").grid(row=1, column=0, sticky='nw', pady=1)
        self._setup_desc = tk.Text(f, width=8, height=4, wrap='word')
        self._setup_desc.grid(row=1, column=1, sticky='ew', padx=(4, 0), pady=1)
        self._setup_desc.insert('1.0', "AWG CH1 -> x24 AMC -> ...")

        self._photo_lbl = ttk.Label(f, text="Photo: none", foreground='gray',
                                    font=('Courier', 8), wraplength=200)
        self._photo_lbl.grid(row=2, column=0, columnspan=2, sticky='w', pady=(4, 1))

        pb = ttk.Frame(f)
        pb.grid(row=3, column=0, columnspan=2, sticky='ew')
        pb.columnconfigure((0, 1), weight=1)
        ttk.Button(pb, text="Browse Photo…", command=self._browse_photo).grid(
            row=0, column=0, sticky='ew', padx=(0, 2))
        ttk.Button(pb, text="Show Setup", command=self._show_setup_photo).grid(
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

    # ── Sweep panel ────────────────────────────────────────────────

    def _build_sweep_panel(self, parent):
        w = ttk.LabelFrame(parent, text="Sweep", padding=8)
        w.pack(fill='x', pady=(0, 6))
        w.columnconfigure(1, weight=1)

        ttk.Label(w, text="Scope CH").grid(row=0, column=0, sticky='w', pady=2)
        self._sweep_chan_var = tk.StringVar(value="1")
        ttk.Combobox(w, textvariable=self._sweep_chan_var, values=["1", "2"],
                     width=4, state='readonly').grid(row=0, column=1, sticky='ew', padx=(4, 0), pady=2)

        ttk.Label(w, text="Averages").grid(row=1, column=0, sticky='w', pady=2)
        self._sweep_avg_var = tk.StringVar(value="1")
        ttk.Combobox(w, textvariable=self._sweep_avg_var,
                     values=["1", "8", "16", "64", "256", "512"],
                     width=6, state='readonly').grid(row=1, column=1, sticky='ew', padx=(4, 0), pady=2)

        ttk.Label(w, text="Settle (ms)").grid(row=2, column=0, sticky='w', pady=2)
        self._settle_var = tk.StringVar(value="200")
        ttk.Entry(w, textvariable=self._settle_var, width=6).grid(
            row=2, column=1, sticky='ew', padx=(4, 0), pady=2)

        ttk.Label(w, text="Cycles shown").grid(row=3, column=0, sticky='w', pady=2)
        self._cycles_var = tk.StringVar(value="8")
        ttk.Entry(w, textvariable=self._cycles_var, width=6).grid(
            row=3, column=1, sticky='ew', padx=(4, 0), pady=2)

        ttk.Label(w, text="AWG mode").grid(row=4, column=0, sticky='w', pady=2)
        self._sweep_mode_var = tk.StringVar(value="DUC (NCO step)")
        ttk.Combobox(w, textvariable=self._sweep_mode_var,
                     values=["Direct (segments)", "DUC (NCO step)"],
                     width=16, state='readonly').grid(
            row=4, column=1, sticky='ew', padx=(4, 0), pady=2)

        # Blank = DEFAULT_FREQS. Comma list ("100, 500, 2000") or
        # "start:stop:step" ("100:4400:100"), all in MHz.
        ttk.Label(w, text="Freqs (MHz)").grid(row=5, column=0, sticky='w', pady=2)
        self._sweep_freqs_var = tk.StringVar(value="100:4400:100")
        ttk.Entry(w, textvariable=self._sweep_freqs_var, width=6).grid(
            row=5, column=1, sticky='ew', padx=(4, 0), pady=2)

        btn = ttk.Button(w, text="Frequency Sweep", command=self._run_sweep)
        btn.grid(row=6, column=0, columnspan=2, sticky='ew', pady=(6, 2))
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
        if busy and self._live_running:
            self._live_running = False
            self._live_gen += 1
        state = 'disabled' if busy else 'normal'
        for btn in self._action_btns:
            try:
                self.root.after(0, lambda b=btn: b.config(state=state))
            except Exception:
                pass

    def _redraw(self):
        self.root.after(0, self._canvas.draw)

    def _status(self, label, text, color):
        self.root.after(0, lambda: label.config(text=text, foreground=color))

    def _std_params(self):
        return (float(self._freq1_var.get()) * 1e6,
                float(self._amp1_var.get()),
                1)

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

    # ── Instrument connections ─────────────────────────────────────

    def _connect_awg(self):
        def connect():
            try:
                self.awg = AWG(sample_rate=self._rate())
                self._awg_needs_reset = True
                self._status(self._awg_lbl, "AWG: connected", "green")
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
        # Reset only on first send after connect / rate change — an unconditional
        # reset on "Send CH1" (*RST + TRAC:DEL:ALL) would silently kill a running CH2.
        reset = self._awg_needs_reset
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
                self._awg_needs_reset = False
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

    def _run_chirp_duc(self):
        if not self._need_awg(): return
        try:
            f_carrier  = float(self._duc_carrier_var.get())   * 1e6
            f_start_bb = float(self._duc_bb_start_var.get())  * 1e6
            f_stop_bb  = float(self._duc_bb_stop_var.get())   * 1e6
            chirp_us   = float(self._duc_chirp_dur_var.get())
            dead_us    = float(self._duc_dead_var.get())
            f_lo       = float(self._duc_lo_var.get())         * 1e6
            detect_us  = float(self._duc_detect_var.get())
            amp        = float(self._duc_amp_var.get())
        except ValueError as e:
            print(f"Invalid DUC chirp parameter: {e}\n")
            return

        self._set_busy(True)
        def run():
            try:
                self.awg.send_chirp_with_lo_duc(
                    f_carrier, f_start_bb, f_stop_bb, chirp_us, dead_us,
                    f_lo, detect_us, amplitude_vpp=amp)
            except Exception as e:
                print(f"DUC chirp error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(run)

    def _stop(self):
        self._sweep_stop.set()
        self._live_running = False
        self._live_gen += 1
        self._set_busy(False)
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
            use_duc = "DUC" in self._sweep_mode_var.get()
            custom_freqs = self._parse_sweep_freqs(self._sweep_freqs_var.get())
        except Exception as e:
            print(f"Sweep param error: {e}\n")
            self._set_busy(False)
            return

        name, desc, photo = self._setup_meta()

        def sweep():
            self._sweep_stop.clear()
            records = []
            n_avg = int(self._sweep_avg_var.get())
            scope_ch = int(self._sweep_chan_var.get())
            try:
                base_freqs = custom_freqs if custom_freqs is not None else DEFAULT_FREQS
                # Drop frequencies above Nyquist — they would silently alias
                # (the buffer wraps) and log a bogus "actual" frequency.
                # DUC: the NCO reaches the 9 GS/s DAC Nyquist (4.5 GHz)
                # regardless of the GUI channel-mode clock.
                dac_rate = 9e9 if use_duc else self.awg.sample_rate
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
                    # Pre-load all segments once; the sweep loop then just switches
                    print("Pre-loading waveform segments…\n")
                    t_pre0 = time.perf_counter()
                    self.awg.send_sine(freqs[0], amp, channel=awg_ch, reset=True)
                    t_pre1 = time.perf_counter()
                    print(f"  send_sine: {(t_pre1-t_pre0):.1f} s\n")
                    segments = self.awg.sweep_preload(freqs, channel=awg_ch)
                    t_pre2 = time.perf_counter()
                    print(f"  sweep_preload: {(t_pre2-t_pre1):.1f} s\n")

                if self.scope is not None:
                    self.scope.setup(channel=scope_ch, n_averages=n_avg)
                    # Start with the vertical scale matched to the commanded
                    # amplitude (~75% of the 10-div screen); measure_vpp_auto
                    # then tracks the signal as it rolls off.
                    self.scope.set_vertical(scope_ch, amp / 7.5)
                    print(f"  scope: CH{scope_ch}, {n_avg}× avg\n")

                ref_vpp = None
                freqs_plot, losses_plot = [], []

                print(f"\n{'Target (MHz)':>14}  {'Actual (MHz)':>14}  {'Vpp (mV)':>10}  {'Loss (dB)':>10}")
                print("-" * 56)

                for (actual, segnum), f in zip(segments, freqs):
                    if self._sweep_stop.is_set():
                        print("Sweep aborted.\n")
                        break

                    t0 = time.perf_counter()
                    if use_duc:
                        self.awg.duc_cw_step(f, channel=awg_ch)
                    else:
                        self.awg.sweep_step(segnum, amp, channel=awg_ch)
                    t1 = time.perf_counter()
                    print(f"  {'duc_cw_step' if use_duc else 'sweep_step'}: "
                          f"{(t1-t0)*1000:.0f} ms\n")

                    if self.scope is not None:
                        t2 = time.perf_counter()
                        self.scope.set_timebase(actual, n_cycles=n_cycles)
                        t3 = time.perf_counter()
                        print(f"  set_timebase: {(t3-t2)*1000:.0f} ms\n")
                        vpp, wt, wv = self.scope.measure_vpp_auto(
                            channel=scope_ch, settle=settle_s)
                        t4 = time.perf_counter()
                        print(f"  measure_vpp: {(t4-t3)*1000:.0f} ms\n")
                        if ref_vpp is None:
                            ref_vpp = vpp
                        loss = 20 * np.log10(vpp / ref_vpp) if vpp > 0 else float('-inf')
                        print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {vpp*1e3:>10.1f}  {loss:>10.2f}")
                        freqs_plot.append(actual / 1e6)
                        losses_plot.append(loss)

                        dt_s = float(wt[1] - wt[0]) if len(wt) > 1 else 0.0
                        records.append({
                            "target_hz": f, "actual_hz": actual,
                            "vpp_v": vpp, "loss_db": loss,
                            "dt_s": dt_s, "voltage": wv,
                        })

                        # Live per-step plot — drawn on the Tk main thread only.
                        # Mutating self._fig from this worker thread races with
                        # the previous iteration's canvas.draw and aborts the
                        # sweep with a matplotlib error on the second step.
                        # Pass list copies: the worker keeps appending.
                        self.root.after(0, self._draw_sweep_step,
                                        wt, wv, actual, vpp,
                                        list(freqs_plot), list(losses_plot),
                                        dac_rate)
                    else:
                        print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {'(no scope)':>10}")

                print("Sweep done.\n")
            except Exception as e:
                print(f"Sweep error: {e}\n")
            finally:
                if records:
                    try:
                        params = {
                            "amplitude_vpp": amp, "awg_ch": awg_ch,
                            "awg_mode": "duc_nco" if use_duc else "direct_segments",
                            "dac_rate_hz": dac_rate,
                            "scope_ch": scope_ch, "n_averages": n_avg,
                            "settle_ms": settle_s * 1000.0, "cycles_shown": n_cycles,
                            "sample_rate_hz": self.awg.sample_rate,
                            "n_samples": self.awg.n_samples,
                        }
                        if use_duc:
                            params["interpolation"] = "X8"
                            params["baseband_rate_hz"] = dac_rate / 8.0
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

    # ── Scope setup ────────────────────────────────────────────────

    def _setup_scope(self):
        if not self._need_scope(): return
        ch = int(self._chan_var.get())
        def do_setup():
            self._set_busy(True)
            try:
                self.scope.setup(channel=ch, n_averages=1)
            except Exception as e:
                print(f"Setup error: {e}\n")
            finally:
                self._set_busy(False)
        self._thread(do_setup)

    def _restore_scope(self):
        """Undo leftover measurement state (NORMal trigger on CH2, averaging,
        armed single-sequence) — e.g. after compare_duc_direct.py."""
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

        def draw(t, v, freqs, spectrum):
            # Main thread only — see _draw_sweep_step.
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
            self._canvas.draw_idle()

        def capture():
            self._set_busy(True)
            try:
                self._apply_timebase()
                t, v = self.scope.get_waveform(channel=ch, max_points=self._scope_max_pts())

                n, dt = len(v), t[1] - t[0]
                freqs    = np.fft.rfftfreq(n, dt)
                spectrum = 20 * np.log10(np.abs(np.fft.rfft(v)) * 2 / n + 1e-12)

                self.root.after(0, draw, t, v, freqs, spectrum)
            except Exception as e:
                print(f"Capture error: {e}\n")
            finally:
                self._set_busy(False)

        self._thread(capture)

    # ── Live view ──────────────────────────────────────────────────

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

    def _live_draw(self, gen, t, v, freqs, spectrum):
        """Runs on the Tk main thread — the only place the live figure is touched.
        Mutating the figure from the worker thread races with canvas draws."""
        if not (self._live_running and self._live_gen == gen):
            return
        if self._live_win is None or not self._live_win.winfo_exists():
            return
        self._live_fig.clear()
        ax1 = self._live_fig.add_subplot(211)
        ax1.plot(t * 1e9, v * 1e3)
        ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("Voltage (mV)"); ax1.grid(True)

        ax2 = self._live_fig.add_subplot(212)
        ax2.plot(freqs * 1e-6, spectrum)
        ax2.set_xlabel("Frequency (MHz)"); ax2.set_ylabel("Amplitude (dBV)")
        ax2.grid(True)

        self._live_fig.tight_layout()
        self._live_canvas.draw_idle()

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

                self.root.after(0, self._live_draw, gen, t, v, freqs, spectrum)

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