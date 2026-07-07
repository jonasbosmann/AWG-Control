import pyvisa
import numpy as np
import time
import threading

SAMPLE_RATE_SINGLE = 9e9    # single-channel max (both DACs interleaved → CH2 disabled)
SAMPLE_RATE_DUAL   = 2.5e9  # dual-channel max (one DAC per channel)
N_SAMPLES = 2048            # must be multiple of 64 for Proteus granularity

_RESOURCE = "TCPIP0::141.51.196.111::5025::SOCKET"


class AWG:
    def __init__(self, sample_rate=SAMPLE_RATE_SINGLE, n_samples=N_SAMPLES):
        self.sample_rate = sample_rate
        self.n_samples = n_samples   # buffer length; must be multiple of 64
        self._active_seg = {}
        self._sweep_amp  = {}        # tracks last :VOLT sent per channel during sweep
        self._lock = threading.Lock()
        rm = pyvisa.ResourceManager()
        self._dev = rm.open_resource(_RESOURCE)
        self._dev.timeout = 10000
        self._dev.read_termination = '\n'
        self._dev.write_termination = '\n'
        print("AWG:", self._dev.query("*IDN?").strip())

    def _cmd(self, c):
        while True:
            e = self._dev.query(":SYST:ERR?")
            if e.startswith("0"):
                break
        self._dev.write(c)
        time.sleep(0.05)
        err = self._dev.query(":SYST:ERR?")
        if err.startswith("0"):
            print(f"  OK    {c.lstrip(':')}")
        else:
            print(f"  ERROR {c.lstrip(':')} -> {err.strip()}")

    def _setup(self, channel=1, reset=True, sample_rate=None):
        rate = sample_rate if sample_rate is not None else self.sample_rate
        if reset:
            self._dev.write("*CLS; *RST")
            time.sleep(0.5)
        self._cmd(f":INST:CHAN {channel}")
        self._cmd(f":FREQ:RAST {rate:.0f}")
        if reset:
            self._cmd(":TRAC:DEL:ALL")
        self._cmd(":INIT:CONT ON")

    def _upload(self, wave_u16, segnum=1):
        n = len(wave_u16)
        self._cmd(f":TRAC:DEF {segnum},{n}")
        self._cmd(f":TRAC:SEL {segnum}")
        data = wave_u16.tobytes()
        nb_str = str(len(data))
        self._dev.write_raw(f":TRAC:DATA #{len(nb_str)}{nb_str}".encode() + data + b"\n")
        time.sleep(0.3)
        self._dev.write("*CLS")
        print("  OK    TRAC:DATA")

    def _play(self, amplitude_vpp=0.5, segnum=1):
        self._cmd(f":FUNC:MODE:SEGM {segnum}")
        self._cmd(f":VOLT {amplitude_vpp:.3f}")
        self._cmd(":OUTP ON")

    @property
    def freq_step_hz(self):
        """Smallest achievable frequency increment at the current buffer size."""
        return self.sample_rate / self.n_samples

    def _sine_u16(self, cycles):
        t = np.arange(self.n_samples)
        wave = np.sin(2 * np.pi * cycles * t / self.n_samples)
        return ((wave + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

    def _square_u16(self, cycles):
        t = np.arange(self.n_samples)
        wave = np.where(np.sin(2 * np.pi * cycles * t / self.n_samples) >= 0, 65535, 0)
        return wave.astype(np.uint16)

    def _ramp_u16(self, cycles):
        t = np.arange(self.n_samples)
        wave = (t * cycles / self.n_samples) % 1.0
        return (wave * 65535).astype(np.uint16)

    def _cycles(self, frequency_hz):
        return max(round(frequency_hz * self.n_samples / self.sample_rate), 1)

    def _resolve(self, frequency_hz, exact=False):
        """Return (cycles, effective_sample_rate, actual_frequency).

        exact=False: keep self.sample_rate, round cycles → frequency error ≤ freq_step/2
        exact=True:  keep cycles integer, back-calculate sample rate → zero frequency error
                     (requires hardware to accept a non-standard clock rate)
        """
        cycles = self._cycles(frequency_hz)
        if exact:
            rate = frequency_hz * self.n_samples / cycles
            # If the required rate exceeds hardware max, try one more cycle
            if rate > SAMPLE_RATE_SINGLE:
                cycles += 1
                rate = frequency_hz * self.n_samples / cycles
        else:
            rate = self.sample_rate
        actual = cycles * rate / self.n_samples
        return cycles, rate, actual

    def send_sine(self, frequency_hz=100e6, amplitude_vpp=0.5, channel=1, reset=True, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact)
        err_hz = actual - frequency_hz
        print(f"[CH{channel}] Sine {frequency_hz/1e6:.6f} MHz -> {actual/1e6:.6f} MHz "
              f"(err {err_hz:+.1f} Hz, Fs={rate/1e9:.6f} GS/s)")
        seg = channel
        with self._lock:
            self._setup(channel, reset, sample_rate=rate)
            self._upload(self._sine_u16(cycles), segnum=seg)
            self._play(amplitude_vpp, segnum=seg)
            self._active_seg[channel] = seg
        return actual

    def update_sine(self, frequency_hz, amplitude_vpp=0.5, channel=1, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact)
        with self._lock:
            cur = self._active_seg.get(channel, channel)
            next_seg = channel + 20 if cur == channel + 10 else channel + 10
            self._cmd(f":INST:CHAN {channel}")
            if rate != self.sample_rate:
                self._cmd(f":FREQ:RAST {rate:.0f}")
            self._upload(self._sine_u16(cycles), segnum=next_seg)
            self._play(amplitude_vpp, segnum=next_seg)
            self._active_seg[channel] = next_seg
        return actual

    def sweep_preload(self, freqs_hz, channel=1):
        """Upload one sine segment per frequency (segments 30, 31, …).
        Returns list of (actual_hz, segnum). Call once before the sweep loop."""
        self._sweep_amp.pop(channel, None)   # force :VOLT on first sweep_step
        results = []
        with self._lock:
            for i, f in enumerate(freqs_hz):
                segnum = 30 + i
                cycles, _, actual = self._resolve(f)
                self._dev.write(f":INST:CHAN {channel}")
                wave = self._sine_u16(cycles)
                n = len(wave)
                self._dev.write(f":TRAC:DEF {segnum},{n}")
                self._dev.write(f":TRAC:SEL {segnum}")
                data = wave.tobytes()
                nb_str = str(len(data))
                self._dev.write_raw(
                    f":TRAC:DATA #{len(nb_str)}{nb_str}".encode() + data + b"\n")
                time.sleep(0.15)
                results.append((actual, segnum))
                print(f"  preload seg {segnum}: {actual/1e6:.3f} MHz")
        return results

    def sweep_step(self, segnum, amplitude_vpp=0.5, channel=1):
        """Switch to a pre-loaded segment and block until the AWG confirms.

        *OPC? blocks until the AWG has processed the segment-switch command.
        The physical output switches within one segment period (<1 µs) after that.
        Avoids re-sending :VOLT/:OUTP ON when unchanged to reduce DAC glitches.
        """
        with self._lock:
            self._dev.write(f":INST:CHAN {channel}")
            self._dev.write(f":FUNC:MODE:SEGM {segnum}")
            prev_amp = self._sweep_amp.get(channel)
            if prev_amp != amplitude_vpp:
                self._dev.write(f":VOLT {amplitude_vpp:.3f}")
                self._dev.write(f":OUTP ON")
                self._sweep_amp[channel] = amplitude_vpp
            self._dev.query("*OPC?")   # block until AWG has processed the switch
            self._active_seg[channel] = segnum

    def _chirp_windowed_u16(self, f_start, f_stop, n_active, n_total, rate, window_frac=0.05):
        t = np.arange(n_active, dtype=np.float64)
        phase = 2 * np.pi * (f_start / rate * t +
                              (f_stop - f_start) / (2 * rate * max(n_active - 1, 1)) * t ** 2)
        sig = np.sin(phase)
        n_win = max(int(window_frac * n_active), 1)
        sigma = n_win / 3.0
        idx = np.arange(n_win, dtype=np.float64)
        win = np.ones(n_active)
        win[:n_win]  = np.exp(-0.5 * ((idx - n_win) / sigma) ** 2)
        win[-n_win:] = np.exp(-0.5 * (idx           / sigma) ** 2)
        sig *= win
        full = np.zeros(n_total)
        full[:n_active] = sig
        return ((full + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

    def _cw_sine_u16(self, cycles, n_samples):
        t = np.arange(n_samples)
        wave = np.sin(2 * np.pi * cycles * t / n_samples)
        return ((wave + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

    def send_chirp_with_lo(self, f_start_hz, f_stop_hz, chirp_us, dead_us,
                            f_lo_hz, detect_us, amplitude_vpp=0.5, window_frac=0.05):
        """CH1: Gaussian-windowed chirp then silence.
           CH2: silence during chirp+dead, then exact CW sine during detection window.
           Both channels share the same buffer → identical period → phase-coherent shots.
           LO frequency is floating-point exact (no integer-cycles quantization).
           Always runs at SAMPLE_RATE_DUAL (2.5 GS/s)."""
        rate = SAMPLE_RATE_DUAL

        n_chirp  = round(chirp_us  * 1e-6 * rate)
        n_dead   = round(dead_us   * 1e-6 * rate)
        n_detect = round(detect_us * 1e-6 * rate)
        n_total  = max(int(np.ceil((n_chirp + n_dead + n_detect) / 64)) * 64, 128)

        # CH1: windowed chirp in first n_chirp samples, zeros for rest
        ch1 = self._chirp_windowed_u16(f_start_hz, f_stop_hz, n_chirp, n_total,
                                        rate, window_frac)

        # CH2: zeros during chirp+dead, exact CW sine during detection window, zeros after
        t_lo  = np.arange(n_detect, dtype=np.float64)
        lo_active = np.sin(2 * np.pi * f_lo_hz / rate * t_lo)
        n_win = max(int(window_frac * n_detect), 1)
        sigma = n_win / 3.0
        idx   = np.arange(n_win, dtype=np.float64)
        lo_active[:n_win]  *= np.exp(-0.5 * ((idx - n_win) / sigma) ** 2)
        lo_active[-n_win:] *= np.exp(-0.5 * (idx           / sigma) ** 2)
        ch2_sig = np.zeros(n_total)
        lo_start = n_chirp + n_dead
        ch2_sig[lo_start:lo_start + n_detect] = lo_active
        ch2 = ((ch2_sig + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

        t_detect_actual = n_detect / rate * 1e6
        t_dead_actual   = (n_total - n_chirp - n_detect) / rate * 1e6
        print(f"[CH1] chirp {f_start_hz/1e6:.3f}→{f_stop_hz/1e6:.3f} MHz  "
              f"active {n_chirp/rate*1e6:.3f} µs  ({n_chirp:,} samp)")
        print(f"[CH2] LO {f_lo_hz/1e6:.6f} MHz (exact)  "
              f"detect {t_detect_actual:.3f} µs  dead {t_dead_actual:.3f} µs  "
              f"period {n_total/rate*1e6:.3f} µs  ({n_total:,} samp)")

        with self._lock:
            self._setup(channel=1, reset=True, sample_rate=rate)
            self._upload(ch1, segnum=1)
            self._play(amplitude_vpp, segnum=1)
            self._active_seg[1] = 1

            self._setup(channel=2, reset=False, sample_rate=rate)
            self._upload(ch2, segnum=2)
            self._play(amplitude_vpp, segnum=2)
            self._active_seg[2] = 2

        return n_chirp / rate, f_lo_hz

    def send_chirp_with_lo_duc(self, f_carrier_chirp_hz, f_start_bb_hz, f_stop_bb_hz,
                               chirp_us, dead_us, f_lo_hz, detect_us,
                               amplitude_vpp=0.5, window_frac=0.05):
        """DUC (IQM ONE) version: simultaneous chirp on CH1 and CW on CH2 at high frequencies.

        CH1: analytic (single-sideband) Gaussian-windowed chirp.
             NCO carrier = f_carrier_chirp_hz.
             Baseband sweeps f_start_bb to f_stop_bb → RF output sweeps
             f_carrier_chirp + f_start_bb to f_carrier_chirp + f_stop_bb.
        CH2: gated CW at exactly f_lo_hz (NCO carrier = f_lo_hz, I=1 Q=0).

        Both channels use IQM ONE with the DAC clock at 9 GS/s and x8 interpolation.
        In DUC mode the DAC runs at the interpolated (high) rate while the segment
        memory holds only the low-rate complex baseband (rate_cx = rate / interp),
        so BOTH channels stay active (the 2.5 GS/s "dual max" is an ARB/direct-mode
        memory limit, not a DUC limit). A 9 GS/s clock puts Nyquist at 4.5 GHz, so
        carriers of 2.0–2.5 GHz are generated cleanly (a 2.5 GS/s clock cannot — the
        carrier would sit above its 1.25 GHz Nyquist, near the sinc null → no output).
        """
        # DUC: DAC clock 9 GS/s, x8 interpolation → complex baseband 1.125 GS/s
        # (≤ the 1.25 GS/s ONE-mode maximum, §5.9). rate/rate_cx selects :SOUR:INT below.
        rate    = 9e9                   # DAC sample clock (:FREQ:RAST)
        rate_cx = rate / 8.0            # 1.125 GS/s complex (baseband) sample rate

        n_chirp  = round(chirp_us  * 1e-6 * rate_cx)
        n_dead   = round(dead_us   * 1e-6 * rate_cx)
        n_detect = round(detect_us * 1e-6 * rate_cx)
        # Segment stores 2×n_total uint16 values; must be multiple of 64 → n_total multiple of 32
        n_total = max(int(np.ceil((n_chirp + n_dead + n_detect) / 32)) * 32, 64)

        def _gauss_win(n):
            n_w = max(int(window_frac * n), 1)
            sig = n_w / 3.0
            idx = np.arange(n_w, dtype=np.float64)
            w = np.ones(n)
            w[:n_w]  = np.exp(-0.5 * ((idx - n_w) / sig) ** 2)
            w[-n_w:] = np.exp(-0.5 * (idx / sig) ** 2)
            return w

        # CH1: analytic chirp — I = cos(phase)·win, Q = sin(phase)·win
        t = np.arange(n_chirp, dtype=np.float64) / rate_cx
        T = t[-1] if n_chirp > 1 else 1e-9
        phase = 2 * np.pi * (f_start_bb_hz * t +
                              (f_stop_bb_hz - f_start_bb_hz) / (2 * T) * t ** 2)
        win1 = _gauss_win(n_chirp)
        I1 = np.zeros(n_total);  I1[:n_chirp] = np.cos(phase) * win1
        Q1 = np.zeros(n_total);  Q1[:n_chirp] = np.sin(phase) * win1

        # CH2: gated CW — I=win during detect window, Q=0 → output at exactly f_lo_hz
        lo_start = n_chirp + n_dead
        win2 = _gauss_win(n_detect)
        I2 = np.zeros(n_total);  I2[lo_start:lo_start + n_detect] = win2
        Q2 = np.zeros(n_total)

        def _to_u16(I, Q):
            iq = np.empty(2 * len(I), dtype=np.float64)
            iq[0::2] = I;  iq[1::2] = Q
            # np.round before astype so midscale (0.0) maps to exactly 32768, not 32767
            return np.round((iq + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

        ch1_u16 = _to_u16(I1, Q1)
        ch2_u16 = _to_u16(I2, Q2)

        print(f"[DUC CH1] {(f_carrier_chirp_hz+f_start_bb_hz)/1e6:.3f}→"
              f"{(f_carrier_chirp_hz+f_stop_bb_hz)/1e6:.3f} MHz  "
              f"carrier {f_carrier_chirp_hz/1e6:.3f} MHz  "
              f"active {n_chirp/rate_cx*1e6:.3f} µs  ({n_chirp:,} cx samp)")
        print(f"[DUC CH2] LO {f_lo_hz/1e6:.6f} MHz (exact NCO)  "
              f"detect {n_detect/rate_cx*1e6:.3f} µs  "
              f"dead {n_dead/rate_cx*1e6:.3f} µs  "
              f"period {n_total/rate_cx*1e6:.3f} µs  ({n_total:,} cx samp)")

        # Interpolation factor: ONE-mode complex (baseband) rate = DAC_rate / INT
        # (manual §5.4/§5.9). complex rate = FREQ:RAST / INT, must be ≤ 1.25 GS/s.
        # rate=9 GS/s, rate_cx=1.125 GS/s → INT X8.
        int_map = {2: "X2", 4: "X4", 8: "X8"}
        int_kw = int_map[round(rate / rate_cx)]

        def _duc_ch(ch, segnum, data_u16, ncof):
            # Ordering matches Tabor's "how to program" DUC recipe and avoids two fw
            # traps found by readback:
            #  - :SOUR:INT is only valid in DUC mode (err 223 "settings conflict" in DIRECT).
            #  - :SOUR:IQM ONE validates complex rate = FREQ:RAST/INT ≤ 1.25 GS/s at the
            #    moment it is issued (err 204 "out of range" if clock is still 9 GHz).
            # So: set clock to the BASEBAND rate first, switch to DUC, set IQM then INT,
            # and only then raise the clock to the interpolated (9 GHz) rate.
            self._cmd(f":INST:CHAN {ch}")
            self._cmd(f":FREQ:RAST {rate_cx:.0f}")   # baseband rate first
            self._cmd(":MODE DUC")
            self._cmd(":SOUR:IQM ONE")               # valid: complex rate = rate_cx ≤ 1.25 GS/s
            self._cmd(f":SOUR:INT {int_kw}")         # now in DUC mode → no conflict
            self._cmd(f":FREQ:RAST {rate:.0f}")      # raise to interpolated DAC clock (9 GHz)
            mode = self._dev.query(":MODE?").strip()
            iqm  = self._dev.query(":SOUR:IQM?").strip()
            print(f"  MODE? -> {mode!r}   IQM? -> {iqm!r}")
            self._cmd(":INIT:CONT ON")
            self._upload(data_u16, segnum=segnum)
            self._cmd(f":FUNC:MODE:SEGM {segnum}")
            self._cmd(f":VOLT {amplitude_vpp:.3f}")
            self._cmd(":NCO:SIXD1 ON")
            self._cmd(f":NCO:CFR1 {ncof:.0f}")
            self._cmd(":OUTP ON")
            self._cmd(f":FREQ:RAST {rate:.0f}")

        def _dump(ch):
            # Read back the state that actually determines DUC behaviour, per channel,
            # AFTER full setup. If IQModulation reads 'NONE' here the interleaved I/Q
            # data is being misinterpreted (§5.9) -> wrong-frequency/garbled output.
            self._dev.write(f":INST:CHAN {ch}")
            print(f"  [CH{ch} state] "
                  f"MODE={self._dev.query(':MODE?').strip()!r} "
                  f"IQM={self._dev.query(':SOUR:IQM?').strip()!r} "
                  f"INT={self._dev.query(':SOUR:INT?').strip()!r} "
                  f"CFR1={self._dev.query(':NCO:CFR1?').strip()!r} "
                  f"RAST={self._dev.query(':FREQ:RAST?').strip()!r} "
                  f"SEGM={self._dev.query(':FUNC:MODE:SEGM?').strip()!r}")

        with self._lock:
            self._dev.write("*CLS; *RST")
            time.sleep(0.5)
            self._dev.query("*OPC?")

            # Diagnostic: confirm the DUC/IQM option is actually installed on this unit.
            print(f"  *OPT? -> {self._dev.query('*OPT?').strip()!r}")

            # NOTE: :IQModulation must be set per-channel *after* :MODE DUC (done in
            # _duc_ch). Setting it here, in DIRECT mode, is silently ignored.
            self._cmd(":INST:CHAN 1")
            self._cmd(":TRAC:DEL:ALL")

            _duc_ch(1, 1, ch1_u16, f_carrier_chirp_hz)
            self._active_seg[1] = 1
            _duc_ch(2, 2, ch2_u16, f_lo_hz)
            self._active_seg[2] = 2

            # Diagnostic: read back final state on both channels.
            _dump(1)
            _dump(2)

        return n_chirp / rate_cx, f_lo_hz

    def send_chirp_with_lo_nco(self, f_carrier_hz, f_start_bb_hz, f_stop_bb_hz,
                                chirp_us, dead_us, f_lo_hz, detect_us,
                                amplitude_vpp=0.5, window_frac=0.05):
        """NCO-mode upconversion — works when IQM ONE is not licensed.

        CH1: real windowed chirp (f_start_bb → f_stop_bb) upconverted by NCO carrier.
             Output is double-sideband:
               upper  = f_carrier + f_start_bb  →  f_carrier + f_stop_bb
               image  = f_carrier - f_stop_bb   →  f_carrier - f_start_bb
        CH2: gaussian-windowed amplitude envelope gating the NCO carrier → pure
             CW at exactly f_lo_hz during the detect window, zero elsewhere.

        Both channels run at SAMPLE_RATE_DUAL (2.5 GS/s real sample rate).
        """
        rate = SAMPLE_RATE_DUAL
        n_chirp  = round(chirp_us  * 1e-6 * rate)
        n_dead   = round(dead_us   * 1e-6 * rate)
        n_detect = round(detect_us * 1e-6 * rate)
        n_total  = max(int(np.ceil((n_chirp + n_dead + n_detect) / 64)) * 64, 128)

        # CH1: real chirp — same waveform as direct mode, NCO does the upconversion
        ch1 = self._chirp_windowed_u16(f_start_bb_hz, f_stop_bb_hz, n_chirp, n_total,
                                        rate, window_frac)

        # CH2: gaussian-windowed amplitude envelope during detect window, zeros elsewhere.
        # In NCO mode: output = envelope(t) × cos(2π f_lo t) → gated CW at f_lo.
        lo_start = n_chirp + n_dead
        n_win    = max(int(window_frac * n_detect), 1)
        sig_     = n_win / 3.0
        idx      = np.arange(n_win, dtype=np.float64)
        lo_env   = np.zeros(n_total)
        lo_env[lo_start:lo_start + n_detect]                         = 1.0
        lo_env[lo_start:lo_start + n_win]                           *= np.exp(-0.5 * ((idx - n_win) / sig_) ** 2)
        lo_env[lo_start + n_detect - n_win:lo_start + n_detect]     *= np.exp(-0.5 * (idx           / sig_) ** 2)
        ch2 = np.round((lo_env + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

        print(f"[NCO CH1] USB {(f_carrier_hz+f_start_bb_hz)/1e6:.1f}→{(f_carrier_hz+f_stop_bb_hz)/1e6:.1f} MHz  "
              f"image {(f_carrier_hz-f_stop_bb_hz)/1e6:.1f}→{(f_carrier_hz-f_start_bb_hz)/1e6:.1f} MHz  "
              f"carrier {f_carrier_hz/1e6:.3f} MHz  active {n_chirp/rate*1e6:.3f} µs")
        print(f"[NCO CH2] LO {f_lo_hz/1e6:.6f} MHz (NCO exact)  "
              f"detect {n_detect/rate*1e6:.3f} µs  period {n_total/rate*1e6:.3f} µs")

        def _nco_ch(ch, segnum, data_u16, ncof):
            self._cmd(f":INST:CHAN {ch}")
            self._cmd(":MODE NCO")
            mode_actual = self._dev.query(":MODE?").strip()
            print(f"  CH{ch} MODE? -> {mode_actual!r}  (expected 'NCO')")
            self._cmd(f":FREQ:RAST {rate:.0f}")
            self._cmd(":INIT:CONT ON")
            self._upload(data_u16, segnum=segnum)
            self._cmd(f":FUNC:MODE:SEGM {segnum}")
            self._cmd(f":VOLT {amplitude_vpp:.3f}")
            self._cmd(":NCO:SIXD1 ON")
            self._cmd(f":NCO:CFR1 {ncof:.0f}")
            self._cmd(":OUTP ON")

        with self._lock:
            self._dev.write("*CLS; *RST")
            time.sleep(0.5)
            self._dev.query("*OPC?")
            self._cmd(":INST:CHAN 1")
            self._cmd(":TRAC:DEL:ALL")
            _nco_ch(1, 1, ch1, f_carrier_hz)
            self._active_seg[1] = 1
            _nco_ch(2, 2, ch2, f_lo_hz)
            self._active_seg[2] = 2

        return n_chirp / rate, f_lo_hz

    def send_iq_sine(self, frequency_hz=100e6, amplitude_vpp=0.5, exact=False):
        """Output I (sine) on CH1 and Q (cosine, 90° shifted) on CH2 simultaneously.

        Both channels share the same DAC clock so they are always phase-locked.
        The rate is computed once from CH1 and reused for CH2 — no mismatch possible.
        Requires dual-channel mode (sample_rate <= SAMPLE_RATE_DUAL = 2.5 GS/s).
        """
        cycles, rate, actual = self._resolve(frequency_hz, exact)
        err_hz = actual - frequency_hz
        print(f"[IQ] {frequency_hz/1e6:.6f} MHz -> {actual/1e6:.6f} MHz "
              f"(err {err_hz:+.1f} Hz, Fs={rate/1e9:.6f} GS/s)")

        t = np.arange(self.n_samples)
        i_wave = ((np.sin(2 * np.pi * cycles * t / self.n_samples) + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)
        q_wave = ((np.cos(2 * np.pi * cycles * t / self.n_samples) + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

        with self._lock:
            # Reset and configure CH1 (sets the shared clock rate for both channels)
            self._setup(channel=1, reset=True, sample_rate=rate)
            self._upload(i_wave, segnum=1)
            self._play(amplitude_vpp, segnum=1)
            self._active_seg[1] = 1

            # Configure CH2 without reset — clock rate already set
            self._setup(channel=2, reset=False, sample_rate=rate)
            self._upload(q_wave, segnum=2)
            self._play(amplitude_vpp, segnum=2)
            self._active_seg[2] = 2

        return actual

    def send_ramp(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact) if frequency_hz else (1, self.sample_rate, self.sample_rate / self.n_samples)
        print(f"[CH{channel}] Ramp -> {actual/1e6:.6f} MHz (err {actual - (frequency_hz or actual):+.1f} Hz)")
        with self._lock:
            self._setup(channel, reset, sample_rate=rate)
            self._upload(self._ramp_u16(cycles), segnum=channel)
            self._play(amplitude_vpp, segnum=channel)

    def send_square(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact) if frequency_hz else (1, self.sample_rate, self.sample_rate / self.n_samples)
        print(f"[CH{channel}] Square -> {actual/1e6:.6f} MHz (err {actual - (frequency_hz or actual):+.1f} Hz)")
        with self._lock:
            self._setup(channel, reset, sample_rate=rate)
            self._upload(self._square_u16(cycles), segnum=channel)
            self._play(amplitude_vpp, segnum=channel)

    def stop(self):
        with self._lock:
            for ch in [1, 2]:
                self._dev.write(f":INST:CHAN {ch}")
                time.sleep(0.05)
                self._dev.write(":INIT:CONT OFF")
                time.sleep(0.05)
                self._dev.write(":ABOR")
                time.sleep(0.05)
                self._dev.write(":OUTP OFF")
                time.sleep(0.05)
        print("AWG stopped")

    def close(self):
        self.stop()
        self._dev.close()