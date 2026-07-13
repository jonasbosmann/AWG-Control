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
        self._rate_sent  = {}        # last :FREQ:RAST actually sent per channel
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
            self._rate_sent.clear()
        self._cmd(f":INST:CHAN {channel}")
        # A previous DUC run leaves the channel in DUC mode (survives reset=False),
        # where real waveforms would be misread as interleaved I/Q — force DIRect.
        if self._dev.query(":MODE?").strip() != "DIR":
            self._cmd(":MODE DIR")
        self._cmd(f":FREQ:RAST {rate:.0f}")
        self._rate_sent[channel] = rate
        if reset:
            self._cmd(":TRAC:DEL:ALL")
        self._cmd(":INIT:CONT ON")

    def _upload(self, wave_u16, segnum=1):
        n = len(wave_u16)
        # Proteus limits (prog. manual p.133-134, P948x): segments 1-128 are
        # "short/fast" segments, minimum 128 points; higher segment numbers are
        # regular segments, minimum 2048 points. Granularity is 32 points.
        min_len = 128 if segnum <= 128 else 2048
        if n < min_len or n % 32:
            raise ValueError(
                f"segment {segnum}: length {n} invalid (min {min_len}, multiple of 32)")
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

    @staticmethod
    def _real_to_u16(wave):
        """Direct-mode offset binary: [-1, 1] → 0..65535 (manual §9.1).
        np.round so midscale (0.0) maps to exactly 32768, not truncated 32767."""
        return np.round((np.asarray(wave) + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

    @staticmethod
    def _gauss_win(n, frac):
        """Gaussian-edged flat-top window of length n; edges span frac·n samples.
        Edge span is capped at n//2 so head and tail can never overlap."""
        w = np.ones(n)
        if n < 2:
            return w
        n_w = min(max(int(frac * n), 1), n // 2)
        sigma = n_w / 3.0
        idx = np.arange(n_w, dtype=np.float64)
        w[:n_w]  = np.exp(-0.5 * ((idx - n_w) / sigma) ** 2)
        w[-n_w:] = np.exp(-0.5 * (idx / sigma) ** 2)
        return w

    def _sine_u16(self, cycles):
        t = np.arange(self.n_samples)
        return self._real_to_u16(np.sin(2 * np.pi * cycles * t / self.n_samples))

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

    MIN_RATE = 1e9   # :FREQ:RAST legal range is 1e9–9e9 (manual §5.10)

    def _resolve(self, frequency_hz, exact=False):
        """Return (cycles, effective_sample_rate, actual_frequency).

        exact=False: keep self.sample_rate, round cycles → frequency error ≤ freq_step/2
        exact=True:  keep cycles integer, back-calculate sample rate → zero frequency error.
                     The rate is kept within [MIN_RATE, self.sample_rate] (self.sample_rate
                     is the max for the current channel mode — 9 GS/s single, 2.5 GS/s dual);
                     if no integer cycle count fits, falls back to the non-exact rate.
        """
        if frequency_hz > self.sample_rate / 2:
            raise ValueError(
                f"{frequency_hz/1e6:.1f} MHz is above Nyquist "
                f"({self.sample_rate/2e6:.0f} MHz at {self.sample_rate/1e9:g} GS/s) — "
                f"output would alias")
        if exact:
            fn = frequency_hz * self.n_samples
            cyc_min = max(int(np.ceil(fn / self.sample_rate)), 1)   # rate ≤ mode max
            cyc_max = min(int(fn // self.MIN_RATE), self.n_samples // 2)  # rate ≥ 1 GS/s, ≤ Nyquist
            if cyc_min <= cyc_max:
                cycles = cyc_min
                rate = fn / cycles
            else:
                print(f"  exact mode impossible for {frequency_hz/1e6:.3f} MHz "
                      f"(needs clock outside {self.MIN_RATE/1e9:g}-"
                      f"{self.sample_rate/1e9:g} GS/s) — using rounded cycles")
                cycles = self._cycles(frequency_hz)
                rate = self.sample_rate
        else:
            cycles = self._cycles(frequency_hz)
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
            # Compare against the rate actually sent to the device (an earlier
            # exact=True call may have moved the clock away from self.sample_rate).
            if rate != self._rate_sent.get(channel):
                self._cmd(f":FREQ:RAST {rate:.0f}")
                self._rate_sent[channel] = rate
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
        sig = np.sin(phase) * self._gauss_win(n_active, window_frac)
        full = np.zeros(n_total)
        full[:n_active] = sig
        return self._real_to_u16(full)

    def _cw_sine_u16(self, cycles, n_samples):
        t = np.arange(n_samples)
        return self._real_to_u16(np.sin(2 * np.pi * cycles * t / n_samples))

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
        lo_active = np.sin(2 * np.pi * f_lo_hz / rate * t_lo) * self._gauss_win(n_detect, window_frac)
        ch2_sig = np.zeros(n_total)
        lo_start = n_chirp + n_dead
        ch2_sig[lo_start:lo_start + n_detect] = lo_active
        ch2 = self._real_to_u16(ch2_sig)

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

        # CH1: analytic chirp — I = cos(phase)·win, Q = sin(phase)·win
        t = np.arange(n_chirp, dtype=np.float64) / rate_cx
        T = t[-1] if n_chirp > 1 else 1e-9
        phase = 2 * np.pi * (f_start_bb_hz * t +
                              (f_stop_bb_hz - f_start_bb_hz) / (2 * T) * t ** 2)
        win1 = self._gauss_win(n_chirp, window_frac)
        I1 = np.zeros(n_total);  I1[:n_chirp] = np.cos(phase) * win1
        Q1 = np.zeros(n_total);  Q1[:n_chirp] = np.sin(phase) * win1

        # CH2: gated CW — I=win during detect window, Q=0 → output at exactly f_lo_hz
        lo_start = n_chirp + n_dead
        win2 = self._gauss_win(n_detect, window_frac)
        I2 = np.zeros(n_total);  I2[lo_start:lo_start + n_detect] = win2
        Q2 = np.zeros(n_total)

        ch1_u16 = self._iq_to_u16(I1, Q1)
        ch2_u16 = self._iq_to_u16(I2, Q2)

        print(f"[DUC CH1] {(f_carrier_chirp_hz+f_start_bb_hz)/1e6:.3f}→"
              f"{(f_carrier_chirp_hz+f_stop_bb_hz)/1e6:.3f} MHz  "
              f"carrier {f_carrier_chirp_hz/1e6:.3f} MHz  "
              f"active {n_chirp/rate_cx*1e6:.3f} µs  ({n_chirp:,} cx samp)")
        print(f"[DUC CH2] LO {f_lo_hz/1e6:.6f} MHz (exact NCO)  "
              f"detect {n_detect/rate_cx*1e6:.3f} µs  "
              f"dead {n_dead/rate_cx*1e6:.3f} µs  "
              f"period {n_total/rate_cx*1e6:.3f} µs  ({n_total:,} cx samp)")

        def _duc_ch(ch, segnum, data_u16, ncof):
            # Shared verified recipe (ordering + IQM readback guard) — see
            # _duc_upload_ch. An earlier local copy set :SOUR:IQM ONE before
            # :SOUR:INT / the final clock, which fw 1.237.0 silently ignores
            # (IQM stayed NONE → garbled multi-tone output).
            self._duc_upload_ch(ch, segnum, data_u16, ncof, rate, rate_cx,
                                amplitude_vpp)

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
            self._rate_sent.clear()

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

    # NOTE: an earlier send_chirp_with_lo_nco used :MODE NCO to "upconvert" an
    # uploaded chirp/envelope. Removed: NCO mode plays NO waveform memory — it
    # internally generates a plain sine at :NCO:CFR (prog. manual §5.2 p.87-88;
    # DUC primer fig 1.4 "no modulation"), so both channels were just CW.
    # Use send_chirp_with_lo_duc (IQM ONE) for upconverted chirps.

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
        i_wave = self._real_to_u16(np.sin(2 * np.pi * cycles * t / self.n_samples))
        q_wave = self._real_to_u16(np.cos(2 * np.pi * cycles * t / self.n_samples))

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
            self._active_seg[channel] = channel
        return actual

    def send_square(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact) if frequency_hz else (1, self.sample_rate, self.sample_rate / self.n_samples)
        print(f"[CH{channel}] Square -> {actual/1e6:.6f} MHz (err {actual - (frequency_hz or actual):+.1f} Hz)")
        with self._lock:
            self._setup(channel, reset, sample_rate=rate)
            self._upload(self._square_u16(cycles), segnum=channel)
            self._play(amplitude_vpp, segnum=channel)
            self._active_seg[channel] = channel
        return actual

    # ── CH2-as-trigger comparison (direct vs DUC) ──────────────────
    # CH1 = signal under test (-> scope CH1); CH2 = start-of-buffer sync pulse
    # (-> scope CH2, used as the scope trigger). Both channels share one buffer
    # period, so the CH2 pulse marks the identical CH1 phase every period ->
    # coherent scope triggering / averaging without the hardware marker.

    def _sync_pulse_u16(self, n_total, n_pulse):
        """Real start-of-buffer pulse: 0 V baseline, +full for the first n_pulse
        samples -> one clean rising edge per period at sample 0."""
        w = np.zeros(n_total)
        w[:max(n_pulse, 1)] = 1.0
        return self._real_to_u16(w)

    @staticmethod
    def _iq_to_u16(I, Q):
        """Interleave complex baseband [I0,Q0,I1,Q1,...] as uint16 for DUC/IQM.

        IQ data must use codes 1..65535 symmetric about 32768, NOT the full
        0..65535 range: with code 0 included the negative half has one more
        level than the positive half, and that DC bias shows up as a residual
        carrier spur at the NCO frequency (~-78 dBc; prog. manual §5.9 p.95,
        DUC primer fig 2.3). Matches Tabor's myQuantization(minLevel=1)."""
        iq = np.empty(2 * len(I), dtype=np.float64)
        iq[0::2] = I
        iq[1::2] = Q
        return (np.round(iq * 32767.0) + 32768.0).clip(1, 65535).astype(np.uint16)

    def _duc_upload_ch(self, ch, segnum, data_u16, ncof, rate, rate_cx, amp):
        """Per-channel DUC/IQM-ONE upload.

        Ordering verified per-command against fw 1.237.0 (2026-07-13 probe):
          1. :MODE DUC, then :SOUR:INT. The firmware flags a spurious
             '22, Invalid argument' on :SOUR:INT while the clock is still low
             but APPLIES the value anyway — so it is verified by readback
             below, not by the error code.
          2. :FREQ:RAST to the interpolated DAC clock — must come after INT
             (with INT still NONE, a 9 GHz clock implies complex rate 9 GS/s
             > 1.25 GS/s → '223, settings conflict' on everything after).
          3. :SOUR:IQM ONE LAST. Sent any earlier (INT still NONE, clock
             still low) the firmware SILENTLY ignores it: no SCPI error,
             readback stays NONE, and the interleaved I/Q segment plays as a
             garbled multi-tone real waveform instead of a clean carrier.
        """
        int_kw = {2: "X2", 4: "X4", 8: "X8"}[round(rate / rate_cx)]
        self._cmd(f":INST:CHAN {ch}")
        self._cmd(":MODE DUC")
        self._dev.write(f":SOUR:INT {int_kw}")   # known-spurious err 22
        time.sleep(0.05)
        self._dev.query(":SYST:ERR?")            # drain it; verified below
        self._cmd(f":FREQ:RAST {rate:.0f}")
        self._cmd(":SOUR:IQM ONE")
        iqm  = self._dev.query(":SOUR:IQM?").strip()
        intp = self._dev.query(":SOUR:INT?").strip()
        if iqm != "ONE" or intp != int_kw:
            raise RuntimeError(
                f"CH{ch} DUC setup failed: IQM={iqm!r} (want 'ONE'), "
                f"INT={intp!r} (want {int_kw!r}) — output would be garbled")
        self._cmd(":INIT:CONT ON")
        self._upload(data_u16, segnum=segnum)
        self._cmd(f":FUNC:MODE:SEGM {segnum}")
        self._cmd(f":VOLT {amp:.3f}")
        self._cmd(":NCO:SIXD1 ON")
        self._cmd(f":NCO:CFR1 {ncof:.0f}")
        self._cmd(":OUTP ON")

    def send_cw_direct_sync(self, freq_hz, amplitude_vpp=0.5,
                            sync_pulse_ns=40, sync_amp_vpp=0.8):
        """Direct-mode CW on CH1 + sync pulse on CH2 (2.5 GS/s dual). Returns actual Hz."""
        rate = SAMPLE_RATE_DUAL
        if freq_hz > rate / 2:
            raise ValueError(f"{freq_hz/1e6:.1f} MHz is above Nyquist "
                             f"({rate/2e6:.0f} MHz) — use the DUC mode instead")
        n_total = self.n_samples
        cycles = max(round(freq_hz * n_total / rate), 1)
        actual = cycles * rate / n_total
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate), 1)
        ch1 = self._cw_sine_u16(cycles, n_total)
        ch2 = self._sync_pulse_u16(n_total, n_pulse)
        print(f"[DIRECT CW] CH1 {actual/1e6:.4f} MHz ({cycles} cyc)  "
              f"CH2 sync {n_pulse/rate*1e9:.1f} ns @ {sync_amp_vpp} Vpp")
        with self._lock:
            self._setup(1, reset=True,  sample_rate=rate)
            self._upload(ch1, segnum=1); self._play(amplitude_vpp, segnum=1)
            self._active_seg[1] = 1
            self._setup(2, reset=False, sample_rate=rate)
            self._upload(ch2, segnum=2); self._play(sync_amp_vpp, segnum=2)
            self._active_seg[2] = 2
        return actual

    def send_chirp_direct_sync(self, f_start_hz, f_stop_hz, chirp_us, dead_us,
                               amplitude_vpp=0.5, sync_pulse_ns=40, sync_amp_vpp=0.8,
                               window_frac=0.05):
        """Direct-mode windowed chirp on CH1 + sync pulse on CH2 (2.5 GS/s dual).
        Returns chirp active duration (s)."""
        rate = SAMPLE_RATE_DUAL
        n_chirp = round(chirp_us * 1e-6 * rate)
        n_dead  = round(dead_us  * 1e-6 * rate)
        n_total = max(int(np.ceil((n_chirp + n_dead) / 64)) * 64, 128)
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate), 1)
        ch1 = self._chirp_windowed_u16(f_start_hz, f_stop_hz, n_chirp, n_total, rate, window_frac)
        ch2 = self._sync_pulse_u16(n_total, n_pulse)
        print(f"[DIRECT CHIRP] CH1 {f_start_hz/1e6:.1f}->{f_stop_hz/1e6:.1f} MHz  "
              f"active {n_chirp/rate*1e6:.3f} us  period {n_total/rate*1e6:.3f} us  "
              f"CH2 sync {n_pulse/rate*1e9:.1f} ns")
        with self._lock:
            self._setup(1, reset=True,  sample_rate=rate)
            self._upload(ch1, segnum=1); self._play(amplitude_vpp, segnum=1)
            self._active_seg[1] = 1
            self._setup(2, reset=False, sample_rate=rate)
            self._upload(ch2, segnum=2); self._play(sync_amp_vpp, segnum=2)
            self._active_seg[2] = 2
        return n_chirp / rate

    def send_cw_duc_sync(self, carrier_hz, amplitude_vpp=0.5, n_total=4096,
                         sync_pulse_ns=40, sync_amp_vpp=0.8, sync_carrier_hz=None):
        """DUC/IQM-ONE CW on CH1 (pure carrier at carrier_hz) + gated-burst sync on
        CH2 at buffer start (9 GS/s DAC, X8). Returns carrier_hz."""
        rate = 9e9
        rate_cx = rate / 8.0
        n_total = max(int(np.ceil(n_total / 32)) * 32, 64)   # 2*n_total multiple of 64
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate_cx), 1)
        sync_carrier = sync_carrier_hz if sync_carrier_hz is not None else carrier_hz

        # CH1: constant complex baseband (I=1,Q=0) -> pure carrier via NCO.
        ch1 = self._iq_to_u16(np.ones(n_total), np.zeros(n_total))
        # CH2: I-envelope gate for the first n_pulse samples -> RF burst at t=0.
        I2 = np.zeros(n_total); I2[:n_pulse] = 1.0
        ch2 = self._iq_to_u16(I2, np.zeros(n_total))
        print(f"[DUC CW] CH1 {carrier_hz/1e6:.4f} MHz  "
              f"CH2 burst {n_pulse/rate_cx*1e9:.1f} ns @ {sync_carrier/1e6:.1f} MHz")
        with self._lock:
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, ch1, carrier_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, ch2, sync_carrier, rate, rate_cx, sync_amp_vpp)
            self._active_seg[2] = 2
        return carrier_hz

    def duc_cw_setup(self, freq_hz, amplitude_vpp=0.5, channel=1, n_total=4096):
        """One-time DUC/IQM-ONE CW setup for an NCO-stepped frequency sweep.

        Uploads a constant complex baseband (I=1, Q=0) once — the output is a
        pure carrier at the NCO frequency. duc_cw_step() then retunes :NCO:CFR1
        only, with no waveform re-upload, so each sweep step costs milliseconds
        and every frequency is exact (no integer-cycles quantization).
        9 GS/s DAC clock, X8 interpolation → NCO reaches the 4.5 GHz Nyquist.
        """
        rate = 9e9
        rate_cx = rate / 8.0
        if not 0 < freq_hz <= rate / 2:
            raise ValueError(f"{freq_hz/1e6:.1f} MHz outside DUC NCO range "
                             f"(0–{rate/2e9:g} GHz]")
        n_total = max(int(np.ceil(n_total / 32)) * 32, 64)
        ch1 = self._iq_to_u16(np.ones(n_total), np.zeros(n_total))
        print(f"[DUC CW sweep] CH{channel} start {freq_hz/1e6:.4f} MHz  "
              f"(9 GS/s DAC, X8, NCO-stepped)")
        with self._lock:
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
            self._rate_sent.clear()
            self._cmd(f":INST:CHAN {channel}"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(channel, 1, ch1, freq_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[channel] = 1
        return freq_hz

    def duc_cw_step(self, freq_hz, channel=1):
        """Retune the NCO carrier of a channel prepared by duc_cw_setup().

        Only :NCO:CFR1 changes — the constant baseband keeps playing. *OPC?
        blocks until the AWG has processed the retune (same pattern as
        sweep_step). Raw writes, no per-command error polling, for speed.
        """
        if not 0 < freq_hz <= 4.5e9:
            raise ValueError(f"{freq_hz/1e6:.1f} MHz outside DUC NCO range "
                             f"(0–4.5 GHz]")
        with self._lock:
            self._dev.write(f":INST:CHAN {channel}")
            self._dev.write(f":NCO:CFR1 {freq_hz:.0f}")
            self._dev.query("*OPC?")
        return freq_hz

    def send_chirp_duc_sync(self, f_carrier_chirp_hz, f_start_bb_hz, f_stop_bb_hz,
                            chirp_us, dead_us, amplitude_vpp=0.5, window_frac=0.05,
                            sync_pulse_ns=40, sync_amp_vpp=0.8, sync_carrier_hz=None):
        """DUC/IQM-ONE analytic chirp on CH1 + gated-burst sync on CH2 at buffer
        start (9 GS/s DAC, X8). Returns chirp active duration (s)."""
        rate = 9e9
        rate_cx = rate / 8.0
        n_chirp = round(chirp_us * 1e-6 * rate_cx)
        n_dead  = round(dead_us  * 1e-6 * rate_cx)
        n_total = max(int(np.ceil((n_chirp + n_dead) / 32)) * 32, 64)
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate_cx), 1)
        sync_carrier = sync_carrier_hz if sync_carrier_hz is not None else f_carrier_chirp_hz

        # CH1: analytic (single-sideband) windowed chirp.
        t = np.arange(n_chirp, dtype=np.float64) / rate_cx
        T = t[-1] if n_chirp > 1 else 1e-9
        phase = 2 * np.pi * (f_start_bb_hz * t +
                             (f_stop_bb_hz - f_start_bb_hz) / (2 * T) * t ** 2)
        win = self._gauss_win(n_chirp, window_frac)
        I1 = np.zeros(n_total); I1[:n_chirp] = np.cos(phase) * win
        Q1 = np.zeros(n_total); Q1[:n_chirp] = np.sin(phase) * win
        ch1 = self._iq_to_u16(I1, Q1)
        # CH2: burst at buffer start.
        I2 = np.zeros(n_total); I2[:n_pulse] = 1.0
        ch2 = self._iq_to_u16(I2, np.zeros(n_total))
        print(f"[DUC CHIRP] CH1 {(f_carrier_chirp_hz+f_start_bb_hz)/1e6:.1f}->"
              f"{(f_carrier_chirp_hz+f_stop_bb_hz)/1e6:.1f} MHz  active {n_chirp/rate_cx*1e6:.3f} us  "
              f"period {n_total/rate_cx*1e6:.3f} us  CH2 burst {n_pulse/rate_cx*1e9:.1f} ns")
        with self._lock:
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, ch1, f_carrier_chirp_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, ch2, sync_carrier, rate, rate_cx, sync_amp_vpp)
            self._active_seg[2] = 2
        return n_chirp / rate_cx

    def stop(self):
        # No :ABORt here — the Proteus rejects it ('209, illegal/unknown scpi'
        # on fw 1.237.0); :INIT:CONT OFF + :OUTP OFF is the whole recipe.
        with self._lock:
            for ch in [1, 2]:
                self._dev.write(f":INST:CHAN {ch}")
                time.sleep(0.05)
                self._dev.write(":INIT:CONT OFF")
                time.sleep(0.05)
                self._dev.write(":OUTP OFF")
                time.sleep(0.05)
        print("AWG stopped")

    def close(self):
        self.stop()
        self._dev.close()