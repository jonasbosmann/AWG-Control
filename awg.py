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
        """Switch to a pre-loaded segment using direct writes (no _cmd overhead)."""
        with self._lock:
            self._dev.write(f":INST:CHAN {channel}")
            self._dev.write(f":FUNC:MODE:SEGM {segnum}")
            self._dev.write(f":VOLT {amplitude_vpp:.3f}")
            self._dev.write(f":OUTP ON")
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