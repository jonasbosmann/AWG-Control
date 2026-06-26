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

    def _setup(self, channel=1, reset=True):
        if reset:
            self._dev.write("*CLS; *RST")
            time.sleep(0.5)
        self._cmd(f":INST:CHAN {channel}")
        self._cmd(f":FREQ:RAST {self.sample_rate:.0f}")
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

    def send_sine(self, frequency_hz=100e6, amplitude_vpp=0.5, channel=1, reset=True):
        cycles = self._cycles(frequency_hz)
        actual = cycles * self.sample_rate / self.n_samples
        print(f"[CH{channel}] Sine {frequency_hz/1e6:.3f} MHz -> {actual/1e6:.6f} MHz "
              f"({cycles} cycles, step {self.freq_step_hz/1e3:.1f} kHz)")
        seg = channel
        with self._lock:
            self._setup(channel, reset)
            self._upload(self._sine_u16(cycles), segnum=seg)
            self._play(amplitude_vpp, segnum=seg)
            self._active_seg[channel] = seg
        return actual

    def update_sine(self, frequency_hz, amplitude_vpp=0.5, channel=1):
        cycles = self._cycles(frequency_hz)
        actual = cycles * self.sample_rate / self.n_samples
        with self._lock:
            cur = self._active_seg.get(channel, channel)
            next_seg = channel + 20 if cur == channel + 10 else channel + 10
            self._cmd(f":INST:CHAN {channel}")
            self._upload(self._sine_u16(cycles), segnum=next_seg)
            self._play(amplitude_vpp, segnum=next_seg)
            self._active_seg[channel] = next_seg
        return actual

    def send_ramp(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True):
        cycles = self._cycles(frequency_hz) if frequency_hz else 1
        actual = cycles * self.sample_rate / self.n_samples
        print(f"[CH{channel}] Ramp -> {actual/1e6:.6f} MHz")
        with self._lock:
            self._setup(channel, reset)
            self._upload(self._ramp_u16(cycles), segnum=channel)
            self._play(amplitude_vpp, segnum=channel)

    def send_square(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True):
        cycles = self._cycles(frequency_hz) if frequency_hz else 1
        actual = cycles * self.sample_rate / self.n_samples
        print(f"[CH{channel}] Square -> {actual/1e6:.6f} MHz")
        with self._lock:
            self._setup(channel, reset)
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