import pyvisa
import numpy as np
import time
import threading

SAMPLE_RATE_SINGLE = 9e9    # single-channel max (both DACs interleaved → CH2 disabled)
SAMPLE_RATE_DUAL   = 2.5e9  # dual-channel max (one DAC per channel)
N_SAMPLES = 2048            # must be multiple of 64 for Proteus granularity

_RESOURCE = "TCPIP0::141.51.196.111::5025::SOCKET"


class AWG:
    def __init__(self, sample_rate=SAMPLE_RATE_SINGLE):
        self.sample_rate = sample_rate
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

    def _sine_u16(self, cycles):
        t = np.arange(N_SAMPLES)
        wave = np.sin(2 * np.pi * cycles * t / N_SAMPLES)
        return ((wave + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

    def _cycles(self, frequency_hz):
        return max(round(frequency_hz * N_SAMPLES / self.sample_rate), 1)

    def send_sine(self, frequency_hz=100e6, amplitude_vpp=0.5, channel=1, reset=True):
        cycles = self._cycles(frequency_hz)
        actual = cycles * self.sample_rate / N_SAMPLES
        print(f"[CH{channel}] Sine {frequency_hz/1e6:.1f} MHz -> {actual/1e6:.3f} MHz ({cycles} cycles)")
        seg = channel
        with self._lock:
            self._setup(channel, reset)
            self._upload(self._sine_u16(cycles), segnum=seg)
            self._play(amplitude_vpp, segnum=seg)
            self._active_seg[channel] = seg
        return actual

    def update_sine(self, frequency_hz, amplitude_vpp=0.5, channel=1):
        cycles = self._cycles(frequency_hz)
        actual = cycles * self.sample_rate / N_SAMPLES
        with self._lock:
            cur = self._active_seg.get(channel, channel)
            next_seg = channel + 20 if cur == channel + 10 else channel + 10
            self._cmd(f":INST:CHAN {channel}")
            self._upload(self._sine_u16(cycles), segnum=next_seg)
            self._play(amplitude_vpp, segnum=next_seg)
            self._active_seg[channel] = next_seg
        return actual

    def send_ramp(self, amplitude_vpp=0.5, channel=1, reset=True):
        print(f"[CH{channel}] Ramp")
        with self._lock:
            self._setup(channel, reset)
            self._upload(np.linspace(0, 65535, N_SAMPLES, dtype=np.uint16), segnum=channel)
            self._play(amplitude_vpp, segnum=channel)

    def send_square(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True):
        cycles = self._cycles(frequency_hz) if frequency_hz else 1
        actual = cycles * self.sample_rate / N_SAMPLES
        print(f"[CH{channel}] Square {cycles} cycle(s) -> {actual/1e6:.3f} MHz")
        spc = N_SAMPLES // cycles
        one_cycle = np.array([0xFFFF] * (spc // 2) + [0] * (spc - spc // 2), dtype=np.uint16)
        with self._lock:
            self._setup(channel, reset)
            self._upload(np.tile(one_cycle, cycles)[:N_SAMPLES], segnum=channel)
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