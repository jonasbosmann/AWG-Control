import pyvisa
import numpy as np
import time
import threading

_RESOURCE = "TCPIP::169.254.4.20::4000::SOCKET"


class Scope:
    def __init__(self):
        self._lock = threading.Lock()
        rm = pyvisa.ResourceManager()
        self._dev = rm.open_resource(_RESOURCE)
        self._dev.read_termination = '\n'
        self._dev.write_termination = '\n'
        self._dev.timeout = 3000
        idn = None
        for attempt in range(2):
            try:
                self._dev.clear()
            except Exception:
                pass
            self._dev.write("*CLS")
            try:
                idn = self._dev.query("*IDN?").strip()
                break
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(0.3)
        print("Scope:", idn)
        self._dev.timeout = 10000

    def setup(self, channel=1, n_averages=1):
        with self._lock:
            self._dev.write(f"CH{channel}:TERmination 50")
            if n_averages > 1:
                self._dev.write("ACQuire:MODe AVErage")
                self._dev.write(f"ACQuire:NUMAVg {n_averages}")
            else:
                self._dev.write("ACQuire:MODe SAMple")
            self._dev.write("TRIGger:A:TYPe EDGE")
            self._dev.write(f"TRIGger:A:EDGE:SOUrce CH{channel}")
            self._dev.write("TRIGger:A:EDGE:SLOpe RISe")
            self._dev.write("TRIGger:A:MODe AUTO")
            self._dev.write(f"TRIGger:A:LEVEL:CH{channel} 0.0")
            self._dev.write("ACQuire:STOPAfter RUNSTop")
            # Do NOT send ACQuire:STATE RUN — that puts the scope into a SCPI-controlled
            # acquisition that blocks all USB reads for up to 30 s waiting for a trigger.
            # The scope runs continuously from the front panel; we only read its buffer.
        mode = f"{n_averages}× avg" if n_averages > 1 else "sample"
        print(f"Scope: CH{channel} @ 50 Ω, {mode}\n")

    def set_timebase(self, freq_hz, n_cycles=8):
        scale = max(1e-10, min(n_cycles / (freq_hz * 10), 1.0))
        with self._lock:
            self._dev.write(f"HORizontal:SCAle {scale:.3e}")

    def get_waveform(self, channel=1, max_points=2500):
        """Stop scope, read frozen ASCII waveform, restart."""
        with self._lock:
            self._dev.write(f"DATa:SOUrce CH{channel}")
            self._dev.write("DATa:ENCdg ASCIi")
            self._dev.write("DATa:STARt 1")
            self._dev.write(f"DATa:STOP {max_points}")
            time.sleep(0.05)
            self._dev.write("ACQuire:STATE STOP")
            time.sleep(0.05)
            try:
                xincr = float(self._dev.query("WFMOutpre:XINcr?"))
                xzero = float(self._dev.query("WFMOutpre:XZEro?"))
                ymult = float(self._dev.query("WFMOutpre:YMUlt?"))
                yzero = float(self._dev.query("WFMOutpre:YZEro?"))
                raw_str = self._dev.query("CURVe?")
            finally:
                self._dev.write("FPAnel:PRESS RUNSTop")
        raw = np.array([int(x) for x in raw_str.split(',')])
        time_s    = xzero + np.arange(len(raw)) * xincr
        voltage_v = raw * ymult + yzero
        return time_s, voltage_v

    def measure_vpp(self, channel=1, settle=0.05):
        time.sleep(settle)
        _, v = self.get_waveform(channel=channel, max_points=500)
        return float(np.ptp(v))

    def close(self):
        self._dev.close()
