import pyvisa
import numpy as np
import time
import threading

_RESOURCE = "TCPIP::169.254.4.20::4000::SOCKET"


class Scope:
    def __init__(self):
        self._lock = threading.Lock()
        self._pre = {}   # cached: xincr, ymult, yzero, rl (cleared on timebase change)
        rm = pyvisa.ResourceManager()
        self._dev = rm.open_resource(_RESOURCE)
        self._dev.read_termination = '\n'
        self._dev.write_termination = '\n'
        self._dev.timeout = 3000
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
            self._dev.write("ACQuire:STATE RUN")
        self._pre.clear()
        mode = f"{n_averages}× avg" if n_averages > 1 else "sample"
        print(f"Scope: CH{channel} @ 50 Ω, {mode}\n")

    def set_timebase_direct(self, seconds_per_div):
        self._pre.clear()
        with self._lock:
            self._dev.write(f"HORizontal:SCAle {seconds_per_div:.3e}")

    def set_timebase(self, freq_hz, n_cycles=8):
        self._pre.clear()
        scale = max(1e-10, min(n_cycles / (freq_hz * 10), 1.0))
        with self._lock:
            self._dev.write(f"HORizontal:SCAle {scale:.3e}")

    def get_waveform(self, channel=1, max_points=10000):
        """Read current waveform buffer without stopping acquisition.

        Preamble (xincr, ymult, yzero, record length) is cached and reused across
        calls until the timebase changes — saves 4 queries per live-view frame.
        """
        with self._lock:
            self._dev.write(f"DATa:SOUrce CH{channel}")
            self._dev.write("DATa:ENCdg ASCIi")
            self._dev.write("DATa:STARt 1")
            if 'xincr' not in self._pre:
                self._pre['xincr'] = float(self._dev.query("WFMOutpre:XINcr?"))
                self._pre['ymult'] = float(self._dev.query("WFMOutpre:YMUlt?"))
                self._pre['yzero'] = float(self._dev.query("WFMOutpre:YZEro?"))
                self._pre['rl']    = int(self._dev.query("HORizontal:RECOrdlength?"))
            pts = min(max_points, self._pre['rl'])
            self._dev.write(f"DATa:STOP {pts}")
            raw_str = self._dev.query("CURVe?")
        raw = np.array([int(x) for x in raw_str.split(',')])
        time_s    = np.arange(len(raw)) * self._pre['xincr']
        voltage_v = raw * self._pre['ymult'] + self._pre['yzero']
        return time_s, voltage_v

    def measure_vpp(self, channel=1, settle=0.0, timeout_ms=1000):
        """Capture one fresh acquisition (SEQuence + *OPC?) and return PK2PK voltage.

        The scope is told to capture after this function is called, so the waveform
        is guaranteed to be at the current AWG frequency — no settle-time guessing.
        TCP command ordering ensures HORizontal:SCAle is applied before the trigger fires.
        """
        if settle > 0:
            time.sleep(settle)
        with self._lock:
            self._dev.write("ACQuire:STOPAfter SEQuence")
            self._dev.write("ACQuire:STATE RUN")
            old_to = self._dev.timeout
            self._dev.timeout = timeout_ms
            try:
                self._dev.query("*OPC?")   # blocks until the one acquisition completes
            except Exception:
                pass   # timeout = no trigger; fall through and read whatever is there
            finally:
                self._dev.timeout = old_to
            self._dev.write(f"DATa:SOUrce CH{channel}")
            self._dev.write("DATa:ENCdg ASCIi")
            self._dev.write("DATa:STARt 1")
            pts = min(500, self._pre.get('rl', 500))
            self._dev.write(f"DATa:STOP {pts}")
            raw_str = self._dev.query("CURVe?")
            # Restart continuous acquisition so live view keeps working
            self._dev.write("ACQuire:STOPAfter RUNSTop")
            self._dev.write("ACQuire:STATE RUN")
        raw = np.array([int(x) for x in raw_str.split(',')])
        ymult = self._pre.get('ymult', 1.0)
        yzero = self._pre.get('yzero', 0.0)
        return float(np.ptp(raw * ymult + yzero))

    def close(self):
        self._dev.close()
