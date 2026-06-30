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

    def measure_vpp(self, channel=1, settle=0.1, acq_timeout=3.0):
        """Capture one fresh acquisition and return RMS-based Vpp.

        Sequence:
          1. sleep(settle) — AWG glitch transients die away while scope runs freely
          2. SEQuence + RUN — arm scope for exactly one acquisition
          3. Poll ACQuire:STATE? until '0' (STOP) — guarantees the acquisition is
             actually complete before we read (unlike *OPC? over LAN, which returns
             as soon as the command is parsed, not when the hardware finishes)
          4. Read CURVe?, compute AC-RMS → Vpp; RMS is robust to residual spikes
          5. Restore continuous mode for live view
        """
        if settle > 0:
            time.sleep(settle)
        with self._lock:
            self._dev.write("ACQuire:STOPAfter SEQuence")
            self._dev.write("ACQuire:STATE RUN")
            deadline = time.perf_counter() + acq_timeout
            while time.perf_counter() < deadline:
                state = self._dev.query("ACQuire:STATE?").strip()
                if state in ('0', '0.0', 'STOP', 'STOPPED'):
                    break
                time.sleep(0.005)
            else:
                print("measure_vpp: acquisition timeout — reading buffer anyway\n")
            if 'ymult' not in self._pre:
                self._pre['xincr'] = float(self._dev.query("WFMOutpre:XINcr?"))
                self._pre['ymult'] = float(self._dev.query("WFMOutpre:YMUlt?"))
                self._pre['yzero'] = float(self._dev.query("WFMOutpre:YZEro?"))
                self._pre['rl']    = int(self._dev.query("HORizontal:RECOrdlength?"))
            self._dev.write(f"DATa:SOUrce CH{channel}")
            self._dev.write("DATa:ENCdg ASCIi")
            self._dev.write("DATa:STARt 1")
            pts = min(2000, self._pre['rl'])
            self._dev.write(f"DATa:STOP {pts}")
            raw_str = self._dev.query("CURVe?")
            self._dev.write("ACQuire:STOPAfter RUNSTop")
            self._dev.write("ACQuire:STATE RUN")
        raw = np.array([int(x) for x in raw_str.split(',')])
        v = raw * self._pre['ymult'] + self._pre['yzero']
        t = np.arange(len(raw)) * self._pre['xincr']
        v_ac = v - np.mean(v)
        vrms = np.sqrt(np.mean(v_ac ** 2))
        vpp = float(2.0 * np.sqrt(2.0) * vrms)    # Vpp = 2√2 · Vrms for sine
        return vpp, t, v

    def close(self):
        self._dev.close()
