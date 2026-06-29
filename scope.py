import pyvisa
import numpy as np
import time
import threading

_RESOURCE = "USB0::0x0699::0x0530::C072786::INSTR"


class Scope:
    def __init__(self):
        self._lock = threading.Lock()
        rm = pyvisa.ResourceManager()
        self._dev = rm.open_resource(_RESOURCE)
        self._dev.read_termination = '\n'
        self._dev.write_termination = '\n'
        self._dev.timeout = 3000   # short timeout just for the initial handshake
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
                time.sleep(0.3)   # endpoint settling time before retry
        print("Scope:", idn)
        self._dev.timeout = 30000  # restore long timeout for waveform transfers

    def setup(self, channel=1, n_averages=1):
        """Set 50 Ohm input, acquisition mode, immediate Vpp measurement, and start acquisition.

        n_averages=1 → SAMPLE mode (instant display update).
        n_averages>1 → AVERAGE mode (slower convergence, lower noise floor).
        """
        with self._lock:
            self._dev.write(f"CH{channel}:TERmination 50")
            if n_averages > 1:
                self._dev.write("ACQuire:MODe AVErage")
                self._dev.write(f"ACQuire:NUMAVg {n_averages}")
            else:
                self._dev.write("ACQuire:MODe SAMple")
            self._dev.write(f"MEASUrement:IMMed:SOUrce1 CH{channel}")
            self._dev.write("MEASUrement:IMMed:TYPE PK2PK")
            self._dev.write("TRIGger:A:MODe AUTO")
            self._dev.write("TRIGger:A:LEVEL 0.0")
            # Continuous-run mode: scope keeps acquiring without blocking the USB
            # interface between shots (single-sequence mode blocks for ~30 s waiting
            # for the auto-trigger, which stalls every subsequent SCPI write).
            self._dev.write("ACQuire:STOPAfter RUNSTop")
            self._dev.write("ACQuire:STATE RUN")
        mode = f"{n_averages}× avg" if n_averages > 1 else "sample"
        print(f"Scope: CH{channel} @ 50 Ω, {mode}, Vpp (IMMed)\n")

    def set_timebase(self, freq_hz, n_cycles=8):
        """Set horizontal scale so ~n_cycles of freq_hz fit on screen (10 divisions)."""
        scale = max(1e-10, min(n_cycles / (freq_hz * 10), 1.0))
        with self._lock:
            self._dev.write(f"HORizontal:SCAle {scale:.3e}")

    def measure_vpp(self, channel=1, settle=0.05):
        """Arm a single acquisition, force-trigger it immediately, read Vpp.

        Uses *TRG (IEEE 488.2 software trigger) so the scope fires at once
        instead of waiting up to 30 s for the hardware auto-trigger fallback.
        """
        time.sleep(settle)
        with self._lock:
            self._dev.write("ACQuire:STATE STOP")
            self._dev.write(f"MEASUrement:IMMed:SOUrce1 CH{channel}")
            self._dev.write("MEASUrement:IMMed:TYPE PK2PK")
            self._dev.write("ACQuire:STATE RUN")
            self._dev.write("*TRG")        # fire immediately — no 30 s auto-trigger wait
            time.sleep(0.05)               # 50 ms >> 200 ns acquisition time
            val = float(self._dev.query("MEASUrement:IMMed:VALue?"))
            self._dev.write("ACQuire:STATE RUN")  # restart continuous for live view
        return val

    def get_waveform(self, channel=1, max_points=10000):
        """Transfer the current waveform from CH<channel>; return (time_s, voltage_v).

        max_points caps the transfer length to keep USB latency manageable.
        Sets DATA:STOP before querying preamble so NR_PT reflects the actual transfer window.
        """
        with self._lock:
            self._dev.write(f"DATa:SOUrce CH{channel}")
            self._dev.write("DATa:ENCdg SRIbinary")
            self._dev.write("DATa:WIDth 2")
            self._dev.write("DATa:STARt 1")
            self._dev.write(f"DATa:STOP {max_points}")

            xincr = float(self._dev.query("WFMOutpre:XINcr?"))
            xzero = float(self._dev.query("WFMOutpre:XZEro?"))
            ymult = float(self._dev.query("WFMOutpre:YMUlt?"))
            yoff  = float(self._dev.query("WFMOutpre:YOFf?"))
            yzero = float(self._dev.query("WFMOutpre:YZEro?"))

            raw = self._dev.query_binary_values("CURVe?", datatype='h', is_big_endian=False,
                                                container=np.array, expect_termination=False)

        if len(raw) == 0:
            raise RuntimeError("Scope returned empty waveform — check ACQ state and channel")
        time_s    = xzero + np.arange(len(raw)) * xincr
        voltage_v = (raw - yoff) * ymult + yzero
        return time_s, voltage_v

    def close(self):
        self._dev.close()