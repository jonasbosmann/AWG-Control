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
        self._dev.timeout = 30000
        self._dev.read_termination = '\n'
        self._dev.write_termination = '\n'
        try:
            self._dev.clear()   # flush stale bytes left in USB buffer from prior session
        except Exception:
            pass
        time.sleep(0.2)
        self._dev.write("*CLS")
        time.sleep(0.1)
        print("Scope:", self._dev.query("*IDN?").strip())

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
            self._dev.write("ACQuire:STATE RUN")
        mode = f"{n_averages}× avg" if n_averages > 1 else "sample"
        print(f"Scope: CH{channel} @ 50 Ω, {mode}, Vpp (IMMed)")

    def measure_vpp(self, settle=0.5, n_readings=1):
        """Read Vpp via immediate measurement.

        n_readings > 1: takes multiple readings in SAMPLE mode and averages them
        in Python — avoids scope-side averaging that doesn't reset between steps.
        """
        time.sleep(settle)
        with self._lock:
            readings = [float(self._dev.query("MEASUrement:IMMed:VALue?"))
                        for _ in range(n_readings)]
        return sum(readings) / len(readings)

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