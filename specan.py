import pyvisa
import numpy as np
import threading

_RESOURCE = "USB0::0x0957::0x0B0B::MY52220585::INSTR"


class SpecAn:
    def __init__(self, resource=_RESOURCE):
        self._lock = threading.Lock()
        self._trace_mode = "WRIT"
        rm = pyvisa.ResourceManager()
        self._dev = rm.open_resource(resource)
        self._dev.read_termination = '\n'
        self._dev.write_termination = '\n'
        self._dev.timeout = 8000
        # A prior script that timed out mid-query (e.g. a MAXH single-sweep
        # hang, see single_sweep()) can leave the USBTMC endpoint wedged or a
        # stale reply queued — one leftover line shifts EVERY following query
        # response by one, same failure mode as the scope over LAN/USB.
        try:
            self._dev.clear()
        except Exception:
            pass
        self._dev.timeout = 300
        try:
            while True:
                self._dev.read()
        except Exception:
            pass
        self._dev.timeout = 8000
        self._dev.write("*CLS")
        idn = self._dev.query("*IDN?").strip()
        if "N9010A" not in idn.upper():
            raise RuntimeError(f"Unexpected instrument: {idn}")
        print("SpecAn:", idn)
        with self._lock:
            self._dev.write(":INST:SEL SA")
            self._dev.write(":FORM:DATA ASCII")

    def set_freq(self, center_hz, span_hz):
        with self._lock:
            self._dev.write(f":FREQ:CENT {center_hz:.0f}")
            self._dev.write(f":FREQ:SPAN {span_hz:.0f}")

    def set_rbw(self, hz=None):
        """None -> couple RBW to span (AUTO). Otherwise set RBW in Hz."""
        with self._lock:
            if hz is None:
                self._dev.write(":BAND:RES:AUTO ON")
            else:
                self._dev.write(f":BAND:RES {hz:.0f}")

    def set_vbw(self, hz=None):
        """None -> couple VBW to RBW (AUTO). Otherwise set VBW in Hz."""
        with self._lock:
            if hz is None:
                self._dev.write(":BAND:VID:AUTO ON")
            else:
                self._dev.write(f":BAND:VID {hz:.0f}")

    def set_ref_level(self, dbm):
        with self._lock:
            self._dev.write(f":DISP:WIND:TRAC:Y:SCAL:RLEV {dbm:.2f}")

    def set_attenuation(self, db=None):
        """None -> AUTO attenuation. Otherwise fixed attenuation in dB."""
        with self._lock:
            if db is None:
                self._dev.write(":POW:ATT:AUTO ON")
            else:
                self._dev.write(f":POW:ATT {db:.0f}")

    def set_points(self, n):
        """Number of points in the swept trace — controlled independently of
        RBW so the two can't be mismatched into undersampling each other."""
        with self._lock:
            self._dev.write(f":SWE:POIN {int(n)}")

    def get_rbw(self):
        """Query the RBW actually in effect — useful after enabling Auto,
        since the resolved value depends on the instrument's own coupling."""
        with self._lock:
            return float(self._dev.query(":BAND:RES?"))

    def get_sweep_time(self):
        """Actual sweep time in seconds for the current RBW/VBW/span, so
        callers can wait a realistic amount instead of guessing a constant."""
        with self._lock:
            return float(self._dev.query(":SWE:TIME?"))

    def set_average_count(self, n):
        """Number of sweeps averaged together when trace mode is AVER."""
        with self._lock:
            self._dev.write(f":AVER:COUN {int(n)}")

    def set_trace_mode(self, mode="NORM"):
        """mode: 'NORM' (clear/write), 'MAXH' (max hold), 'MINH', or 'AVER'."""
        code = {"NORM": "WRIT", "WRIT": "WRIT", "MAXH": "MAXH",
                "MINH": "MINH", "AVER": "AVER"}[mode.upper()]
        with self._lock:
            self._dev.write(f":TRAC1:TYPE {code}")
        self._trace_mode = code

    def clear_trace(self):
        """Restart accumulation (needed for MAXH/AVER) by toggling through WRITe."""
        with self._lock:
            mode = self._dev.query(":TRAC1:TYPE?").strip()
            if mode != "WRIT":
                self._dev.write(":TRAC1:TYPE WRIT")
                self._dev.write(f":TRAC1:TYPE {mode}")

    def single_sweep(self):
        """Trigger one sweep and block until it completes.

        Only works in Normal ('WRIT') trace mode. Verified on this unit
        (fw A.11.04): with trace type MAXH/MINH/AVER, ':INIT:IMM;*OPC?' never
        completes — not a transport issue, the sweep engine itself never
        signals done for a held/accumulating trace in single-sweep mode
        (confirmed via chained *OPC?, separate *OPC?, and STAT:OPER:COND?
        polling — all hang the same way). Those modes are accumulation-over-
        many-sweeps by nature anyway: use start_continuous() + get_trace()
        instead (i.e. the GUI's Live View), not single_sweep().
        """
        if self._trace_mode != "WRIT":
            raise RuntimeError(
                f"single_sweep() is not supported in trace mode {self._trace_mode!r} "
                "(hangs on this instrument) - use start_continuous() + get_trace() "
                "instead for MAXH/MINH/AVER."
            )
        with self._lock:
            self._dev.write(":INIT:CONT OFF")
            self._dev.query(":INIT:IMM;*OPC?")

    def start_continuous(self):
        with self._lock:
            self._dev.write(":INIT:CONT ON")

    def get_trace(self):
        """Read the current trace. Returns (freqs_hz, amplitudes_dbm)."""
        with self._lock:
            start = float(self._dev.query(":FREQ:START?"))
            stop  = float(self._dev.query(":FREQ:STOP?"))
            n     = int(self._dev.query(":SWE:POIN?"))
            raw   = self._dev.query(":TRAC:DATA? TRACE1")
        amps = np.array([float(x) for x in raw.split(',')])
        freqs = np.linspace(start, stop, n)
        return freqs, amps

    def sweep_once(self):
        """Single sweep + read, as one call."""
        self.single_sweep()
        return self.get_trace()

    def close(self):
        self._dev.close()


def find_peaks(freqs, amps, exclude_hz=None, exclude_width_hz=0.0,
                min_prominence_db=10.0, min_spacing_hz=0.0):
    """Local-maximum peak finder for a swept SA trace (no scipy dependency).

    A point counts as a peak if it is a strict local maximum and stands at
    least ``min_prominence_db`` above the higher of the two valley floors
    reached by walking outward until a higher-or-equal point (or the trace
    edge) is hit — a simplified topographic-prominence test, adequate for
    picking real bumps out of a noise floor without over-engineering the
    rigorous "key col" algorithm.

    ``exclude_hz``/``exclude_width_hz`` blank out a window (e.g. around the
    carrier) so the main signal itself isn't reported as a spur. Detections
    within ``min_spacing_hz`` of each other are merged, keeping the stronger.

    Returns a list of (freq_hz, amp_dbm) sorted by frequency.
    """
    n = len(amps)
    candidates = []
    for i in range(1, n - 1):
        if amps[i] <= amps[i - 1] or amps[i] <= amps[i + 1]:
            continue
        if exclude_hz is not None and abs(freqs[i] - exclude_hz) <= exclude_width_hz / 2:
            continue
        left = i
        while left > 0 and amps[left - 1] <= amps[i]:
            left -= 1
        right = i
        while right < n - 1 and amps[right + 1] <= amps[i]:
            right += 1
        left_floor = min(amps[left:i + 1]) if left < i else amps[i]
        right_floor = min(amps[i:right + 1]) if right > i else amps[i]
        prominence = amps[i] - max(left_floor, right_floor)
        if prominence >= min_prominence_db:
            candidates.append((freqs[i], amps[i]))

    candidates.sort(key=lambda c: c[0])
    merged = []
    for f, a in candidates:
        if merged and (f - merged[-1][0]) <= min_spacing_hz:
            if a > merged[-1][1]:
                merged[-1] = (f, a)
        else:
            merged.append((f, a))
    return merged
