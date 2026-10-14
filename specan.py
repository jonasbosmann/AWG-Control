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

    def _recover(self):
        """Clear a wedged transaction and drain anything left queued.

        A query that times out MID-TRANSACTION may still have its reply
        arrive afterwards. Retrying without draining then reads that stale
        reply, and every following query is answered by the one before it --
        silently, with plausible numbers. So recovery has to drain, not just
        retry. Same failure this class guards against in __init__.
        """
        try:
            self._dev.clear()
        except Exception:                            # noqa: BLE001
            pass
        t = self._dev.timeout
        self._dev.timeout = 300
        try:
            while True:
                self._dev.read()
        except Exception:                            # noqa: BLE001
            pass
        self._dev.timeout = t
        try:
            self._dev.write("*CLS")
        except Exception:                            # noqa: BLE001
            pass

    def sweep_once(self, retries=1):
        """Single sweep + read, as one call.

        Retries once through _recover() on a VISA timeout. Observed
        2026-07-31: a single transient timeout on :INIT:IMM killed a 4-minute
        sweep, with the instrument found afterwards in a perfectly healthy
        state (SA mode, no errors, sweeps completing in 0.00 s). Losing a run
        to that is not worth it; silently accepting shifted data would be
        much worse, hence the drain.
        """
        for attempt in range(int(retries) + 1):
            try:
                self.single_sweep()
                return self.get_trace()
            except pyvisa.errors.VisaIOError:
                if attempt >= retries:
                    raise
                print("  (analyzer timed out -- clearing and retrying)")
                self._recover()

    # ── IQ Analyzer (BASIC) mode ──────────────────────────────────────
    # The swept trace above is a magnitude-only measurement: it cannot see
    # phase, so it cannot answer "what does this stage's output phase do when
    # its drive level moves?" -- i.e. AM-to-PM. The Waveform measurement in
    # BASIC mode returns raw time-domain I/Q instead, from which envelope and
    # phase come out together, sample by sample.
    #
    # Use: iq_setup() once, iq_capture() per point, iq_teardown() at the end.
    # Mode switches take seconds, so they are deliberately NOT per-capture.

    def _drain(self, max_n=20):
        """Empty the SCPI error queue and return whatever was in it.
        Called from inside an already-locked section."""
        errs = []
        for _ in range(max_n):
            e = self._dev.query(":SYST:ERR?").strip()
            if e.startswith("0") or e.startswith("+0"):
                break
            errs.append(e)
        return errs

    def _try(self, cmd):
        """Write a command that may not exist on this firmware/option set.
        Returns the error text, or '' if it took. Never raises -- used only
        for the nice-to-haves, so a missing one degrades the run instead of
        killing it."""
        self._dev.write(cmd)
        e = self._dev.query(":SYST:ERR?").strip()
        return "" if (e.startswith("0") or e.startswith("+0")) else e

    def _read_iq(self, big_endian):
        raw = self._dev.query_binary_values(":READ:WAV0?", datatype='f',
                                            is_big_endian=big_endian,
                                            container=np.array)
        x = np.asarray(raw, dtype=np.float64)
        if x.size < 64 or x.size % 2:
            raise RuntimeError(f"IQ record has {x.size} values -- expected an "
                               f"even count of at least 64 interleaved I,Q")
        z = x[0::2] + 1j * x[1::2]
        mag = np.abs(z)
        med = float(np.median(mag))
        if not np.isfinite(z).all() or mag.max() <= 0 or \
                (med > 0 and mag.max() / med > 1e6):
            raise RuntimeError("IQ record is not physical (non-finite, or a "
                               "1e6 spread between peak and median) -- "
                               "suspect the binary byte order")
        return z

    def iq_setup(self, center_hz, if_bw_hz=5e6, duration_s=1e-3,
                 atten_db=None):
        """Enter BASIC mode and configure a time-domain I/Q capture.

        THE SAMPLE RATE IS NOT SETTABLE ON THIS UNIT and is not what the
        instrument says it is. Probed on fw A.11.04, 2026-07-31:

          * ':WAV:SRAT <value>' is rejected -- '-110, Illegal command:
            WAV:SRAT'. Only the QUERY form parses.
          * ':WAV:SRAT?' answers, but it echoes the IF BANDWIDTH, not the
            sample rate. With the IF BW at 100 kHz it reported 1e5 while the
            record really arrived at 1.364 MS/s. ':WAV:BAND?',
            ':WAV:BAND:RES?' and ':WAV:BWID:RES?' all alias to the same
            parameter, so none of them is a second opinion.
          * The real knob is ':WAV:DIF:BANDwidth'. Rate follows it, but not
            by a fixed ratio: 100 kHz -> 1.364 MS/s (x13.6), 1 MHz ->
            7.5 MS/s (x7.5).

        So the rate is DERIVED from a real capture -- pairs returned divided
        by ':WAV:SWE:TIME?' -- and that derived value is what iq_capture()
        reports. A believed-but-wrong sample rate is the kind of error that
        produces a plausible answer on the wrong frequency axis, so it is
        settled here against data rather than read off a query that lies.

        if_bw_hz should be set well WIDER than any modulation being measured.
        The digital IF filter is Gaussian (':WAV:BAND:TYPE?' -> GAUS), and
        any asymmetry of its response about the tone converts AM into PM and
        back -- exactly the quantity an AM-to-PM measurement is trying to
        attribute to the DUT. Running 25x wide makes its contribution
        negligible; the cost is noise floor, which is cheap here.

        Reference level is deliberately not touched. Attenuation is what sets
        the front-end headroom, and an AM run made at a level already verified
        in swept mode creates no new worst case: the modulation only goes
        DOWN from the peak that was already measured safely.
        """
        self._iq_prev_timeout = self._dev.timeout
        with self._lock:
            self._dev.timeout = 30000
            self._dev.write("*CLS")
            self._dev.write(":INST:SEL BASIC")
            self._dev.query("*OPC?")
            mode = self._dev.query(":INST:SEL?").strip().upper()
            if "BASIC" not in mode and "IQ" not in mode:
                errs = self._drain()
                self._dev.write(":INST:SEL SA")
                self._dev.timeout = self._iq_prev_timeout
                raise RuntimeError(
                    f"could not enter IQ Analyzer mode: :INST:SEL? -> {mode!r}"
                    + (f"; errors {errs}" if errs else "")
                    + ". BASIC mode is a licensed option -- without it this "
                      "instrument cannot capture I/Q and AM-to-PM is not "
                      "measurable on it.")
            self._dev.write(":CONF:WAV")
            self._dev.query("*OPC?")
            self._dev.write(":INIT:CONT OFF")
            self._dev.write(f":FREQ:CENT {center_hz:.0f}")
            if atten_db is not None:
                self._dev.write(":POW:ATT:AUTO OFF")
                self._dev.write(f":POW:ATT {atten_db:.0f}")
            soft = {}
            for tag, c in (("aver_off", ":WAV:AVER:STAT OFF"),
                           ("elec_att_off", ":POW:EATT:STAT OFF")):
                e = self._try(c)
                if e:
                    soft[tag] = e
            self._dev.write(f":WAV:DIF:BAND {if_bw_hz:.0f}")
            self._dev.write(f":WAV:SWE:TIME {duration_s:.9f}")
            self._dev.write(":FORM:DATA REAL,32")
            self._dev.write(":FORM:BORD SWAP")
            errs = self._drain()
            if errs:
                self._dev.write(":INST:SEL SA")
                self._dev.timeout = self._iq_prev_timeout
                raise RuntimeError(f"IQ setup rejected: {'; '.join(errs)}")

            dif = float(self._dev.query(":WAV:DIF:BAND?"))
            tsw = float(self._dev.query(":WAV:SWE:TIME?"))
            shape = self._dev.query(":WAV:BAND:TYPE?").strip()
            att = float(self._dev.query(":POW:ATT?"))
            self._dev.timeout = 120000

            # Probe capture. Byte order is the one thing here that fails
            # SILENTLY -- wrong-endian float32 still parses, just into
            # nonsense -- so it is settled now, against a real record, rather
            # than discovered later in a phase number that looks plausible.
            self._iq_big_endian = False
            try:
                try:
                    z = self._read_iq(False)
                except RuntimeError:
                    self._dev.write(":FORM:BORD NORM")
                    self._drain()
                    z = self._read_iq(True)
                    self._iq_big_endian = True
            except Exception:
                # Never leave the analyzer parked in BASIC mode: every other
                # script here uses the swept trace, and a mode left switched
                # would break the next run rather than this one.
                self._dev.write(":FORM:DATA ASCII")
                self._dev.write(":INST:SEL SA")
                self._drain()
                self._dev.timeout = self._iq_prev_timeout
                del self._iq_big_endian
                raise
            fs = len(z) / tsw                    # DERIVED, see the docstring
            self._iq_fs = fs
            self._iq_tsw = tsw
            claimed = float(self._dev.query(":WAV:SRAT?"))
        info = dict(sample_rate_hz=fs, duration_s=tsw, dif_bw_hz=dif,
                    if_filter=shape, atten_db=att, center_hz=float(center_hz),
                    n_samples=int(len(z)), srat_claimed_hz=claimed,
                    big_endian=bool(self._iq_big_endian), soft_errors=soft)
        return info

    def iq_capture(self, center_hz=None):
        """One I/Q record. Returns (z_complex, sample_rate_hz).

        Retuning only writes :FREQ:CENT -- the capture configuration set up by
        iq_setup() stays put, so points in a sweep are all taken under
        identical acquisition conditions.
        """
        if not hasattr(self, "_iq_fs"):
            raise RuntimeError("iq_capture() before iq_setup()")
        with self._lock:
            if center_hz is not None:
                self._dev.write(f":FREQ:CENT {center_hz:.0f}")
            self._dev.write("*CLS")
            z = self._read_iq(self._iq_big_endian)
            errs = self._drain()
        if errs:
            raise RuntimeError(f"IQ capture reported: {'; '.join(errs)}")
        # Recomputed per record rather than reused: the returned length is the
        # only handle on the rate, so a record that comes back a different
        # length must change the rate with it, not be silently mislabelled.
        return z, len(z) / self._iq_tsw

    def iq_teardown(self):
        """Back to swept SA mode, with the trace type that was in effect."""
        with self._lock:
            try:
                self._dev.write(":FORM:DATA ASCII")
                self._dev.write(":INST:SEL SA")
                self._dev.query("*OPC?")
                self._dev.write(f":TRAC1:TYPE {self._trace_mode}")
                self._dev.write(":INIT:CONT ON")
                self._drain()
            finally:
                self._dev.timeout = getattr(self, "_iq_prev_timeout", 8000)
                for a in ("_iq_fs", "_iq_tsw", "_iq_big_endian"):
                    if hasattr(self, a):
                        delattr(self, a)

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
