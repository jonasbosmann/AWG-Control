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
        for attempt in range(3):
            try:
                self._dev.clear()
            except Exception:
                pass
            self._dev.write("*CLS")
            # Drain stale unread replies a previous session can leave on the
            # scope's socket server — one leftover line shifts EVERY following
            # query response by one (seen as *IDN? returning '1').
            self._dev.timeout = 300
            try:
                while True:
                    self._dev.read()
            except Exception:
                pass
            self._dev.timeout = 3000
            try:
                idn = self._dev.query("*IDN?").strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.3)
                continue
            if "TEKTRONIX" in idn.upper():
                break
            # a stale reply was consumed as the IDN — drain again and retry
        print("Scope:", idn)
        self._dev.timeout = 10000

    def setup(self, channel=1, n_averages=1, trigger_channel=None, trigger_level=0.0,
              trigger_mode="AUTO", trigger_slope="RISe"):
        """Configure acquisition and edge trigger.

        trigger_channel: if given, trigger on that channel while measuring `channel`
        (e.g. measure CH1, trigger on CH2 sync pulse). Defaults to `channel`.
        trigger_level: edge trigger level (V) on the trigger channel — use ~half the
        sync-pulse amplitude when triggering on a 0 V-baseline pulse.
        trigger_mode: 'AUTO' (free-runs if no trigger — WASHES OUT coherent averaging)
        or 'NORMal' (only acquire/average on genuine triggers — required for coherent
        averaging on a CH2 sync pulse).
        trigger_slope: 'RISe' (default) or 'FALL'. For a start-of-buffer sync pulse that
        shares its buffer with the signal under test (both start at sample 0), triggering
        on the pulse's FALLing edge instead of its RISing edge moves the trigger reference
        point to the END of the pulse rather than its start — giving a guaranteed
        pulse-width's worth of extra margin before the signal of interest, and a cleaner/
        more certain edge to detect than a possibly slow rising transition.
        """
        trig_ch = trigger_channel if trigger_channel is not None else channel
        with self._lock:
            self._dev.write("MEASUrement:DELETEALL")
            self._dev.write(f"CH{channel}:TERmination 50")
            if trig_ch != channel:
                self._dev.write(f"CH{trig_ch}:TERmination 50")
            if n_averages > 1:
                self._dev.write("ACQuire:MODe AVErage")
                self._dev.write(f"ACQuire:NUMAVg {n_averages}")
            else:
                self._dev.write("ACQuire:MODe SAMple")
            self._dev.write("TRIGger:A:TYPe EDGE")
            self._dev.write(f"TRIGger:A:EDGE:SOUrce CH{trig_ch}")
            self._dev.write(f"TRIGger:A:EDGE:SLOpe {trigger_slope}")
            self._dev.write(f"TRIGger:A:MODe {trigger_mode}")
            self._dev.write(f"TRIGger:A:LEVEL:CH{trig_ch} {trigger_level:.3f}")
            self._dev.write("ACQuire:STOPAfter RUNSTop")
            self._dev.write("ACQuire:STATE RUN")
        self._pre.clear()
        mode = f"{n_averages}× avg" if n_averages > 1 else "sample"
        trig = f", trig CH{trig_ch}@{trigger_level:.2f}V {trigger_mode} {trigger_slope}" if trig_ch != channel else ""
        print(f"Scope: CH{channel} @ 50 Ω, {mode}{trig}\n")

    def set_timebase_direct(self, seconds_per_div):
        self._pre.clear()
        with self._lock:
            self._dev.write(f"HORizontal:SCAle {seconds_per_div:.3e}")

    def set_record_length(self, n_points):
        """Set the acquisition record length (memory depth) directly.

        Only takes effect reliably in MANUAL horizontal mode (see
        set_max_sample_rate) -- in the default AUTO/CONSTANT modes the scope
        recalculates record length/sample rate itself whenever HORizontal:
        SCAle changes, silently overriding a plain RECOrdlength write.
        """
        self._pre.clear()
        with self._lock:
            self._dev.write(f"HORizontal:RECOrdlength {int(n_points)}")

    def set_active_channels(self, channels, n_channels=4):
        """Turn ON the given channels and OFF all others.

        This scope's real-time sample rate is tiered by how many channels
        are active (e.g. 1 active -> fastest tier, 2 -> half that, 4 ->
        quarter) -- leaving unused channels switched on from an earlier
        session silently caps the achievable rate well below spec even
        with HORizontal:MODE:SAMPLERate MAX requested. `channels`: iterable
        of channel numbers to enable (e.g. [1, 2]); everything else in
        1..n_channels is turned off.
        """
        self._pre.clear()
        wanted = set(channels)
        with self._lock:
            for ch in range(1, n_channels + 1):
                self._dev.write(f"SELect:CH{ch} {'ON' if ch in wanted else 'OFF'}")

    def set_max_sample_rate(self, capture_seconds):
        """Force MANUAL horizontal mode with sample rate pinned at the
        scope's maximum, and record length sized to capture exactly
        `capture_seconds` at whatever that maximum turns out to be --
        so every capture runs as fast as the hardware allows regardless of
        what's on the input, without capturing far more time than needed
        (a fixed large record length overshoots badly once the achieved
        rate is known: 100000 samples at this scope's actual 12.5 GS/s is
        8 us of data for a 1 us chirp). Crop to the exact region of interest
        afterward in software (chirp_quality.py's find_burst_window())
        rather than trading sample rate for a specific on-screen window.

        Plain HORizontal:RECOrdlength writes (set_record_length) don't
        reliably stick in the scope's default AUTO/CONSTANT horizontal
        mode -- SCAle changes there recompute record length/sample rate
        automatically and silently override it. MANUAL mode is what makes
        both settings actually hold.

        Returns the achieved sample rate (Hz), queried after requesting MAX
        rather than assumed, so this adapts automatically if the channel
        count or scope model changes what "max" actually is.
        """
        self._pre.clear()
        with self._lock:
            self._dev.write("HORizontal:MODE MANual")
            self._dev.write("HORizontal:MODE:SAMPLERate MAX")
            rate = float(self._dev.query("HORizontal:SAMPLERate?"))
            record_length = max(int(round(capture_seconds * rate)), 1)
            self._dev.write(f"HORizontal:MODE:RECOrdlength {record_length}")
        return rate

    def set_timebase(self, freq_hz, n_cycles=8):
        self._pre.clear()
        scale = max(1e-10, min(n_cycles / (freq_hz * 10), 1.0))
        with self._lock:
            self._dev.write(f"HORizontal:SCAle {scale:.3e}")

    def set_vertical(self, channel, volts_per_div):
        """Set vertical scale (V/div), clamped to the MSO64B 50 Ω input range."""
        volts_per_div = min(max(volts_per_div, 1e-3), 1.0)
        self._pre.clear()
        with self._lock:
            self._dev.write(f"CH{channel}:SCAle {volts_per_div:.4e}")
        return volts_per_div

    def set_horizontal_position(self, percent):
        """Trigger position as % of the record (default 50). Set ~10 before
        capturing a triggered burst/chirp so most of the record is post-trigger."""
        self._pre.clear()
        with self._lock:
            self._dev.write(f"HORizontal:POSition {percent:.0f}")

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

    def measure_vpp(self, channel=1, settle=0.1, acq_timeout=30.0):
        """Capture one fresh acquisition and return RMS-based Vpp.

        Sequence:
          1. sleep(settle) — AWG glitch transients die away while scope runs freely
          2. STOP, then SEQuence + RUN — arm scope for exactly one acquisition.
             The explicit STOP matters in average mode: the scope free-runs
             between calls accumulating averages of the OLD signal, and arming
             SEQuence while already running can complete the sequence with that
             stale average mixed in. STOP→RUN restarts averaging from zero, so
             every averaged waveform is acquired after the settle sleep.
          3. Poll ACQuire:STATE? until '0' (STOP) — guarantees the acquisition is
             actually complete before we read (unlike *OPC? over LAN, which returns
             as soon as the command is parsed, not when the hardware finishes)
          4. Read CURVe?, compute AC-RMS → Vpp; RMS is robust to residual spikes
          5. Restore continuous mode for live view
        """
        if settle > 0:
            time.sleep(settle)
        with self._lock:
            self._dev.write("ACQuire:STATE STOP")
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
            # Select the data source FIRST — the preamble describes the currently
            # selected source, so querying it before DATa:SOUrce can return another
            # channel's vertical scaling. Always re-query here (cheap per sweep
            # step) so front-panel scale changes can't leave the cache stale.
            self._dev.write(f"DATa:SOUrce CH{channel}")
            self._dev.write("DATa:ENCdg ASCIi")
            self._dev.write("DATa:STARt 1")
            self._pre['xincr'] = float(self._dev.query("WFMOutpre:XINcr?"))
            self._pre['ymult'] = float(self._dev.query("WFMOutpre:YMUlt?"))
            self._pre['yzero'] = float(self._dev.query("WFMOutpre:YZEro?"))
            self._pre['rl']    = int(self._dev.query("HORizontal:RECOrdlength?"))
            pts = self._pre['rl']   # download full record — consistent cycle count
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

    def measure_vpp_auto(self, channel=1, settle=0.1, target_frac=0.75,
                         max_retries=2):
        """measure_vpp with vertical autoscale.

        Re-measures with the channel rescaled so the waveform fills
        ~target_frac of the 10-division screen whenever the current scale is
        badly matched — too big (clipping distorts the RMS-based Vpp) or too
        small (quantization/noise dominates). The scale persists between
        calls, so across a smooth sweep most points measure once.
        """
        vpp, t, v = self.measure_vpp(channel, settle=settle)
        for _ in range(max_retries):
            if vpp <= 0:
                break
            with self._lock:
                scale = float(self._dev.query(f"CH{channel}:SCAle?"))
            full = 10.0 * scale
            if 0.3 * full <= vpp <= 0.95 * full:
                break
            new_scale = self.set_vertical(channel, vpp / (10.0 * target_frac))
            if abs(new_scale - scale) < 0.05 * scale:
                break   # clamped at an input-range limit — can't improve
            print(f"  vscale {scale*1e3:.3g} -> {new_scale*1e3:.3g} mV/div, "
                  f"re-measuring\n")
            vpp, t, v = self.measure_vpp(channel, settle=settle)
        return vpp, t, v

    def restore(self, channel=1):
        """Return the scope to normal free-running operation: AUTO edge trigger on
        `channel`, Sample mode, continuous acquisition. Undoes the NORMal-trigger /
        Average / alternate-trigger-source state left by comparison measurements."""
        with self._lock:
            self._dev.write("ACQuire:MODe SAMple")
            self._dev.write("TRIGger:A:TYPe EDGE")
            self._dev.write(f"TRIGger:A:EDGE:SOUrce CH{channel}")
            self._dev.write("TRIGger:A:MODe AUTO")
            self._dev.write(f"TRIGger:A:LEVEL:CH{channel} 0.0")
            self._dev.write("HORizontal:POSition 50")
            self._dev.write("ACQuire:STOPAfter RUNSTop")
            self._dev.write("ACQuire:STATE RUN")
        self._pre.clear()
        print(f"Scope: restored to normal (AUTO trigger CH{channel}, Sample, running)\n")

    def close(self):
        self._dev.close()
