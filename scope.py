import pyvisa
import numpy as np
import time
import threading

_RESOURCE = "TCPIP::169.254.4.20::4000::SOCKET"


class Scope:
    def __init__(self):
        self._lock = threading.Lock()
        self._pre = {}   # cached: xincr, ymult, yzero, rl (cleared on timebase change)
        self._width = None   # bytes per returned sample; see get_waveform
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

    def reference_status(self):
        """Is the scope's timebase locked to the external 10 MHz?

        Unlike the Proteus, the MSO64B DOES expose a real lock state:
        `ROSc:SOUrce?` -> INTERnal|EXTERnal and `ROSc:STATe?` -> LOCKED|
        UNLOCKed (bench-confirmed 2026-10-19). Worth checking at connect
        time: the setting is nonvolatile but `File > Default Setup` clears
        it back to internal, and a silent fallback is invisible on screen
        while costing 350-875 kHz of line position at 350 GHz.

        Returns {'source': str, 'state': str, 'external': bool,
                 'locked': bool}.
        """
        with self._lock:
            src = self._dev.query("ROSc:SOUrce?").strip().upper()
            try:
                state = self._dev.query("ROSc:STATe?").strip().upper()
            except Exception:
                state = ""
        return {"source": src, "state": state,
                "external": src.startswith("EXT"),
                "locked": state.startswith("LOCK")}

    def aux_out_source(self):
        """What the rear AUX OUT connector is currently emitting."""
        with self._lock:
            return self._dev.query("AUXout:SOUrce?").strip().upper()

    def set_aux_out(self, source="REFOut"):
        """Choose what the rear AUX OUT connector emits.

        REFOut    10 MHz reference clock, derived from whichever reference
                  the timebase is using. With the scope locked to the
                  external OCXO this is a buffered copy of it -- which is
                  how a reference gets onto a scope channel for the AWG beat
                  test without spending a third splitter arm.
        ATRIGger  a pulse on each A-trigger event. Power-on default.

        Bench-probed 2026-10-19: only those two are accepted on this unit.
        REFerence / REFCLK / CLOCK / AFG all return ESR=32 and are ignored
        (no AFG option fitted). Equivalent UI path is
        Utility > I/O > AUX OUT > Reference Clock.

        Levels: >=1.0 V high, <=0.25 V low into 50 ohm (~0.75-1.0 Vpp), so a
        scope channel at 50 ohm and ~200 mV/div shows it comfortably.
        """
        expect = {"REFOUT": "REFOUT", "ATRIGGER": "ATRIGGER",
                  "ATRIG": "ATRIGGER"}.get(source.upper())
        if expect is None:
            raise ValueError(
                f"{source!r}: valid values on this unit are REFOut and ATRIGger")
        with self._lock:
            self._dev.write("*CLS")
            self._dev.write(f"AUXout:SOUrce {source}")
            got = self._dev.query("AUXout:SOUrce?").strip().upper()
            esr = self._dev.query("*ESR?").strip()
        if got != expect:
            raise RuntimeError(
                f"AUX OUT not set: asked {source!r}, reads {got!r} (ESR={esr})")
        return got

    def require_external_reference(self):
        """Raise unless the timebase is locked to the external reference."""
        st = self.reference_status()
        if not (st["external"] and st["locked"]):
            raise RuntimeError(
                f"scope is NOT locked to the external 10 MHz: "
                f"ROSc:SOUrce?={st['source']!r} ROSc:STATe?={st['state']!r}. "
                "Acquisition badge > Timebase Reference Source > External "
                "(1 ppm). A File > Default Setup clears this.")
        return st

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
        """Set the horizontal display scale (time/div) and verify it stuck --
        reads the value back plus drains :SYSTem:ERRor? afterward, since a
        write that's silently ignored (e.g. rejected in the scope's current
        horizontal mode) looks identical to success unless checked for."""
        self._pre.clear()
        with self._lock:
            self._dev.write(f"HORizontal:SCAle {seconds_per_div:.3e}")
            actual = float(self._dev.query("HORizontal:SCAle?"))
            err = self._dev.query("*ESR?").strip()
            errmsg = self._dev.query("EVMsg?").strip() if err != "0" else ""
        if abs(actual - seconds_per_div) > 0.01 * seconds_per_div:
            print(f"  WARNING: set_timebase_direct({seconds_per_div:.3e}) requested but "
                  f"scope reports {actual:.3e} s/div -- write likely rejected/ignored"
                  f"{' -- ' + errmsg if errmsg else ''}")
        return actual

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

    def save_state(self):
        """Snapshot the few settings a measurement helper is likely to
        disturb, so it can hand the scope back as it found it. Channel
        display states are included because turning the others off to chase
        a sample-rate tier is rude if the user was watching them."""
        with self._lock:
            st = {"horiz_mode": self._dev.query("HORizontal:MODE?").strip(),
                  "scale": self._dev.query("HORizontal:MODE:SCAle?").strip(),
                  "reclen": self._dev.query("HORizontal:MODE:RECOrdlength?").strip(),
                  "trig_src": self._dev.query("TRIGger:A:EDGE:SOUrce?").strip(),
                  "display": {ch: self._dev.query(f"SELect:CH{ch}?").strip()
                              for ch in (1, 2, 3, 4)}}
        return st

    def restore_state(self, st):
        """Put back what save_state() captured."""
        with self._lock:
            for ch, on in st["display"].items():
                self._dev.write(f"SELect:CH{ch} {'ON' if on in ('1', 'ON') else 'OFF'}")
            self._dev.write(f"HORizontal:MODE {st['horiz_mode']}")
            self._dev.write(f"HORizontal:MODE:RECOrdlength {st['reclen']}")
            self._dev.write(f"HORizontal:MODE:SCAle {st['scale']}")
            self._dev.write(f"TRIGger:A:EDGE:SOUrce {st['trig_src']}")
        self._pre.clear()

    def set_max_sample_rate(self, capture_seconds, interpolate=2.0):
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

        `interpolate` multiplies the real-time ceiling. The ADC here runs at
        25 GS/s, but asking for 50 GS/s measurably IMPROVES shot-to-shot
        coherence -- 0.9860 against 0.9612 at 2.25 GHz, bench-measured
        2026-10-20 -- because the interpolated samples land on a finer grid
        relative to the trigger, and the scope's averager stacks on that
        grid. The extra samples carry no new information and double the
        bytes, so pass interpolate=1.0 when transfer time matters more than
        the last percent of coherence.

        Returns the achieved sample rate (Hz), queried after setting it
        rather than assumed.
        """
        self._pre.clear()
        with self._lock:
            self._dev.write("HORizontal:MODE MANual")
            # "SAMPLERate MAX" is NOT accepted on this firmware -- it raises a
            # command error (ESR 32) and is silently ignored, so this used to
            # leave the rate at whatever the previous session had set and
            # return that, which looked exactly like a successful maximise.
            # Bench-found 2026-10-20 after it quietly ran a measurement at
            # 6.25 GS/s. ACQuire:MAXSamplerate? gives the real-time ceiling
            # (25 GS/s on this unit); writing a huge number is no good either,
            # because the scope happily accepts 2500 GS/s of INTERPOLATED
            # samples.
            try:
                top = float(self._dev.query("ACQuire:MAXSamplerate?"))
            except Exception:
                top = float(self._dev.query("HORizontal:SAMPLERate?"))
            want = top * max(float(interpolate), 1.0)
            self._dev.write(f"HORizontal:MODE:SAMPLERate {want:.0f}")
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

    def get_waveform(self, channel=1, max_points=10000, fresh=False,
                     acq_timeout=10.0, absolute_time=False, width=None):
        """Read the waveform buffer.

        Preamble (xincr, ymult, yzero, record length) is cached and reused across
        calls until the timebase changes — saves 4 queries per live-view frame.

        channel may be an int (1-4) or a source name such as "MATH1" or
        "REF1", which is how an on-screen math average is read back rather
        than recomputed here.

        fresh=False (default) reads whatever is in the buffer without arming an
        acquisition. That is right for live view, and WRONG for anything that
        needs independent shots: if the scope is not currently triggering, every
        call returns the IDENTICAL stale frame. Verified 2026-10-19 — five
        consecutive calls returned byte-identical arrays, which silently faked a
        coherence of exactly 1.000 in a shot-to-shot phase measurement.

        fresh=True arms exactly one acquisition and waits for it to complete
        (same handshake as measure_vpp), so successive calls really are
        successive shots. Use it for averaging, coherence, or any statistics
        over repeats. It raises on timeout rather than handing back a stale
        frame, because a silent stale frame is the failure mode that matters.

        absolute_time=True puts t=0 at the TRIGGER instant instead of at the
        first sample, using WFMOutpre:XZEro -- queried every call and never
        cached, because it is the one preamble field that legitimately changes
        shot to shot. The trigger interpolator places the trigger BETWEEN two
        samples, so without this correction each shot's time origin is
        quantised to the sample grid; at 800 MHz and 12.5 GS/s that alone is
        +-23 deg of apparent shot-to-shot phase. Required for coherence or any
        phase statistic, irrelevant for amplitude and live view.

        width is bytes per returned sample (DATa:WIDth). The default of 1 is
        8 bits, which is BELOW this scope's own ADC and far below what an
        averaged record holds: measured 2026-10-20, a 64-average read at
        width 1 came back bit-identical across a quiet stretch -- RMS exactly
        0.000 mV -- because every sample truncated to the same code, making
        the noise improvement from averaging invisible. Pass width=2 for
        anything where sub-LSB detail matters (averaging, noise, SNR). The
        preamble's ymult/yzero follow the width, so the volts are right
        either way; the cache is cleared whenever the width changes.
        """
        if width is not None and width != self._width:
            with self._lock:
                self._dev.write(f"DATa:WIDth {int(width)}")
            self._width = int(width)
            self._pre.clear()       # ymult/yzero scale with the width
        if fresh:
            with self._lock:
                self._dev.write("ACQuire:STATE STOP")
                self._dev.write("ACQuire:STOPAfter SEQuence")
                self._dev.write("ACQuire:STATE RUN")
                deadline = time.perf_counter() + acq_timeout
                while time.perf_counter() < deadline:
                    if self._dev.query("ACQuire:STATE?").strip() in (
                            '0', '0.0', 'STOP', 'STOPPED'):
                        break
                    time.sleep(0.005)
                else:
                    self._dev.write("ACQuire:STOPAfter RUNSTop")
                    self._dev.write("ACQuire:STATE RUN")
                    raise RuntimeError(
                        f"no trigger within {acq_timeout:.0f} s on the configured "
                        "source — refusing to return a stale frame")
                self._pre.clear()       # force a fresh preamble for this shot
        with self._lock:
            src = channel if isinstance(channel, str) else f"CH{channel}"
            # The preamble is per SOURCE, not just per timebase: ymult/yzero
            # differ between channels at different V/div, and a MATH source
            # returns volts directly (ymult 1.0) rather than ADC codes.
            # Reusing one source's scaling for another silently returns a
            # waveform scaled by the wrong factor -- MATH1 read straight
            # after CH3 came back as 0.0 mVpp that way.
            if self._pre.get('src') != src:
                self._pre.clear()
                self._pre['src'] = src
            self._dev.write(f"DATa:SOUrce {src}")
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
            # per-shot, never cached: this is the sub-sample trigger position
            xzero = (float(self._dev.query("WFMOutpre:XZEro?"))
                     if absolute_time else 0.0)
            if fresh:                       # back to free-running for live view
                self._dev.write("ACQuire:STOPAfter RUNSTop")
                self._dev.write("ACQuire:STATE RUN")
        # float, not int: CH sources return integer codes but MATH returns
        # floating point, and dtype=float on the split list is also quicker
        # than a per-sample Python int().
        raw = np.array(raw_str.split(','), dtype=float)
        time_s    = xzero + np.arange(len(raw)) * self._pre['xincr']
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
