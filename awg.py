import pyvisa
import numpy as np
import time
import threading

SAMPLE_RATE_SINGLE = 9e9    # single-channel max (both DACs interleaved → CH2 disabled)
SAMPLE_RATE_DUAL   = 2.5e9  # dual-channel max (one DAC per channel)
N_SAMPLES = 2048            # must be multiple of 64 for Proteus granularity

_RESOURCE = "TCPIP0::141.51.196.111::5025::SOCKET"


def widen_for_flat_plateau(f_start, f_stop, window_frac=0.05):
    """Given a DESIRED flat/full-amplitude plateau [f_start, f_stop], return
    the wider (f_start_cmd, f_stop_cmd) that should actually be COMMANDED to
    the AWG so the plateau -- not the tapered edges -- covers the requested
    range.

    The chirp's Gaussian amplitude taper (_gauss_win, window_frac fraction
    of the sweep tapered on EACH side) only leaves the middle
    (1 - 2*window_frac) of whatever bandwidth is actually commanded at full
    amplitude; the outer window_frac on each side ramps from ~0 up to full
    scale. Since a chirp's instantaneous frequency is linear in time, that
    same window_frac fraction of TIME maps directly onto window_frac of the
    SWEPT FREQUENCY RANGE too -- so if you ask for e.g. "1.70-1.80 GHz" and
    command exactly that, the AWG is NOT actually at full power at either
    1.70 or 1.80 GHz, only somewhere inside that range. Confirmed on real
    bench data 2026-07-24 (chirp_bench/pipeline_view.py FFT-panel
    investigation): a nominal 1.70-1.80 GHz commanded segment's -3dB
    amplitude plateau only spanned ~1.706-1.793 GHz, not the full 100 MHz.

    Solving B_total*(1-2*window_frac) = B_design for B_total:
        B_total = B_design / (1 - 2*window_frac)
        margin (per side) = (B_total - B_design) / 2

    Returns (f_start_cmd, f_stop_cmd), each pushed outward by that margin.
    Keeps the commanded DURATION unchanged (sweep rate goes up slightly,
    ~11% at window_frac=0.05) -- deliberate choice for diagnostic/QC scans,
    where matching a specific physical sweep rate doesn't matter, over
    extending duration instead (which would ripple into dead-time/capture-
    window budgets elsewhere). Revisit if this margin logic is ever applied
    to real excitation-chirp design, where sweep rate (Hz/s) affects
    adiabatic-passage efficiency and preserving it might matter more than
    preserving duration."""
    b_design = f_stop - f_start
    margin = b_design * window_frac / (1 - 2 * window_frac)
    return f_start - margin, f_stop + margin


class AWG:
    def __init__(self, sample_rate=SAMPLE_RATE_SINGLE, n_samples=N_SAMPLES):
        self.sample_rate = sample_rate
        self.n_samples = n_samples   # buffer length; must be multiple of 64
        self._active_seg = {}
        self._sweep_amp  = {}        # tracks last :VOLT sent per channel during sweep
        self._rate_sent  = {}        # last :FREQ:RAST actually sent per channel
        self._duc_n_total = {}       # complex samples per DUC buffer, per channel
        self._ref_cfg = None         # (freq_mhz, bypass) once an external ref
                                     # is requested; re-applied after every *RST
        self._byp_supported = None   # None=unprobed, False=absent on this fw
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

    def _drain_errs(self, context):
        """Empty the SCPI error queue; raise if anything was in it.

        Guard for raw-write fast paths: fw 1.237.0 rejects some commands
        silently or with queued errors only, so unchecked writes can corrupt
        a measurement with no trace (see the DUC IQM=NONE incident)."""
        errs = []
        for _ in range(10):
            e = self._dev.query(":SYST:ERR?").strip()
            if e.startswith("0"):
                break
            errs.append(e)
        if errs:
            raise RuntimeError(f"AWG error after {context}: {'; '.join(errs)}")

    # ── 10 MHz reference ──────────────────────────────────────────────
    @staticmethod
    def _rosc_freq_mhz(raw):
        """Normalise a :ROSC:FREQ? reply ('10M', '1e7', '10000000') to MHz."""
        s = raw.strip().upper().replace("HZ", "").replace(" ", "")
        if s.endswith("M"):
            s = s[:-1]
        else:
            try:
                v = float(s)
            except ValueError:
                return None
            return v / 1e6 if v > 1e3 else v
        try:
            return float(s)
        except ValueError:
            return None

    def _query_bypass_locked(self):
        """Read :ROSC:BYP?, or None if this firmware has no such command.

        CONFIRMED ABSENT on fw 1.237.0 (bench, 2026-10-19): the query gets no
        reply at all, times out after the full 10 s, and leaves
        '208, syntax error in scpi' in the queue. :ROSC:BYPass only appears in
        Programming Manual rev 1.6. So probe at most once, with a short
        timeout, and remember the answer.
        """
        if self._byp_supported is False:
            return None
        old = self._dev.timeout
        try:
            self._dev.timeout = 1500
            val = self._dev.query(":ROSC:BYP?").strip().upper()
            self._byp_supported = True
            return val
        except Exception:
            self._byp_supported = False
            return None
        finally:
            self._dev.timeout = old
            for _ in range(10):     # clear the queued syntax error
                try:
                    if self._dev.query(":SYST:ERR?").startswith("0"):
                        break
                except Exception:
                    break

    def _reference_status_locked(self):
        src = self._dev.query(":ROSC:SOUR?").strip().upper()
        raw = self._dev.query(":ROSC:FREQ?").strip()
        return {"source": src, "freq_raw": raw,
                "freq_mhz": self._rosc_freq_mhz(raw),
                "bypass": self._query_bypass_locked()}

    def reference_status(self):
        """What reference the AWG is CONFIGURED to use.

        There is no lock-status query on this instrument, so this reports
        what was accepted, NOT whether the PLL actually acquired. Confirm a
        real lock on the bench with check_reference.py.
        """
        with self._lock:
            return self._reference_status_locked()

    def set_reference_external(self, freq_mhz=10, bypass=None):
        """Lock the AWG's clock to the external 10 MHz reference.

        Bench chain (as built 2026-10-14): Wenzel 501-27501-11 OCXO ->
        ZX10R-2-183-S+ 2-way splitter -> 6 dB pad -> AWG REF IN (SMP,
        50 ohm AC coupled, 0.6-1.7 Vpp window; the pad lands ~0.98 Vpp
        there). The other splitter arm feeds the scope's rear Ref In, so
        both instruments run off one timebase.

        Two separate things this buys, worth not conflating:
          * ABSOLUTE frequency accuracy, set by this instrument's reference
            alone. The internal TCXO's 1-2.5 ppm is 350-875 kHz of line
            position at 350 GHz after the x24 chain -- it shifts every line
            equally and so survives every internal consistency check.
          * AWG<->scope COHERENCE, set by the DIFFERENCE between the two
            clocks. chirp_quality.align_and_average() can only remove a
            constant per-shot time OFFSET; independent clocks also STRETCH
            the timebase differently shot to shot, and no time-shift
            alignment can undo that. It has to die in hardware.

        :ROSC:FREQ is sent FIRST, deliberately: it defaults to 100M, so
        switching the source first leaves the PLL hunting for a 100 MHz
        input that is not on the cable. Both settings are read back
        afterwards because fw 1.237.0 accepts and silently ignores some
        commands (see the :SOUR:IQM trap in _duc_upload_ch).

        Order no longer matters: `*RST` restores :ROSC:SOUR to its INT
        default, but every reset in this file goes through _reset_locked(),
        which re-applies and re-verifies whatever was requested here. Call it
        once and it stays.

        bypass: NOT AVAILABLE on this unit. :ROSC:BYPass appears only in
        Programming Manual rev 1.6; fw 1.237.0 answers the query with a
        timeout and '208, syntax error in scpi' (bench-confirmed 2026-10-19).
        Passing bypass= raises rather than hanging. Left in place for a
        future firmware.
        """
        with self._lock:
            got = self._apply_reference_locked(freq_mhz, bypass)
            self._ref_cfg = (freq_mhz, bypass)
        byp = f", BYP {got['bypass']}" if got["bypass"] else ""
        print(f"AWG reference: EXT {freq_mhz} MHz{byp} "
              f"(configured; lock itself is not queryable)")
        return got

    def _apply_reference_locked(self, freq_mhz, bypass=None):
        """Send and verify the external-reference settings. Caller holds _lock."""
        freq_kw = {10: "10M", 100: "100M"}[freq_mhz]
        self._cmd(f":ROSC:FREQ {freq_kw}")          # MUST precede :ROSC:SOUR
        self._cmd(":ROSC:SOUR EXT")
        if bypass is not None:
            if self._byp_supported is None:
                self._query_bypass_locked()         # probe once
            if self._byp_supported is False:
                raise RuntimeError(
                    ":ROSC:BYPass does not exist on this firmware (1.237.0) -- "
                    "it was added in Programming Manual rev 1.6. Call "
                    "set_reference_external() without bypass=.")
            self._cmd(f":ROSC:BYP {'ON' if bypass else 'OFF'}")
        got = self._reference_status_locked()
        ok_freq = (got["freq_mhz"] is not None
                   and abs(got["freq_mhz"] - freq_mhz) < 0.01)
        if "EXT" not in got["source"] or not ok_freq:
            raise RuntimeError(
                f"external reference NOT applied: :ROSC:SOUR? -> {got['source']!r}, "
                f":ROSC:FREQ? -> {got['freq_raw']!r} "
                f"(wanted EXT / {freq_mhz} MHz). Is the 10 MHz cable on REF IN?")
        return got

    def set_reference_internal(self):
        """Back to the internal TCXO (1-2.5 ppm).

        Mainly for the INT/EXT toggle test: a static readback cannot tell
        'locked' apart from 'command ignored', but watching the beat against
        the reference start and stop can.
        """
        with self._lock:
            # Clear the remembered request FIRST. Otherwise _reset_locked()
            # and _ensure_reference_locked() keep re-asserting the external
            # reference behind your back -- which silently invalidated the
            # INT-vs-EXT control measurement on 2026-10-19 (both read 0 ppm
            # because "internal" was never actually internal).
            self._ref_cfg = None
            self._cmd(":ROSC:SOUR INT")
            got = self._reference_status_locked()
        if "INT" not in got["source"]:
            raise RuntimeError(
                f"internal reference NOT applied: :ROSC:SOUR? -> {got['source']!r}")
        print("AWG reference: INT (internal TCXO, 1-2.5 ppm)")
        return got

    def _reset_locked(self, opc=False):
        """`*CLS; *RST` plus everything a reset silently clears. Holds _lock.

        *RST returns :ROSC:SOUR to its INT default, so EVERY reset drops an
        external reference lock. Six paths in this file reset inline instead
        of going through _setup(), and each one was quietly un-locking the
        AWG mid-measurement -- `duc_cw_setup()` among them, which is what the
        GUI calls for every DUC send. Route all resets through here.
        """
        self._dev.write("*CLS; *RST")
        time.sleep(0.5)
        if opc:
            self._dev.query("*OPC?")
        self._rate_sent.clear()
        if self._ref_cfg is not None:
            self._apply_reference_locked(*self._ref_cfg)

    def _setup(self, channel=1, reset=True, sample_rate=None):
        rate = sample_rate if sample_rate is not None else self.sample_rate
        if reset:
            self._reset_locked()
        self._cmd(f":INST:CHAN {channel}")
        # A previous DUC run leaves the channel in DUC mode (survives reset=False),
        # where real waveforms would be misread as interleaved I/Q — force DIRect.
        if self._dev.query(":MODE?").strip() != "DIR":
            self._cmd(":MODE DIR")
        self._cmd(f":FREQ:RAST {rate:.0f}")
        self._rate_sent[channel] = rate
        if reset:
            self._cmd(":TRAC:DEL:ALL")
        self._cmd(":INIT:CONT ON")

    def _upload(self, wave_u16, segnum=1):
        n = len(wave_u16)
        # Proteus limits (prog. manual p.133-134, P948x): segments 1-128 are
        # "short/fast" segments, minimum 128 points; higher segment numbers are
        # regular segments, minimum 2048 points. Granularity is 32 points.
        min_len = 128 if segnum <= 128 else 2048
        if n < min_len or n % 32:
            raise ValueError(
                f"segment {segnum}: length {n} invalid (min {min_len}, multiple of 32)")
        self._cmd(f":TRAC:DEF {segnum},{n}")
        self._cmd(f":TRAC:SEL {segnum}")
        data = wave_u16.tobytes()
        nb_str = str(len(data))
        self._dev.write_raw(f":TRAC:DATA #{len(nb_str)}{nb_str}".encode() + data + b"\n")
        time.sleep(0.3)
        # Check instead of *CLS — the old *CLS silently WIPED upload errors,
        # so a rejected transfer played a stale/garbage segment without trace.
        self._drain_errs(f"TRAC:DATA (segment {segnum})")
        print("  OK    TRAC:DATA")

    # ── markers ───────────────────────────────────────────────────────
    MARKER_POINTS_PER_BYTE = 8      # 9 GS/s model: 1 marker byte per 8 points

    @staticmethod
    def _marker_format(rate, marker):
        """(waveform points per marker byte, byte value for 'high').

        BENCH-DETERMINED 2026-10-19 at BOTH operating points, and the manual
        is wrong about both of them.

        Manual Table 8.1 claims bit0..3 = M1..M4 with bits 4-7 "N/A". In fact
        **each byte holds TWO marker points**: low nibble bits0-3 = M1..M4 for
        the first point, high nibble bits4-7 = the same four for the second.
        Setting only bit0 gives a ~0.89 ns sub-pulse per byte and adjacent
        "high" bytes do NOT merge -- 16 bytes of 0x01 came out as 16 separate
        0.89 ns pulses 1.8 ns apart. 0x11 merges them into one clean pulse.

        Manual S8.3.1/S8.3.2 also give different points-per-byte for the
        9 GS/s and 2.5 GS/s models. Measured, the rule is the same for both:
        **marker bytes = (:TRAC:DEF length) / 4.**
            2.5 GS/s Direct : 2048-point segment -> 512 bytes, 1.6 ns each
            9   GS/s DUC    : 8192-long  segment -> 2048 bytes, 1.8 ns each
        In DUC that works out as one marker point per complex baseband sample
        (0.889 ns = 8 DAC samples), which matches the known 8-sample marker
        granularity. Too short an array misbehaves badly rather than erroring:
        1024 bytes where 2048 were needed produced a single 1.85 us pulse.

        `rate` is kept in the signature for the day a model or mode breaks
        this unified rule; it is currently unused.
        """
        return 4, (1 << (marker - 1)) | (1 << (marker - 1 + 4))

    def marker_pulse(self, channel, segnum, n_wave_points, marker=1,
                     high_frac=0.05, delay_bytes=0, level_vpp=1.0,
                     offset_v=0.0, enable=True, points_per_byte=None):
        """Put a start-of-segment pulse on `marker` of `channel`.

        This is the trigger coherent averaging needs: the marker is played
        from the same segment memory, on the same sample clock, as the
        waveform -- so it is phase-rigid to the output even when the AWG's
        own trigger jitters (3.6 ns on this model). The scope triggers on it
        instead of on the RF, which removes cycle ambiguity entirely.

        IMPORTANT -- markers belong to the CHANNEL'S segment. A marker on CH1
        is rigid to whatever CH1 is playing, NOT to CH2. Put the marker on the
        same channel as the signal, or configure both channels inside one
        reset so their segments stay aligned.

        n_wave_points is the segment length in the units :TRAC:DEF was given.
        Format (Desktop Manual ch.8, 9 GS/s model): one marker byte per eight
        waveform points, bit0=M1, bit1=M2, bit2=M3, bit3=M4, bits 4-7 unused.

        UNVERIFIED IN DUC MODE: a DUC/IQM-ONE segment is defined with
        2*n_total uint16 (interleaved I/Q), and the manual does not say
        whether "waveform points" there means the complex samples or the
        uint16 count. That is a factor of two in pulse position and width.
        Override `points_per_byte` (8 or 16) once the bench says which.

        high_frac: fraction of the segment the pulse stays high.
        delay_bytes: whole marker-bytes of delay from segment start; each is
        `points_per_byte` samples. Sub-byte trim is :MARK:DEL:FINE, not this.
        """
        ppb, high_val = self._marker_format(self.sample_rate, marker)
        if points_per_byte:
            ppb = points_per_byte
        n_bytes = int(n_wave_points) // ppb
        if n_bytes < 1:
            raise ValueError(
                f"segment of {n_wave_points} points gives no marker bytes at "
                f"{ppb} points/byte")
        n_high = max(int(round(n_bytes * high_frac)), 1)
        bit = high_val
        data = np.zeros(n_bytes, dtype=np.uint8)
        start = int(delay_bytes) % n_bytes
        idx = (np.arange(n_high) + start) % n_bytes
        data[idx] = bit

        with self._lock:
            self._cmd(f":INST:CHAN {channel}")
            self._cmd(f":TRAC:SEL {segnum}")
            self._cmd(f":MARK:SEL {marker}")
            self._cmd(f":MARK:VOLT:PTOP {level_vpp:.3f}")
            self._cmd(f":MARK:VOLT:OFFS {offset_v:.3f}")
            raw = data.tobytes()
            nb = str(len(raw))
            self._dev.write_raw(f":MARK:DATA #{len(nb)}{nb}".encode() + raw + b"\n")
            time.sleep(0.2)
            self._drain_errs(f"MARK:DATA (CH{channel} M{marker}, seg {segnum})")
            self._cmd(f":MARK {'ON' if enable else 'OFF'}")
        print(f"  marker CH{channel} M{marker}: {n_bytes} bytes, "
              f"{n_high} high ({high_frac*100:.1f}%), {level_vpp} Vpp, "
              f"{'ON' if enable else 'OFF'}")
        return {"n_bytes": n_bytes, "n_high": n_high,
                "points_per_byte": ppb, "bit": bit}

    def marker_off(self, channel, marker=1):
        """Disable a marker output without touching its data."""
        with self._lock:
            self._cmd(f":INST:CHAN {channel}")
            self._cmd(f":MARK:SEL {marker}")
            self._cmd(":MARK OFF")

    def _ensure_reference_locked(self):
        """Re-assert the external reference if the configuration cleared it.

        Entering 9 GS/s DIRECT mode (interleaved DACs, CH2 disabled) resets
        :ROSC back to INT/100M. Measured 2026-10-19: setting the reference
        BEFORE that configuration survives 0/3, setting it AFTER survives
        3/3. 2.5 GS/s Direct and 9 GS/s DUC never clear it. So the reference
        has to be re-checked at the END of a configuration sequence, not just
        after *RST. Cheap: two queries, and it only writes when it has to.
        Caller holds _lock.
        """
        if self._ref_cfg is None:
            return
        want_mhz = self._ref_cfg[0]
        st = self._reference_status_locked()
        if "EXT" in st["source"] and st["freq_mhz"] is not None                 and abs(st["freq_mhz"] - want_mhz) < 0.01:
            return
        self._apply_reference_locked(*self._ref_cfg)

    def _play(self, amplitude_vpp=0.5, segnum=1):
        self._cmd(f":FUNC:MODE:SEGM {segnum}")
        self._cmd(f":VOLT {amplitude_vpp:.3f}")
        self._cmd(":OUTP ON")
        self._ensure_reference_locked()

    @property
    def freq_step_hz(self):
        """Smallest achievable frequency increment at the current buffer size."""
        return self.sample_rate / self.n_samples

    @staticmethod
    def _real_to_u16(wave):
        """Direct-mode offset binary: [-1, 1] → 0..65535 (manual §9.1).
        np.round so midscale (0.0) maps to exactly 32768, not truncated 32767."""
        return np.round((np.asarray(wave) + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)

    @staticmethod
    def _gauss_win(n, frac):
        """Gaussian-edged flat-top window of length n; edges span frac·n samples.
        Edge span is capped at n//2 so head and tail can never overlap."""
        w = np.ones(n)
        if n < 2:
            return w
        n_w = min(max(int(frac * n), 1), n // 2)
        sigma = n_w / 3.0
        idx = np.arange(n_w, dtype=np.float64)
        w[:n_w]  = np.exp(-0.5 * ((idx - n_w) / sigma) ** 2)
        w[-n_w:] = np.exp(-0.5 * (idx / sigma) ** 2)
        return w

    def _sine_u16(self, cycles):
        t = np.arange(self.n_samples)
        return self._real_to_u16(np.sin(2 * np.pi * cycles * t / self.n_samples))

    def _square_u16(self, cycles):
        t = np.arange(self.n_samples)
        wave = np.where(np.sin(2 * np.pi * cycles * t / self.n_samples) >= 0, 65535, 0)
        return wave.astype(np.uint16)

    def _ramp_u16(self, cycles):
        t = np.arange(self.n_samples)
        wave = (t * cycles / self.n_samples) % 1.0
        return (wave * 65535).astype(np.uint16)

    def _cycles(self, frequency_hz):
        return max(round(frequency_hz * self.n_samples / self.sample_rate), 1)

    MIN_RATE = 1e9   # :FREQ:RAST legal range is 1e9–9e9 (manual §5.10)

    def _resolve(self, frequency_hz, exact=False):
        """Return (cycles, effective_sample_rate, actual_frequency).

        exact=False: keep self.sample_rate, round cycles → frequency error ≤ freq_step/2
        exact=True:  keep cycles integer, back-calculate sample rate → zero frequency error.
                     The rate is kept within [MIN_RATE, self.sample_rate] (self.sample_rate
                     is the max for the current channel mode — 9 GS/s single, 2.5 GS/s dual);
                     if no integer cycle count fits, falls back to the non-exact rate.
        """
        if frequency_hz > self.sample_rate / 2:
            raise ValueError(
                f"{frequency_hz/1e6:.1f} MHz is above Nyquist "
                f"({self.sample_rate/2e6:.0f} MHz at {self.sample_rate/1e9:g} GS/s) — "
                f"output would alias")
        if exact:
            fn = frequency_hz * self.n_samples
            cyc_min = max(int(np.ceil(fn / self.sample_rate)), 1)   # rate ≤ mode max
            cyc_max = min(int(fn // self.MIN_RATE), self.n_samples // 2)  # rate ≥ 1 GS/s, ≤ Nyquist
            if cyc_min <= cyc_max:
                cycles = cyc_min
                rate = fn / cycles
            else:
                print(f"  exact mode impossible for {frequency_hz/1e6:.3f} MHz "
                      f"(needs clock outside {self.MIN_RATE/1e9:g}-"
                      f"{self.sample_rate/1e9:g} GS/s) — using rounded cycles")
                cycles = self._cycles(frequency_hz)
                rate = self.sample_rate
        else:
            cycles = self._cycles(frequency_hz)
            rate = self.sample_rate
        actual = cycles * rate / self.n_samples
        return cycles, rate, actual

    def send_sine(self, frequency_hz=100e6, amplitude_vpp=0.5, channel=1, reset=True, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact)
        err_hz = actual - frequency_hz
        print(f"[CH{channel}] Sine {frequency_hz/1e6:.6f} MHz -> {actual/1e6:.6f} MHz "
              f"(err {err_hz:+.1f} Hz, Fs={rate/1e9:.6f} GS/s)")
        seg = channel
        with self._lock:
            self._setup(channel, reset, sample_rate=rate)
            self._upload(self._sine_u16(cycles), segnum=seg)
            self._play(amplitude_vpp, segnum=seg)
            self._active_seg[channel] = seg
        return actual

    def update_sine(self, frequency_hz, amplitude_vpp=0.5, channel=1, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact)
        with self._lock:
            cur = self._active_seg.get(channel, channel)
            next_seg = channel + 20 if cur == channel + 10 else channel + 10
            self._cmd(f":INST:CHAN {channel}")
            # Compare against the rate actually sent to the device (an earlier
            # exact=True call may have moved the clock away from self.sample_rate).
            if rate != self._rate_sent.get(channel):
                self._cmd(f":FREQ:RAST {rate:.0f}")
                self._rate_sent[channel] = rate
            self._upload(self._sine_u16(cycles), segnum=next_seg)
            self._play(amplitude_vpp, segnum=next_seg)
            self._active_seg[channel] = next_seg
        return actual

    def sweep_preload(self, freqs_hz, channel=1):
        """Upload one sine segment per frequency (segments 30, 31, …).
        Returns list of (actual_hz, segnum). Call once before the sweep loop."""
        self._sweep_amp.pop(channel, None)   # force :VOLT on first sweep_step
        results = []
        with self._lock:
            for i, f in enumerate(freqs_hz):
                segnum = 30 + i
                cycles, _, actual = self._resolve(f)
                self._dev.write(f":INST:CHAN {channel}")
                wave = self._sine_u16(cycles)
                n = len(wave)
                self._dev.write(f":TRAC:DEF {segnum},{n}")
                self._dev.write(f":TRAC:SEL {segnum}")
                data = wave.tobytes()
                nb_str = str(len(data))
                self._dev.write_raw(
                    f":TRAC:DATA #{len(nb_str)}{nb_str}".encode() + data + b"\n")
                time.sleep(0.15)
                self._drain_errs(f"preload of segment {segnum} "
                                 f"({actual/1e6:.1f} MHz)")
                results.append((actual, segnum))
                print(f"  preload seg {segnum}: {actual/1e6:.3f} MHz")
        return results

    def sweep_step(self, segnum, amplitude_vpp=0.5, channel=1):
        """Switch to a pre-loaded segment and block until the AWG confirms.

        *OPC? blocks until the AWG has processed the segment-switch command.
        The physical output switches within one segment period (<1 µs) after that.
        Avoids re-sending :VOLT/:OUTP ON when unchanged to reduce DAC glitches.
        """
        with self._lock:
            self._dev.write(f":INST:CHAN {channel}")
            self._dev.write(f":FUNC:MODE:SEGM {segnum}")
            prev_amp = self._sweep_amp.get(channel)
            if prev_amp != amplitude_vpp:
                self._dev.write(f":VOLT {amplitude_vpp:.3f}")
                self._dev.write(f":OUTP ON")
                self._sweep_amp[channel] = amplitude_vpp
            self._dev.query("*OPC?")   # block until AWG has processed the switch
            self._drain_errs(f"sweep_step to segment {segnum}")
            self._active_seg[channel] = segnum

    def _chirp_windowed_u16(self, f_start, f_stop, n_active, n_total, rate, window_frac=0.05):
        t = np.arange(n_active, dtype=np.float64)
        phase = 2 * np.pi * (f_start / rate * t +
                              (f_stop - f_start) / (2 * rate * max(n_active - 1, 1)) * t ** 2)
        sig = np.sin(phase) * self._gauss_win(n_active, window_frac)
        full = np.zeros(n_total)
        full[:n_active] = sig
        return self._real_to_u16(full)

    def _cw_sine_u16(self, cycles, n_samples):
        t = np.arange(n_samples)
        return self._real_to_u16(np.sin(2 * np.pi * cycles * t / n_samples))

    # ── General dual-channel waveform builder ─────────────────────
    #
    # One code path behind every "play something on CH1 and/or CH2" case:
    # chirp, CW, square or ramp, in DIRECT mode (real samples) or DUC mode
    # (complex baseband on an exact NCO carrier), each channel gated to a
    # window of ONE shared buffer period.  send_chirp_with_lo() and
    # send_chirp_with_lo_duc() are thin wrappers over it, so the
    # bench-proven CP-FTMW shot is built by exactly this code rather than
    # by a parallel copy that can drift out of step with it.
    #
    # A channel spec is a plain dict:
    #   kind           'off' | 'cw' | 'chirp' | 'square' | 'ramp'
    #   freq_hz        cw/square/ramp: output frequency (direct mode) or
    #                  baseband offset from the carrier (duc; 0 = pure carrier)
    #   f_start_hz,
    #   f_stop_hz      chirp: output sweep (direct) or baseband sweep (duc)
    #   carrier_hz     duc only: NCO carrier, exact to the Hz
    #   sideband       duc only: 'upper' | 'lower' | 'both'
    #                  ('both' feeds the real baseband to I with Q=0 →
    #                   carrier ± f, i.e. a DSB/two-tone signal)
    #   gate           'continuous' | 'chirp' | 'detect'
    #   amplitude_vpp  per channel (:VOLT is per channel on this instrument)

    GRAN_DIRECT  = 64     # direct: segment length must be a multiple of 64
    GRAN_DUC     = 32     # duc: 2 u16 per sample, so n·32 → 2n is a multiple of 64
    DUC_DAC_RATE = 9e9    # DUC DAC clock; X8 interpolation → 1.125 GS/s complex
    DUC_INTERP   = 8
    SIGNAL_KINDS = ("off", "cw", "chirp", "square", "ramp")

    @staticmethod
    def _analytic(x):
        """Single-sideband (analytic) version of a real baseband array.

        Same construction as scipy.signal.hilbert, kept local to avoid adding
        a scipy dependency: zero the negative-frequency half of the spectrum
        and double the positive half. Needed for square/ramp basebands in DUC
        mode — feeding the real waveform straight into I would put the
        modulation on BOTH sides of the carrier.
        """
        n = len(x)
        X = np.fft.fft(x)
        h = np.zeros(n)
        h[0] = 1.0
        if n % 2 == 0:
            h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[1:(n + 1) // 2] = 2.0
        return np.fft.ifft(X * h)

    @staticmethod
    def snap_continuous(freq_hz, n_win, rate):
        """Frequency a continuously repeating buffer can actually hold.

        A waveform that plays continuously has to close on itself: the buffer
        repeats, so a non-integer number of cycles steps the phase at the wrap
        and sprays spurs across the band. Snapped to the nearest whole cycle,
        the worst-case error is half of rate/n_win. A GATED burst is windowed
        to zero at both ends, never wraps, and so keeps its exact frequency —
        which is what the CP-FTMW LO needs.

        Shared with the GUI so what it previews is what the builder sends.
        """
        if freq_hz == 0:
            return 0.0
        cycles = max(round(abs(freq_hz) * n_win / rate), 1)
        return float(np.sign(freq_hz) * cycles * rate / n_win)

    def _plan_timing(self, mode, timing, rate_eff):
        """Sample counts for one shared buffer period: chirp | dead | detect.

        The granularity round-up lands AFTER the detect window, so the chirp
        and the detect window both start exactly where the requested timing
        says they do; `pad_s` reports the leftover tail.

        timing["n_cycles"] = K repeats that chirp|dead|detect cycle K times
        inside ONE buffer, which is the CP-FTMW architecture: a gas pulse
        lasts hundreds of µs but a chirp/FID cycle only ~11 µs, so K cycles
        fit inside one pulse and the scope pays its ~21 ms rearm once per
        pulse instead of once per FID. One marker still fires per buffer.
        The round-up is applied to ONE cycle, not to the whole buffer, so
        every cycle starts on a granularity boundary and the cycle period is
        exactly n_cycle/rate — see _build_channel, which places cycle k at
        k*n_cycle, and phasecycle.split_cycles, which cuts there.

        With K > 1 the carrier's achievable coherent grid changes, because
        coherent_step_hz keys off the 2-adic valuation of n_total and
        n_total = K * n_cycle. That is handled where it belongs (the carrier
        is snapped against plan["n_total"]), but it does mean the usable
        carrier step moves when K does.
        """
        t = timing or {}
        chirp_us  = float(t.get("chirp_us",  1.0))
        dead_us   = float(t.get("dead_us",   0.1))
        detect_us = float(t.get("detect_us", 10.0))
        n_cycles  = max(int(t.get("n_cycles", 1) or 1), 1)
        n_chirp  = round(chirp_us  * 1e-6 * rate_eff)
        n_dead   = round(dead_us   * 1e-6 * rate_eff)
        n_detect = round(detect_us * 1e-6 * rate_eff)
        gran    = self.GRAN_DUC if mode == "duc" else self.GRAN_DIRECT
        floor_n = 64 if mode == "duc" else 128
        n_cycle = max(int(np.ceil((n_chirp + n_dead + n_detect) / gran)) * gran, floor_n)
        n_total = n_cycle * n_cycles
        return {
            "mode": mode, "rate_eff": rate_eff,
            "n_chirp": n_chirp, "n_dead": n_dead, "n_detect": n_detect,
            "n_cycles": n_cycles, "n_cycle": n_cycle,
            "t_cycle": n_cycle / rate_eff,
            "n_total": n_total,
            "period_s": n_total / rate_eff,
            "pad_s": (n_cycle - n_chirp - n_dead - n_detect) / rate_eff,
            "windows": {
                "continuous": (0, n_cycle),
                "chirp":      (0, max(n_chirp, 1)),
                "detect":     (n_chirp + n_dead, max(n_detect, 1)),
            },
        }

    def _build_channel(self, mode, spec, plan, window_frac):
        """One channel's samples (uint16, ready for :TRAC:DATA) + an info dict."""
        kind  = spec.get("kind", "off")
        gate  = spec.get("gate", "continuous")
        if gate not in plan["windows"]:
            raise ValueError(f"unknown gate {gate!r}")
        start, n_win = plan["windows"][gate]
        gated = gate != "continuous"
        rate  = plan["rate_eff"]
        env   = self._gauss_win(n_win, window_frac) if gated else np.ones(n_win)
        t     = np.arange(n_win, dtype=np.float64) / rate
        info  = {"kind": kind, "gate": gate, "n_win": n_win, "t_win_s": n_win / rate}

        if kind == "chirp":
            f0 = float(spec.get("f_start_hz", 0.0))
            f1 = float(spec.get("f_stop_hz",  0.0))
            T  = t[-1] if n_win > 1 else 1e-9
            phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t ** 2)
            info.update(f_start_hz=f0, f_stop_hz=f1,
                        chirp_rate_hz_per_s=(f1 - f0) / max(T, 1e-12))
        else:
            f = float(spec.get("freq_hz", 0.0))
            # Gated bursts keep the exact frequency; a continuously repeating
            # buffer gets snapped to a whole cycle count — see snap_continuous.
            f_act = f if gated else self.snap_continuous(f, n_win, rate)
            phase = 2 * np.pi * f_act * t
            info.update(freq_hz=f, freq_actual_hz=f_act)

        def shape(phi):
            """One cycle's samples with the whole waveform advanced by phi.

            phi goes INSIDE the phase argument rather than multiplying the
            finished waveform. For a complex baseband the two are the same,
            but a real direct-mode waveform multiplied by exp(i.phi) and
            then taken real is just scaled by cos(phi) -- an amplitude
            error wearing a phase's clothes. Inside the argument it is exact
            for every kind and both modes.
            """
            p = phase + phi if phi else phase
            if kind in ("cw", "chirp"):
                return (np.exp(1j * p) if mode == "duc" else np.sin(p)) * env
            if kind == "square":
                real = np.where(np.sin(p) >= 0, 1.0, -1.0)
                return (self._analytic(real) if mode == "duc" else real) * env
            if kind == "ramp":
                real = 2.0 * ((p / (2 * np.pi)) % 1.0) - 1.0
                return (self._analytic(real) if mode == "duc" else real) * env
            raise ValueError(f"unknown signal kind {kind!r} "
                             f"(expected one of {self.SIGNAL_KINDS})")

        n_cycles = plan.get("n_cycles", 1)
        n_cycle  = plan.get("n_cycle", plan["n_total"])
        phases   = spec.get("phases") or [0.0]
        carrier  = float(spec.get("carrier_hz", 0.0))
        sideband = spec.get("sideband", "upper")
        if mode == "duc" and sideband not in ("upper", "lower", "both"):
            raise ValueError(f"unknown sideband {sideband!r}")

        def cycle_phase(k):
            """Programmed phase for cycle k, with the NCO's advance removed.

            The DUC NCO free-runs across the whole buffer and is not reset at
            a cycle boundary, so by cycle k the carrier has already turned
            f_c * k * t_cycle times on top of whatever the baseband carries.
            Subtracting it here makes the phase actually EMITTED by cycle k
            the phase that was asked for. Measured 2026-10-20: without this
            the per-cycle phases come out as an arbitrary ramp.

            The advance is reduced to a fraction of a turn before becoming
            radians -- f_c * k * t_cycle reaches ~1e5 turns, and carrying
            that in radians throws away precision for no reason.
            """
            phi = float(phases[k % len(phases)])
            if mode == "duc" and carrier:
                phi -= 2 * np.pi * ((carrier * (k * plan["t_cycle"])) % 1.0)
            return phi

        emitted = [float(phases[k % len(phases)]) for k in range(n_cycles)]
        if mode == "duc" and sideband == "lower":
            # conj() mirrors the baseband, which inverts the phase sense: a
            # cycle programmed +90 deg is emitted at -90. Whoever recombines
            # the cycles needs the emitted phase, not the programmed one.
            emitted = [-p for p in emitted]
        info["phases_rad"] = [float(phases[k % len(phases)])
                              for k in range(n_cycles)]
        info["phases_emitted_rad"] = (None if (mode == "duc" and
                                               sideband == "both") else emitted)
        info["n_cycles"] = n_cycles

        if mode == "duc":
            I_full = np.zeros(plan["n_total"])
            Q_full = np.zeros(plan["n_total"])
            for k in range(n_cycles):
                z = shape(cycle_phase(k))
                if sideband == "lower":
                    z = np.conj(z)
                elif sideband == "both":
                    z = np.real(z).astype(np.complex128)   # DSB: carrier ± f
                s = start + k * n_cycle
                I_full[s:s + n_win] = np.real(z)
                Q_full[s:s + n_win] = np.imag(z)
            # The analytic square/ramp overshoots ±1 (Gibbs); scale it back so
            # _iq_to_u16 does not clip the peaks into a flat top.
            peak = max(np.abs(I_full).max(initial=0.0),
                       np.abs(Q_full).max(initial=0.0))
            if peak > 1.0:
                I_full, Q_full = I_full / peak, Q_full / peak
                info["scaled_by"] = 1.0 / peak
            info["sideband"] = sideband
            return self._iq_to_u16(I_full, Q_full), info

        full = np.zeros(plan["n_total"])
        for k in range(n_cycles):
            s = start + k * n_cycle
            full[s:s + n_win] = np.real(shape(cycle_phase(k)))
        return self._real_to_u16(full), info

    def _check_limits(self, mode, specs, rate_eff):
        """Reject anything that would alias or exceed the NCO range, with the
        number that is wrong named in the message — a silent alias looks like
        a real measurement."""
        for ch, s in specs.items():
            kind = s.get("kind", "off")
            freqs = ([s.get("f_start_hz", 0.0), s.get("f_stop_hz", 0.0)]
                     if kind == "chirp" else [s.get("freq_hz", 0.0)])
            if mode == "duc":
                carrier = float(s.get("carrier_hz", 0.0))
                nco_max = self.DUC_DAC_RATE / 2
                if not 0 < carrier <= nco_max:
                    raise ValueError(
                        f"CH{ch} carrier {carrier/1e6:.1f} MHz outside the DUC NCO "
                        f"range (0–{nco_max/1e9:g} GHz]")
                bb_max = rate_eff / 2
                for f in freqs:
                    if abs(f) > bb_max:
                        raise ValueError(
                            f"CH{ch} baseband {f/1e6:.1f} MHz exceeds the complex "
                            f"Nyquist limit ({bb_max/1e6:.1f} MHz at "
                            f"{rate_eff/1e9:g} GS/s complex)")
                    if not 0 < carrier + f <= nco_max:
                        raise ValueError(
                            f"CH{ch} output {(carrier + f)/1e6:.1f} MHz outside "
                            f"0–{nco_max/1e9:g} GHz")
            else:
                for f in freqs:
                    if abs(f) > rate_eff / 2:
                        raise ValueError(
                            f"CH{ch} {f/1e6:.1f} MHz is above Nyquist "
                            f"({rate_eff/2e6:.0f} MHz at {rate_eff/1e9:g} GS/s) — "
                            f"output would alias")

    def send_waveforms(self, specs, mode="duc", timing=None, rate=None,
                       window_frac=0.05, verbose=True, coherent=True):
        """Play one spec per channel, both built into ONE shared buffer period.

        mode='duc'    : 9 GS/s DAC clock, X8 interpolation (1.125 GS/s complex
                        baseband), IQM ONE. Each channel's carrier is its own
                        exact NCO frequency; both channels stay usable because
                        the 2.5 GS/s dual-channel ceiling is an ARB/direct-mode
                        memory limit, not a DUC one.
        mode='direct' : real samples at `rate` (defaults to 2.5 GS/s whenever a
                        second channel is active — one DAC per channel).

        coherent=True (DUC only) puts every carrier on the grid that makes
        the output repeat buffer to buffer, which is what coherent averaging
        needs -- the DUC NCO free-runs and is not reset at segment start, so
        an off-grid carrier randomises the phase of every shot. For a GATED
        waveform this is free: the baseband is shifted by the same residual
        and the absolute frequency is held exactly. For a CONTINUOUS one the
        output moves by up to half a grid step (grid = 1/period), which is
        unavoidable. Set coherent=False for a measurement that wants an exact
        frequency and does not care about shot-to-shot phase, e.g. a
        chain-response sweep. See _snap_specs_to_grid.

        Returns the plan dict: period, window sample counts, the per-channel
        info (including the frequency actually produced after integer-cycle
        snapping), and plan["coherent_grid"] recording any carrier that moved.
        """
        mode  = mode.lower()
        specs = {int(ch): dict(s or {"kind": "off"}) for ch, s in specs.items()}
        active = {ch: s for ch, s in specs.items() if s.get("kind", "off") != "off"}
        if not active:
            raise ValueError("nothing to send — every channel spec is 'off'")

        if mode == "duc":
            rate_dac = self.DUC_DAC_RATE
            rate_eff = rate_dac / self.DUC_INTERP
        elif mode == "direct":
            rate_dac = rate_eff = float(
                rate if rate else (SAMPLE_RATE_DUAL if len(active) > 1
                                   else self.sample_rate))
            if len(active) > 1 and rate_eff > SAMPLE_RATE_DUAL:
                raise ValueError(
                    f"{rate_eff/1e9:g} GS/s is single-channel only — two active "
                    f"channels need ≤ {SAMPLE_RATE_DUAL/1e9:g} GS/s in direct mode")
        else:
            raise ValueError(f"mode must be 'duc' or 'direct', not {mode!r}")

        self._check_limits(mode, active, rate_eff)
        plan = self._plan_timing(mode, timing, rate_eff)
        # Must run AFTER the period is known and BEFORE the samples are
        # built, so a shifted baseband is actually the one uploaded.
        plan["rate_dac"] = rate_dac
        plan["coherent_grid"] = (
            self._snap_specs_to_grid(active, plan, verbose)
            if (mode == "duc" and coherent) else [])
        plan["grid_hz"] = (self.coherent_step_hz(
            plan["n_total"], rate_dac, int(round(rate_dac / rate_eff)))
            if mode == "duc" else None)
        if plan["coherent_grid"]:
            # the baseband moved, so Nyquist/NCO headroom has to be re-checked
            self._check_limits(mode, active, rate_eff)
        built = {ch: self._build_channel(mode, s, plan, window_frac)
                 for ch, s in active.items()}
        plan["channels"] = {ch: info for ch, (_, info) in built.items()}
        plan["rate_dac"] = rate_dac

        if verbose:
            print(f"[{mode.upper()}] period {plan['period_s']*1e6:.3f} µs "
                  f"({plan['n_total']:,} samp @ {rate_eff/1e9:g} GS/s"
                  f"{', complex' if mode == 'duc' else ''}), "
                  f"rep rate {1/plan['period_s']/1e3:.2f} kHz")
            for ch in sorted(built):
                s, info = active[ch], built[ch][1]
                base = f"[CH{ch}] {info['kind']}"
                if mode == "duc":
                    carrier = float(s.get("carrier_hz", 0.0))
                    if info["kind"] == "chirp":
                        span = (f"{(carrier+info['f_start_hz'])/1e6:.3f}→"
                                f"{(carrier+info['f_stop_hz'])/1e6:.3f} MHz")
                    else:
                        span = f"{(carrier+info['freq_actual_hz'])/1e6:.6f} MHz"
                    base += (f" {span}  carrier {carrier/1e6:.6f} MHz "
                             f"({info['sideband']} sideband)")
                else:
                    base += (f" {info['f_start_hz']/1e6:.3f}→{info['f_stop_hz']/1e6:.3f} MHz"
                             if info["kind"] == "chirp"
                             else f" {info['freq_actual_hz']/1e6:.6f} MHz")
                print(f"{base}  gate {info['gate']} "
                      f"({info['t_win_s']*1e6:.3f} µs, {info['n_win']:,} samp)  "
                      f"{s.get('amplitude_vpp', 0.5)} Vpp")

        with self._lock:
            if mode == "duc":
                self._reset_locked(opc=True)
                if verbose:
                    # Confirms the DUC/IQM option is actually installed on this unit.
                    print(f"  *OPT? -> {self._dev.query('*OPT?').strip()!r}")
                # :IQModulation is per channel and only sticks after :MODE DUC,
                # so every DUC command lives in _duc_upload_ch — see its docstring
                # for the ordering the firmware actually honours.
                self._cmd(":INST:CHAN 1")
                self._cmd(":TRAC:DEL:ALL")
                for ch in sorted(built):
                    self._duc_upload_ch(ch, ch, built[ch][0],
                                        float(active[ch].get("carrier_hz", 0.0)),
                                        rate_dac, rate_eff,
                                        float(active[ch].get("amplitude_vpp", 0.5)))
                    self._active_seg[ch] = ch
                if verbose:
                    for ch in sorted(built):
                        self._dump_duc_state(ch)
            else:
                for i, ch in enumerate(sorted(built)):
                    self._setup(ch, reset=(i == 0), sample_rate=rate_eff)
                    self._upload(built[ch][0], segnum=ch)
                    self._play(float(active[ch].get("amplitude_vpp", 0.5)), segnum=ch)
                    self._active_seg[ch] = ch
            for ch in sorted(set(specs) - set(active)):
                self._dev.write(f":INST:CHAN {ch}")
                self._dev.write(":OUTP OFF")
        return plan

    def _dump_duc_state(self, ch):
        """Read back the state that actually determines DUC behaviour, AFTER setup.

        _duc_upload_ch already raises if IQM/INT came out wrong, so this is not
        the guard — it is the bench record of what the channel ended up in, the
        thing that made the 2026-07 'IQM stayed NONE → garbled multi-tone'
        failure visible in the first place.
        """
        self._dev.write(f":INST:CHAN {ch}")
        print(f"  [CH{ch} state] "
              f"MODE={self._dev.query(':MODE?').strip()!r} "
              f"IQM={self._dev.query(':SOUR:IQM?').strip()!r} "
              f"INT={self._dev.query(':SOUR:INT?').strip()!r} "
              f"CFR1={self._dev.query(':NCO:CFR1?').strip()!r} "
              f"RAST={self._dev.query(':FREQ:RAST?').strip()!r} "
              f"SEGM={self._dev.query(':FUNC:MODE:SEGM?').strip()!r}")

    def retune(self, carriers, amplitudes=None):
        """Fast retune of channels already set up by send_waveforms(mode='duc').

        Writes only :NCO:CFR1 (plus :VOLT when an amplitude is given), so
        moving the carrier costs milliseconds instead of the seconds a
        *RST + re-upload takes. The baseband keeps playing untouched, which
        means the chirp SHAPE is preserved and only its centre moves.

        NOTE ON THE COHERENT GRID: send_waveforms() snaps each carrier onto
        the achievable grid and shifts that channel's baseband to compensate.
        This path cannot do the second half -- the baseband is already in
        segment memory -- so a carrier handed here is used exactly as given
        and may well be off grid, which costs shot-to-shot phase. It warns
        when that happens. Re-send rather than retune when the result is
        going to be averaged.
        """
        amplitudes = amplitudes or {}
        for ch, f in carriers.items():
            self.duc_cw_step(float(f), channel=int(ch),
                             amplitude_vpp=amplitudes.get(ch), warn_grid=True)
        return carriers

    def send_chirp_with_lo(self, f_start_hz, f_stop_hz, chirp_us, dead_us,
                            f_lo_hz, detect_us, amplitude_vpp=0.5, window_frac=0.05):
        """CH1: Gaussian-windowed chirp then silence.
           CH2: silence during chirp+dead, then exact CW sine during detection window.
           Both channels share the same buffer → identical period → phase-coherent shots.
           LO frequency is floating-point exact (no integer-cycles quantization).
           Always runs at SAMPLE_RATE_DUAL (2.5 GS/s).

           Thin wrapper over send_waveforms(mode='direct') — kept because the
           GUI and the bench scripts call it by name."""
        plan = self.send_waveforms(
            {1: {"kind": "chirp", "f_start_hz": f_start_hz, "f_stop_hz": f_stop_hz,
                 "gate": "chirp", "amplitude_vpp": amplitude_vpp},
             2: {"kind": "cw", "freq_hz": f_lo_hz, "gate": "detect",
                 "amplitude_vpp": amplitude_vpp}},
            mode="direct", rate=SAMPLE_RATE_DUAL, window_frac=window_frac,
            timing={"chirp_us": chirp_us, "dead_us": dead_us, "detect_us": detect_us})
        return plan["n_chirp"] / plan["rate_eff"], f_lo_hz

    def send_chirp_with_lo_duc(self, f_carrier_chirp_hz, f_start_bb_hz, f_stop_bb_hz,
                               chirp_us, dead_us, f_lo_hz, detect_us,
                               amplitude_vpp=0.5, window_frac=0.05):
        """DUC (IQM ONE): upconverted chirp on CH1, gated exact CW on CH2.

        CH1 baseband sweeps f_start_bb → f_stop_bb on an NCO carrier at
        f_carrier_chirp_hz, so the RF output sweeps carrier+start → carrier+stop.
        CH2 is a gated carrier at exactly f_lo_hz (constant baseband, I=1 Q=0).

        Thin wrapper over send_waveforms(mode='duc') — kept because the GUI and
        the bench scripts call it by name. The samples it produces are identical
        to the pre-refactor implementation (see tests/test_awg_builder.py).
        """
        plan = self.send_waveforms(
            {1: {"kind": "chirp", "carrier_hz": f_carrier_chirp_hz,
                 "f_start_hz": f_start_bb_hz, "f_stop_hz": f_stop_bb_hz,
                 "gate": "chirp", "sideband": "upper",
                 "amplitude_vpp": amplitude_vpp},
             2: {"kind": "cw", "carrier_hz": f_lo_hz, "freq_hz": 0.0,
                 "gate": "detect", "sideband": "upper",
                 "amplitude_vpp": amplitude_vpp}},
            mode="duc", window_frac=window_frac,
            timing={"chirp_us": chirp_us, "dead_us": dead_us, "detect_us": detect_us})
        return plan["n_chirp"] / plan["rate_eff"], f_lo_hz

    # NOTE: an earlier send_chirp_with_lo_nco used :MODE NCO to "upconvert" an
    # uploaded chirp/envelope. Removed: NCO mode plays NO waveform memory — it
    # internally generates a plain sine at :NCO:CFR (prog. manual §5.2 p.87-88;
    # DUC primer fig 1.4 "no modulation"), so both channels were just CW.
    # Use send_chirp_with_lo_duc (IQM ONE) for upconverted chirps.

    def send_iq_sine(self, frequency_hz=100e6, amplitude_vpp=0.5, exact=False):
        """Output I (sine) on CH1 and Q (cosine, 90° shifted) on CH2 simultaneously.

        Both channels share the same DAC clock so they are always phase-locked.
        The rate is computed once from CH1 and reused for CH2 — no mismatch possible.
        Requires dual-channel mode (sample_rate <= SAMPLE_RATE_DUAL = 2.5 GS/s).
        """
        cycles, rate, actual = self._resolve(frequency_hz, exact)
        err_hz = actual - frequency_hz
        print(f"[IQ] {frequency_hz/1e6:.6f} MHz -> {actual/1e6:.6f} MHz "
              f"(err {err_hz:+.1f} Hz, Fs={rate/1e9:.6f} GS/s)")

        t = np.arange(self.n_samples)
        i_wave = self._real_to_u16(np.sin(2 * np.pi * cycles * t / self.n_samples))
        q_wave = self._real_to_u16(np.cos(2 * np.pi * cycles * t / self.n_samples))

        with self._lock:
            # Reset and configure CH1 (sets the shared clock rate for both channels)
            self._setup(channel=1, reset=True, sample_rate=rate)
            self._upload(i_wave, segnum=1)
            self._play(amplitude_vpp, segnum=1)
            self._active_seg[1] = 1

            # Configure CH2 without reset — clock rate already set
            self._setup(channel=2, reset=False, sample_rate=rate)
            self._upload(q_wave, segnum=2)
            self._play(amplitude_vpp, segnum=2)
            self._active_seg[2] = 2

        return actual

    def send_ramp(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact) if frequency_hz else (1, self.sample_rate, self.sample_rate / self.n_samples)
        print(f"[CH{channel}] Ramp -> {actual/1e6:.6f} MHz (err {actual - (frequency_hz or actual):+.1f} Hz)")
        with self._lock:
            self._setup(channel, reset, sample_rate=rate)
            self._upload(self._ramp_u16(cycles), segnum=channel)
            self._play(amplitude_vpp, segnum=channel)
            self._active_seg[channel] = channel
        return actual

    def send_square(self, frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True, exact=False):
        cycles, rate, actual = self._resolve(frequency_hz, exact) if frequency_hz else (1, self.sample_rate, self.sample_rate / self.n_samples)
        print(f"[CH{channel}] Square -> {actual/1e6:.6f} MHz (err {actual - (frequency_hz or actual):+.1f} Hz)")
        with self._lock:
            self._setup(channel, reset, sample_rate=rate)
            self._upload(self._square_u16(cycles), segnum=channel)
            self._play(amplitude_vpp, segnum=channel)
            self._active_seg[channel] = channel
        return actual

    # ── CH2-as-trigger comparison (direct vs DUC) ──────────────────
    # CH1 = signal under test (-> scope CH1); CH2 = start-of-buffer sync pulse
    # (-> scope CH2, used as the scope trigger). Both channels share one buffer
    # period, so the CH2 pulse marks the identical CH1 phase every period ->
    # coherent scope triggering / averaging without the hardware marker.

    def _sync_pulse_u16(self, n_total, n_pulse):
        """Real start-of-buffer pulse: 0 V baseline, +full for the first n_pulse
        samples -> one clean rising edge per period at sample 0."""
        w = np.zeros(n_total)
        w[:max(n_pulse, 1)] = 1.0
        return self._real_to_u16(w)

    @staticmethod
    def _iq_to_u16(I, Q):
        """Interleave complex baseband [I0,Q0,I1,Q1,...] as uint16 for DUC/IQM.

        IQ data must use codes 1..65535 symmetric about 32768, NOT the full
        0..65535 range: with code 0 included the negative half has one more
        level than the positive half, and that DC bias shows up as a residual
        carrier spur at the NCO frequency (~-78 dBc; prog. manual §5.9 p.95,
        DUC primer fig 2.3). Matches Tabor's myQuantization(minLevel=1)."""
        iq = np.empty(2 * len(I), dtype=np.float64)
        iq[0::2] = I
        iq[1::2] = Q
        return (np.round(iq * 32767.0) + 32768.0).clip(1, 65535).astype(np.uint16)

    def _duc_upload_ch(self, ch, segnum, data_u16, ncof, rate, rate_cx, amp):
        """Per-channel DUC/IQM-ONE upload.

        Ordering verified per-command against fw 1.237.0 (2026-07-13 probe):
          1. :MODE DUC, then :SOUR:INT. The firmware flags a spurious
             '22, Invalid argument' on :SOUR:INT while the clock is still low
             but APPLIES the value anyway — so it is verified by readback
             below, not by the error code.
          2. :FREQ:RAST to the interpolated DAC clock — must come after INT
             (with INT still NONE, a 9 GHz clock implies complex rate 9 GS/s
             > 1.25 GS/s → '223, settings conflict' on everything after).
          3. :SOUR:IQM ONE LAST. Sent any earlier (INT still NONE, clock
             still low) the firmware SILENTLY ignores it: no SCPI error,
             readback stays NONE, and the interleaved I/Q segment plays as a
             garbled multi-tone real waveform instead of a clean carrier.
        """
        int_kw = {2: "X2", 4: "X4", 8: "X8"}[round(rate / rate_cx)]
        # One warning site covering every DUC path; silent when on grid.
        self._warn_off_grid(ncof, len(data_u16) // 2, rate,
                            round(rate / rate_cx), ch)
        self._cmd(f":INST:CHAN {ch}")
        self._cmd(":MODE DUC")
        self._dev.write(f":SOUR:INT {int_kw}")   # known-spurious err 22
        time.sleep(0.05)
        self._dev.query(":SYST:ERR?")            # drain it; verified below
        self._cmd(f":FREQ:RAST {rate:.0f}")
        self._cmd(":SOUR:IQM ONE")
        iqm  = self._dev.query(":SOUR:IQM?").strip()
        intp = self._dev.query(":SOUR:INT?").strip()
        if iqm != "ONE" or intp != int_kw:
            raise RuntimeError(
                f"CH{ch} DUC setup failed: IQM={iqm!r} (want 'ONE'), "
                f"INT={intp!r} (want {int_kw!r}) — output would be garbled")
        self._cmd(":INIT:CONT ON")
        self._upload(data_u16, segnum=segnum)
        self._cmd(f":FUNC:MODE:SEGM {segnum}")
        if not 0 < amp <= self.MAX_VPP:
            raise ValueError(
                f"CH{ch} amplitude {amp:.3f} Vpp is outside this unit's "
                f"0.001-{self.MAX_VPP} Vpp range; the instrument would reject "
                f"the write and silently keep the previous level")
        self._cmd(f":VOLT {amp:.3f}")
        self._duc_n_total[ch] = len(data_u16) // 2
        self._cmd(":NCO:SIXD1 ON")
        self._cmd(f":NCO:CFR1 {ncof:.0f}")
        self._cmd(":OUTP ON")
        self._ensure_reference_locked()

    def send_cw_direct_sync(self, freq_hz, amplitude_vpp=0.5,
                            sync_pulse_ns=40, sync_amp_vpp=0.5):
        """Direct-mode CW on CH1 + sync pulse on CH2 (2.5 GS/s dual). Returns actual Hz."""
        rate = SAMPLE_RATE_DUAL
        if freq_hz > rate / 2:
            raise ValueError(f"{freq_hz/1e6:.1f} MHz is above Nyquist "
                             f"({rate/2e6:.0f} MHz) — use the DUC mode instead")
        n_total = self.n_samples
        cycles = max(round(freq_hz * n_total / rate), 1)
        actual = cycles * rate / n_total
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate), 1)
        ch1 = self._cw_sine_u16(cycles, n_total)
        ch2 = self._sync_pulse_u16(n_total, n_pulse)
        print(f"[DIRECT CW] CH1 {actual/1e6:.4f} MHz ({cycles} cyc)  "
              f"CH2 sync {n_pulse/rate*1e9:.1f} ns @ {sync_amp_vpp} Vpp")
        with self._lock:
            self._setup(1, reset=True,  sample_rate=rate)
            self._upload(ch1, segnum=1); self._play(amplitude_vpp, segnum=1)
            self._active_seg[1] = 1
            self._setup(2, reset=False, sample_rate=rate)
            self._upload(ch2, segnum=2); self._play(sync_amp_vpp, segnum=2)
            self._active_seg[2] = 2
        return actual

    def send_chirp_direct_sync(self, f_start_hz, f_stop_hz, chirp_us, dead_us,
                               amplitude_vpp=0.5, sync_pulse_ns=40, sync_amp_vpp=0.5,
                               window_frac=0.05):
        """Direct-mode windowed chirp on CH1 + sync pulse on CH2 (2.5 GS/s dual).
        Returns chirp active duration (s)."""
        rate = SAMPLE_RATE_DUAL
        n_chirp = round(chirp_us * 1e-6 * rate)
        n_dead  = round(dead_us  * 1e-6 * rate)
        n_total = max(int(np.ceil((n_chirp + n_dead) / 64)) * 64, 128)
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate), 1)
        ch1 = self._chirp_windowed_u16(f_start_hz, f_stop_hz, n_chirp, n_total, rate, window_frac)
        ch2 = self._sync_pulse_u16(n_total, n_pulse)
        print(f"[DIRECT CHIRP] CH1 {f_start_hz/1e6:.1f}->{f_stop_hz/1e6:.1f} MHz  "
              f"active {n_chirp/rate*1e6:.3f} us  period {n_total/rate*1e6:.3f} us  "
              f"CH2 sync {n_pulse/rate*1e9:.1f} ns")
        with self._lock:
            self._setup(1, reset=True,  sample_rate=rate)
            self._upload(ch1, segnum=1); self._play(amplitude_vpp, segnum=1)
            self._active_seg[1] = 1
            self._setup(2, reset=False, sample_rate=rate)
            self._upload(ch2, segnum=2); self._play(sync_amp_vpp, segnum=2)
            self._active_seg[2] = 2
        return n_chirp / rate

    def send_cw_duc_sync(self, carrier_hz, amplitude_vpp=0.5, n_total=4096,
                         sync_pulse_ns=40, sync_amp_vpp=0.5, sync_carrier_hz=None,
                         coherent=True):
        """DUC/IQM-ONE CW on CH1 (pure carrier at carrier_hz) + gated-burst sync on
        CH2 at buffer start (9 GS/s DAC, X8). Returns the carrier ACTUALLY used.

        coherent=True snaps the carrier to the 1/period grid so its phase
        repeats shot to shot. The baseband here is a constant, so there is
        nothing to absorb the residual with and the output really does move,
        by up to half a grid step -- for n_total=4096 that is up to 137 kHz.
        Pass a larger n_total for a finer grid, or coherent=False when an
        exact frequency matters more than shot-to-shot phase.
        """
        rate = 9e9
        rate_cx = rate / 8.0
        n_total = max(int(np.ceil(n_total / 32)) * 32, 64)   # 2*n_total multiple of 64
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate_cx), 1)
        step = self.coherent_step_hz(n_total, rate, 8)
        if coherent:
            snapped, residual = self.snap_duc_carrier(carrier_hz, n_total,
                                                      rate, 8)
            if abs(residual) > self.GRID_TOL_HZ:
                print(f"  [grid {step/1e3:.3f} kHz] carrier "
                      f"{carrier_hz/1e6:.6f} -> {snapped/1e6:.6f} MHz "
                      f"({residual:+.1f} Hz); a constant baseband cannot "
                      f"absorb this, the output moves")
                carrier_hz = snapped
        sync_carrier = sync_carrier_hz if sync_carrier_hz is not None else carrier_hz
        if coherent:
            sync_carrier = self.snap_duc_carrier(sync_carrier, n_total,
                                                 rate, 8)[0]

        # CH1: constant complex baseband (I=1,Q=0) -> pure carrier via NCO.
        ch1 = self._iq_to_u16(np.ones(n_total), np.zeros(n_total))
        # CH2: I-envelope gate for the first n_pulse samples -> RF burst at t=0.
        I2 = np.zeros(n_total); I2[:n_pulse] = 1.0
        ch2 = self._iq_to_u16(I2, np.zeros(n_total))
        print(f"[DUC CW] CH1 {carrier_hz/1e6:.4f} MHz  "
              f"CH2 burst {n_pulse/rate_cx*1e9:.1f} ns @ {sync_carrier/1e6:.1f} MHz")
        with self._lock:
            self._reset_locked(opc=True)
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, ch1, carrier_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, ch2, sync_carrier, rate, rate_cx, sync_amp_vpp)
            self._active_seg[2] = 2
        return carrier_hz

    # -- DUC phase-coherent carrier grid ------------------------------
    #
    # BENCH-MEASURED 2026-10-20 (run_marker_coherence.py, run_nco_resolution.py).
    #
    # The DUC NCO free-runs: it is NOT reset at segment start. The baseband
    # buffer repeats every T = n_total / (rate/interp), so the OUTPUT repeats
    # shot to shot only if the carrier fits a whole number of cycles into that
    # buffer. Measured at 808.59 MHz, 25 shots spread over 60 s, triggered on
    # the AWG's own marker:
    #     on grid    coherence 0.998   (99.8% of amplitude survives averaging)
    #     100 kHz off          0.208   (the random-phase floor for 25 shots)
    # so this is not a refinement, it is the difference between averaging
    # working and not working at all.
    #
    # The NCO quantises its tuning word to 9e9 / 2^32 = 2.095 Hz (measured:
    # coherence holds for a request up to 1.0 Hz off a grid point and dies by
    # 1.5 Hz, bracketing q/2). Every grid point is k * rate / (interp*n_total),
    # which for a power-of-two n_total is exactly representable in that word,
    # and the integer-Hz ':NCO:CFR1 %.0f' write rounds by at most 0.5 Hz --
    # less than q/2 -- so it always lands back on the grid point. Writing
    # whole Hz is therefore safe; asking for an off-grid carrier is not.
    #
    # Note what this does NOT allow: moving a CW carrier off the grid by
    # putting the remainder in the baseband. A baseband tone must itself
    # repeat in the buffer, so it can only sit on the same grid. For a CHIRP
    # it does work -- the baseband chirp is a transient, not a periodic tone,
    # so its start/stop can be shifted by the residual while the buffer still
    # repeats exactly. Finer CW spacing needs a LONGER buffer, nothing else.

    # The grid is just "a whole number of carrier cycles per buffer period",
    # so it is 1/period. Everything below restates that in different units.
    # DIRECT mode needs none of it: there the whole waveform lives in the
    # buffer and repeats by construction. Only the DUC has a carrier running
    # free underneath the buffer.
    # A carrier that averages coherently has to satisfy TWO conditions, and
    # the second one is easy to miss.
    #
    #   1. BUFFER GRID. The NCO free-runs and is not reset at segment start,
    #      so the output only repeats if the carrier fits a whole number of
    #      cycles into the buffer:  f = k * rate_cx / n_total.
    #   2. NCO WORD. The carrier must also be a value the NCO can actually
    #      hold. Measured 2026-10-20: the tuning quantum is 2.095 Hz, i.e.
    #      rate / 2**32 (coherence survives a request 1.0 Hz off a grid point
    #      and dies by 1.5 Hz, bracketing q/2; 2**31 and 2**33 are both ruled
    #      out).
    #
    # Writing n_total = 2**a * d with d odd, condition 2 forces k to be a
    # multiple of d, so the ACHIEVABLE carrier spacing is
    #
    #       rate_cx / 2**a        NOT rate_cx / n_total
    #
    # These are equal only when n_total is a power of two. They diverge
    # violently otherwise: the default CP-FTMW timing gives n_total = 12512 =
    # 2**5 * 391, where the buffer grid is 89.9 kHz but the achievable step is
    # 35.2 MHz. Snapping to the buffer grid there leaves up to 1.05 Hz of
    # unrepresentable residual, which walks a full cycle of phase in about a
    # second and destroys any average longer than that.
    #
    # Coarse is not a problem for a GATED waveform -- the baseband absorbs the
    # whole residual and the absolute frequency is held exactly. It only
    # limits a CONTINUOUS tone. Use a power-of-two n_total when a CW carrier
    # needs fine placement.
    #
    # DIRECT mode needs none of this: the whole waveform lives in the buffer
    # and repeats by construction. Only the DUC has a carrier running free
    # underneath it.
    # Instrument-queried 2026-10-20: ":VOLT? MAX" is 0.55 on this unit in
    # BOTH direct and DUC mode. Asking for more returns "204, data out of
    # range in scpi" and the amplitude silently stays where it was -- the
    # sync-burst defaults were 0.8 Vpp and had been failing exactly that way.
    MAX_VPP = 0.55

    NCO_BITS = 32         # measured 2026-10-20; q = 9 GS/s / 2**32 = 2.095 Hz
    GRID_TOL_HZ = 1.0     # under q/2 -> the NCO lands on the grid point anyway

    @staticmethod
    def nco_step_hz(rate=9e9):
        """The NCO's own frequency quantum."""
        return rate / 2 ** AWG.NCO_BITS

    @staticmethod
    def buffer_grid_hz(n_total, rate=9e9, interp=8):
        """Carrier spacing that repeats buffer to buffer, ignoring the NCO."""
        return (rate / interp) / n_total

    @staticmethod
    def coherent_step_hz(n_total, rate=9e9, interp=8):
        """Spacing of carriers that are on the buffer grid AND representable.

        Equals buffer_grid_hz() when n_total is a power of two, and is
        coarser by the odd part of n_total otherwise.
        """
        n = int(n_total)
        a = (n & -n).bit_length() - 1              # power of two dividing n
        a = min(a, AWG.NCO_BITS - 3)               # interp 8 = 2**3
        return (rate / interp) / 2 ** a

    @staticmethod
    def snap_duc_carrier(freq_hz, n_total, rate=9e9, interp=8):
        """Nearest ACHIEVABLE coherent carrier. Returns (snapped, residual).

        residual = asked minus achievable. For a GATED waveform add it to the
        baseband and the absolute frequency is preserved exactly; for a
        CONTINUOUS one there is nothing to add it to and the output moves.
        """
        step = AWG.coherent_step_hz(n_total, rate, interp)
        snapped = round(freq_hz / step) * step
        return snapped, freq_hz - snapped

    @staticmethod
    def duc_grid_hz(n_total, rate=9e9, interp=8):
        """Deprecated alias for coherent_step_hz()."""
        return AWG.coherent_step_hz(n_total, rate, interp)

    @staticmethod
    def _warn_off_grid(freq_hz, n_total, rate=9e9, interp=8, ch=None):
        """Warn if a DUC carrier will not repeat shot to shot.

        Called from _duc_upload_ch so it covers EVERY DUC path rather than
        the few that remember to check. Silent once the carrier is on the
        achievable grid, which is the normal case after snapping.
        """
        snapped, residual = AWG.snap_duc_carrier(freq_hz, n_total, rate, interp)
        if abs(residual) <= AWG.GRID_TOL_HZ:
            return
        step = AWG.coherent_step_hz(n_total, rate, interp)
        where = f"CH{ch} " if ch is not None else ""
        print(f"  !! {where}carrier {freq_hz/1e6:.6f} MHz is {residual:+.1f} Hz"
              f" off the {step/1e3:.3f} kHz achievable coherent grid -- its"
              " phase will NOT repeat shot to shot and coherent averaging"
              f" will cancel it. Nearest usable carrier {snapped/1e6:.6f} MHz.")

    def _snap_specs_to_grid(self, active, plan, verbose=True):
        """Put every DUC carrier on the achievable coherent grid, in place.

        The carrier moves onto coherent_step_hz(), and the SAME residual is
        then subtracted back out in the baseband, so the output frequency
        stays where it was asked for. How exactly depends on the channel:

          GATED (chirp or burst) -- windowed to zero at both ends, so it
            never wraps and its baseband frequency is unconstrained. The
            absolute frequency is held EXACTLY, whatever the carrier did.

          CONTINUOUS -- the buffer wraps, so _build_channel snaps the
            baseband to a whole cycle count. That is the buffer grid, which
            is FINER than the achievable carrier step whenever n_total is not
            a power of two. Carrier on the coarse grid plus baseband on the
            fine grid still repeats, so the output lands on the fine buffer
            grid: residual error at most half of buffer_grid_hz(), not half
            of the carrier step. This is why the residual is pushed into the
            baseband here rather than left in the carrier -- with the default
            CP-FTMW timing it is the difference between a 45 kHz error and a
            17.6 MHz one.

        Double sideband is the exception: carrier+f and carrier-f cannot both
        be held by one shift, so it is snapped and left uncompensated.

        Returns a list of per-channel dicts recording what moved, including
        output_error_hz -- how far the output really ended up from the
        request, which is 0 for every gated channel.
        """
        n_total = plan["n_total"]
        rate_eff = plan["rate_eff"]
        rate_dac = plan.get("rate_dac", self.DUC_DAC_RATE)
        interp = int(round(rate_dac / rate_eff))
        step = self.coherent_step_hz(n_total, rate_dac, interp)
        buf = self.buffer_grid_hz(n_total, rate_dac, interp)
        if verbose and step > buf * 1.5:
            print(f"  [grid] n_total {n_total} is not a power of two, so only "
                  f"every {round(step/buf)}th buffer-grid point is a usable "
                  f"carrier:\n         carrier step {step/1e6:.3f} MHz, "
                  f"buffer grid {buf/1e3:.1f} kHz. The baseband takes up the "
                  f"difference.")
        report = []
        for ch in sorted(active):
            spec = active[ch]
            carrier = float(spec.get("carrier_hz", 0.0))
            if carrier == 0.0:
                continue
            snapped, residual = self.snap_duc_carrier(carrier, n_total,
                                                      rate_dac, interp)
            if abs(residual) <= self.GRID_TOL_HZ:
                continue
            kind = spec.get("kind", "off")
            sideband = spec.get("sideband", "upper")
            gated = spec.get("gate", "continuous") != "continuous"
            spec["carrier_hz"] = snapped

            shift, err = 0.0, -residual
            if sideband in ("upper", "lower"):
                shift = residual if sideband == "upper" else -residual
                if kind == "chirp":
                    spec["f_start_hz"] = float(spec.get("f_start_hz", 0.0)) + shift
                    spec["f_stop_hz"] = float(spec.get("f_stop_hz", 0.0)) + shift
                    err = 0.0
                else:
                    f_new = float(spec.get("freq_hz", 0.0)) + shift
                    spec["freq_hz"] = f_new
                    # A continuous tone gets rounded to a whole cycle count by
                    # _build_channel; that rounding is the only error left.
                    err = 0.0 if gated else (
                        self.snap_continuous(f_new, n_total, rate_eff) - f_new)
            report.append({"channel": ch, "carrier_asked_hz": carrier,
                           "carrier_hz": snapped, "residual_hz": residual,
                           "compensated": shift != 0.0,
                           "baseband_shift_hz": shift,
                           "output_error_hz": err})
            if verbose:
                if not shift:
                    tail = (f"DSB cannot be compensated, output moves "
                            f"{-residual:+.1f} Hz")
                elif err == 0.0:
                    tail = "baseband shifted, output frequency held exactly"
                else:
                    tail = (f"baseband shifted; continuous, so the output "
                            f"lands on the {buf/1e3:.1f} kHz buffer grid "
                            f"({err:+.1f} Hz)")
                print(f"  [grid {step/1e3:.3f} kHz] CH{ch} carrier "
                      f"{carrier/1e6:.6f} -> {snapped/1e6:.6f} MHz "
                      f"({residual:+.1f} Hz); {tail}")
        return report
    def duc_cw_setup(self, freq_hz, amplitude_vpp=0.5, channel=1, n_total=4096):
        """One-time DUC/IQM-ONE CW setup for an NCO-stepped frequency sweep.

        Uploads a constant complex baseband (I=1, Q=0) once — the output is a
        pure carrier at the NCO frequency. duc_cw_step() then retunes :NCO:CFR1
        only, with no waveform re-upload, so each sweep step costs milliseconds
        and every frequency is exact (no integer-cycles quantization).
        9 GS/s DAC clock, X8 interpolation → NCO reaches the 4.5 GHz Nyquist.
        """
        rate = 9e9
        rate_cx = rate / 8.0
        if not 0 < freq_hz <= rate / 2:
            raise ValueError(f"{freq_hz/1e6:.1f} MHz outside DUC NCO range "
                             f"(0–{rate/2e9:g} GHz]")
        n_total = max(int(np.ceil(n_total / 32)) * 32, 64)
        ch1 = self._iq_to_u16(np.ones(n_total), np.zeros(n_total))
        print(f"[DUC CW sweep] CH{channel} start {freq_hz/1e6:.4f} MHz  "
              f"(9 GS/s DAC, X8, NCO-stepped)")
        with self._lock:
            self._reset_locked(opc=True)
            self._rate_sent.clear()
            self._cmd(f":INST:CHAN {channel}"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(channel, 1, ch1, freq_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[channel] = 1
        return freq_hz

    def duc_cw_setup_dual(self, freq_hz, amplitude_vpp=0.5, n_total=4096):
        """Identical DUC CW on CH1 AND CH2 at the same NCO frequency.

        Purpose: CH1 drives the DUT (e.g. a mixer IF port) where it can't be
        observed, while CH2 carries a monitor copy for a scope, so an
        automated sweep can be watched/verified step by step instead of
        trusted blind.

        A plain second call to duc_cw_setup() cannot do this -- that function
        begins with '*CLS; *RST', which would wipe whichever channel was set
        up first. Both channels are therefore configured inside ONE reset
        here, the same structure send_cw_duc_sync()/send_chirp_duc_sync()
        use. Both share the 9 GS/s DAC clock and X8 interpolation; dual
        channels are fine in DUC mode (the 2.5 GS/s dual-channel ceiling is
        an ARB/direct-mode memory limit, not a DUC one).

        Retune BOTH afterwards with duc_cw_step(f, channel=1) and
        duc_cw_step(f, channel=2) -- NCO writes are per-channel.
        """
        rate = 9e9
        rate_cx = rate / 8.0
        if not 0 < freq_hz <= rate / 2:
            raise ValueError(f"{freq_hz/1e6:.1f} MHz outside DUC NCO range "
                             f"(0-{rate/2e9:g} GHz]")
        n_total = max(int(np.ceil(n_total / 32)) * 32, 64)
        data = self._iq_to_u16(np.ones(n_total), np.zeros(n_total))
        print(f"[DUC CW dual] CH1+CH2 both {freq_hz/1e6:.4f} MHz "
              f"(CH2 = monitor copy of CH1)")
        with self._lock:
            self._reset_locked(opc=True)
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, data, freq_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, data, freq_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[2] = 2
        return freq_hz

    def duc_iq_setup_dual(self, i_arr, q_arr, carrier_hz, amplitude_vpp=0.5):
        """Upload the SAME arbitrary complex baseband to CH1 and CH2, one reset.

        Built for FID-style transient tests: CH2 carries the signal and CH1 is
        an aligned copy whose only job is to clock the marker. Both segments
        have to be the same length, or they loop at different rates and the
        marker walks relative to the signal.

        i_arr/q_arr are the complex baseband, each scaled to +-1, at
        rate/interp = 1.125 GS/s. The length must be a multiple of 32.

        The CARRIER must sit on the coherent grid (duc_grid_hz) or the output
        will not repeat shot to shot. A TRANSIENT baseband -- one that has
        decayed to zero before the buffer ends -- may sit anywhere, because it
        never has to join up across the buffer boundary. That is what lets a
        FID be placed at an arbitrary absolute frequency and still average
        coherently, and it is the one case where the grid residual really can
        be pushed into the baseband. A CW baseband tone cannot do this.
        """
        rate = 9e9
        rate_cx = rate / 8.0
        i_arr = np.asarray(i_arr, dtype=float)
        q_arr = np.asarray(q_arr, dtype=float)
        if i_arr.size != q_arr.size:
            raise ValueError("I and Q must be the same length")
        n_total = int(i_arr.size)
        if n_total % 32 or n_total < 64:
            raise ValueError(f"baseband length {n_total} must be a multiple "
                             f"of 32 and at least 64")
        if not 0 < carrier_hz <= rate / 2:
            raise ValueError(f"{carrier_hz/1e6:.1f} MHz outside DUC NCO range "
                             f"(0-{rate/2e9:g} GHz]")
        peak = float(max(np.abs(i_arr).max(), np.abs(q_arr).max()))
        if peak > 1.0 + 1e-9:        # tolerate float roundoff
            raise ValueError(f"baseband peaks at {peak:.3f}; scale to +-1 "
                             f"before upload or it will clip")
        data = self._iq_to_u16(i_arr, q_arr)
        print(f"[DUC IQ dual] CH1+CH2 carrier {carrier_hz/1e6:.4f} MHz, "
              f"{n_total} complex samples = {n_total/rate_cx*1e6:.4f} us buffer")
        with self._lock:
            self._reset_locked(opc=True)
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1")
            self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, data, carrier_hz, rate, rate_cx,
                                amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, data, carrier_hz, rate, rate_cx,
                                amplitude_vpp)
            self._active_seg[2] = 2
        return carrier_hz

    def duc_cw_step(self, freq_hz, channel=1, amplitude_vpp=None,
                    warn_grid=False):
        """Retune the NCO carrier of a channel prepared by duc_cw_setup().

        Only :NCO:CFR1 changes — the constant baseband keeps playing. *OPC?
        blocks until the AWG has processed the retune (same pattern as
        sweep_step). Raw writes, no per-command error polling, for speed.

        amplitude_vpp, if given, also retunes the output level in the same
        transaction. That is what makes a PRE-EMPHASISED sweep possible: a
        correction table is a different amplitude at every frequency, and
        re-uploading a waveform per point would cost seconds each. Sent with
        6 decimals because the correction this chain needs is measured in
        thousandths of a dB — whether the instrument honours that many is a
        separate question, and one worth measuring rather than assuming.
        """
        if not 0 < freq_hz <= 4.5e9:
            raise ValueError(f"{freq_hz/1e6:.1f} MHz outside DUC NCO range "
                             f"(0–4.5 GHz]")
        # This path only moves the NCO -- the baseband already in segment
        # memory cannot be re-shifted, so an off-grid carrier here really
        # is off grid. Silent by default because frequency sweeps step
        # this hundreds of times and do not care about shot-to-shot
        # phase; retune() turns it on, which is the interactive route.
        if warn_grid and channel in self._duc_n_total:
            self._warn_off_grid(freq_hz, self._duc_n_total[channel],
                                self.DUC_DAC_RATE, self.DUC_INTERP, channel)
        with self._lock:
            self._dev.write(f":INST:CHAN {channel}")
            self._dev.write(f":NCO:CFR1 {freq_hz:.0f}")
            if amplitude_vpp is not None:
                self._dev.write(f":VOLT {amplitude_vpp:.6f}")
            self._dev.query("*OPC?")
            self._drain_errs(f"NCO retune to {freq_hz/1e6:.1f} MHz")
        return freq_hz

    def duc_am_setup(self, carrier_hz, mod_hz=200e3, depth_db=6.0,
                     amplitude_vpp=0.5, channel=1, n_total=16384):
        """DUC/IQM-ONE carrier with a sinusoidal AMPLITUDE modulation.

        Built for the AM-to-PM measurement: swing a stage's drive by a known
        amount and watch what its output PHASE does. The slope (degrees per
        dB) is the number that says whether running an amplifier or a
        multiplier near compression is affordable. In this chain it is not a
        detail -- a phase error at the AWG is multiplied by 192 on the way to
        310-370 GHz and nothing downstream can absorb it, whereas an
        amplitude error is multiplied into a 7 dB window that has room for it.

        The envelope is normalised so its PEAK is full scale. amplitude_vpp
        therefore means the same thing it means for duc_cw_setup() -- the
        level at the TOP of the swing -- so an AM run made at the CW
        operating amplitude explores DOWNWARD from the operating point and
        never drives anything harder than the CW run already did. That is
        what makes this safe to point at a chain already sitting close to a
        compression limit: no new worst case is created.

        I carries the envelope, Q is zero, so this is ordinary double-sideband
        AM -- sidebands at carrier +- mod_hz and no phase modulation imposed
        at all. Whatever phase modulation comes back is the DUT's, plus the
        AWG's own, which needs a thru reference to separate.

        An integer number of modulation cycles is fitted into the buffer, so
        looping playback is continuous: without that, the wrap discontinuity
        once per buffer would spray energy across the whole analysis band and
        land in the same place as the phase modulation being measured.

        Retune the carrier afterwards with duc_cw_step() -- it writes only
        :NCO:CFR1 and leaves this baseband playing.
        """
        rate = 9e9
        rate_cx = rate / 8.0
        if not 0 < carrier_hz <= rate / 2:
            raise ValueError(f"{carrier_hz/1e6:.1f} MHz outside DUC NCO range "
                             f"(0-{rate/2e9:g} GHz]")
        if not 0.0 < depth_db <= 40.0:
            raise ValueError(f"depth_db {depth_db} outside (0, 40] dB")
        n_total = max(int(np.ceil(n_total / 32)) * 32, 64)
        cycles = max(int(round(mod_hz * n_total / rate_cx)), 1)
        spc = n_total / cycles
        if spc < 16:
            raise ValueError(
                f"{mod_hz/1e3:.1f} kHz gives only {spc:.1f} baseband samples "
                f"per modulation cycle in an {n_total}-sample buffer -- the "
                f"envelope would be a staircase. Lower mod_hz or raise n_total.")
        actual_mod = cycles * rate_cx / n_total

        # depth_db is the PEAK-TO-TROUGH swing: (1+m)/(1-m) in amplitude.
        ratio = 10 ** (depth_db / 20.0)
        m = (ratio - 1.0) / (ratio + 1.0)
        k = np.arange(n_total, dtype=np.float64)
        env = (1.0 + m * np.cos(2 * np.pi * cycles * k / n_total)) / (1.0 + m)
        data = self._iq_to_u16(env, np.zeros(n_total))

        print(f"[DUC AM] CH{channel} {carrier_hz/1e6:.4f} MHz, "
              f"AM {actual_mod/1e3:.3f} kHz ({cycles} cyc in "
              f"{n_total/rate_cx*1e6:.3f} us), depth {depth_db:.2f} dB p-p "
              f"(m={m:.4f}), peak {amplitude_vpp:.4f} Vpp")
        with self._lock:
            self._reset_locked(opc=True)
            self._rate_sent.clear()
            self._cmd(f":INST:CHAN {channel}"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(channel, 1, data, carrier_hz, rate, rate_cx,
                                amplitude_vpp)
            self._active_seg[channel] = 1
        return dict(carrier_hz=float(carrier_hz), mod_hz=float(actual_mod),
                    depth_db=float(depth_db), m=float(m),
                    n_total=int(n_total), cycles=int(cycles),
                    samples_per_cycle=float(spc), trough_db=float(-depth_db),
                    peak_vpp=float(amplitude_vpp))

    def send_chirp_duc_sync(self, f_carrier_chirp_hz, f_start_bb_hz, f_stop_bb_hz,
                            chirp_us, dead_us, amplitude_vpp=0.5, window_frac=0.05,
                            sync_pulse_ns=40, sync_amp_vpp=0.5, sync_carrier_hz=None,
                            coherent=True):
        """DUC/IQM-ONE analytic chirp on CH1 + gated-burst sync on CH2 at buffer
        start (9 GS/s DAC, X8). Returns chirp active duration (s).

        coherent=True snaps the carrier to the grid that makes the output
        repeat buffer to buffer (1/period) and shifts the baseband span by the
        same residual, so the chirp still sweeps exactly the absolute
        frequencies asked for. The chirp is windowed to zero at both ends and
        never wraps, so nothing constrains its baseband -- this correction is
        free. Without it the free-running NCO randomises the carrier phase of
        every shot and coherent averaging cancels the FID (measured
        2026-10-20: 0.998 on grid against 0.21-0.37 off it).
        """
        rate = 9e9
        rate_cx = rate / 8.0
        n_chirp = round(chirp_us * 1e-6 * rate_cx)
        n_dead  = round(dead_us  * 1e-6 * rate_cx)
        n_total = max(int(np.ceil((n_chirp + n_dead) / 32)) * 32, 64)
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate_cx), 1)
        step = self.coherent_step_hz(n_total, rate, 8)

        if coherent:
            snapped, residual = self.snap_duc_carrier(f_carrier_chirp_hz,
                                                      n_total, rate, 8)
            if abs(residual) > self.GRID_TOL_HZ:
                # upper sideband (exp(+j.phase)), so output = carrier + baseband
                f_start_bb_hz += residual
                f_stop_bb_hz += residual
                print(f"  [grid {step/1e3:.3f} kHz] carrier "
                      f"{f_carrier_chirp_hz/1e6:.6f} -> {snapped/1e6:.6f} MHz "
                      f"({residual:+.1f} Hz); baseband shifted, chirp span "
                      f"held exactly")
                f_carrier_chirp_hz = snapped
        sync_carrier = sync_carrier_hz if sync_carrier_hz is not None else f_carrier_chirp_hz
        if coherent:
            # CH2's burst has a DC baseband, so there is nothing to shift --
            # its carrier just snaps. It is a sync pulse; its exact frequency
            # does not matter, but its phase repeating does.
            sync_carrier = self.snap_duc_carrier(sync_carrier, n_total,
                                                 rate, 8)[0]

        # CH1: analytic (single-sideband) windowed chirp.
        t = np.arange(n_chirp, dtype=np.float64) / rate_cx
        T = t[-1] if n_chirp > 1 else 1e-9
        phase = 2 * np.pi * (f_start_bb_hz * t +
                             (f_stop_bb_hz - f_start_bb_hz) / (2 * T) * t ** 2)
        win = self._gauss_win(n_chirp, window_frac)
        I1 = np.zeros(n_total); I1[:n_chirp] = np.cos(phase) * win
        Q1 = np.zeros(n_total); Q1[:n_chirp] = np.sin(phase) * win
        ch1 = self._iq_to_u16(I1, Q1)
        # CH2: burst at buffer start.
        I2 = np.zeros(n_total); I2[:n_pulse] = 1.0
        ch2 = self._iq_to_u16(I2, np.zeros(n_total))
        print(f"[DUC CHIRP] CH1 {(f_carrier_chirp_hz+f_start_bb_hz)/1e6:.1f}->"
              f"{(f_carrier_chirp_hz+f_stop_bb_hz)/1e6:.1f} MHz  active {n_chirp/rate_cx*1e6:.3f} us  "
              f"period {n_total/rate_cx*1e6:.3f} us  CH2 burst {n_pulse/rate_cx*1e9:.1f} ns")
        with self._lock:
            self._reset_locked(opc=True)
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, ch1, f_carrier_chirp_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, ch2, sync_carrier, rate, rate_cx, sync_amp_vpp)
            self._active_seg[2] = 2
        return n_chirp / rate_cx

    def stop(self):
        # No :ABORt here — the Proteus rejects it ('209, illegal/unknown scpi'
        # on fw 1.237.0); :INIT:CONT OFF + :OUTP OFF is the whole recipe.
        with self._lock:
            for ch in [1, 2]:
                self._dev.write(f":INST:CHAN {ch}")
                time.sleep(0.05)
                self._dev.write(":INIT:CONT OFF")
                time.sleep(0.05)
                self._dev.write(":OUTP OFF")
                time.sleep(0.05)
        print("AWG stopped")

    def close(self):
        self.stop()
        self._dev.close()