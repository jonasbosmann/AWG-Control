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

    def set_reference_external(self, freq_mhz=10):
        """Lock the AWG's clock to an external 10/100 MHz reference (e.g.
        the scope's REF OUT) instead of its own internal 1 ppm TCXO.

        Needs a physical cable: scope REF OUT (BNC) -> AWG REF IN (BNC).
        Without this, even software-aligned averaging
        (chirp_quality.align_and_average()) can only correct a constant
        per-shot time OFFSET between AWG and scope -- independent,
        unsynchronized clocks also drift the whole waveform's timebase
        slightly differently on every shot (worse at higher frequency,
        since more phase accumulates per ppm of clock mismatch over the
        same duration), which a simple time-shift alignment cannot undo.
        Sharing one reference removes that drift at the source.
        """
        freq_kw = {10: "10M", 100: "100M"}[freq_mhz]
        with self._lock:
            self._cmd(":ROSC:SOUR EXT")
            self._cmd(f":ROSC:FREQ {freq_kw}")

    def _setup(self, channel=1, reset=True, sample_rate=None):
        rate = sample_rate if sample_rate is not None else self.sample_rate
        if reset:
            self._dev.write("*CLS; *RST")
            time.sleep(0.5)
            self._rate_sent.clear()
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

    def _play(self, amplitude_vpp=0.5, segnum=1):
        self._cmd(f":FUNC:MODE:SEGM {segnum}")
        self._cmd(f":VOLT {amplitude_vpp:.3f}")
        self._cmd(":OUTP ON")

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
        """
        t = timing or {}
        chirp_us  = float(t.get("chirp_us",  1.0))
        dead_us   = float(t.get("dead_us",   0.1))
        detect_us = float(t.get("detect_us", 10.0))
        n_chirp  = round(chirp_us  * 1e-6 * rate_eff)
        n_dead   = round(dead_us   * 1e-6 * rate_eff)
        n_detect = round(detect_us * 1e-6 * rate_eff)
        gran    = self.GRAN_DUC if mode == "duc" else self.GRAN_DIRECT
        floor_n = 64 if mode == "duc" else 128
        n_total = max(int(np.ceil((n_chirp + n_dead + n_detect) / gran)) * gran, floor_n)
        return {
            "mode": mode, "rate_eff": rate_eff,
            "n_chirp": n_chirp, "n_dead": n_dead, "n_detect": n_detect,
            "n_total": n_total,
            "period_s": n_total / rate_eff,
            "pad_s": (n_total - n_chirp - n_dead - n_detect) / rate_eff,
            "windows": {
                "continuous": (0, n_total),
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

        if kind in ("cw", "chirp"):
            z = (np.exp(1j * phase) if mode == "duc" else np.sin(phase)) * env
        elif kind == "square":
            real = np.where(np.sin(phase) >= 0, 1.0, -1.0)
            z = (self._analytic(real) if mode == "duc" else real) * env
        elif kind == "ramp":
            real = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
            z = (self._analytic(real) if mode == "duc" else real) * env
        else:
            raise ValueError(f"unknown signal kind {kind!r} "
                             f"(expected one of {self.SIGNAL_KINDS})")

        if mode == "duc":
            sideband = spec.get("sideband", "upper")
            if sideband == "lower":
                z = np.conj(z)
            elif sideband == "both":
                z = np.real(z).astype(np.complex128)   # DSB: carrier ± f
            elif sideband != "upper":
                raise ValueError(f"unknown sideband {sideband!r}")
            I, Q = np.real(z).copy(), np.imag(z).copy()
            # The analytic square/ramp overshoots ±1 (Gibbs); scale it back so
            # _iq_to_u16 does not clip the peaks into a flat top.
            peak = max(np.abs(I).max(initial=0.0), np.abs(Q).max(initial=0.0))
            if peak > 1.0:
                I, Q = I / peak, Q / peak
                info["scaled_by"] = 1.0 / peak
            I_full = np.zeros(plan["n_total"]); I_full[start:start + n_win] = I
            Q_full = np.zeros(plan["n_total"]); Q_full[start:start + n_win] = Q
            info["sideband"] = sideband
            return self._iq_to_u16(I_full, Q_full), info

        full = np.zeros(plan["n_total"])
        full[start:start + n_win] = np.real(z)
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
                       window_frac=0.05, verbose=True):
        """Play one spec per channel, both built into ONE shared buffer period.

        mode='duc'    : 9 GS/s DAC clock, X8 interpolation (1.125 GS/s complex
                        baseband), IQM ONE. Each channel's carrier is its own
                        exact NCO frequency; both channels stay usable because
                        the 2.5 GS/s dual-channel ceiling is an ARB/direct-mode
                        memory limit, not a DUC one.
        mode='direct' : real samples at `rate` (defaults to 2.5 GS/s whenever a
                        second channel is active — one DAC per channel).

        Returns the plan dict: period, window sample counts, and the per-channel
        info (including the frequency actually produced after integer-cycle
        snapping) so a caller can display what the hardware was really told.
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
                self._dev.write("*CLS; *RST")
                time.sleep(0.5)
                self._dev.query("*OPC?")
                self._rate_sent.clear()
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
        """
        amplitudes = amplitudes or {}
        for ch, f in carriers.items():
            self.duc_cw_step(float(f), channel=int(ch),
                             amplitude_vpp=amplitudes.get(ch))
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
        self._cmd(f":VOLT {amp:.3f}")
        self._cmd(":NCO:SIXD1 ON")
        self._cmd(f":NCO:CFR1 {ncof:.0f}")
        self._cmd(":OUTP ON")

    def send_cw_direct_sync(self, freq_hz, amplitude_vpp=0.5,
                            sync_pulse_ns=40, sync_amp_vpp=0.8):
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
                               amplitude_vpp=0.5, sync_pulse_ns=40, sync_amp_vpp=0.8,
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
                         sync_pulse_ns=40, sync_amp_vpp=0.8, sync_carrier_hz=None):
        """DUC/IQM-ONE CW on CH1 (pure carrier at carrier_hz) + gated-burst sync on
        CH2 at buffer start (9 GS/s DAC, X8). Returns carrier_hz."""
        rate = 9e9
        rate_cx = rate / 8.0
        n_total = max(int(np.ceil(n_total / 32)) * 32, 64)   # 2*n_total multiple of 64
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate_cx), 1)
        sync_carrier = sync_carrier_hz if sync_carrier_hz is not None else carrier_hz

        # CH1: constant complex baseband (I=1,Q=0) -> pure carrier via NCO.
        ch1 = self._iq_to_u16(np.ones(n_total), np.zeros(n_total))
        # CH2: I-envelope gate for the first n_pulse samples -> RF burst at t=0.
        I2 = np.zeros(n_total); I2[:n_pulse] = 1.0
        ch2 = self._iq_to_u16(I2, np.zeros(n_total))
        print(f"[DUC CW] CH1 {carrier_hz/1e6:.4f} MHz  "
              f"CH2 burst {n_pulse/rate_cx*1e9:.1f} ns @ {sync_carrier/1e6:.1f} MHz")
        with self._lock:
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, ch1, carrier_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, ch2, sync_carrier, rate, rate_cx, sync_amp_vpp)
            self._active_seg[2] = 2
        return carrier_hz

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
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
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
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
            self._rate_sent.clear()
            self._cmd(":INST:CHAN 1"); self._cmd(":TRAC:DEL:ALL")
            self._duc_upload_ch(1, 1, data, freq_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[1] = 1
            self._duc_upload_ch(2, 2, data, freq_hz, rate, rate_cx, amplitude_vpp)
            self._active_seg[2] = 2
        return freq_hz

    def duc_cw_step(self, freq_hz, channel=1, amplitude_vpp=None):
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
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
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
                            sync_pulse_ns=40, sync_amp_vpp=0.8, sync_carrier_hz=None):
        """DUC/IQM-ONE analytic chirp on CH1 + gated-burst sync on CH2 at buffer
        start (9 GS/s DAC, X8). Returns chirp active duration (s)."""
        rate = 9e9
        rate_cx = rate / 8.0
        n_chirp = round(chirp_us * 1e-6 * rate_cx)
        n_dead  = round(dead_us  * 1e-6 * rate_cx)
        n_total = max(int(np.ceil((n_chirp + n_dead) / 32)) * 32, 64)
        n_pulse = max(int(sync_pulse_ns * 1e-9 * rate_cx), 1)
        sync_carrier = sync_carrier_hz if sync_carrier_hz is not None else f_carrier_chirp_hz

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
            self._dev.write("*CLS; *RST"); time.sleep(0.5); self._dev.query("*OPC?")
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