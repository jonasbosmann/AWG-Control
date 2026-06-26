import pyvisa
import numpy as np
import time

AWG_IP = "141.51.196.111"
RESOURCE = f"TCPIP0::{AWG_IP}::5025::SOCKET"

SAMPLE_RATE_SINGLE = 9e9    # single-channel max (both DACs interleaved → CH2 disabled)
SAMPLE_RATE_DUAL   = 2.5e9  # dual-channel max per manual (one DAC per channel)
SAMPLE_RATE = SAMPLE_RATE_SINGLE  # active rate — overridden in __main__ for dual-channel
N_SAMPLES = 2048    # must be multiple of 64 for Proteus granularity

rm = pyvisa.ResourceManager()
awg = rm.open_resource(RESOURCE)
awg.timeout = 10000
awg.read_termination = '\n'
awg.write_termination = '\n'

print("Connected:", awg.query("*IDN?"))


def cmd(c):
    """Drain stale errors, send command, report result."""
    while True:
        e = awg.query(":SYST:ERR?")
        if e.startswith("0"):
            break
    awg.write(c)
    time.sleep(0.05)
    err = awg.query(":SYST:ERR?")
    label = c.lstrip(":")
    if err.startswith("0"):
        print(f"  OK    {label}")
    else:
        print(f"  ERROR {label} -> {err.strip()}")


def _setup(channel=1, reset=True):
    if reset:
        awg.write("*CLS; *RST")
        time.sleep(0.5)
    cmd(f":INST:CHAN {channel}")           # select channel first
    cmd(f":FREQ:RAST {SAMPLE_RATE:.0f}")  # per-channel setting
    if reset:
        cmd(":TRAC:DEL:ALL")
    cmd(":INIT:CONT ON")


def _upload(wave_u16, segnum=1):
    n = len(wave_u16)
    cmd(f":TRAC:DEF {segnum},{n}")
    cmd(f":TRAC:SEL {segnum}")
    data = wave_u16.tobytes()
    nb = len(data)
    nb_str = str(nb)
    awg.write_raw(f":TRAC:DATA #{len(nb_str)}{nb_str}".encode() + data + b"\n")
    time.sleep(0.3)
    awg.write("*CLS")
    print("  OK    TRAC:DATA")


def _play(amplitude_vpp=0.5, segnum=1):
    cmd(f":FUNC:MODE:SEGM {segnum}")
    cmd(f":VOLT {amplitude_vpp:.3f}")
    cmd(":OUTP ON")


def _make_sine_u16(cycles):
    t = np.arange(N_SAMPLES)
    wave = np.sin(2 * np.pi * cycles * t / N_SAMPLES)
    return ((wave + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)


def send_sine(frequency_hz=100e6, amplitude_vpp=0.5, channel=1, reset=True):
    cycles = max(round(frequency_hz * N_SAMPLES / SAMPLE_RATE), 1)
    actual_freq = cycles * SAMPLE_RATE / N_SAMPLES
    print(f"[CH{channel}] Sine: {frequency_hz/1e6:.1f} MHz -> {actual_freq/1e6:.3f} MHz ({cycles} cycles/buffer)")
    _setup(channel, reset)
    _upload(_make_sine_u16(cycles), segnum=channel)
    _play(amplitude_vpp, segnum=channel)
    print(f"[CH{channel}] Sine running")


def send_ramp(amplitude_vpp=0.5, channel=1, reset=True):
    print(f"[CH{channel}] Ramp: sawtooth 0 -> full scale")
    _setup(channel, reset)
    _upload(np.linspace(0, 65535, N_SAMPLES, dtype=np.uint16), segnum=channel)
    _play(amplitude_vpp, segnum=channel)
    print(f"[CH{channel}] Ramp running")


def send_square(frequency_hz=None, amplitude_vpp=0.5, channel=1, reset=True):
    cycles = 1
    if frequency_hz is not None:
        cycles = max(round(frequency_hz * N_SAMPLES / SAMPLE_RATE), 1)
    actual_freq = cycles * SAMPLE_RATE / N_SAMPLES
    print(f"[CH{channel}] Square: {cycles} cycle(s)/buffer -> {actual_freq/1e6:.3f} MHz")

    samples_per_cycle = N_SAMPLES // cycles
    half = samples_per_cycle // 2
    one_cycle = np.array([0xFFFF] * half + [0] * (samples_per_cycle - half), dtype=np.uint16)
    wave_u16 = np.tile(one_cycle, cycles)[:N_SAMPLES]

    _setup(channel, reset)
    _upload(wave_u16, segnum=channel)
    _play(amplitude_vpp, segnum=channel)
    print(f"[CH{channel}] Square running")


def freq_sweep(channel=1, amplitude_vpp=0.5, dwell=3.0, reset=True, freqs_hz=None):
    """
    Step through sine frequencies on one channel, holding each for `dwell`
    seconds so you can read the amplitude on the scope. Press Ctrl-C to abort.

    Run once with CH2 off and once with CH2 active to compare single- vs
    dual-channel bandwidth.
    """
    if freqs_hz is None:
        # Covers 10 MHz to 4 GHz (scope limit); Nyquist at 9 GS/s is 4.5 GHz
        freqs_hz = [10e6, 50e6, 100e6, 200e6, 500e6,
                    1e9, 1.5e9, 2e9, 2.5e9, 3e9, 3.5e9, 4e9]

    print(f"\n=== Freq sweep CH{channel} @ {SAMPLE_RATE/1e9:.0f} GS/s | {dwell}s/step ===")
    print(f"{'Target (MHz)':>14}  {'Actual (MHz)':>14}  {'Cycles':>8}")
    print("-" * 44)

    _setup(channel, reset)
    try:
        for f in freqs_hz:
            cycles = max(round(f * N_SAMPLES / SAMPLE_RATE), 1)
            actual = cycles * SAMPLE_RATE / N_SAMPLES
            _upload(_make_sine_u16(cycles), segnum=channel)
            _play(amplitude_vpp, segnum=channel)
            print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {cycles:>8d}  <- measure now")
            time.sleep(dwell)
    except KeyboardInterrupt:
        print("\nSweep aborted.")

    print("=== Sweep done ===\n")


def stop():
    for ch in [1, 2]:
        awg.write(f":INST:CHAN {ch}")
        awg.write(":OUTP OFF")
        awg.write(":ABOR")
    print("Output stopped")


def close():
    stop()
    awg.close()


if __name__ == "__main__":
    # Change MODE to select what happens when you run this file:
    #   "sine"   -- sine wave on CH1
    #   "square" -- square wave on CH1
    #   "ramp"   -- ramp on CH1
    #   "both"   -- sine on CH1, square on CH2
    #   "sweep"  -- bandwidth sweep on CH1
    #   "stop"   -- silence all outputs
  
    if MODE == "sine":
        send_sine(frequency_hz=500e6, amplitude_vpp=0.5, channel=1)
    elif MODE == "square":
        send_square(frequency_hz=500e6, amplitude_vpp=0.5, channel=1)
    elif MODE == "ramp":
        send_ramp(amplitude_vpp=0.5, channel=1)
    elif MODE == "both":
        SAMPLE_RATE = SAMPLE_RATE_DUAL
        send_sine(frequency_hz=500e6, amplitude_vpp=0.5, channel=1)
        send_sine(frequency_hz=500e6, amplitude_vpp=0.5, channel=2, reset=False)
    elif MODE == "sweep":
        # Single-channel baseline — 9 GS/s, CH2 off (Nyquist = 4.5 GHz):
        # SAMPLE_RATE = SAMPLE_RATE_SINGLE
        # freq_sweep(channel=1, amplitude_vpp=0.5, dwell=3.0)

        # Dual-channel — 4.5 GS/s per channel, CH2 active (Nyquist = 2.25 GHz):
        SAMPLE_RATE = SAMPLE_RATE_DUAL
        dual_freqs = [10e6, 50e6, 100e6, 200e6, 500e6, 750e6, 1e9, 1.1e9, 1.2e9]
        send_sine(frequency_hz=100e6, amplitude_vpp=0.5, channel=2, reset=True)
        freq_sweep(channel=1, amplitude_vpp=0.5, dwell=3.0, reset=False, freqs_hz=dual_freqs)
    elif MODE == "stop":
        stop()