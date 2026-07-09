import numpy as np
import matplotlib.pyplot as plt
from awg import AWG, SAMPLE_RATE_SINGLE, SAMPLE_RATE_DUAL
from scope import Scope

DEFAULT_FREQS = [10e6, 50e6, 100e6, 200e6, 500e6, 1e9, 1.5e9, 2e9, 3e9, 4e9]


def amplitude_sweep(freqs_hz=None, amplitude_vpp=0.5, channel=1, settle=2.0, n_averages=512):
    """
    Sweep sine frequencies on AWG CH<channel>, read Vpp from scope at each step.
    Returns list of (target_hz, actual_hz, vpp_volts, loss_db).
    """
    if freqs_hz is None:
        freqs_hz = DEFAULT_FREQS

    awg = AWG(sample_rate=SAMPLE_RATE_SINGLE)
    scope = Scope()
    scope.setup(channel=channel, n_averages=n_averages)

    results = []
    ref_vpp = None

    print(f"\n{'Target (MHz)':>14}  {'Actual (MHz)':>14}  {'Vpp (mV)':>10}  {'Loss (dB)':>10}")
    print("-" * 56)

    actual = awg.send_sine(freqs_hz[0], amplitude_vpp, channel=channel, reset=True)
    vpp, _, _ = scope.measure_vpp(channel=channel, settle=settle)
    ref_vpp = vpp
    results.append((freqs_hz[0], actual, vpp, 0.0))
    print(f"{freqs_hz[0]/1e6:>14.1f}  {actual/1e6:>14.3f}  {vpp*1e3:>10.1f}  {'0.00':>10}")

    try:
        for f in freqs_hz[1:]:
            actual = awg.update_sine(f, amplitude_vpp, channel=channel)
            vpp, _, _ = scope.measure_vpp(channel=channel, settle=settle)
            loss_db = 20 * np.log10(vpp / ref_vpp) if vpp > 0 else float('-inf')
            results.append((f, actual, vpp, loss_db))
            print(f"{f/1e6:>14.1f}  {actual/1e6:>14.3f}  {vpp*1e3:>10.1f}  {loss_db:>10.2f}")
    except KeyboardInterrupt:
        print("\nSweep aborted.")
    finally:
        awg.close()
        scope.close()

    return results


def plot_waveform(channel=1, title=None):
    """Capture and plot the current waveform from the scope."""
    scope = Scope()
    t, v = scope.get_waveform(channel=channel)
    scope.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(t * 1e9, v * 1e3)
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title(title or f"CH{channel} waveform")
    axes[0].grid(True)

    n = len(v)
    dt = t[1] - t[0]
    freqs = np.fft.rfftfreq(n, dt)
    spectrum = np.abs(np.fft.rfft(v)) * 2 / n
    axes[1].plot(freqs * 1e-6, 20 * np.log10(spectrum + 1e-12))
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Amplitude (dBV)")
    axes[1].set_title("Spectrum")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    return t, v


if __name__ == "__main__":
    MODE = "waveform"

    if MODE == "sweep":
        amplitude_sweep(
            freqs_hz=DEFAULT_FREQS,
            amplitude_vpp=0.5,
            channel=1,
            settle=2.0,
            n_averages=512,
        )
    elif MODE == "waveform":
        plot_waveform(channel=1)