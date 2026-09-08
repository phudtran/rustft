"""Compare rustft's STFT/ISTFT against PyTorch for accuracy and speed.

Two things this deliberately does that the older version of this script did not:

* The window is float64. `torch.hann_window` returns float32 by default, and mixing
  that with a float64 signal puts ~1e-7 of error into *PyTorch's own* roundtrip. It
  looks like a rustft bug and is not one.
* The `Stft` object is built once, outside the timing loop, because `torch.stft`
  likewise reuses a cached plan. Planning an FFT per call measures the planner.
"""

import time

import numpy as np
import torch
from rustft import Stft

TEST_CASES = [
    # (channels, signal length, n_fft, hop length)
    (2, 16384, 1024, 512),
    (2, 261120, 6144, 1024),
    (2, 65536, 4096, 2048),
]


def generate_test_signal(num_channels, signal_length, sample_rate=44100, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.arange(signal_length) / sample_rate
    tone = sum(np.sin(2 * np.pi * f * t) for f in (440, 880, 1320))
    signal = np.empty((num_channels, signal_length))
    for channel in range(num_channels):
        noisy = tone + rng.normal(0, 0.1, signal_length)
        signal[channel] = noisy / np.max(np.abs(noisy))
    return signal


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def compare(num_channels, signal_length, n_fft, hop_length, dtype="float64", num_trials=10):
    torch_dtype = getattr(torch, dtype)
    np_dtype = getattr(np, dtype)
    window = torch.hann_window(n_fft, periodic=True, dtype=torch_dtype)
    stft = Stft(n_fft, hop_length, "hann", True, dtype)
    rng = np.random.default_rng(0)

    totals = {k: 0.0 for k in (
        "stft_diff", "istft_diff", "stft_scale", "rust_roundtrip", "torch_roundtrip",
        "rust_stft_torch_istft", "torch_stft_rust_istft",
        "t_rust_stft", "t_rust_istft", "t_torch_stft", "t_torch_istft",
    )}

    for _ in range(num_trials):
        signal = generate_test_signal(num_channels, signal_length, rng=rng).astype(np_dtype)
        tensor = torch.from_numpy(signal)

        rust_spec, dt = timed(stft.forward, signal)
        totals["t_rust_stft"] += dt
        torch_spec, dt = timed(
            lambda: torch.stft(tensor, n_fft, hop_length, window=window,
                               center=True, return_complex=True))
        totals["t_torch_stft"] += dt

        rust_back, dt = timed(stft.inverse, rust_spec)
        totals["t_rust_istft"] += dt
        torch_back, dt = timed(
            lambda: torch.istft(torch_spec, n_fft, hop_length, window=window, center=True))
        totals["t_torch_istft"] += dt

        torch_spec = torch_spec.numpy()
        torch_back = torch_back.numpy()
        # Widen only for the arithmetic below; every transform above ran in `dtype`.
        signal = signal.astype(np.float64)
        cross_rust = stft.inverse(torch_spec)
        cross_torch = torch.istft(torch.from_numpy(rust_spec), n_fft, hop_length,
                                  window=window, center=True).numpy()

        reference = signal[:, : rust_back.shape[1]]
        totals["stft_scale"] += np.abs(torch_spec).max()
        totals["stft_diff"] += np.abs(rust_spec - torch_spec.astype(np.complex128)).max()
        totals["istft_diff"] += np.abs(rust_back - torch_back.astype(np.float64)).max()
        totals["rust_roundtrip"] += np.abs(reference - rust_back).max()
        totals["torch_roundtrip"] += np.abs(reference - torch_back).max()
        totals["rust_stft_torch_istft"] += np.abs(reference - cross_torch).max()
        totals["torch_stft_rust_istft"] += np.abs(reference - cross_rust).max()

    avg = {k: v / num_trials for k, v in totals.items()}
    relative = avg["stft_diff"] / avg["stft_scale"]
    print(f"  max |rustft - PyTorch|, STFT           : {avg['stft_diff']:.3e}"
          f"   ({relative:.2e} of peak {avg['stft_scale']:.1f})")
    print(f"  max |rustft - PyTorch|, ISTFT          : {avg['istft_diff']:.3e}")
    print(f"  rustft roundtrip error                 : {avg['rust_roundtrip']:.3e}")
    print(f"  PyTorch roundtrip error                : {avg['torch_roundtrip']:.3e}")
    print(f"  rustft STFT -> PyTorch ISTFT           : {avg['rust_stft_torch_istft']:.3e}")
    print(f"  PyTorch STFT -> rustft ISTFT           : {avg['torch_stft_rust_istft']:.3e}")
    print()
    rust_total = avg["t_rust_stft"] + avg["t_rust_istft"]
    torch_total = avg["t_torch_stft"] + avg["t_torch_istft"]
    print(f"  {'':<22}{'rustft':>12}{'PyTorch':>12}{'ratio':>10}")
    for label, r, t in (
        ("STFT", avg["t_rust_stft"], avg["t_torch_stft"]),
        ("ISTFT", avg["t_rust_istft"], avg["t_torch_istft"]),
        ("STFT + ISTFT", rust_total, torch_total),
    ):
        print(f"  {label:<22}{r * 1e3:>10.3f}ms{t * 1e3:>10.3f}ms{t / r:>9.2f}x")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads\n")
    for dtype in ("float64", "float32"):
        print(f"================  {dtype} on both sides  ================\n")
        for num_channels, signal_length, n_fft, hop_length in TEST_CASES:
            print(f"{num_channels} channels, {signal_length} samples, "
                  f"n_fft {n_fft}, hop {hop_length}")
            compare(num_channels, signal_length, n_fft, hop_length, dtype)
            print()
