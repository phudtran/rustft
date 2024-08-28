import numpy as np
import torch
from rustft import rust_stft, rust_istft, rust_stft_roundtrip
import time

def generate_test_signal(num_channels, signal_length, sample_rate=44100):
    t = np.linspace(0, signal_length / sample_rate, signal_length, endpoint=False)
    frequencies = [440, 880, 1320]  # A4, A5, E6
    signal = np.zeros((num_channels, signal_length))
    for channel in range(num_channels):
        channel_signal = np.zeros(signal_length)
        for freq in frequencies:
            channel_signal += np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.1, signal_length)
        channel_signal += noise
        channel_signal = channel_signal / np.max(np.abs(channel_signal))
        signal[channel] = channel_signal
    return signal

def compare_stft_istft(num_channels, signal_length, n_fft, hop_length, num_trials=10):
    total_rust_diff = 0
    total_pytorch_diff = 0
    total_rust_stft_pytorch_istft_diff = 0
    total_pytorch_stft_rust_istft_diff = 0

    total_rust_time = 0
    total_pytorch_time = 0
    total_rust_stft_time = 0
    total_pytorch_istft_time = 0
    total_pytorch_stft_time = 0
    total_rust_istft_time = 0

    window = torch.hann_window(window_length=n_fft, periodic=True)

    for _ in range(num_trials):
        signal = generate_test_signal(num_channels, signal_length)

        # Rust STFT -> ISTFT
        start_time = time.time()
        rust_roundtrip = rust_stft_roundtrip(signal, n_fft, hop_length)
        total_rust_time += time.time() - start_time

        # PyTorch STFT -> ISTFT
        start_time = time.time()
        pytorch_stft_result = torch.stft(
            torch.from_numpy(signal),
            n_fft, hop_length,
            return_complex=True,
            window=window,
            center=True,
        )
        total_pytorch_stft_time += time.time() - start_time

        start_time = time.time()
        pytorch_roundtrip = torch.istft(
            pytorch_stft_result,
            n_fft,
            hop_length,
            window=window,
            center=True
        ).numpy()
        total_pytorch_istft_time += time.time() - start_time

        total_pytorch_time += total_pytorch_stft_time + total_pytorch_istft_time

        # Rust STFT -> PyTorch ISTFT
        start_time = time.time()
        rust_stft_result = rust_stft(signal, n_fft, hop_length, None)
        total_rust_stft_time += time.time() - start_time

        start_time = time.time()
        pytorch_istft_rust_stft = torch.istft(
            torch.from_numpy(rust_stft_result),
            n_fft,
            hop_length,
            window=window,
            center=True
        ).numpy()
        total_pytorch_istft_time += time.time() - start_time

        # PyTorch STFT -> Rust ISTFT
        start_time = time.time()
        rust_istft_pytorch_stft = rust_istft(
            pytorch_stft_result.numpy(),
            n_fft,
            hop_length)
        total_rust_istft_time += time.time() - start_time

        # Compare results to original signal
        rust_diff = np.mean(np.abs(signal - rust_roundtrip))
        pytorch_diff = np.mean(np.abs(signal - pytorch_roundtrip))
        rust_stft_pytorch_istft_diff = np.mean(np.abs(signal - pytorch_istft_rust_stft))
        pytorch_stft_rust_istft_diff = np.mean(np.abs(signal - rust_istft_pytorch_stft))

        total_rust_diff += rust_diff
        total_pytorch_diff += pytorch_diff
        total_rust_stft_pytorch_istft_diff += rust_stft_pytorch_istft_diff
        total_pytorch_stft_rust_istft_diff += pytorch_stft_rust_istft_diff

    avg_rust_diff = total_rust_diff / num_trials
    avg_pytorch_diff = total_pytorch_diff / num_trials
    avg_rust_stft_pytorch_istft_diff = total_rust_stft_pytorch_istft_diff / num_trials
    avg_pytorch_stft_rust_istft_diff = total_pytorch_stft_rust_istft_diff / num_trials

    avg_rust_time = total_rust_time / num_trials
    avg_pytorch_time = total_pytorch_time / num_trials
    avg_rust_stft_time = total_rust_stft_time / num_trials
    avg_pytorch_istft_time = total_pytorch_istft_time / num_trials
    avg_pytorch_stft_time = total_pytorch_stft_time / num_trials
    avg_rust_istft_time = total_rust_istft_time / num_trials

    print(f"Average Rust roundtrip error: {avg_rust_diff}")
    print(f"Average PyTorch roundtrip error: {avg_pytorch_diff}")
    print(f"Average roundtrip error (Rust STFT -> PyTorch ISTFT): {avg_rust_stft_pytorch_istft_diff}")
    print(f"Average roundtrip error (PyTorch STFT -> Rust ISTFT): {avg_pytorch_stft_rust_istft_diff}")
    print(f"\nAverage run times:")
    print(f"Rust STFT + ISTFT: {avg_rust_time:.6f} seconds")
    print(f"PyTorch STFT + ISTFT: {avg_pytorch_time:.6f} seconds")
    print(f"Rust STFT: {avg_rust_stft_time:.6f} seconds")
    print(f"PyTorch ISTFT: {avg_pytorch_istft_time:.6f} seconds")
    print(f"PyTorch STFT: {avg_pytorch_stft_time:.6f} seconds")
    print(f"Rust ISTFT: {avg_rust_istft_time:.6f} seconds")

if __name__ == "__main__":
    test_cases = [
        (2, 16384, 1024, 512),
        (4, 32768, 2048, 1024),
        (8, 65536, 4096, 2048)
    ]
    for num_channels, signal_length, n_fft, hop_length in test_cases:
        print(f"\nTesting with: {num_channels} channels, signal length {signal_length}, n_fft {n_fft}, hop_length {hop_length}")
        compare_stft_istft(num_channels, signal_length, n_fft, hop_length)