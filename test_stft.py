import numpy as np
import torch
from rustfft_test import rust_stft, rust_istft

def generate_test_signal(num_channels, signal_length):
    return np.random.random((num_channels, signal_length))

def compare_stft_istft(num_channels, signal_length, n_fft, hop_length, num_trials=10):
    total_rust_diff = 0
    total_pytorch_diff = 0
    total_rust_stft_pytorch_istft_diff = 0
    total_pytorch_stft_rust_istft_diff = 0
    window = torch.hann_window(window_length=n_fft, periodic=True)
    for _ in range(num_trials):
        signal = generate_test_signal(num_channels, signal_length)

        # Rust STFT -> ISTFT
        rust_stft_result = rust_stft(signal, n_fft, hop_length, None)
        rust_roundtrip = rust_istft(rust_stft_result,n_fft, hop_length, signal_length)

        # PyTorch STFT -> ISTFT
        pytorch_stft_result = torch.stft(
            torch.from_numpy(signal),
            n_fft, hop_length,
            return_complex=True,
            window=window,
            center=True,)
        pytorch_roundtrip = torch.istft(
            pytorch_stft_result,
            n_fft,
            hop_length,
            window=window,
            center=True).numpy()

        # Rust STFT -> PyTorch ISTFT
        pytorch_istft_rust_stft = torch.istft(
            torch.from_numpy(rust_stft_result),
            n_fft,
            hop_length,
            window=window,
            center=True
            ).numpy()

        # PyTorch STFT -> Rust ISTFT
        rust_istft_pytorch_stft = rust_istft(
            pytorch_stft_result.numpy(),
            n_fft,
            hop_length,
            signal_length,
            None)

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

    print(f"Average Rust roundtrip error: {avg_rust_diff}")
    print(f"Average PyTorch roundtrip error: {avg_pytorch_diff}")
    print(f"Average roundtrip error (Rust STFT -> PyTorch ISTFT): {avg_rust_stft_pytorch_istft_diff}")
    print(f"Average roundtrip error (PyTorch STFT -> Rust ISTFT): {avg_pytorch_stft_rust_istft_diff}")

if __name__ == "__main__":
    test_cases = [
        (2, 16384, 1024, 512),
        (4, 32768, 2048, 1024),
        (8, 65536, 4096, 2048)
    ]

    for num_channels, signal_length, n_fft, hop_length in test_cases:
        print(f"\nTesting with: {num_channels} channels, signal length {signal_length}, n_fft {n_fft}, hop_length {hop_length}")
        compare_stft_istft(num_channels, signal_length, n_fft, hop_length)