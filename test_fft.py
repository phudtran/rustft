import numpy as np
import torch
from rustfft_test import rust_fft, rust_ifft, rust_fft_roundtrip_test

def generate_test_signal(size):
    return np.random.random(size)

def compare_fft_ifft_roundtrip(signal_size, num_trials=100):
    total_rust_diff = 0
    total_pytorch_diff = 0
    total_rust_fft_pytorch_ifft_diff = 0
    total_pytorch_fft_rust_ifft_diff = 0

    for _ in range(num_trials):
        signal = generate_test_signal(signal_size)

        # Rust roundtrip
        rust_roundtrip = rust_fft_roundtrip_test(signal)

        # PyTorch roundtrip
        pytorch_fft_result = torch.fft.fft(torch.from_numpy(signal))
        pytorch_roundtrip = torch.fft.ifft(pytorch_fft_result).numpy().real

        # PyTorch FFT -> Rust IFFT
        rust_roundtrip_pytorch_fft = rust_ifft(pytorch_fft_result.numpy())

        # Rust FFT -> PyTorch IFFT
        rust_fft_result = torch.from_numpy(rust_fft(signal))
        pytorch_roundtrip_rust_fft = torch.fft.ifft(rust_fft_result).numpy().real

        # Compare results to original signal
        rust_diff = np.abs(signal - rust_roundtrip)
        pytorch_diff = np.abs(signal - pytorch_roundtrip)
        rust_fft_pytorch_ifft_diff = np.abs(signal - pytorch_roundtrip_rust_fft)
        pytorch_fft_rust_ifft_diff = np.abs(signal - rust_roundtrip_pytorch_fft)

        total_rust_diff += np.mean(rust_diff)
        total_pytorch_diff += np.mean(pytorch_diff)
        total_rust_fft_pytorch_ifft_diff += np.mean(rust_fft_pytorch_ifft_diff)
        total_pytorch_fft_rust_ifft_diff += np.mean(pytorch_fft_rust_ifft_diff)

    avg_rust_diff = total_rust_diff / num_trials
    avg_pytorch_diff = total_pytorch_diff / num_trials
    avg_rust_fft_pytorch_ifft_diff = total_rust_fft_pytorch_ifft_diff / num_trials
    avg_pytorch_fft_rust_ifft_diff = total_pytorch_fft_rust_ifft_diff / num_trials

    print(f"Average Rust roundtrip error: {avg_rust_diff}")
    print(f"Average PyTorch roundtrip error: {avg_pytorch_diff}")
    print(f"Average roundtrip error (Rust FFT -> PyTorch IFFT): {avg_rust_fft_pytorch_ifft_diff}")
    print(f"Average roundtrip error (PyTorch FFT -> Rust IFFT): {avg_pytorch_fft_rust_ifft_diff}")

if __name__ == "__main__":
    signal_sizes = [1024, 4096, 16384]
    for size in signal_sizes:
        print(f"\nTesting with signal size: {size}")
        compare_fft_ifft_roundtrip(size)