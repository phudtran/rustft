import numpy as np
import torch
import time
from rustfft_test import rust_fft, rust_ifft

def generate_test_signal(size):
    return np.random.random(size)

def compare_fft_ifft_roundtrip(signal_size, num_trials=100):
    total_rust_diff = 0
    total_pytorch_diff = 0
    total_rust_fft_pytorch_ifft_diff = 0
    total_pytorch_fft_rust_ifft_diff = 0


    for _ in range(num_trials):
        signal = generate_test_signal(signal_size)
        # ----  round trip tests ---

        # Rust FFT -> IFFT
        rust_fft_result = np.array(rust_fft(signal))
        rust_fft_complex = rust_fft_result[::2] + 1j * rust_fft_result[1::2]
        rust_roundtrip = np.array(rust_ifft(rust_fft_result))

        # PyTorch FFT -> IFFT
        torch_signal = torch.from_numpy(signal)
        pytorch_fft_result = torch.fft.fft(torch_signal)
        pytorch_roundtrip = torch.fft.ifft(pytorch_fft_result).real.numpy()

        # PyTorch FFT -> Rust IFFT
        pytorch_fft_result_numpy = pytorch_fft_result.numpy()
        pytorch_fft_result_flat = np.concatenate((pytorch_fft_result_numpy.real, pytorch_fft_result_numpy.imag))
        rust_roundtrip_pytorch_fft = np.array(rust_ifft(pytorch_fft_result_flat))

        # Rust FFT -> PyTorch IFFT
        rust_fft_complex_torch = torch.from_numpy(rust_fft_complex)
        pytorch_roundtrip_rust_fft = torch.fft.ifft(rust_fft_complex_torch).real.numpy()

        # Compare results to original signal
        rust_diff = np.abs(signal - rust_roundtrip)
        pytorch_diff = np.abs(signal - pytorch_roundtrip)
        rust_fft_pytorch_ifft_diff = np.abs(signal - pytorch_roundtrip_rust_fft)
        pytorch_fft_rust_ifft_diff = np.abs(signal - rust_roundtrip_pytorch_fft)


        total_rust_diff += np.mean(rust_diff)
        total_pytorch_diff += np.mean(pytorch_diff)
        total_rust_fft_pytorch_ifft_diff = np.mean(rust_fft_pytorch_ifft_diff)
        total_pytorch_fft_rust_ifft_diff = np.mean(pytorch_fft_rust_ifft_diff)

    # ----  round trip tests ----

    avg_rust_diff = total_rust_diff / num_trials
    avg_pytorch_diff = total_pytorch_diff / num_trials
    avg_total_rust_fft_pytorch_ifft_diff = total_rust_fft_pytorch_ifft_diff / num_trials
    avg_total_pytorch_fft_rust_ifft_diff = total_pytorch_fft_rust_ifft_diff / num_trials

    print(f"Average Rust roundtrip difference: {avg_rust_diff}")
    print(f"Average PyTorch roundtrip difference: {avg_pytorch_diff}")
    print(f"Average roundtrip difference (Rust FFT -> PyTorch IFFT): {avg_total_rust_fft_pytorch_ifft_diff}")
    print(f"Average roundtrip difference (PyTorch FFT -> Rust IFFT): {avg_total_pytorch_fft_rust_ifft_diff}")

if __name__ == "__main__":
    signal_sizes = [1024, 4096, 16384]
    for size in signal_sizes:
        print(f"\nTesting with signal size: {size}")
        compare_fft_ifft_roundtrip(size)