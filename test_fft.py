import numpy as np
import torch
import time
from rustfft_test import rust_fft, rust_ifft

def generate_test_signal(size):
    return np.random.random(size)

def compare_fft_ifft_roundtrip(signal_size, num_trials=100):
    total_rust_diff = 0
    total_pytorch_diff = 0
    total_time_rust = 0
    total_time_pytorch = 0
    total_fft_diff = 0
    total_ifft_diff = 0
    total_time_rust_fft = 0
    total_time_rust_ifft = 0
    total_time_pytorch_fft = 0
    total_time_pytorch_ifft = 0

    for _ in range(num_trials):
        signal = generate_test_signal(signal_size)
        # ----  FFT and IFFT tests ----

         # Rust FFT
        start_time = time.time()
        rust_fft_result = np.array(rust_fft(signal))
        rust_fft_result = rust_fft_result[::2] + 1j * rust_fft_result[1::2]
        total_time_rust_fft += time.time() - start_time

        # PyTorch FFT
        torch_signal = torch.from_numpy(signal)
        start_time = time.time()
        pytorch_fft_result = torch.fft.fft(torch_signal).numpy()
        total_time_pytorch_fft += time.time() - start_time

        # Compare FFT results
        fft_diff = np.abs(rust_fft_result - pytorch_fft_result)
        total_fft_diff += np.mean(fft_diff)

        # Prepare input for IFFT (use PyTorch FFT result)
        ifft_input = np.concatenate((pytorch_fft_result.real, pytorch_fft_result.imag))

        # Rust IFFT
        start_time = time.time()
        rust_ifft_result = np.array(rust_ifft(ifft_input))
        total_time_rust_ifft += time.time() - start_time

        # PyTorch IFFT
        torch_fft_result = torch.from_numpy(pytorch_fft_result)
        start_time = time.time()
        pytorch_ifft_result = torch.fft.ifft(torch_fft_result).numpy().real
        total_time_pytorch_ifft += time.time() - start_time

        # Compare IFFT results
        ifft_diff = np.abs(rust_ifft_result - pytorch_ifft_result)
        total_ifft_diff += np.mean(ifft_diff)
        
        # ----  round trip tests ----


        # Rust FFT -> IFFT
        start_time = time.time()
        rust_fft_result = np.array(rust_fft(signal))
        rust_fft_complex = rust_fft_result[::2] + 1j * rust_fft_result[1::2]
        rust_roundtrip = np.array(rust_ifft(rust_fft_result))
        total_time_rust += time.time() - start_time

        # PyTorch FFT -> IFFT
        torch_signal = torch.from_numpy(signal)
        start_time = time.time()
        pytorch_fft_result = torch.fft.fft(torch_signal)
        pytorch_roundtrip = torch.fft.ifft(pytorch_fft_result).real.numpy()
        total_time_pytorch += time.time() - start_time

        # PyTorch FFT -> Rust IFFT
        pytorch_fft_result_numpy = pytorch_fft_result.numpy()
        pytorch_fft_result_flat = np.concatenate((pytorch_fft_result_numpy.real, pytorch_fft_result_numpy.imag))
        rust_roundtrip_pytorch_fft = np.array(rust_ifft(pytorch_fft_result_flat))
        pytorch_fft_rust_ifft_diff = np.abs(signal - rust_roundtrip_pytorch_fft)
        avg_pytorch_fft_rust_ifft_diff = np.mean(pytorch_fft_rust_ifft_diff)


        # Rust FFT -> PyTorch IFFT
        rust_fft_complex_torch = torch.from_numpy(rust_fft_complex)
        pytorch_roundtrip_rust_fft = torch.fft.ifft(rust_fft_complex_torch).real.numpy()
        rust_fft_pytorch_ifft = np.abs(signal - pytorch_roundtrip_rust_fft)
        avg_rust_fft_pytorch_ifft = np.mean(rust_fft_pytorch_ifft)


        # Compare results to original signal
        rust_diff = np.abs(signal - rust_roundtrip)
        pytorch_diff = np.abs(signal - pytorch_roundtrip)
        rust_fft_pytorch_ifft_diff = np.abs(signal - pytorch_roundtrip_rust_fft)
        pytorch_fft_rust_ifft_diff = np.abs(signal - rust_roundtrip_pytorch_fft)
        total_rust_diff += np.mean(rust_diff)
        total_pytorch_diff += np.mean(pytorch_diff)
        total_rust_fft_pytorch_ifft_diff = np.mean(rust_fft_pytorch_ifft_diff)
        total_pytorch_fft_rust_ifft_diff = np.mean(pytorch_fft_rust_ifft_diff)


    # ----  FFT and IFFT tests ----

    avg_fft_diff = total_fft_diff / num_trials
    avg_ifft_diff = total_ifft_diff / num_trials
    avg_time_rust_fft = total_time_rust_fft / num_trials
    avg_time_rust_ifft = total_time_rust_ifft / num_trials
    avg_time_pytorch_fft = total_time_pytorch_fft / num_trials
    avg_time_pytorch_ifft = total_time_pytorch_ifft / num_trials

    print(f"Average FFT difference: {avg_fft_diff}")
    print(f"Average IFFT difference: {avg_ifft_diff}")
    print(f"Average time (Rust FFT): {avg_time_rust_fft:.6f} seconds")
    print(f"Average time (Rust IFFT): {avg_time_rust_ifft:.6f} seconds")
    print(f"Average time (PyTorch FFT): {avg_time_pytorch_fft:.6f} seconds")
    print(f"Average time (PyTorch IFFT): {avg_time_pytorch_ifft:.6f} seconds")

    # ----  round trip tests ----

    avg_rust_diff = total_rust_diff / num_trials
    avg_pytorch_diff = total_pytorch_diff / num_trials
    avg_time_rust = total_time_rust / num_trials
    avg_time_pytorch = total_time_pytorch / num_trials
    avg_total_rust_fft_pytorch_ifft_diff = total_rust_fft_pytorch_ifft_diff / num_trials
    avg_total_pytorch_fft_rust_ifft_diff = total_pytorch_fft_rust_ifft_diff / num_trials

    print(f"Average Rust roundtrip difference: {avg_rust_diff}")
    print(f"Average PyTorch roundtrip difference: {avg_pytorch_diff}")
    print(f"Average time (Rust FFT+IFFT): {avg_time_rust:.6f} seconds")
    print(f"Average time (PyTorch FFT+IFFT): {avg_time_pytorch:.6f} seconds")
    print(f"Average roundtrip difference (Rust FFT -> PyTorch IFFT): {avg_total_rust_fft_pytorch_ifft_diff}")
    print(f"Average roundtrip difference (PyTorch FFT -> Rust IFFT): {avg_total_pytorch_fft_rust_ifft_diff}")

if __name__ == "__main__":
    signal_sizes = [1024, 4096, 16384]
    for size in signal_sizes:
        print(f"\nTesting with signal size: {size}")
        compare_fft_ifft_roundtrip(size)