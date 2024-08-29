# WIP STFT/ISTFT Library

Goal to get outputs in line with PyTorch.
The error is high when n_fft != 2\* hop_length

# Install

```
cargo add rustft
```

# Performance

WIP. No optimizations, super slow.

# How to run test

```
cd pyo3
source ~/.venv/bin/activate
pip install numpy torch
cargo build --release
maturin develop
python3 test_stft.py

```

# Benchmarks

```
Testing with: 2 channels, signal length 16384, n_fft 1024, hop_length 512
Average STFT difference (Rust vs PyTorch): 4.771385448526834e-07
Average Rust roundtrip error: 1.3051805968152608e-16
Average PyTorch roundtrip error: 9.61298997794433e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.780793284997755e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.5507429217807917e-08

Average run times:

Rust STFT + ISTFT: 0.051980 seconds
PyTorch STFT + ISTFT: 0.006521 seconds
Rust STFT: 0.024872 seconds
PyTorch ISTFT: 0.000851 seconds
PyTorch STFT: 0.000356 seconds
Rust ISTFT: 0.025898 seconds

Testing with: 4 channels, signal length 32768, n_fft 6144, hop_length 1024
Average STFT difference (Rust vs PyTorch): 9.272865197326369e-07
Average Rust roundtrip error: 0.6275651337767195
Average PyTorch roundtrip error: 1.3115320925365482e-08
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.4475060099035079e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.6275651084766845

Average run times:

Rust STFT + ISTFT: 0.823908 seconds
PyTorch STFT + ISTFT: 0.069491 seconds
Rust STFT: 0.409617 seconds
PyTorch ISTFT: 0.009064 seconds
PyTorch STFT: 0.004068 seconds
Rust ISTFT: 0.405551 seconds

Testing with: 8 channels, signal length 65536, n_fft 4096, hop_length 2048
Average STFT difference (Rust vs PyTorch): 7.44958479031276e-07
Average Rust roundtrip error: 1.2947973644518254e-16
Average PyTorch roundtrip error: 9.658558043450567e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7407385948571054e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.506891813552541e-08

Average run times:

Rust STFT + ISTFT: 0.911379 seconds
PyTorch STFT + ISTFT: 0.059759 seconds
Rust STFT: 0.436219 seconds
PyTorch ISTFT: 0.008553 seconds
PyTorch STFT: 0.003107 seconds
Rust ISTFT: 0.462101 seconds
```
