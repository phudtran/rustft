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
Testing with: 4 channels, signal length 16384, n_fft 1024, hop_length 512
Average STFT difference (Rust vs PyTorch): 4.783036912791149e-07
Average Rust roundtrip error: 0.08131700542121276
Average PyTorch roundtrip error: 9.63681898072814e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7856050059654996e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.08131700853854766

Average run times:

Rust STFT + ISTFT: 0.106578 seconds
PyTorch STFT + ISTFT: 0.018235 seconds
Rust STFT: 0.059887 seconds
PyTorch ISTFT: 0.010675 seconds
PyTorch STFT: 0.008297 seconds
Rust ISTFT: 0.056084 seconds

Testing with: 2 channels, signal length 261120, n_fft 6144, hop_length 1024
Average STFT difference (Rust vs PyTorch): 9.116106659546072e-07
Average Rust roundtrip error: 0.0793338731554039
Average PyTorch roundtrip error: 1.3156109949201987e-08
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.4382259670138511e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.07933387623695336

Average run times:

Rust STFT + ISTFT: 3.219443 seconds
PyTorch STFT + ISTFT: 0.030186 seconds
Rust STFT: 1.539922 seconds
PyTorch ISTFT: 0.031425 seconds
PyTorch STFT: 0.014956 seconds
Rust ISTFT: 1.632065 seconds

Testing with: 8 channels, signal length 65536, n_fft 4096, hop_length 2048
Average STFT difference (Rust vs PyTorch): 7.420832296988986e-07
Average Rust roundtrip error: 0.07946276505915406
Average PyTorch roundtrip error: 9.62317103791877e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.734334297600013e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.07946276805843747

Average run times:

Rust STFT + ISTFT: 0.923372 seconds
PyTorch STFT + ISTFT: 0.010154 seconds
Rust STFT: 0.437315 seconds
PyTorch ISTFT: 0.009968 seconds
PyTorch STFT: 0.005478 seconds
Rust ISTFT: 0.486644 seconds
```
