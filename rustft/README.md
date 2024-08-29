# WIP STFT/ISTFT Library

Goal to get outputs in line with PyTorch.
The error is still too high.

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
Average STFT difference (Rust vs PyTorch): 4.7681456430939274e-07
Average Rust roundtrip error: 0.3231553895378749
Average PyTorch roundtrip error: 9.598797280322882e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7784312354085827e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.32315538956071693

Average run times:

Rust STFT + ISTFT: 0.051144 seconds
PyTorch STFT + ISTFT: 0.021410 seconds
Rust STFT: 0.025497 seconds
PyTorch ISTFT: 0.001597 seconds
PyTorch STFT: 0.000933 seconds
Rust ISTFT: 0.025861 seconds

Testing with: 4 channels, signal length 32768, n_fft 2048, hop_length 1024
Average STFT difference (Rust vs PyTorch): 5.97084863815579e-07
Average Rust roundtrip error: 0.32008460333928596
Average PyTorch roundtrip error: 9.699180397974077e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7605382232922285e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.32008460335027145

Average run times:

Rust STFT + ISTFT: 0.225601 seconds
PyTorch STFT + ISTFT: 0.023577 seconds
Rust STFT: 0.108802 seconds
PyTorch ISTFT: 0.003290 seconds
PyTorch STFT: 0.001194 seconds
Rust ISTFT: 0.113689 seconds

Testing with: 8 channels, signal length 65536, n_fft 4096, hop_length 2048
Average STFT difference (Rust vs PyTorch): 7.438268093038237e-07
Average Rust roundtrip error: 0.31818092221733296
Average PyTorch roundtrip error: 9.642855556690204e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7379899977190086e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.318180922222956

Average run times:

Rust STFT + ISTFT: 0.904157 seconds
PyTorch STFT + ISTFT: 0.063348 seconds
Rust STFT: 0.434657 seconds
PyTorch ISTFT: 0.008514 seconds
PyTorch STFT: 0.003525 seconds
Rust ISTFT: 0.457826 seconds
```
