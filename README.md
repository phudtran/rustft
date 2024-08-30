# WIP STFT/ISTFT Library

Handles windows Hann Windows that are non-COLA compliant but still NOLA compliant.
For example n_fft == 6144 and hop_length == 1024

# Install

```
cargo add rustft
```

# Example

See `pyo3/lib.rs`

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
Average STFT difference (Rust vs PyTorch): 4.746677862528417e-07
Average Rust roundtrip error: 9.911936998122515e-17
Average PyTorch roundtrip error: 9.561478833379476e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7710999088383453e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.470317770150175e-08

Average run times:

Rust STFT + ISTFT: 0.125835 seconds
PyTorch STFT + ISTFT: 0.010600 seconds
Rust STFT: 0.050728 seconds
PyTorch ISTFT: 0.001436 seconds
PyTorch STFT: 0.000531 seconds
Rust ISTFT: 0.074708 seconds

Testing with: 2 channels, signal length 261120, n_fft 6144, hop_length 1024
Average STFT difference (Rust vs PyTorch): 9.059638180348471e-07
Average Rust roundtrip error: 1.451597103641284e-16
Average PyTorch roundtrip error: 1.307342735780541e-08
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.4292311512782457e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 6.774415643315167e-09

Average run times:

Rust STFT + ISTFT: 3.608590 seconds
PyTorch STFT + ISTFT: 0.215071 seconds
Rust STFT: 1.536361 seconds
PyTorch ISTFT: 0.029943 seconds
PyTorch STFT: 0.011420 seconds
Rust ISTFT: 2.046302 seconds

Testing with: 2 channels, signal length 65536, n_fft 4096, hop_length 2048
Average STFT difference (Rust vs PyTorch): 7.438915845846038e-07
Average Rust roundtrip error: 1.1081883623829144e-16
Average PyTorch roundtrip error: 9.644176858092867e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7381988721350482e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.4266159964957597e-08

Average run times:

Rust STFT + ISTFT: 1.069613 seconds
PyTorch STFT + ISTFT: 0.072821 seconds
Rust STFT: 0.426307 seconds
PyTorch ISTFT: 0.009796 seconds
PyTorch STFT: 0.004240 seconds
Rust ISTFT: 0.631240 seconds
```
