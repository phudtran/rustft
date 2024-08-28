# Pyo3 bindings

# WIP

Currently broken until
https://github.com/PyO3/rust-numpy/pull/439 is merged

# Performance

WIP. No optimizations, super slow.

# How to run test

```
source ~/.venv/bin/activate
pip install numpy torch
cargo build --release
maturin develop
python3 test_stft.py

```

# Benchmarks

```
Testing with: 2 channels, signal length 16384, n_fft 1024, hop_length 512
Average Rust roundtrip error: 1.2937405892934302e-16
Average PyTorch roundtrip error: 9.561864467600023e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.771066294604266e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.5421215870497426e-08

Average run times:
Rust STFT + ISTFT: 0.058335 seconds
PyTorch STFT + ISTFT: 0.005722 seconds
Rust STFT: 0.029399 seconds
PyTorch ISTFT: 0.000716 seconds
PyTorch STFT: 0.000272 seconds
Rust ISTFT: 0.026649 seconds

Testing with: 4 channels, signal length 32768, n_fft 2048, hop_length 1024
Average Rust roundtrip error: 1.2302921848087881e-16
Average PyTorch roundtrip error: 9.679753242194043e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7571041522967382e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.5074709989037257e-08

Average run times:
Rust STFT + ISTFT: 0.224398 seconds
PyTorch STFT + ISTFT: 0.021696 seconds
Rust STFT: 0.117436 seconds
PyTorch ISTFT: 0.002956 seconds
PyTorch STFT: 0.001173 seconds
Rust ISTFT: 0.106868 seconds

Testing with: 8 channels, signal length 65536, n_fft 4096, hop_length 2048
Average Rust roundtrip error: 1.379474655196007e-16
Average PyTorch roundtrip error: 9.650499519246785e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7393561622267553e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.5055924811920244e-08

Average run times:
Rust STFT + ISTFT: 0.986159 seconds
PyTorch STFT + ISTFT: 0.062225 seconds
Rust STFT: 0.510961 seconds
PyTorch ISTFT: 0.009001 seconds
PyTorch STFT: 0.003260 seconds
Rust ISTFT: 0.475791 seconds
```
