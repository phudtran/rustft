# How to run

```
source ~/.venv/bin/activate
pip install numpy torch
cargo build --release
maturin develop
python3 test_fft.py

```

# Results

```
Testing with signal size: 1024
Average Rust roundtrip difference: 1.0282363339681605e-16
Average PyTorch roundtrip difference: 1.0393012100614381e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 8.510987054011209e-19
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.0028309373507784184

Testing with signal size: 4096
Average Rust roundtrip difference: 1.175554262918319e-16
Average PyTorch roundtrip difference: 1.1838921164379114e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.1453621864420011e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.002843724461265158

Testing with signal size: 16384
Average Rust roundtrip difference: 1.2930157217572626e-16
Average PyTorch roundtrip difference: 1.3205981075901912e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.4472555814530704e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.002912105748527018
```
