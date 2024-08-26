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
Average Rust roundtrip difference: 1.0429630181957132e-16
Average PyTorch roundtrip difference: 1.0643646608608087e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.056513681879298e-16
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 1.0587661965959314e-16

Testing with signal size: 4096
Average Rust roundtrip difference: 1.1817968110362884e-16
Average PyTorch roundtrip difference: 1.2021159614766607e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.1923411112680955e-16
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 1.2145750432132071e-16

Testing with signal size: 16384
Average Rust roundtrip difference: 1.2716147295870007e-16
Average PyTorch roundtrip difference: 1.312618791867515e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.292748748883689e-16
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 1.3139329370154005e-16
```
