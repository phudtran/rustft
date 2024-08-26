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
Average Rust roundtrip difference: 1.0789151622353326e-16
Average PyTorch roundtrip difference: 1.1709378294661755e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.004872985718332864
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 4.989937375572853

Testing with signal size: 4096
Average Rust roundtrip difference: 1.194176726956002e-16
Average PyTorch roundtrip difference: 1.3139718159156325e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.0050154637837731
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 20.543339658334617

Testing with signal size: 16384
Average Rust roundtrip difference: 1.2692186583368034e-16
Average PyTorch roundtrip difference: 1.4479730753893462e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.0050241207559530736
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 82.31519446553516
```
