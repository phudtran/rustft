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
Average Rust roundtrip difference: 1.0525155169616684e-16
Average PyTorch roundtrip difference: 1.160989823646898e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.4525487987356361e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.0029092289747573623

Testing with signal size: 4096
Average Rust roundtrip difference: 1.1610568440398084e-16
Average PyTorch roundtrip difference: 1.28897091924686e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.4234869038026785e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.0028725365060352404

Testing with signal size: 16384
Average Rust roundtrip difference: 1.2986990997516133e-16
Average PyTorch roundtrip difference: 1.459866765596829e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.7037095303724374e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.002953246464695906
```
