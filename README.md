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
Average Rust roundtrip difference: 1.0572329399064591e-16
Average PyTorch roundtrip difference: 1.161424297029473e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.004979554027795324
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.15934572888945037

Testing with signal size: 4096
Average Rust roundtrip difference: 1.161745291243513e-16
Average PyTorch roundtrip difference: 1.2870693382833326e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.004929797466797684
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.3155070378750518

Testing with signal size: 16384
Average Rust roundtrip difference: 1.3016321303333657e-16
Average PyTorch roundtrip difference: 1.460541064760903e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.0049478770226842024
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.6333282589035779
```
