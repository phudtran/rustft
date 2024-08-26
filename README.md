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
Average Rust roundtrip difference: 0.1660183779029597
Average PyTorch roundtrip difference: 1.1627914739697414e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.003236004506459895
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.0032278275267588826

Testing with signal size: 4096
Average Rust roundtrip difference: 0.16645052851788375
Average PyTorch roundtrip difference: 1.2828566100614384e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.003158114948556162
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.003098265066608156

Testing with signal size: 16384
Average Rust roundtrip difference: 0.16659173069958502
Average PyTorch roundtrip difference: 1.4374930141859136e-16
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 0.0031669848694140974
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.0031452365763146774
```
