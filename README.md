# How to run

```
source ~/.venv/bin/activate
pip install numpy torch
cargo build --release
maturin develop
python3 test_fft.py

```

# Results

### FFT/IFFT

```
Testing with signal size: 1024
Average Rust roundtrip error: 1.0863422224025661e-16
Average PyTorch roundtrip error: 1.0955702016706044e-16
Average roundtrip error (Rust FFT -> PyTorch IFFT): 1.0855936782112538e-16
Average roundtrip error (PyTorch FFT -> Rust IFFT): 1.0996885817392732e-16

Testing with signal size: 4096
Average Rust roundtrip error: 1.1710504299190903e-16
Average PyTorch roundtrip error: 1.1739377243612858e-16
Average roundtrip error (Rust FFT -> PyTorch IFFT): 1.1804875548566953e-16
Average roundtrip error (PyTorch FFT -> Rust IFFT): 1.1887165249378961e-16

Testing with signal size: 16384
Average Rust roundtrip error: 1.267817319418804e-16
Average PyTorch roundtrip error: 1.3074565250859276e-16
Average roundtrip error (Rust FFT -> PyTorch IFFT): 1.2911309113222218e-16
Average roundtrip error (PyTorch FFT -> Rust IFFT): 1.3041009577432653e-16
```

### STFT/ISTFT

```
Testing with: 2 channels, signal length 16384, n_fft 1024, hop_length 512
Average Rust roundtrip error: 1.0804457850576575e-16
Average PyTorch roundtrip error: 1.4826852848950367e-08
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 2.7412625609263545e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 2.273012613578353e-08

Testing with: 4 channels, signal length 32768, n_fft 2048, hop_length 1024
Average Rust roundtrip error: 0.24998884648421238
Average PyTorch roundtrip error: 1.511156196794707e-08
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 2.7479478628299385e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.24998885050404357

Testing with: 8 channels, signal length 65536, n_fft 4096, hop_length 2048
Average Rust roundtrip error: 0.37505129583172775
Average PyTorch roundtrip error: 1.514057545556388e-08
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 2.727195581803125e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 0.37505129786651914
```
