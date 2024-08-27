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

### STFT/ISTFT

```
Testing with: 2 channels, signal length 16384, n_fft 1024, hop_length 512
Average Rust roundtrip difference: 0.25024201755773356
Average PyTorch roundtrip difference: 1.4834728567372855e-08
Average roundtrip difference (Rust STFT -> PyTorch ISTFT): 2.7409904686815138e-08
Average roundtrip difference (PyTorch STFT -> Rust ISTFT): 0.25024202170859045

Testing with: 4 channels, signal length 32768, n_fft 2048, hop_length 1024
Average Rust roundtrip difference: 0.3747784041116478
Average PyTorch roundtrip difference: 1.5103139005472542e-08
Average roundtrip difference (Rust STFT -> PyTorch ISTFT): 2.745574728982313e-08
Average roundtrip difference (PyTorch STFT -> Rust ISTFT): 0.3747784061264924

Testing with: 8 channels, signal length 65536, n_fft 4096, hop_length 2048
Average Rust roundtrip difference: 0.4375103958237623
Average PyTorch roundtrip difference: 1.5139098415594048e-08
Average roundtrip difference (Rust STFT -> PyTorch ISTFT): 2.7265447528499924e-08
Average roundtrip difference (PyTorch STFT -> Rust ISTFT): 0.437510396841536
```
