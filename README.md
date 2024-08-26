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
Average FFT difference: 2.6430453561220717e-15
Average IFFT difference: 0.29103046005426647
Average time (Rust FFT): 0.000783 seconds
Average time (Rust IFFT): 0.000637 seconds
Average time (PyTorch FFT): 0.000009 seconds
Average time (PyTorch IFFT): 0.000009 seconds
Average Rust roundtrip difference: 1.0647747148045305e-16
Average PyTorch roundtrip difference: 1.070500758113132e-16
Average time (Rust FFT+IFFT): 0.001421 seconds
Average time (PyTorch FFT+IFFT): 0.000017 seconds
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 9.318209452744557e-19
Average roundtrip difference (PyTorch FFT -> PyTorch IFFT): 0.002976334268420765

Testing with signal size: 4096
Average FFT difference: 6.302036440331763e-15
Average IFFT difference: 0.29183621840553353
Average time (Rust FFT): 0.003285 seconds
Average time (Rust IFFT): 0.002718 seconds
Average time (PyTorch FFT): 0.000020 seconds
Average time (PyTorch IFFT): 0.000028 seconds
Average Rust roundtrip difference: 1.1740511923655043e-16
Average PyTorch roundtrip difference: 1.1901525475389798e-16
Average time (Rust FFT+IFFT): 0.005998 seconds
Average time (PyTorch FFT+IFFT): 0.000048 seconds
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.0661857816974055e-18
Average roundtrip difference (PyTorch FFT -> PyTorch IFFT): 0.00290670821869978

Testing with signal size: 16384
Average FFT difference: 1.4520727685636978e-14
Average IFFT difference: 0.29179986992099627
Average time (Rust FFT): 0.013862 seconds
Average time (Rust IFFT): 0.011656 seconds
Average time (PyTorch FFT): 0.000089 seconds
Average time (PyTorch IFFT): 0.000134 seconds
Average Rust roundtrip difference: 1.2832457777223466e-16
Average PyTorch roundtrip difference: 1.3181994141774518e-16
Average time (Rust FFT+IFFT): 0.025505 seconds
Average time (PyTorch FFT+IFFT): 0.000226 seconds
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.3370939174045245e-18
Average roundtrip difference (PyTorch FFT -> PyTorch IFFT): 0.002910664033079555
```
