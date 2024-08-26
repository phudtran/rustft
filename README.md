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
Average FFT difference: 2.6466977383587384e-15
Average IFFT difference: 0.2910249298331762
Average time (Rust FFT): 0.000760 seconds
Average time (Rust IFFT): 0.000620 seconds
Average time (PyTorch FFT): 0.000008 seconds
Average time (PyTorch IFFT): 0.000008 seconds
Average Rust roundtrip difference: 1.0596021334115767e-16
Average PyTorch roundtrip difference: 1.0718986377618095e-16
Average time (Rust FFT+IFFT): 0.001387 seconds
Average time (PyTorch FFT+IFFT): 0.000016 seconds
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.2553705904666534e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.0029133628791726545

Testing with signal size: 4096
Average FFT difference: 6.29602883795488e-15
Average IFFT difference: 0.2916517995674905
Average time (Rust FFT): 0.003248 seconds
Average time (Rust IFFT): 0.002697 seconds
Average time (PyTorch FFT): 0.000020 seconds
Average time (PyTorch IFFT): 0.000027 seconds
Average Rust roundtrip difference: 1.1840510727583756e-16
Average PyTorch roundtrip difference: 1.1978880018143967e-16
Average time (Rust FFT+IFFT): 0.005954 seconds
Average time (PyTorch FFT+IFFT): 0.000049 seconds
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.7596558907792374e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.002969099309800841

Testing with signal size: 16384
Average FFT difference: 1.4512421283084802e-14
Average IFFT difference: 0.29138542453241434
Average time (Rust FFT): 0.013765 seconds
Average time (Rust IFFT): 0.011569 seconds
Average time (PyTorch FFT): 0.000088 seconds
Average time (PyTorch IFFT): 0.000132 seconds
Average Rust roundtrip difference: 1.2784740023938662e-16
Average PyTorch roundtrip difference: 1.317630262299945e-16
Average time (Rust FFT+IFFT): 0.025348 seconds
Average time (PyTorch FFT+IFFT): 0.000228 seconds
Average roundtrip difference (Rust FFT -> PyTorch IFFT): 1.3364739951662526e-18
Average roundtrip difference (PyTorch FFT -> Rust IFFT): 0.0029214745671869336
```
