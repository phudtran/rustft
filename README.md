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
Average FFT difference: 2.6461929295764306e-15
Average IFFT difference: 0.29258012610368017
Average time (Rust FFT): 0.000859 seconds
Average time (Rust IFFT): 0.000703 seconds
Average time (PyTorch FFT): 0.000064 seconds
Average time (PyTorch IFFT): 0.000016 seconds

Testing with signal size: 4096
Average FFT difference: 6.291434261458335e-15
Average IFFT difference: 0.2916950593071173
Average time (Rust FFT): 0.003422 seconds
Average time (Rust IFFT): 0.002953 seconds
Average time (PyTorch FFT): 0.000023 seconds
Average time (PyTorch IFFT): 0.000030 seconds

Testing with signal size: 16384
Average FFT difference: 1.451280174643235e-14
Average IFFT difference: 0.29141259727761487
Average time (Rust FFT): 0.014347 seconds
Average time (Rust IFFT): 0.012194 seconds
Average time (PyTorch FFT): 0.000089 seconds
Average time (PyTorch IFFT): 0.000131 seconds
```
