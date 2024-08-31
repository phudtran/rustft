# Pyo3 Binding for Benchmarks

# WIP

https://github.com/PyO3/rust-numpy/pull/439 needs to be merged to update to ndarray 0.16

# Performance

WIP. No optimizations, super slow.

# How to run test

```bash
source ~/.venv/bin/activate
pip install numpy torch
cargo build --release
maturin develop
python3 test_stft.py
```
