# Pyo3 Binding for Benchmarks

# WIP

Currently broken until
https://github.com/PyO3/rust-numpy/pull/439 is merged

# Performance

WIP. No optimizations, super slow.

# How to run test

```
source ~/.venv/bin/activate
pip install numpy torch
cargo build --release
maturin develop
python3 test_stft.py

```
