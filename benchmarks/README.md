# PyO3 bindings for the PyTorch benchmarks

These bindings exist so `test_stft.py` can compare this repository's `rustft`
against PyTorch in one process. They build the crate from `../rustft` by path —
depending on the published crate here would benchmark whatever was last
released rather than the working tree.

## Running

```bash
python3 -m venv ../.venv
../.venv/bin/pip install numpy torch maturin
../.venv/bin/maturin develop --release
../.venv/bin/python test_stft.py
```

`maturin develop --release` matters: a debug build is roughly an order of
magnitude slower and the timings are meaningless.

## Two things the harness is careful about

**Use a float64 window.** `torch.hann_window` returns float32 by default. Paired
with a float64 signal it puts about `1e-7` of error into PyTorch's *own*
roundtrip, which is easy to misread as a rustft bug. Pass
`dtype=torch.float64`.

**Build the `Stft` once, outside the timing loop.** It plans its FFTs in the
constructor, and `torch.stft` likewise reuses a cached plan. Constructing one
per call times the planner rather than the transform.

## API

```python
from rustft import Stft

stft = Stft(n_fft=1024, hop_length=256, window="hann", periodic=True)
spectrogram = stft.forward(signal)   # (channels, n_fft // 2 + 1, frames) complex128
recovered = stft.inverse(spectrogram) # (channels, samples) float64
recovered = stft.roundtrip(signal)    # forward then inverse, without leaving Rust
```

`test_fft.py` additionally exercises the one-shot `rust_fft` / `rust_ifft` /
`rust_fft_roundtrip` helpers.
