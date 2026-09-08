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

**Match the window dtype to the signal.** `torch.hann_window` returns float32 by
default. Paired with a float64 signal it puts about `1e-7` of error into
PyTorch's *own* roundtrip, which is easy to misread as a rustft bug.

**Build the `Stft` once, outside the timing loop.** It plans its FFTs in the
constructor, and `torch.stft` likewise reuses a cached plan. Constructing one
per call times the planner rather than the transform.

## API

```python
from rustft import Stft

stft = Stft(n_fft=1024, hop_length=256, window="hann", periodic=True,
            dtype="float64")          # or "float32"
spectrogram = stft.forward(signal)    # (channels, n_fft // 2 + 1, frames) complex128
recovered = stft.inverse(spectrogram) # (channels, samples) float64
recovered = stft.roundtrip(signal)    # forward then inverse, without leaving Rust
```

`dtype` fixes the precision the transform runs in and the array dtype the
methods accept; passing an array of the other dtype is an error rather than a
silent conversion, since the two differ by ~2e-7 relative and a silent cast
would look like a bug. `test_stft.py` reports both precisions.

`Stft.with_window(hop_length, window)` takes a window array verbatim, for window
shapes this crate does not provide. It is no longer needed to match PyTorch:
rustft generates bit-identical windows itself in both precisions. See the
top-level README.

`test_fft.py` additionally exercises the one-shot `rust_fft` / `rust_ifft` /
`rust_fft_roundtrip` helpers.
