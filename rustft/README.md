# rustft

Short-time Fourier transform and its inverse in Rust, numerically matching
`torch.stft` / `torch.istft`.

See the [repository README](https://github.com/phudtran/rustft) for benchmarks,
the conformance harness, and the full description of what "matching PyTorch"
means here.

```rust
use ndarray::Array2;
use rustft::{Stft, WindowFunction};

let stft = Stft::new(1024, 256, WindowFunction::Hann::<f64>, true);
let signal = Array2::from_shape_fn((2, 44100), |(_, i)| (i as f64 * 0.01).sin());

let spectrogram = stft.forward(signal.view()).unwrap();
let recovered = stft.inverse(spectrogram.view()).unwrap();
```
