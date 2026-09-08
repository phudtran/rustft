# Rustft

Short-time Fourier transform and its inverse, numerically matching `torch.stft`
and `torch.istft`. Handles non-COLA-compliant windows (they must be
[NOLA](https://gauss256.github.io/blog/cola.html)).

# Install

```bash
cargo add rustft
```

Optional features:

| feature | effect |
| --- | --- |
| `parallel` | Spread the per-frame FFTs and the transposes across cores with rayon. Engaged automatically only for inputs large enough to pay for it. |

# Example

```rust
use ndarray::Array2;
use rustft::{Stft, WindowFunction};

fn main() {
    // Plan once; forward and inverse both borrow, so one Stft serves every block.
    let stft = Stft::new(1024, 256, WindowFunction::Hann::<f64>, true);

    let input = Array2::from_shape_fn((2, 44100), |(_, i)| (i as f64 * 0.01).sin());
    let spectrogram = stft.forward(input.view()).unwrap();   // (2, 513, 173)
    let recovered = stft.inverse(spectrogram.view()).unwrap(); // (2, 44032)
}
```

`inverse` returns `hop_length * (frames - 1)` samples: like `torch.istft`, it
trims the centring pad off both ends, so a roundtrip is shorter than its input
whenever the signal length is not a multiple of the hop.

# Matching PyTorch

`forward` is `torch.stft(..., center=True, pad_mode="reflect", onesided=True,
return_complex=True)` and `inverse` is `torch.istft(..., center=True)`. Both
agree with PyTorch to within float64 rounding — worst observed difference `4e-13`
on a spectrogram with values of order `100`, i.e. about one part in `10^15`.

`conformance/` holds the harness that checks this, case by case, against a real
PyTorch install:

```bash
cargo build -p conformance --release
python3 conformance/conformance.py
```

One trap worth knowing about when comparing yourself: **`torch.hann_window`
returns float32 by default.** Applied to a float64 signal, that alone puts about
`1e-7` of error into PyTorch's own roundtrip, and it is easy to mistake for a bug
on the Rust side. Build the window with `dtype=torch.float64`.

That trap was most of the `~5e-7` disagreement this README used to report. The
old benchmark script built its window with `torch.hann_window(...)`, whose float32
values are only accurate to about `6e-8` — which is why the *PyTorch* roundtrip
in those numbers was `9.6e-9` rather than the `1e-16` a float64 pipeline gives.
It also linked against `rustft = "0.0.8"` from crates.io rather than the working
tree, so it was not measuring this code at all.

Underneath that, three genuine defects are fixed:

- **Odd `n_fft` produced a completely wrong inverse** (errors of `0.1` to `0.7`,
  not rounding noise). The Hermitian spectrum was rebuilt as though bin
  `n_fft / 2` were Nyquist: its imaginary part was discarded and it was never
  mirrored into the upper half. An odd-length transform has no Nyquist bin.
- **A signal shorter than `n_fft / 2` panicked** inside the reflect padding
  instead of returning an error. `torch.stft` rejects those inputs cleanly.
- **A window that fails NOLA produced silent infinities and NaNs** — overlap-add
  divides by the window envelope, and nothing checked it was non-zero.
  `torch.istft` raises here, and now so does `inverse`.

Two smaller ones: `update` left the old window in place when only `n_fft`
changed, so the next transform windowed with the wrong length; and a
`hop_length` of zero panicked with a divide-by-zero deep inside `forward`.

## API changes

- `update` now returns `Result<(), String>` and rejects a zero `n_fft` or
  `hop_length` rather than corrupting the transform.
- `Stft::try_new` is the fallible constructor; `Stft::new` still panics on
  invalid parameters.
- `n_fft()`, `hop_length()`, `window()` and `num_frames()` accessors were added.

# Benchmarks

## Against PyTorch

`benchmarks/` builds PyO3 bindings around this crate and times both libraries in
one process. See `benchmarks/README.md` for how to run it. PyTorch 2.8 on 12
threads, `rustft` with `--features parallel`, f64 throughout:

| shape | rustft STFT+ISTFT | PyTorch STFT+ISTFT | |
| --- | --- | --- | --- |
| 2 ch, 16384 samples, n_fft 1024, hop 512 | 0.35 ms | 0.87 ms | 2.5x |
| 2 ch, 261120 samples, n_fft 6144, hop 1024 | 8.89 ms | 27.79 ms | 3.1x |
| 2 ch, 65536 samples, n_fft 4096, hop 2048 | 1.45 ms | 2.68 ms | 1.9x |

Agreement over those same runs: the largest difference between the two STFTs is
`4.3e-13` on values of order `100`, and between the two ISTFTs `1.2e-15`. Both
libraries reconstruct the original signal to about `1e-15`.

## Against the previous version

`cargo bench -p rustft`, best of three runs each, same machine and session:

| benchmark | before | after | `--features parallel` |
| --- | --- | --- | --- |
| forward, 2 ch, 16384, n_fft 1024, hop 512 | 0.490 ms | 0.162 ms (3.0x) | 0.161 ms |
| forward, 2 ch, 261120, n_fft 6144, hop 1024 | 33.06 ms | 9.45 ms (3.5x) | 4.04 ms (8.2x) |
| forward, 2 ch, 65536, n_fft 4096, hop 2048 | 2.035 ms | 0.771 ms (2.6x) | 0.569 ms (3.6x) |
| inverse, 2 ch, 16384, n_fft 1024, hop 512 | 0.897 ms | 0.167 ms (5.4x) | 0.160 ms |
| inverse, 2 ch, 261120, n_fft 6144, hop 1024 | 48.73 ms | 8.22 ms (5.9x) | 4.43 ms (11.0x) |
| inverse, 2 ch, 65536, n_fft 4096, hop 2048 | 3.820 ms | 0.807 ms (4.7x) | 0.823 ms (4.6x) |

Where it came from:

- The release profile was set to `opt-level = "z"`, optimising a numerics
  library for binary size. That alone accounted for roughly half the time.
- A real-input FFT (`realfft`) replaces the full complex transform, halving both
  the arithmetic and the memory traffic.
- Scratch buffers are allocated once per call instead of per frame, and
  `process_with_scratch` replaces `process`, which allocates internally on every
  call.
- Frames are transformed into a frame-major staging buffer and transposed in one
  blocked pass, rather than scattered into the frequency-major output one bin at
  a time.
- The inverse computes its window envelope once per call rather than once per
  channel, folds the `1 / n_fft` normalisation into the window, and writes
  straight into the trimmed output instead of building a padded signal and
  shifting it down afterwards.

`--features parallel` roughly doubles that again on the largest shape. It is
deliberately inert below a size threshold — on the 16384-sample case it makes no
difference at all, because rayon's scheduling would cost more than the FFTs it
spread out.
