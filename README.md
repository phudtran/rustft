# Rustft

Short-time Fourier transform and its inverse in Rust, matching `torch.stft` and
`torch.istft` bit for bit — same options, same defaults, same output.

```bash
cargo add rustft
```

| feature | default | effect |
| --- | --- | --- |
| `pocketfft` | off | Use a vendored pocketfft — the FFT PyTorch itself uses on CPU — instead of rustfft. **Required for bit-identical output.** Needs a C++ toolchain. |
| `parallel` | off | Spread the per-frame FFTs and the transposes across cores with rayon, for inputs large enough to pay for it. Does not affect the results. |

## Matching PyTorch bit for bit

```bash
cargo add rustft --features pocketfft
```

`pocketfft` is the only flag that matters for exactness, and it is the whole
requirement — windows are generated natively, so nothing has to be handed in
from Python, and `parallel` can be on or off without changing a single bit.
Without `pocketfft` the output is still correct, just not identical: about one
ulp, `1.3e-15` in `f64` and `2.1e-07` in `f32`.

Two things it cannot promise on their own, both worth checking on your target
with `conformance/conformance.py`:

* PyTorch must be using pocketfft itself. It does when built without MKL, which
  is the case on macOS/aarch64; x86 wheels are often built with MKL, which
  dispatches FFTs elsewhere. Check `torch.__config__.show()` for `USE_MKL`.
* `f64` windows rely on both sides resolving `cos` to the same libm. That holds
  where PyTorch has no vectorised `f64` cosine. The `f32` path does not depend
  on this.

# Example

```rust
use ndarray::Array2;
use rustft::{Stft, WindowFunction};

fn main() {
    // Plan once; forward and inverse both borrow, so one Stft serves every block.
    let stft = Stft::new(1024, 256, WindowFunction::Hann::<f64>, true);

    let input = Array2::from_shape_fn((2, 44100), |(_, i)| (i as f64 * 0.01).sin());
    let spectrogram = stft.forward(input.view()).unwrap();    // (2, 513, 173)
    let recovered = stft.inverse(spectrogram.view()).unwrap(); // (2, 44032)
}
```

Samples run along the last axis; every axis before it is carried through, so a
mono `(samples,)` signal, a `(channels, samples)` one and a batched
`(batch, channels, samples)` one all work with the same call:

```rust
# use ndarray::{Array1, Array3};
# use rustft::{Stft, WindowFunction};
# let stft = Stft::new(256, 64, WindowFunction::Hann::<f64>, true);
let mono = Array1::from_shape_fn(4096, |i| (i as f64 * 0.01).sin());
assert_eq!(stft.forward(mono.view()).unwrap().dim(), (129, 65));

let batched = Array3::from_shape_fn((8, 2, 4096), |(_, _, i)| (i as f64 * 0.01).sin());
assert_eq!(stft.forward(batched.view()).unwrap().dim(), (8, 2, 129, 65));
```

`inverse` trims the centring pad off both ends, as `torch.istft` does, so a
roundtrip is shorter than its input unless the signal length is a multiple of
the hop.

Every `torch.stft` option is available through the builder, with the same
defaults (`hop_length = n_fft / 4`, an all-ones window):

```rust
use rustft::{PadMode, Stft, WindowFunction};

let stft = Stft::builder(1024)
    .hop_length(256)
    .win_length(400)                              // zero-padded, centred in n_fft
    .window(WindowFunction::Hann::<f32>, true)
    .center(false)
    .pad_mode(PadMode::Constant)                  // reflect, constant, replicate, circular
    .normalized(true)
    .onesided(false)
    .build()
    .unwrap();
```

`inverse_with_length(spectrogram, Some(n))` is `torch.istft`'s `length`.
`Stft::with_window(hop, samples)` takes a window verbatim, for shapes this crate
does not provide. `Stft<T>` is generic over `f32` and `f64`.

# Accuracy

Against PyTorch 2.8, both libraries running end to end in the same precision,
with rustft generating its own windows:

| | `--features pocketfft` | default (rustfft) |
| --- | --- | --- |
| `f64` | **0 — bit-identical** | 1.3e-15 |
| `f32` | **0 — bit-identical** | 2.1e-07 |

Worst relative difference over the conformance suite. Verified across:

| sweep | checks |
| --- | --- |
| Option space: `win_length`, `center`, four pad modes, `normalized`, `onesided`, `length` | 4944/4944 per precision |
| Shapes and signals: prime `n_fft`, `hop = 1`, `hop = n_fft - 1`, sizes down to 7, amplitudes 1e6 and 1e-8, impulses, zeros | 1240/1240 per precision |
| Windows: Hann, Hamming, Blackman, Bartlett, every length 2–500 plus larger, periodic and symmetric | 4056/4056 per precision |

Each check covers the forward transform, the inverse, and the mixed pipelines
(rustft's STFT into PyTorch's ISTFT and the reverse). All results hold with and
without `--features parallel`.

```bash
cargo build -p conformance --release --features rustft/pocketfft
python3 conformance/conformance.py
```

Measured against PyTorch 2.8 on macOS/aarch64.

# Performance

Against PyTorch on 12 threads, `--features pocketfft,parallel` — the
bit-identical configuration. Best of 30 runs on both sides:

| shape | | `f64` rustft | PyTorch | | `f32` rustft | PyTorch | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 ch, 16384, n_fft 1024 | forward | 0.193 ms | 0.266 ms | 1.4x | 0.159 ms | 0.166 ms | 1.0x |
| | inverse | 0.206 ms | 0.446 ms | 2.2x | 0.157 ms | 0.316 ms | 2.0x |
| | **both** | **0.399 ms** | **0.712 ms** | **1.8x** | **0.316 ms** | **0.482 ms** | **1.5x** |
| 2 ch, 261120, n_fft 6144 | forward | 5.465 ms | 11.241 ms | 2.1x | 3.539 ms | 5.515 ms | 1.6x |
| | inverse | 5.729 ms | 14.362 ms | 2.5x | 3.282 ms | 7.796 ms | 2.4x |
| | **both** | **11.194 ms** | **25.603 ms** | **2.3x** | **6.821 ms** | **13.311 ms** | **2.0x** |
| 2 ch, 65536, n_fft 4096 | forward | 0.701 ms | 0.863 ms | 1.2x | 0.555 ms | 0.424 ms | 0.8x |
| | inverse | 0.959 ms | 1.516 ms | 1.6x | 0.814 ms | 0.891 ms | 1.1x |
| | **both** | **1.659 ms** | **2.379 ms** | **1.4x** | **1.369 ms** | **1.315 ms** | **1.0x** |

The inverse is the consistent win; the forward is closer, and PyTorch takes one
`f32` shape. None of this is Python overhead — a PyTorch call dispatches in about
a microsecond against transforms that take hundreds, so it is compute against
compute, and both are calling the same pocketfft.

The default rustfft backend is faster still, at one ulp instead of zero.
`cargo bench -p rustft`, best of several runs, `f64`:

| benchmark | before | default | `parallel` |
| --- | --- | --- | --- |
| forward, 2 ch, 16384, n_fft 1024 | 0.490 ms | 0.167 ms | 0.167 ms |
| forward, 2 ch, 261120, n_fft 6144 | 33.06 ms | 9.53 ms | 3.93 ms |
| forward, 2 ch, 65536, n_fft 4096 | 2.035 ms | 0.772 ms | 0.577 ms |
| inverse, 2 ch, 16384, n_fft 1024 | 0.897 ms | 0.168 ms | 0.168 ms |
| inverse, 2 ch, 261120, n_fft 6144 | 48.73 ms | 8.36 ms | 4.40 ms |
| inverse, 2 ch, 65536, n_fft 4096 | 3.820 ms | 0.843 ms | 0.832 ms |

Where that came from: the release profile was `opt-level = "z"`, optimising a
numerics library for size; a real-input FFT replaces the full complex transform;
scratch buffers are allocated once per call rather than per frame; frames are
transformed into a frame-major staging buffer and transposed in one blocked pass
instead of scattered into the output a bin at a time; and the inverse computes
its window envelope once per call rather than once per channel.

`--features pocketfft` costs roughly 1.3-2x the default backend, and `parallel`
brings most of that back. Figures were taken on a machine with background load;
the single-threaded ones repeat consistently, the `parallel` ones vary by 10-30%
between runs and would sit at the better end on a quiet machine.
