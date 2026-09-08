# Conformance harness

Diffs `rustft` against PyTorch case by case, in float64, without going through
the PyO3 bindings: `conformance.py` generates a signal, runs `torch.stft` /
`torch.istft` on it, shells out to the `conformance` binary to run the same
transforms through `rustft`, and reports the largest absolute difference.

```bash
cargo build -p conformance --release
../.venv/bin/python conformance.py
```

Add `--features rustft/parallel` to the build to check the parallel paths, which
have their own blocking and overlap-add and so are worth testing separately.

Two more switches worth knowing:

* `--features rustft/pocketfft` builds against the FFT PyTorch itself uses. The
  worst difference drops from ~1e-15 to exactly zero.
* `--share-window` passes PyTorch's own window samples to `Stft::with_window`
  instead of letting rustft build one. That matters only in f32, where PyTorch's
  vectorised cosine cannot be reproduced; with both switches on, every case is
  bit-identical in both precisions.

The cases deliberately cover the awkward shapes: hop lengths that do not divide
`n_fft`, odd `n_fft` (no Nyquist bin), signal lengths that are not a multiple of
the hop, and non-power-of-two transform sizes.

Every case runs twice, once in `f64` and once in `f32`, with both libraries in
the same precision end to end -- same signal bits, same window dtype. Mixing
them is the classic trap: the float32 default of `torch.hann_window` injects
~1e-7 of error on its own, which swamps everything this harness measures.

Tolerances are relative to the transform's peak, because the two precisions land
four decades apart: `f64` agrees to ~1e-15, `f32` to ~2e-7.
