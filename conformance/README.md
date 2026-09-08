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

The cases deliberately cover the awkward shapes: hop lengths that do not divide
`n_fft`, odd `n_fft` (no Nyquist bin), signal lengths that are not a multiple of
the hop, and non-power-of-two transform sizes.

Every window here is built with `dtype=torch.float64`. The float32 default of
`torch.hann_window` injects ~1e-7 of error on its own, which swamps everything
this harness is trying to measure.
