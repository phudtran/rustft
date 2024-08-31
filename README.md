# WIP STFT/ISTFT Library

Handles non-COLA compliant windows (Must be (NOLA)[https://gauss256.github.io/blog/cola.html]).

For example
n_fft: 6144
hop_length: 1024

# Install

```
cargo add rustft
```

# Example

```
use ndarray::ArrayView2;
use rustft::{Stft, WindowFunction};

fn main() {
    // Initialize a new STFT object
    let stft = Stft::new(1024, 256, WindowFunction::Hann::<f64>, true);
    // Create a 2D array of f64
    let data = vec![0.0; 2048];
    let input = ArrayView2::from_shape((2, 1024), &data).unwrap();
    let expected_output = input.clone();
    // Perform the forward STFT
    let stft_res = stft.forward(input).unwrap();
    // perform the inverse STFT
    let istft_res = stft.inverse(stft_res.view()).unwrap();
    assert_eq!(expected_output, istft_res);
}
```

# Performance

WIP. No optimizations, super slow.

# How to run benchmarks

```
cd benchmarks
source ~/.venv/bin/activate
pip install numpy torch
cargo build --release
maturin develop
python3 test_stft.py

```

# Benchmarks

```
Testing with: 2 channels, signal length 16384, n_fft 1024, hop_length 512
Average STFT difference (Rust vs PyTorch): 4.770876278062766e-07
Average Rust roundtrip error: 9.973346089885864e-17
Average PyTorch roundtrip error: 9.614115577385553e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.7807017136265533e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.4784528280914146e-08

Average run times:

Rust STFT + ISTFT: 0.058088 seconds
PyTorch STFT + ISTFT: 0.013779 seconds
Rust STFT: 0.023892 seconds
PyTorch ISTFT: 0.001202 seconds
PyTorch STFT: 0.000659 seconds
Rust ISTFT: 0.033392 seconds

Testing with: 2 channels, signal length 261120, n_fft 6144, hop_length 1024
Average STFT difference (Rust vs PyTorch): 9.033896440199904e-07
Average Rust roundtrip error: 1.4512502510819078e-16
Average PyTorch roundtrip error: 1.3035270142443053e-08
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.4251233051816704e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 6.755024010548921e-09

Average run times:

Rust STFT + ISTFT: 3.393714 seconds
PyTorch STFT + ISTFT: 0.209490 seconds
Rust STFT: 1.488826 seconds
PyTorch ISTFT: 0.029740 seconds
PyTorch STFT: 0.010761 seconds
Rust ISTFT: 1.860682 seconds

Testing with: 2 channels, signal length 65536, n_fft 4096, hop_length 2048
Average STFT difference (Rust vs PyTorch): 7.407661646859121e-07
Average Rust roundtrip error: 1.1003355500607053e-16
Average PyTorch roundtrip error: 9.603641780202305e-09
Average roundtrip error (Rust STFT -> PyTorch ISTFT): 1.730898875409103e-08
Average roundtrip error (PyTorch STFT -> Rust ISTFT): 1.4207531055181427e-08

Average run times:

Rust STFT + ISTFT: 0.255585 seconds
PyTorch STFT + ISTFT: 0.020741 seconds
Rust STFT: 0.105303 seconds
PyTorch ISTFT: 0.002952 seconds
PyTorch STFT: 0.001078 seconds
Rust ISTFT: 0.146071 seconds
```
