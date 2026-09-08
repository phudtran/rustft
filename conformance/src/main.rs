//! Ground-truth harness: read raw f64 arrays produced by Python, run rustft, write
//! raw f64 arrays back out so `conformance.py` can diff them against PyTorch.
use ndarray::{Array2, Array3};
use rustft::{Stft, WindowFunction};
use rustfft::num_complex::Complex;
use std::fs;

fn read_f64(path: &str) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_f64(path: &str, data: &[f64]) {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n_fft: usize = a[1].parse().unwrap();
    let hop: usize = a[2].parse().unwrap();
    let win = a[3].as_str();
    let periodic: bool = a[4].parse().unwrap();
    let channels: usize = a[5].parse().unwrap();
    let length: usize = a[6].parse().unwrap();
    let dir = a[7].as_str();

    let window = match win {
        "hann" => WindowFunction::Hann,
        "hamming" => WindowFunction::Hamming,
        "blackman" => WindowFunction::Blackman,
        "rectangular" => WindowFunction::Rectangular,
        "bartlett" => WindowFunction::Bartlett,
        other => panic!("unknown window {other}"),
    };

    let stft = Stft::new(n_fft, hop, window, periodic);

    // 1. forward on the signal Python generated
    let sig = read_f64(&format!("{dir}/in.bin"));
    assert_eq!(sig.len(), channels * length);
    let input = Array2::from_shape_vec((channels, length), sig).unwrap();
    let spec = stft.forward(input.view()).expect("forward failed");
    let (c, f, t) = spec.dim();
    let mut flat = Vec::with_capacity(c * f * t * 2);
    for v in spec.iter() {
        flat.push(v.re);
        flat.push(v.im);
    }
    write_f64(&format!("{dir}/rust_stft.bin"), &flat);
    fs::write(format!("{dir}/rust_stft.shape"), format!("{c} {f} {t}")).unwrap();

    // 2. inverse of PyTorch's own spectrogram, so the two halves are tested apart
    let raw = read_f64(&format!("{dir}/torch_stft.bin"));
    let vals: Vec<Complex<f64>> = raw.chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect();
    let ref_spec = Array3::from_shape_vec((c, f, t), vals).unwrap();
    let recon = match stft.inverse(ref_spec.view()) {
        Ok(recon) => recon,
        // Refusing is a valid answer — torch.istft refuses some spectrograms too, and
        // the harness checks the two libraries agree about which.
        Err(e) => {
            fs::write(format!("{dir}/rust_istft.shape"), format!("error {e}")).unwrap();
            return;
        }
    };
    let (rc, rl) = recon.dim();
    write_f64(
        &format!("{dir}/rust_istft.bin"),
        recon.as_standard_layout().as_slice().unwrap(),
    );
    fs::write(format!("{dir}/rust_istft.shape"), format!("{rc} {rl}")).unwrap();
}
