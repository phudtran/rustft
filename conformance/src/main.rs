//! Ground-truth harness: read raw f64 arrays produced by Python, run rustft, write
//! raw f64 arrays back out so `conformance.py` can diff them against PyTorch.
//!
//! The wire format is always f64. When asked for `f32`, this narrows the input, runs
//! the whole transform in `Stft<f32>`, and widens the result again — so what the diff
//! measures is the arithmetic precision, not the file format.
use ndarray::{Array2, Array3};
use rustfft::num_complex::Complex;
use rustft::{Stft, WindowFunction};
use rustfft::{num_traits::Float, FftNum};
use std::fs;

fn read_f64(path: &str) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_f64(path: &str, data: impl Iterator<Item = f64>) {
    let bytes: Vec<u8> = data.flat_map(f64::to_le_bytes).collect();
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

fn narrow<T: Float>(v: f64) -> T {
    T::from(v).expect("f64 narrows into the working type")
}

fn widen<T: Float>(v: T) -> f64 {
    v.to_f64().expect("working type widens into f64")
}

struct Args {
    n_fft: usize,
    hop: usize,
    window: String,
    periodic: bool,
    channels: usize,
    length: usize,
    dir: String,
    /// When set, the window is read from this file rather than generated. In f32 the
    /// generated window cannot match PyTorch's (see `Stft::with_window`), so passing
    /// PyTorch's own samples across is the only way to compare the transforms alone.
    window_file: Option<String>,
}

fn run<T>(args: &Args)
where
    T: Float + FftNum + ndarray::ScalarOperand,
{
    let window = match args.window.as_str() {
        "hann" => WindowFunction::Hann,
        "hamming" => WindowFunction::Hamming,
        "blackman" => WindowFunction::Blackman,
        "rectangular" => WindowFunction::Rectangular,
        "bartlett" => WindowFunction::Bartlett,
        other => panic!("unknown window {other}"),
    };
    let stft = match &args.window_file {
        Some(path) => {
            let samples: Vec<T> = read_f64(path).into_iter().map(narrow::<T>).collect();
            assert_eq!(samples.len(), args.n_fft, "window file has the wrong length");
            Stft::<T>::with_window(args.hop, samples).expect("with_window failed")
        }
        None => Stft::<T>::new(args.n_fft, args.hop, window, args.periodic),
    };
    let dir = &args.dir;

    // 1. forward on the signal Python generated
    let sig = read_f64(&format!("{dir}/in.bin"));
    assert_eq!(sig.len(), args.channels * args.length);
    let input = Array2::from_shape_vec(
        (args.channels, args.length),
        sig.into_iter().map(narrow::<T>).collect(),
    )
    .unwrap();
    let spec = stft.forward(input.view()).expect("forward failed");
    let (c, f, t) = spec.dim();
    write_f64(
        &format!("{dir}/rust_stft.bin"),
        spec.iter().flat_map(|v| [widen(v.re), widen(v.im)]),
    );
    fs::write(format!("{dir}/rust_stft.shape"), format!("{c} {f} {t}")).unwrap();

    // 2. inverse of PyTorch's own spectrogram, so the two halves are tested apart
    let raw = read_f64(&format!("{dir}/torch_stft.bin"));
    let vals: Vec<Complex<T>> = raw
        .chunks_exact(2)
        .map(|p| Complex::new(narrow(p[0]), narrow(p[1])))
        .collect();
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
        recon.iter().copied().map(widen),
    );
    fs::write(format!("{dir}/rust_istft.shape"), format!("{rc} {rl}")).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "--cosf" {
        // Dump rustft's f32 cosine for the arguments in a[2], for diffing against torch.
        let args: Vec<f32> = read_f64(&a[2]).into_iter().map(|v| v as f32).collect();
        let out: Vec<u8> = args
            .iter()
            .flat_map(|&x| rustft::sleef_cosf_for_test(x).to_le_bytes())
            .collect();
        fs::write(&a[3], out).unwrap();
        return;
    }
    let args = Args {
        n_fft: a[1].parse().unwrap(),
        hop: a[2].parse().unwrap(),
        window: a[3].clone(),
        periodic: a[4].parse().unwrap(),
        channels: a[5].parse().unwrap(),
        length: a[6].parse().unwrap(),
        dir: a[7].clone(),
        window_file: a.get(9).cloned(),
    };
    match a.get(8).map(String::as_str).unwrap_or("f64") {
        "f64" => run::<f64>(&args),
        "f32" => run::<f32>(&args),
        other => panic!("unknown precision {other}"),
    }
}
