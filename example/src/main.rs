use ndarray::Array2;
use rustft::{Stft, WindowFunction};

fn main() {
    // Plan once; forward and inverse both borrow, so one Stft serves every block.
    let n_fft = 1024;
    let hop_length = 256;
    let stft = Stft::new(n_fft, hop_length, WindowFunction::Hann::<f64>, true);

    // Two channels of a 440 Hz tone at 44.1 kHz.
    let input = Array2::from_shape_fn((2, 44100), |(channel, i)| {
        let t = i as f64 / 44100.0;
        (2.0 * std::f64::consts::PI * 440.0 * t).sin() * (1.0 - 0.2 * channel as f64)
    });

    let spectrogram = stft.forward(input.view()).unwrap();
    println!(
        "{:?} samples -> {:?} spectrogram (channels, bins, frames)",
        input.dim(),
        spectrogram.dim()
    );

    let recovered = stft.inverse(spectrogram.view()).unwrap();

    // Overlap-add reconstructs hop_length * (frames - 1) samples: the centring pad is
    // trimmed off both ends, exactly as torch.istft does.
    let error = input
        .slice(ndarray::s![.., ..recovered.dim().1])
        .iter()
        .zip(recovered.iter())
        .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
    println!("recovered {} samples, worst error {error:e}", recovered.dim().1);
    assert!(error < 1e-12);
}
