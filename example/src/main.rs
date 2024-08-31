use ndarray::ArrayView2;
use rustft::{Stft, WindowFunction};

fn main() {
    // Initialize a new STFT object
    let n_fft = 1024;
    let hop_length = 256;
    let stft = Stft::new(n_fft, hop_length, WindowFunction::Hann::<f64>, true);
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
