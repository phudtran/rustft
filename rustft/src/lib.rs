use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

pub fn fft(input: ArrayView1<f64>) -> Result<Array1<Complex<f64>>, String> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let fft = planner.plan_fft_forward(len);
    let mut buffer: Vec<Complex<f64>> = input
        .to_vec()
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut buffer);
    let array = Array1::from_vec(buffer);
    Ok(array.into())
}

pub fn ifft(input: ArrayView1<Complex<f64>>) -> Result<Array1<f64>, String> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let ifft = planner.plan_fft_inverse(len);
    let mut buffer = input.to_vec();
    ifft.process(&mut buffer);
    // Normalize and extract real parts
    let scale = len as f64;
    let result: Array1<f64> = buffer.into_iter().map(|c| c.re / scale).collect();
    Ok(result)
}

pub fn stft(
    input: ArrayView2<f64>,
    n_fft: usize,
    hop_length: usize,
    window: Option<Array1<f64>>,
) -> Result<Array3<Complex<f64>>, String> {
    let num_channels = input.shape()[0];
    let signal_length = input.shape()[1];

    // For center padding (reflection)
    let pad_length = n_fft / 2;
    let padded_length = signal_length + 2 * pad_length;

    // Calculate the number of frames
    let num_frames = (padded_length - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let window: Array1<f64> = match window {
        Some(w) => w,
        None => Array1::from_vec(hann_window(n_fft, true)),
    };

    let mut output = Array3::zeros((num_channels, n_freqs, num_frames));

    for (ch, channel) in input.outer_iter().enumerate() {
        // Pad the entire channel once
        let padded_channel = pad_reflect(channel.to_owned(), pad_length);

        for frame in 0..num_frames {
            let start = frame * hop_length;
            let mut buffer = Array1::zeros(n_fft);

            for i in 0..n_fft {
                buffer[i] = Complex::new(padded_channel[start + i] * window[i], 0.0);
            }

            fft.process(
                buffer
                    .as_slice_mut()
                    .expect("Failed to get mutable slice of buffer"),
            );

            for (freq, &value) in buffer.iter().take(n_freqs).enumerate() {
                output[[ch, freq, frame]] = value;
            }
        }
    }

    Ok(output)
}

pub fn istft(
    input: ArrayView3<Complex<f64>>,
    n_fft: usize,
    hop_length: usize,
) -> Result<Array2<f64>, String> {
    let num_channels = input.shape()[0];
    let num_frames = input.shape()[2];
    let padded_length = (num_frames - 1) * hop_length + n_fft;
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let scale_factor = 1.0 / (n_fft as f64);

    let mut output: Array2<f64> = Array2::zeros((num_channels, padded_length));

    for (ch, channel) in input.outer_iter().enumerate() {
        for frame in 0..num_frames {
            let start = frame * hop_length;
            let mut full_spectrum = vec![Complex::new(0.0, 0.0); n_fft];

            // Reconstruct full spectrum with correct Nyquist handling
            for (i, &value) in channel.slice(s![.., frame]).iter().enumerate() {
                if i == 0 || i == n_fft / 2 {
                    // DC and Nyquist components are always real
                    full_spectrum[i] = Complex::new(value.re, 0.0);
                } else if i < n_fft / 2 {
                    // Positive frequencies
                    full_spectrum[i] = value;
                    // Negative frequencies (complex conjugate)
                    full_spectrum[n_fft - i] = value.conj();
                }
            }
            // Perform IFFT
            ifft.process(&mut full_spectrum);

            // Accumulate
            for (i, &value) in full_spectrum.iter().enumerate() {
                if start + i < padded_length {
                    output[[ch, start + i]] += value.re * scale_factor;
                }
            }
        }
    }
    // Remove padding
    remove_padding(&mut output, n_fft);
    Ok(output)
}

pub fn hann_window(size: usize, periodic: bool) -> Vec<f64> {
    if periodic {
        (0..size)
            .map(|n| {
                let cos_term = (2.0 * PI * n as f64 / size as f64).cos();
                0.5 * (1.0 - cos_term)
            })
            .collect()
    } else {
        (0..size)
            .map(|n| {
                let cos_term = (2.0 * PI * n as f64 / (size - 1) as f64).cos();
                0.5 * (1.0 - cos_term)
            })
            .collect()
    }
}

fn pad_reflect(signal: Array1<f64>, pad_length: usize) -> Array1<f64> {
    let signal_length = signal.len();
    let mut padded = Array1::zeros(signal_length + 2 * pad_length);

    // Copy the original signal
    padded
        .slice_mut(s![pad_length..pad_length + signal_length])
        .assign(&signal);

    // Reflect at the beginning
    for i in 0..pad_length {
        padded[pad_length - 1 - i] = signal[i + 1];
    }

    // Reflect at the end
    for i in 0..pad_length {
        padded[pad_length + signal_length + i] = signal[signal_length - 2 - i];
    }
    padded
}

fn remove_padding(padded: &mut Array2<f64>, n_fft: usize) {
    let (num_channels, padded_length) = padded.dim();
    let start = n_fft / 2;
    let end = padded_length - n_fft / 2;

    // Removing front padding
    for ch in 0..num_channels {
        for i in 0..(end - start) {
            padded[[ch, i]] = padded[[ch, i + start]];
        }
    }

    // Truncate
    padded.slice_collapse(s![.., ..(end - start)]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_reverse_padding_multi_channel() {
        let mut padded = arr2(&[
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
            [8.0, 7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0],
        ]);

        let n_fft = 4;
        remove_padding(&mut padded, n_fft);
        let expected = arr2(&[[1.0, 2.0, 3.0, 4.0], [6.0, 7.0, 8.0, 9.0]]);
        assert_eq!(padded, expected);
    }

    #[test]
    fn test_apply_padding() {
        let signal = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let pad_length = 2;
        let padded = pad_reflect(signal, pad_length);
        let expected = arr1(&[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
        assert_eq!(padded, expected);
    }
}
