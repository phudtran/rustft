use numpy::ndarray::{s, Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

#[pyfunction]
fn rust_fft(py: Python, input: &PyArray1<f64>) -> PyResult<Py<PyArray1<Complex<f64>>>> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let fft = planner.plan_fft_forward(len);
    let mut buffer: Vec<Complex<f64>> = input
        .to_vec()?
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut buffer);
    // Convert to PyArray without normalization
    let array = PyArray1::from_vec(py, buffer);
    Ok(array.to_owned())
}

#[pyfunction]
fn rust_ifft(py: Python, input: &PyArray1<Complex<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let ifft = planner.plan_fft_inverse(len);
    let mut buffer = input.to_vec()?;
    ifft.process(&mut buffer);
    // Normalize and extract real parts
    let scale = len as f64;
    let result: Vec<f64> = buffer.into_iter().map(|c| c.re / scale).collect();
    // Convert to PyArray
    let array = PyArray1::from_vec(py, result);
    Ok(array.to_owned())
}

#[pyfunction]
fn rust_fft_roundtrip_test(py: Python, input: &PyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let fft_result = rust_fft(py, input)?;
    let ifft_result = rust_ifft(py, &fft_result.as_ref(py))?;
    Ok(ifft_result)
}

#[pyfunction]
fn rust_stft(
    py: Python,
    input: &PyArray2<f64>,
    n_fft: usize,
    hop_length: usize,
    window: Option<&PyArray1<f64>>,
) -> PyResult<Py<PyArray3<Complex<f64>>>> {
    let input_array: Array2<f64> = input.readonly().as_array().to_owned();
    let num_channels = input_array.shape()[0];
    let signal_length = input_array.shape()[1];

    // Apply centering padding
    let pad_length = n_fft / 2;
    let padded_length = signal_length + 2 * pad_length;

    // Calculate the number of frames
    let num_frames = (padded_length - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let window: Array1<f64> = match window {
        Some(w) => w.readonly().as_array().to_owned(),
        None => Array1::from_vec(hann_window(n_fft, true)),
    };

    let mut output = Array3::zeros((num_channels, n_freqs, num_frames));

    for (ch, channel) in input_array.outer_iter().enumerate() {
        for frame in 0..num_frames {
            let start = frame * hop_length;
            let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];

            for i in 0..n_fft {
                let padded_index = start + i;
                if padded_index < pad_length {
                    buffer[i] = Complex::new(0.0, 0.0);
                } else if padded_index >= pad_length && padded_index < pad_length + signal_length {
                    buffer[i] = Complex::new(channel[padded_index - pad_length] * window[i], 0.0);
                } else {
                    buffer[i] = Complex::new(0.0, 0.0);
                }
            }

            fft.process(&mut buffer);

            for (freq, &value) in buffer.iter().take(n_freqs).enumerate() {
                output[[ch, freq, frame]] = value;
            }
        }
    }

    let py_output = PyArray3::from_owned_array(py, output);
    Ok(py_output.to_owned())
}

#[pyfunction]
fn rust_istft(
    py: Python,
    input: &PyArray3<Complex<f64>>,
    n_fft: usize,
    hop_length: usize,
    original_length: usize,
    window: Option<&PyArray1<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let input_array = input.readonly().as_array().to_owned();
    let num_channels = input_array.shape()[0];
    let n_freqs = input_array.shape()[1];
    let num_frames = input_array.shape()[2];

    let pad_length = n_fft / 2;
    let padded_length = (num_frames - 1) * hop_length + n_fft;

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let window: Array1<f64> = match window {
        Some(w) => w.readonly().as_array().to_owned(),
        None => Array1::from_vec(hann_window(n_fft, true)),
    };

    let mut output = Array2::zeros((num_channels, padded_length));
    let mut window_sum: Array1<f64> = Array1::zeros(padded_length);

    for (ch, channel) in input_array.outer_iter().enumerate() {
        for frame in 0..num_frames {
            let start = frame * hop_length;
            let mut full_spectrum = vec![Complex::new(0.0, 0.0); n_fft];

            for (i, &value) in channel.slice(s![.., frame]).iter().enumerate() {
                full_spectrum[i] = value;
                if i > 0 && i < n_freqs - 1 {
                    full_spectrum[n_fft - i] = value.conj();
                }
            }

            ifft.process(&mut full_spectrum);

            for (i, &value) in full_spectrum.iter().enumerate() {
                if start + i < padded_length {
                    output[[ch, start + i]] += value.re * window[i];
                    window_sum[start + i] += window[i].powi(2);
                }
            }
        }
    }

    // Normalize by the window sum
    for mut channel in output.outer_iter_mut() {
        for (i, sample) in channel.iter_mut().enumerate() {
            if window_sum[i] > 0.0 {
                *sample /= window_sum[i] as f64;
            }
        }
    }

    // Trim the output to match the original signal length
    let trimmed_output = output
        .slice(s![.., pad_length..pad_length + original_length])
        .to_owned();

    // Apply scaling factor
    let scaling_factor = 1.0 / (n_fft as f64);
    let scaled_output = trimmed_output.mapv(|x: f64| x * scaling_factor);

    let py_output = PyArray2::from_owned_array(py, scaled_output);
    Ok(py_output.to_owned())
}

fn hann_window(size: usize, periodic: bool) -> Vec<f64> {
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

#[pyfunction]
fn rust_stft_roundtrip_test(
    py: Python,
    input: &PyArray2<f64>,
    n_fft: usize,
    hop_length: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let stft_result = rust_stft(py, input, n_fft, hop_length, None)?;
    let istft_result = rust_istft(
        py,
        &stft_result.as_ref(py),
        n_fft,
        hop_length,
        input.len(),
        None,
    )?;
    Ok(istft_result)
}

#[pymodule]
fn rustfft_test(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_fft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ifft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_fft_roundtrip_test, m)?)?;
    m.add_function(wrap_pyfunction!(rust_stft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_istft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_stft_roundtrip_test, m)?)?;
    Ok(())
}
