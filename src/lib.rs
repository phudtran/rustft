use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

#[pyfunction]
fn rust_fft(input: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let fft = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    // Convert to PyTorch-like "half-complex" format
    let mut result = Vec::with_capacity(len);
    result.push(buffer[0].re); // DC component (real)
    for i in 1..len / 2 {
        result.push(buffer[i].re);
        result.push(buffer[i].im);
    }
    if len % 2 == 0 {
        result.push(buffer[len / 2].re); // Nyquist frequency (real)
    }
    // Pad with zeros to maintain input length
    result.resize(len, 0.0);
    Ok(result)
}

#[pyfunction]
fn rust_fft_roundtrip_test(input: Vec<f64>) -> PyResult<Vec<f64>> {
    let fft_result = rust_fft(input.clone())?;
    let ifft_result = rust_ifft(fft_result)?;
    Ok(ifft_result)
}

#[pyfunction]
fn rust_ifft(input: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let ifft = planner.plan_fft_inverse(len);

    // Convert from PyTorch-like "half-complex" format to full complex
    let mut buffer = Vec::with_capacity(len);
    buffer.push(Complex::new(input[0], 0.0)); // DC component
    for i in 1..len / 2 {
        buffer.push(Complex::new(input[2 * i - 1], input[2 * i]));
    }
    if len % 2 == 0 {
        buffer.push(Complex::new(input[len - 1], 0.0)); // Nyquist frequency
    }
    // Fill the rest with complex conjugates
    for i in (len / 2 + 1..len).rev() {
        buffer.push(buffer[len - i].conj());
    }

    ifft.process(&mut buffer);

    // Normalize and extract real parts
    let scale = 1.0 / len as f64;
    let result: Vec<f64> = buffer.into_iter().map(|c| c.re * scale).collect();
    Ok(result)
}

#[pymodule]
fn rustfft_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_fft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ifft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_fft_roundtrip_test, m)?)?;
    Ok(())
}
