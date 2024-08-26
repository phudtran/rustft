use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

#[pyfunction]
fn rust_fft(input: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let fft = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();

    fft.process(&mut buffer);

    let result: Vec<f64> = buffer.into_iter().flat_map(|c| vec![c.re, c.im]).collect();

    Ok(result)
}

#[pyfunction]
fn rust_ifft(input: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut planner = FftPlanner::new();
    let len = input.len() / 2;
    let ifft = planner.plan_fft_inverse(len);

    let mut buffer: Vec<Complex<f64>> = input
        .chunks(2)
        .map(|chunk| Complex::new(chunk[0], chunk[1]))
        .collect();

    ifft.process(&mut buffer);

    // Normalize the output
    let scale = 1.0 / len as f64;
    buffer.iter_mut().for_each(|x| *x *= scale);

    let result: Vec<f64> = buffer
        .into_iter()
        .map(|c| c.re) // We only return the real part for IFFT
        .collect();

    Ok(result)
}

#[pymodule]
fn rustfft_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_fft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ifft, m)?)?;
    Ok(())
}
