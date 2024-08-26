use numpy::PyArray1;
use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

#[pyfunction]
fn rust_fft(py: Python, input: &PyArray1<f64>) -> PyResult<Py<PyArray1<Complex<f64>>>> {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let fft = planner.plan_fft_forward(len);

    let input_slice = input.to_vec().unwrap();
    let mut buffer: Vec<Complex<f64>> = input_slice.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    // Normalize the FFT output
    let scale = (len as f64).sqrt();
    buffer.iter_mut().for_each(|x| *x /= scale);

    // Convert to PyArray
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
    let scale = (len as f64).sqrt();
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

#[pymodule]
fn rustfft_test(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_fft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ifft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_fft_roundtrip_test, m)?)?;
    Ok(())
}
