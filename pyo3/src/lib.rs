use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use stft::{fft, ifft, istft, stft};

#[pyfunction]
fn rust_fft(py: Python, input: &PyArray1<f64>) -> PyResult<Py<PyArray1<Complex<f64>>>> {
    let binding = input.readonly();
    let input_array = binding.as_array();
    let fft_res = fft(input_array)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Error in fft: {}", e))
        })?
        .to_vec();
    let py_result = PyArray1::from_vec_bound(py, fft_res);
    Ok(py_result.into())
}

#[pyfunction]
fn rust_ifft(py: Python, input: &PyArray1<Complex<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    // Convert input to  Array1<Complex<f64>> for ifft
    let binding = input.readonly();
    let input_array = binding.as_array();
    let ifft_res = ifft(input_array)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Error in ifft: {}", e))
        })?
        .to_vec();
    let py_result = PyArray1::from_vec_bound(py, ifft_res);
    Ok(py_result.into())
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
    let window_array = match window {
        Some(w) => Some(w.readonly().as_array().to_owned()),
        None => None,
    };
    let binding = input.readonly();
    let input_array = binding.as_array();
    let stft_res = stft(input_array, n_fft, hop_length, window_array).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Error in stft: {}", e))
    })?;
    let py_result = PyArray3::from_owned_array_bound(py, stft_res);
    Ok(py_result.into())
}

#[pyfunction]
fn rust_istft(
    py: Python,
    input: &PyArray3<Complex<f64>>,
    n_fft: usize,
    hop_length: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let input_array = binding.as_array();
    let istft_res = istft(input_array, n_fft, hop_length).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Error in istft: {}", e))
    })?;
    let py_output = PyArray2::from_owned_array_bound(py, istft_res);
    Ok(py_output.into())
}

#[pyfunction]
fn rust_stft_roundtrip(
    py: Python,
    input: &PyArray2<f64>,
    n_fft: usize,
    hop_length: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let stft_result = rust_stft(py, input, n_fft, hop_length, None)?;
    let istft_result = rust_istft(py, &stft_result.as_ref(py), n_fft, hop_length)?;
    Ok(istft_result)
}

#[pymodule]
fn rustft(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_fft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ifft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_fft_roundtrip_test, m)?)?;
    m.add_function(wrap_pyfunction!(rust_stft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_istft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_stft_roundtrip, m)?)?;
    Ok(())
}
