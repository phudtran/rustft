//! Python bindings used to benchmark this repository's `rustft` against PyTorch.
//!
//! The planner is built once per `Stft` object, as `torch.stft` also reuses its plan
//! cache; constructing one per call would time FFT planning rather than the transform.

use ::rustft::{fft, ifft, Stft as RustStft, WindowFunction};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn window_from_name(name: &str) -> PyResult<WindowFunction<f64>> {
    Ok(match name {
        "hann" => WindowFunction::Hann,
        "hamming" => WindowFunction::Hamming,
        "blackman" => WindowFunction::Blackman,
        "bartlett" => WindowFunction::Bartlett,
        "rectangular" => WindowFunction::Rectangular,
        other => return Err(PyValueError::new_err(format!("unknown window {other:?}"))),
    })
}

#[pyclass]
struct Stft {
    inner: RustStft<f64>,
}

#[pymethods]
impl Stft {
    #[new]
    #[pyo3(signature = (n_fft, hop_length, window = "hann", periodic = true))]
    fn new(n_fft: usize, hop_length: usize, window: &str, periodic: bool) -> PyResult<Self> {
        let inner = RustStft::try_new(n_fft, hop_length, window_from_name(window)?, periodic)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// `(channels, samples)` -> `(channels, n_fft / 2 + 1, frames)`.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
        let input = input.as_array();
        let spectrogram = py
            .detach(|| self.inner.forward(input))
            .map_err(PyValueError::new_err)?;
        Ok(spectrogram.into_pyarray(py))
    }

    /// `(channels, n_fft / 2 + 1, frames)` -> `(channels, samples)`.
    fn inverse<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<'py, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let input = input.as_array();
        let signal = py
            .detach(|| self.inner.inverse(input))
            .map_err(PyValueError::new_err)?;
        Ok(signal.into_pyarray(py))
    }

    fn roundtrip<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let input = input.as_array();
        let signal = py
            .detach(|| {
                self.inner
                    .forward(input)
                    .and_then(|spectrogram| self.inner.inverse(spectrogram.view()))
            })
            .map_err(PyValueError::new_err)?;
        Ok(signal.into_pyarray(py))
    }
}

/// Single forward FFT of a real signal. Plans on every call, unlike [`Stft`].
#[pyfunction]
fn rust_fft<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let spectrum = fft(input.as_array()).map_err(PyValueError::new_err)?;
    Ok(spectrum.into_pyarray(py))
}

/// Single inverse FFT, normalised and reduced to its real part.
#[pyfunction]
fn rust_ifft<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = ifft(input.as_array()).map_err(PyValueError::new_err)?;
    Ok(signal.into_pyarray(py))
}

#[pyfunction]
fn rust_fft_roundtrip<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spectrum = fft(input.as_array()).map_err(PyValueError::new_err)?;
    let signal = ifft(spectrum.view()).map_err(PyValueError::new_err)?;
    Ok(signal.into_pyarray(py))
}

#[pymodule]
fn rustft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Stft>()?;
    m.add_function(wrap_pyfunction!(rust_fft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ifft, m)?)?;
    m.add_function(wrap_pyfunction!(rust_fft_roundtrip, m)?)?;
    Ok(())
}
