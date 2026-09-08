//! Python bindings used to benchmark this repository's `rustft` against PyTorch.
//!
//! The planner is built once per `Stft` object, as `torch.stft` also reuses its plan
//! cache; constructing one per call would time FFT planning rather than the transform.

use ::rustft::{fft, ifft, Complex, FftNum, Float, PadMode, Stft as RustStft, WindowFunction};
use numpy::{
    Complex64, Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn pad_from_name(name: &str) -> PyResult<PadMode> {
    Ok(match name {
        "reflect" => PadMode::Reflect,
        "constant" => PadMode::Constant,
        "replicate" => PadMode::Replicate,
        "circular" => PadMode::Circular,
        other => return Err(PyValueError::new_err(format!("unknown pad_mode {other:?}"))),
    })
}

fn window_from_name<T: Float + FftNum>(name: &str) -> PyResult<WindowFunction<T>> {
    Ok(match name {
        "hann" => WindowFunction::Hann,
        "hamming" => WindowFunction::Hamming,
        "blackman" => WindowFunction::Blackman,
        "bartlett" => WindowFunction::Bartlett,
        "rectangular" => WindowFunction::Rectangular,
        other => return Err(PyValueError::new_err(format!("unknown window {other:?}"))),
    })
}

/// The transform is planned for one precision, chosen at construction, so the plan and
/// the window are built once rather than per call.
enum Planned {
    F32(RustStft<f32>),
    F64(RustStft<f64>),
}

/// An array of the wrong dtype is a mistake worth naming precisely: in f32 the results
/// differ from f64 by about 2e-7 relative, which is easy to misread as a bug.
fn wrong_dtype(array: &Bound<'_, PyAny>, planned: &str) -> PyErr {
    let got = array
        .getattr("dtype")
        .and_then(|d| d.str())
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "an unknown type".to_string());
    PyValueError::new_err(format!(
        "this Stft was planned for {planned}, but the array is {got}. Build it with \
         dtype=\"{planned}\", or convert the array."
    ))
}

fn forward_typed<'py, T>(
    py: Python<'py>,
    stft: &RustStft<T>,
    input: &Bound<'py, PyAny>,
    planned: &str,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Float + FftNum + Element,
    Complex<T>: Element,
{
    let array = input
        .extract::<PyReadonlyArrayDyn<T>>()
        .map_err(|_| wrong_dtype(input, planned))?;
    let array = array.as_array();
    let spectrogram = py
        .detach(|| stft.forward(array))
        .map_err(PyValueError::new_err)?;
    Ok(spectrogram.into_pyarray(py).into_any())
}

fn inverse_typed<'py, T>(
    py: Python<'py>,
    stft: &RustStft<T>,
    input: &Bound<'py, PyAny>,
    planned: &str,
    length: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Float + FftNum + Element,
    Complex<T>: Element,
{
    let array = input
        .extract::<PyReadonlyArrayDyn<Complex<T>>>()
        .map_err(|_| wrong_dtype(input, planned))?;
    let array = array.as_array();
    let signal = py
        .detach(|| stft.inverse_with_length(array, length))
        .map_err(PyValueError::new_err)?;
    Ok(signal.into_pyarray(py).into_any())
}

fn roundtrip_typed<'py, T>(
    py: Python<'py>,
    stft: &RustStft<T>,
    input: &Bound<'py, PyAny>,
    planned: &str,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Float + FftNum + Element,
    Complex<T>: Element,
{
    let array = input
        .extract::<PyReadonlyArrayDyn<T>>()
        .map_err(|_| wrong_dtype(input, planned))?;
    let array = array.as_array();
    let signal = py
        .detach(|| {
            stft.forward(array)
                .and_then(|spectrogram| stft.inverse(spectrogram.view()))
        })
        .map_err(PyValueError::new_err)?;
    Ok(signal.into_pyarray(py).into_any())
}

#[allow(clippy::too_many_arguments)]
fn build<T>(
    n_fft: usize,
    hop_length: usize,
    window: Option<&[T]>,
    window_name: &str,
    win_length: Option<usize>,
    periodic: bool,
    center: bool,
    pad_mode: &str,
    normalized: bool,
    onesided: bool,
) -> PyResult<RustStft<T>>
where
    T: Float + FftNum,
{
    let mut builder = RustStft::<T>::builder(n_fft)
        .hop_length(hop_length)
        .center(center)
        .pad_mode(pad_from_name(pad_mode)?)
        .normalized(normalized)
        .onesided(onesided);
    builder = match window {
        Some(samples) => builder.window_samples(samples.to_vec()),
        None => builder.window(window_from_name(window_name)?, periodic),
    };
    if let Some(win_length) = win_length {
        builder = builder.win_length(win_length);
    }
    builder.build().map_err(PyValueError::new_err)
}

#[pyclass]
struct Stft {
    inner: Planned,
}

#[pymethods]
impl Stft {
    /// `dtype` selects the precision the transform runs in, and the array dtype that
    /// `forward`, `inverse` and `roundtrip` will accept.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n_fft, hop_length, window = "hann", periodic = true,
                        dtype = "float64", win_length = None, center = true,
                        pad_mode = "reflect", normalized = false, onesided = true))]
    fn new(
        n_fft: usize,
        hop_length: usize,
        window: &str,
        periodic: bool,
        dtype: &str,
        win_length: Option<usize>,
        center: bool,
        pad_mode: &str,
        normalized: bool,
        onesided: bool,
    ) -> PyResult<Self> {
        let inner = match dtype {
            "float32" => Planned::F32(build::<f32>(
                n_fft, hop_length, None, window, win_length, periodic, center, pad_mode,
                normalized, onesided,
            )?),
            "float64" => Planned::F64(build::<f64>(
                n_fft, hop_length, None, window, win_length, periodic, center, pad_mode,
                normalized, onesided,
            )?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "dtype must be \"float32\" or \"float64\", got {other:?}"
                )))
            }
        };
        Ok(Self { inner })
    }

    /// Plan a transform that uses `window` exactly as given, rather than building one.
    ///
    /// The precision follows the window's dtype. This is the way to hand rustft another
    /// framework's window bits verbatim -- in f32 PyTorch builds its windows with a
    /// vectorised cosine that no independent implementation reproduces exactly.
    #[staticmethod]
    #[pyo3(signature = (hop_length, window, n_fft = None, center = true,
                        pad_mode = "reflect", normalized = false, onesided = true))]
    fn with_window(
        hop_length: usize,
        window: &Bound<'_, PyAny>,
        n_fft: Option<usize>,
        center: bool,
        pad_mode: &str,
        normalized: bool,
        onesided: bool,
    ) -> PyResult<Self> {
        if let Ok(w) = window.extract::<PyReadonlyArray1<f32>>() {
            let w = w.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(Self {
                inner: Planned::F32(build::<f32>(
                    n_fft.unwrap_or(w.len()), hop_length, Some(w), "", None, false,
                    center, pad_mode, normalized, onesided,
                )?),
            });
        }
        let w = window
            .extract::<PyReadonlyArray1<f64>>()
            .map_err(|_| PyValueError::new_err("window must be a 1-D float32 or float64 array"))?;
        let w = w.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Planned::F64(build::<f64>(
                n_fft.unwrap_or(w.len()), hop_length, Some(w), "", None, false,
                center, pad_mode, normalized, onesided,
            )?),
        })
    }

    /// The analysis window actually in use.
    fn window<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.inner {
            Planned::F32(stft) => stft.window().to_vec().into_pyarray(py).into_any(),
            Planned::F64(stft) => stft.window().to_vec().into_pyarray(py).into_any(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match self.inner {
            Planned::F32(_) => "float32",
            Planned::F64(_) => "float64",
        }
    }

    /// Transforms the last axis; any leading axes are carried through.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            Planned::F32(stft) => forward_typed(py, stft, input, "float32"),
            Planned::F64(stft) => forward_typed(py, stft, input, "float64"),
        }
    }

    /// Inverts the last two axes; any leading axes are carried through.
    #[pyo3(signature = (input, length = None))]
    fn inverse<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        length: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            Planned::F32(stft) => inverse_typed(py, stft, input, "float32", length),
            Planned::F64(stft) => inverse_typed(py, stft, input, "float64", length),
        }
    }

    /// Forward then inverse without returning the spectrogram to Python.
    fn roundtrip<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            Planned::F32(stft) => roundtrip_typed(py, stft, input, "float32"),
            Planned::F64(stft) => roundtrip_typed(py, stft, input, "float64"),
        }
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
