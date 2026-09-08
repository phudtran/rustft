//! Short-time Fourier transform and its inverse, numerically matching
//! `torch.stft` / `torch.istft` with `center=True` and `pad_mode="reflect"`.
//!
//! ```
//! use ndarray::Array2;
//! use rustft::{Stft, WindowFunction};
//!
//! let stft = Stft::new(1024, 256, WindowFunction::Hann::<f64>, true);
//! let signal = Array2::from_shape_fn((2, 8192), |(_, i)| (i as f64 * 0.01).sin());
//!
//! let spectrogram = stft.forward(signal.view()).unwrap();
//! let recovered = stft.inverse(spectrogram.view()).unwrap();
//! assert_eq!(recovered.dim(), signal.dim());
//! ```

mod backend;
mod sleef;

/// Exposed for the conformance harness, which diffs this against `torch.cos`.
#[doc(hidden)]
pub fn sleef_cosf_for_test(d: f32) -> f32 {
    sleef::cosf(d)
}

use backend::{Forward, Inverse, Planner};
use ndarray::{s, Array, Array1, Array2, Array3, ArrayView, ArrayView1, ArrayView2, ArrayView3, Axis, Dimension, IxDyn};
use rustfft::FftPlanner;

pub use rustfft::num_complex::Complex;
pub use rustfft::num_traits::Float;
pub use rustfft::FftNum;
use std::f64::consts::PI;

/// Smallest window-envelope value `inverse` will divide by. `torch.istft` uses the
/// same constant to decide a window fails the NOLA condition and is not invertible.
const NOLA_EPSILON: f64 = 1e-11;

/// Frames per parallel work unit. Large enough that the per-unit buffer setup and
/// rayon's own bookkeeping stay negligible next to the FFTs in the block.
#[cfg(feature = "parallel")]
const FRAME_BLOCK: usize = 8;

/// Transformed samples per channel below which rayon's scheduling costs more than the
/// FFTs it would spread out. Measured on the benchmark shapes: at a few tens of
/// thousands of samples the parallel path is several times slower, and it pulls ahead
/// somewhere above 2^17.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 1 << 17;

/// Matrix size, in bytes, above which the spectrogram transpose is worth splitting
/// across cores. Below it the matrix still fits in cache, where the blocked sequential
/// transpose already runs at memory speed and rayon only adds latency.
#[cfg(feature = "parallel")]
const PARALLEL_TRANSPOSE_BYTES: usize = 4 << 20;

/// How [`Stft`] extends the signal when `center` is on, mirroring `torch.stft`'s
/// `pad_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PadMode {
    /// Mirror the signal without repeating the edge sample. PyTorch's default.
    #[default]
    Reflect,
    /// Extend with zeros.
    Constant,
    /// Repeat the edge sample.
    Replicate,
    /// Wrap around to the opposite end.
    Circular,
}

/// Configures the options `Stft::new` fixes at their PyTorch defaults.
///
/// ```
/// use rustft::{PadMode, Stft, WindowFunction};
///
/// let stft = Stft::builder(1024)
///     .hop_length(256)
///     .win_length(400)
///     .window(WindowFunction::Hann::<f64>, true)
///     .center(false)
///     .pad_mode(PadMode::Constant)
///     .build()
///     .unwrap();
/// ```
pub struct StftBuilder<T>
where
    T: Float + FftNum,
{
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: WindowSource<T>,
    center: bool,
    pad_mode: PadMode,
    normalized: bool,
    onesided: bool,
}

enum WindowSource<T>
where
    T: Float + FftNum,
{
    Function(WindowFunction<T>, bool),
    Samples(Vec<T>),
}

impl<T> StftBuilder<T>
where
    T: Float + FftNum,
{
    /// Samples between the start of consecutive frames. Defaults to `n_fft / 4`, as
    /// `torch.stft` does.
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Length of the window before zero-padding. Defaults to `n_fft`; anything shorter
    /// is centred inside `n_fft` with zeros either side.
    pub fn win_length(mut self, win_length: usize) -> Self {
        self.win_length = Some(win_length);
        self
    }

    /// Build the window from one of the [`WindowFunction`]s. Defaults to
    /// [`WindowFunction::Rectangular`], matching `torch.stft`'s `window=None`.
    pub fn window(mut self, function: WindowFunction<T>, periodic: bool) -> Self {
        self.window = WindowSource::Function(function, periodic);
        self
    }

    /// Use these samples as the window verbatim. Sets `win_length` to their count.
    pub fn window_samples(mut self, samples: Vec<T>) -> Self {
        self.window = WindowSource::Samples(samples);
        self
    }

    /// Pad the signal by `n_fft / 2` either side so frame `t` is centred on sample
    /// `t * hop_length`. On by default.
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// How to extend the signal when `center` is on.
    pub fn pad_mode(mut self, pad_mode: PadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// Scale the forward transform by `1 / sqrt(n_fft)`, and the inverse back up. Off
    /// by default.
    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Return only the `n_fft / 2 + 1` non-redundant bins. On by default; turning it
    /// off returns the full `n_fft`, whose upper half is the conjugate mirror.
    pub fn onesided(mut self, onesided: bool) -> Self {
        self.onesided = onesided;
        self
    }

    pub fn build(self) -> Result<Stft<T>, String> {
        let n_fft = self.n_fft;
        let hop_length = self.hop_length.unwrap_or(n_fft / 4);
        validate_params(n_fft, hop_length)?;

        let (window, win_length, function, periodic) = match self.window {
            WindowSource::Function(function, periodic) => {
                let win_length = self.win_length.unwrap_or(n_fft);
                (
                    function.new(win_length, periodic),
                    win_length,
                    Some(function),
                    periodic,
                )
            }
            WindowSource::Samples(samples) => {
                let win_length = samples.len();
                (samples, win_length, None, false)
            }
        };
        if win_length == 0 || win_length > n_fft {
            return Err(format!(
                "win_length must be between 1 and n_fft = {n_fft}, got {win_length}"
            ));
        }
        // A short window sits centred inside n_fft with zeros either side, which is what
        // torch.stft does before transforming.
        let window = if win_length == n_fft {
            window
        } else {
            let mut padded = vec![T::zero(); n_fft];
            let left = (n_fft - win_length) / 2;
            padded[left..left + win_length].copy_from_slice(&window);
            padded
        };

        let mut planner = Planner::new();
        let forward = planner.plan_forward(n_fft);
        let inverse = planner.plan_inverse(n_fft, inverse_scale::<T>(n_fft, self.normalized));
        Ok(Stft {
            inverse_window: scaled_window(&window, n_fft, self.normalized),
            nyquist: nyquist_bin(n_fft),
            n_fft,
            hop_length,
            window,
            window_function: function,
            window_periodic: periodic,
            win_length,
            center: self.center,
            pad_mode: self.pad_mode,
            normalized: self.normalized,
            onesided: self.onesided,
            forward,
            inverse,
            planner,
        })
    }
}

/// A planned short-time Fourier transform.
///
/// Planning the FFT, building the window and sizing the scratch buffers all happen
/// once, in [`Stft::new`]. [`Stft::forward`] and [`Stft::inverse`] take `&self`, so a
/// single instance can be shared across threads and reused for every block of audio.
pub struct Stft<T>
where
    T: Float + FftNum,
{
    n_fft: usize,
    hop_length: usize,
    window: Vec<T>,
    /// The window the inverse re-applies to each frame. When the FFT backend leaves
    /// normalisation to us it is `window[i] / n_fft`, folding both into one multiply;
    /// when the backend normalises (pocketfft, which takes a scale factor for free) it
    /// is the plain window, which is also the order `torch.istft` applies them in.
    inverse_window: Vec<T>,
    /// Index of the Nyquist bin, which a real signal leaves purely real. Odd transform
    /// lengths have no Nyquist bin, and their top bin keeps its imaginary part.
    nyquist: Option<usize>,
    /// `None` when the window was supplied verbatim by [`Stft::with_window`], which
    /// leaves nothing to regenerate if `n_fft` later changes.
    window_function: Option<WindowFunction<T>>,
    window_periodic: bool,
    /// The un-padded window length. When it is shorter than `n_fft`, `window` holds it
    /// zero-padded and centred, as `torch.stft` does.
    win_length: usize,
    center: bool,
    pad_mode: PadMode,
    normalized: bool,
    onesided: bool,
    forward: Forward<T>,
    inverse: Inverse<T>,
    planner: Planner<T>,
}

impl<T> Stft<T>
where
    T: Float + FftNum,
{
    /// Plan a transform of `n_fft` points advancing `hop_length` samples per frame.
    ///
    /// # Panics
    /// If `n_fft` or `hop_length` is zero. Use [`Stft::try_new`] to handle that as an
    /// error instead.
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        window_function: WindowFunction<T>,
        window_periodic: bool,
    ) -> Self {
        Self::try_new(n_fft, hop_length, window_function, window_periodic)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible [`Stft::new`].
    pub fn try_new(
        n_fft: usize,
        hop_length: usize,
        window_function: WindowFunction<T>,
        window_periodic: bool,
    ) -> Result<Self, String> {
        Self::builder(n_fft)
            .hop_length(hop_length)
            .window(window_function, window_periodic)
            .build()
    }

    /// Start from PyTorch's defaults and change only what you need: `win_length`,
    /// `center`, `pad_mode`, `normalized` and `onesided` all have the same meaning and
    /// the same defaults as in `torch.stft`.
    pub fn builder(n_fft: usize) -> StftBuilder<T> {
        StftBuilder {
            n_fft,
            hop_length: None,
            win_length: None,
            // torch.stft's `window=None` means an all-ones window.
            window: WindowSource::Function(WindowFunction::Rectangular, true),
            center: true,
            pad_mode: PadMode::Reflect,
            normalized: false,
            onesided: true,
        }
    }

    /// Plan a transform that uses `window` exactly as given, `n_fft` being its length.
    ///
    /// The built-in [`WindowFunction`]s are evaluated in PyTorch's order, and on a
    /// platform where both resolve `cos` to the same libm — macOS/aarch64, where PyTorch
    /// has no vectorised f64 cosine — the `f64` windows come out bit-identical. In `f32`
    /// they cannot: PyTorch uses SLEEF there and Rust uses libm's `cosf`, which disagree
    /// on about 5% of arguments by up to 1 ulp. Passing the samples across removes that
    /// as a variable, and is also the way to use a window shape this crate does not
    /// provide.
    pub fn with_window(hop_length: usize, window: Vec<T>) -> Result<Self, String> {
        let n_fft = window.len();
        Self::builder(n_fft)
            .hop_length(hop_length)
            .window_samples(window)
            .build()
    }

    /// Re-plan in place, keeping the cached FFT plans for sizes already seen.
    ///
    /// Changing `n_fft` or `window_periodic` rebuilds the window, so the window always
    /// matches the transform length even when only `n_fft` is supplied. A transform
    /// built by [`Stft::with_window`] has no function to rebuild from, so changing its
    /// `n_fft` requires supplying a `window_function` in the same call.
    pub fn update(
        &mut self,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window_function: Option<WindowFunction<T>>,
        window_periodic: Option<bool>,
    ) -> Result<(), String> {
        let n_fft = n_fft.unwrap_or(self.n_fft);
        let hop_length = hop_length.unwrap_or(self.hop_length);
        validate_params(n_fft, hop_length)?;

        let window_function = match (window_function, self.window_function) {
            (Some(function), _) | (None, Some(function)) => function,
            (None, None) => {
                if n_fft != self.n_fft {
                    return Err(format!(
                        "cannot change n_fft to {n_fft}: this Stft uses a window supplied \
                         to with_window, so there is nothing to rebuild it from. Pass a \
                         window_function too, or build a new Stft."
                    ));
                }
                self.hop_length = hop_length;
                return Ok(());
            }
        };
        let window_periodic = window_periodic.unwrap_or(self.window_periodic);
        let rebuild_window = n_fft != self.n_fft
            || window_periodic != self.window_periodic
            || !self
                .window_function
                .is_some_and(|current| window_function.same_as(&current));

        if n_fft != self.n_fft {
            self.forward = self.planner.plan_forward(n_fft);
            self.inverse = self
                .planner
                .plan_inverse(n_fft, inverse_scale::<T>(n_fft, self.normalized));
            self.nyquist = nyquist_bin(n_fft);
            self.n_fft = n_fft;
        }
        self.hop_length = hop_length;
        if rebuild_window {
            self.window = window_function.new(n_fft, window_periodic);
            self.inverse_window = scaled_window(&self.window, n_fft, self.normalized);
            self.window_function = Some(window_function);
            self.window_periodic = window_periodic;
        }
        Ok(())
    }

    /// The transform length.
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Samples between the start of consecutive frames.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// The analysis window, `n_fft` samples long and zero-padded if `win_length` is
    /// shorter.
    pub fn window(&self) -> &[T] {
        &self.window
    }

    /// The window length before zero-padding.
    pub fn win_length(&self) -> usize {
        self.win_length
    }

    /// Whether frames are centred on their sample index.
    pub fn center(&self) -> bool {
        self.center
    }

    /// Number of frequency bins [`Stft::forward`] produces per frame.
    pub fn num_bins(&self) -> usize {
        if self.onesided {
            self.n_fft / 2 + 1
        } else {
            self.n_fft
        }
    }

    /// Number of frames [`Stft::forward`] will produce for a signal this long.
    pub fn num_frames(&self, signal_length: usize) -> Result<usize, String> {
        self.check_signal_length(signal_length)?;
        Ok((self.padded_length(signal_length) - self.n_fft) / self.hop_length + 1)
    }

    fn padded_length(&self, signal_length: usize) -> usize {
        signal_length + if self.center { 2 * (self.n_fft / 2) } else { 0 }
    }

    /// The signal has to be long enough for the padding mode to draw on, and long
    /// enough to hold one frame. `torch.stft` rejects the same inputs, so refuse them
    /// here rather than panicking on the copy.
    fn check_signal_length(&self, signal_length: usize) -> Result<(), String> {
        if !self.center {
            if signal_length < self.n_fft {
                return Err(format!(
                    "signal length {signal_length} is shorter than n_fft = {} and \
                     center is off, so not one frame fits",
                    self.n_fft
                ));
            }
            return Ok(());
        }
        let pad = self.n_fft / 2;
        let limit = match self.pad_mode {
            // Reflection cannot mirror the edge sample itself; wrapping can.
            PadMode::Reflect => pad >= signal_length,
            PadMode::Circular => pad > signal_length,
            PadMode::Constant | PadMode::Replicate => signal_length == 0,
        };
        if limit {
            return Err(format!(
                "signal length {signal_length} is too short: {:?} padding of n_fft / 2 = \
                 {pad} needs more samples per channel",
                self.pad_mode
            ));
        }
        Ok(())
    }

    /// Forward STFT of a `(channels, samples)` signal.
    ///
    /// Returns `(channels, n_fft / 2 + 1, frames)`, the layout and values of
    /// `torch.stft(..., center=True, onesided=True, return_complex=True)`.
    /// Forward STFT over the last axis, matching `torch.stft`.
    ///
    /// The samples run along the last axis and every axis before it is carried through
    /// untouched, so a `(samples,)` signal gives `(bins, frames)`, `(channels, samples)`
    /// gives `(channels, bins, frames)`, and a batched `(batch, channels, samples)`
    /// gives `(batch, channels, bins, frames)`. `bins` is `n_fft / 2 + 1`, or `n_fft`
    /// when `onesided` is off.
    ///
    /// ```
    /// # use ndarray::{Array1, Array2, Array3};
    /// # use rustft::{Stft, WindowFunction};
    /// let stft = Stft::new(256, 64, WindowFunction::Hann::<f64>, true);
    ///
    /// let mono = Array1::from_shape_fn(4096, |i| (i as f64 * 0.01).sin());
    /// assert_eq!(stft.forward(mono.view()).unwrap().dim(), (129, 65));
    ///
    /// let stereo = Array2::from_shape_fn((2, 4096), |(_, i)| (i as f64 * 0.01).sin());
    /// assert_eq!(stft.forward(stereo.view()).unwrap().dim(), (2, 129, 65));
    ///
    /// let batched = Array3::from_shape_fn((8, 2, 4096), |(_, _, i)| (i as f64 * 0.01).sin());
    /// assert_eq!(stft.forward(batched.view()).unwrap().dim(), (8, 2, 129, 65));
    /// ```
    pub fn forward<D>(&self, input: ArrayView<T, D>) -> Result<Array<Complex<T>, D::Larger>, String>
    where
        D: Dimension,
    {
        let (leading, signal_length) = split_last_axis(input.shape(), "signal")?;
        let rows = leading.iter().product::<usize>().max(1);
        let flat = input
            .to_shape((rows, signal_length))
            .map_err(|e| format!("could not view the input as ({rows}, {signal_length}): {e}"))?;

        let spectrogram = self.forward_rows(flat.view())?;
        let (_, bins, frames) = spectrogram.dim();
        let mut shape = leading.to_vec();
        shape.extend_from_slice(&[bins, frames]);
        reshape(spectrogram, shape)
    }

    /// Inverse STFT over the last two axes, matching `torch.istft(..., length=None)`.
    ///
    /// Undoes the shape change [`Stft::forward`] makes: `(bins, frames)` gives
    /// `(samples,)`, `(channels, bins, frames)` gives `(channels, samples)`, and so on.
    /// `n_fft / 2` samples are trimmed from each end when `center` is on.
    ///
    /// Fails if the window and hop length do not satisfy NOLA — that is, if overlap-add
    /// leaves some output sample with no window energy, so nothing there can be
    /// recovered. `torch.istft` rejects the same case.
    pub fn inverse<D>(&self, input: ArrayView<Complex<T>, D>) -> Result<Array<T, D::Smaller>, String>
    where
        D: Dimension,
    {
        self.inverse_with_length(input, None)
    }

    /// [`Stft::inverse`], trimmed or zero-extended to `length` samples.
    ///
    /// This is `torch.istft`'s `length` argument, including its behaviour when `length`
    /// runs past what the frames cover: the tail is zeros rather than an error.
    pub fn inverse_with_length<D>(
        &self,
        input: ArrayView<Complex<T>, D>,
        length: Option<usize>,
    ) -> Result<Array<T, D::Smaller>, String>
    where
        D: Dimension,
    {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(format!(
                "a spectrogram needs at least a bin and a frame axis, got {} dimension(s)",
                shape.len()
            ));
        }
        let (leading, frames) = split_last_axis(shape, "spectrogram")?;
        let (leading, bins) = split_last_axis(leading, "spectrogram")?;
        let rows = leading.iter().product::<usize>().max(1);
        let flat = input
            .to_shape((rows, bins, frames))
            .map_err(|e| format!("could not view the input as ({rows}, {bins}, {frames}): {e}"))?;

        let signal = self.inverse_rows(flat.view(), length)?;
        let mut shape = leading.to_vec();
        shape.push(signal.dim().1);
        reshape(signal, shape)
    }

    fn forward_rows(&self, input: ArrayView2<T>) -> Result<Array3<Complex<T>>, String> {
        let (num_channels, signal_length) = input.dim();
        self.check_signal_length(signal_length)?;

        let pad = if self.center { self.n_fft / 2 } else { 0 };
        let padded_length = self.padded_length(signal_length);
        let num_frames = (padded_length - self.n_fft) / self.hop_length + 1;
        let bins = self.num_bins();

        let mut output = Array3::zeros((num_channels, bins, num_frames));
        let mut padded = vec![T::zero(); padded_length];
        // Frames are transformed into `staging` frame-major so each spectrum lands in a
        // contiguous run, then transposed in one blocked pass. Writing straight into the
        // frequency-major output would touch a fresh cache line per bin.
        let mut staging = vec![Complex::new(T::zero(), T::zero()); num_frames * bins];
        let parallel = self.use_parallel(num_frames);

        for (ch, channel) in input.outer_iter().enumerate() {
            pad_into(channel, pad, self.pad_mode, &mut padded);
            self.transform_frames(&padded, num_frames, &mut staging)?;
            let mut out_channel = output.index_axis_mut(Axis(0), ch);
            let out_channel = out_channel
                .as_slice_mut()
                .expect("freshly allocated Array3 is contiguous");
            transpose_into(&staging, out_channel, num_frames, bins, parallel);
        }
        Ok(output)
    }

    fn inverse_rows(
        &self,
        input: ArrayView3<Complex<T>>,
        length: Option<usize>,
    ) -> Result<Array2<T>, String> {
        let (num_channels, bins, num_frames) = input.dim();
        let expected = self.num_bins();
        if bins != expected {
            return Err(format!(
                "expected {expected} frequency bins for n_fft = {} ({}), got {bins}",
                self.n_fft,
                if self.onesided { "onesided" } else { "two-sided" }
            ));
        }
        if num_frames == 0 {
            return Err("spectrogram has no frames".to_string());
        }

        let n_freqs = self.n_fft / 2 + 1;
        let covered = (num_frames - 1) * self.hop_length + self.n_fft;
        let start = if self.center { self.n_fft / 2 } else { 0 };
        let output_length = match length {
            Some(length) => length,
            None => covered - start - if self.center { self.n_fft / 2 } else { 0 },
        };
        // The envelope depends only on the window, the hop and the frame count, so it is
        // shared by every channel instead of being rebuilt for each one.
        let envelope = self.window_envelope(num_frames, start, output_length, covered)?;

        let mut output = Array2::zeros((num_channels, output_length));
        let mut spectra = vec![Complex::new(T::zero(), T::zero()); num_frames * n_freqs];
        // The parallel overlap-add materialises every frame before summing them; the
        // sequential one reuses a single frame. Either way the buffer outlives the
        // channel loop so it is allocated once per call, not once per channel.
        let parallel = self.use_parallel(num_frames);
        let frame_capacity = if parallel {
            num_frames * self.n_fft
        } else {
            self.n_fft
        };
        let mut frames = vec![T::zero(); frame_capacity];

        for ch in 0..num_channels {
            // A two-sided spectrogram's upper half is redundant; torch.istft drops it
            // and inverts the first n_fft / 2 + 1 bins, so do the same.
            let channel = input.slice(s![ch, ..n_freqs, ..]);
            // `as_standard_layout` borrows when the slab is already row-major and copies
            // when it is not. `to_owned` would not do: it preserves the source's memory
            // order, so a transposed or reversed view stays non-contiguous.
            let contiguous = channel.as_standard_layout();
            let slab = contiguous
                .as_slice()
                .expect("as_standard_layout is row-major");
            transpose_into(slab, &mut spectra, n_freqs, num_frames, parallel);
            let mut row = output.index_axis_mut(Axis(0), ch);
            let row = row.as_slice_mut().expect("freshly allocated Array2 is contiguous");
            self.overlap_add(&mut spectra, num_frames, start, row, &mut frames)?;
            for (sample, &weight) in row.iter_mut().zip(envelope.iter()) {
                *sample = *sample / weight;
            }
        }
        Ok(output)
    }

    /// Window, transform and store each frame of `padded` into `staging`, frame-major.
    fn transform_frames(
        &self,
        padded: &[T],
        num_frames: usize,
        staging: &mut [Complex<T>],
    ) -> Result<(), String> {
        #[cfg(feature = "parallel")]
        if self.use_parallel(num_frames) {
            return self.transform_frames_parallel(padded, num_frames, staging);
        }

        let mut frame = vec![T::zero(); self.n_fft];
        let mut scratch = self.forward.scratch();
        for (f, spectrum) in staging.chunks_exact_mut(self.num_bins()).enumerate() {
            debug_assert!(f < num_frames);
            self.transform_frame(padded, f, &mut frame, spectrum, &mut scratch)?;
        }
        Ok(())
    }

    /// Every frame reads a distinct window of `padded` and writes a distinct run of
    /// `staging`, so blocks of frames are fully independent.
    #[cfg(feature = "parallel")]
    fn transform_frames_parallel(
        &self,
        padded: &[T],
        num_frames: usize,
        staging: &mut [Complex<T>],
    ) -> Result<(), String> {
        use rayon::prelude::*;

        let bins = self.num_bins();
        staging
            .par_chunks_mut(bins * FRAME_BLOCK)
            .enumerate()
            .try_for_each_init(
                || (vec![T::zero(); self.n_fft], self.forward.scratch()),
                |(frame, scratch), (block, chunk)| {
                    let base = block * FRAME_BLOCK;
                    for (offset, spectrum) in chunk.chunks_exact_mut(bins).enumerate() {
                        debug_assert!(base + offset < num_frames);
                        self.transform_frame(padded, base + offset, frame, spectrum, scratch)?;
                    }
                    Ok(())
                },
            )
    }

    /// Whether a transform of this size is worth spreading across cores. Always false
    /// unless the `parallel` feature is on.
    fn use_parallel(&self, num_frames: usize) -> bool {
        #[cfg(feature = "parallel")]
        {
            num_frames.saturating_mul(self.n_fft) >= PARALLEL_THRESHOLD
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = num_frames;
            false
        }
    }

    fn transform_frame(
        &self,
        padded: &[T],
        index: usize,
        frame: &mut [T],
        spectrum: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), String> {
        let start = index * self.hop_length;
        let source = &padded[start..start + self.n_fft];
        for ((slot, &sample), &weight) in frame.iter_mut().zip(source).zip(&self.window) {
            *slot = sample * weight;
        }
        let n_freqs = self.n_fft / 2 + 1;
        self.forward
            .process(frame, &mut spectrum[..n_freqs], scratch)?;
        if !self.onesided {
            // A real signal's spectrum is conjugate-symmetric, and PyTorch's own
            // two-sided output satisfies that bit for bit, so the upper half is a
            // mirror rather than a second transform.
            for k in 1..self.n_fft.div_ceil(2) {
                spectrum[self.n_fft - k] = spectrum[k].conj();
            }
        }
        if let Some(scale) = self.forward_scale() {
            for value in spectrum.iter_mut() {
                *value = *value * scale;
            }
        }
        Ok(())
    }

    /// `1 / sqrt(n_fft)`, formed entirely in the working precision and multiplied
    /// through after the transform — the order `torch.stft` uses.
    ///
    /// Every step matters at the last bit. Taking the reciprocal in `f64` and narrowing
    /// gives `0.044676706` where `f32` throughout gives `0.044676702`, and dividing
    /// elementwise instead of multiplying by the reciprocal disagrees on two thirds of
    /// an f64 spectrogram.
    fn forward_scale(&self) -> Option<T> {
        self.normalized.then(|| {
            T::one()
                / T::from(self.n_fft as f64)
                    .expect("n_fft is representable")
                    .sqrt()
        })
    }

    /// Overlap-add every frame of `spectra` into `row`, which holds the signal with the
    /// centring pad already removed.
    fn overlap_add(
        &self,
        spectra: &mut [Complex<T>],
        num_frames: usize,
        trim: usize,
        row: &mut [T],
        frames: &mut [T],
    ) -> Result<(), String> {
        #[cfg(feature = "parallel")]
        if self.use_parallel(num_frames) {
            return self.overlap_add_parallel(spectra, num_frames, trim, row, frames);
        }

        let n_freqs = self.n_fft / 2 + 1;
        let frame = &mut frames[..self.n_fft];
        let mut scratch = self.inverse.scratch();

        for (f, spectrum) in spectra.chunks_exact_mut(n_freqs).enumerate() {
            debug_assert!(f < num_frames);
            self.inverse_frame(spectrum, frame, &mut scratch)?;
            let frame_start = f * self.hop_length;
            let Some((lo, hi)) = overlap_bounds(frame_start, self.n_fft, trim, row.len()) else {
                continue;
            };
            let target = &mut row[frame_start + lo - trim..frame_start + hi - trim];
            for (sample, &value) in target.iter_mut().zip(&frame[lo..hi]) {
                *sample = *sample + value;
            }
        }
        Ok(())
    }

    /// Parallel overlap-add, in two passes.
    ///
    /// The inverse FFTs are independent per frame but their outputs overlap, so they
    /// cannot be summed in place concurrently. Materialising every frame first keeps the
    /// expensive pass embarrassingly parallel and lets the summing pass split the output
    /// into disjoint segments without recomputing a single transform.
    #[cfg(feature = "parallel")]
    fn overlap_add_parallel(
        &self,
        spectra: &mut [Complex<T>],
        num_frames: usize,
        trim: usize,
        row: &mut [T],
        frames: &mut [T],
    ) -> Result<(), String> {
        use rayon::prelude::*;

        let n_freqs = self.n_fft / 2 + 1;
        spectra
            .par_chunks_mut(n_freqs * FRAME_BLOCK)
            .zip(frames.par_chunks_mut(self.n_fft * FRAME_BLOCK))
            .try_for_each_init(
                || self.inverse.scratch(),
                |scratch, (spectra, frames)| {
                    for (spectrum, frame) in spectra
                        .chunks_exact_mut(n_freqs)
                        .zip(frames.chunks_exact_mut(self.n_fft))
                    {
                        self.inverse_frame(spectrum, frame, scratch)?;
                    }
                    Ok::<(), String>(())
                },
            )?;

        let segments = rayon::current_num_threads().max(1);
        let segment_length = row.len().div_ceil(segments).max(self.n_fft);
        row.par_chunks_mut(segment_length)
            .enumerate()
            .for_each(|(segment, chunk)| {
                let base = segment * segment_length;
                // Frame `f` covers output samples `[f * hop - trim, f * hop - trim + n_fft)`,
                // so only a contiguous run of frames reaches into this segment.
                let first = match (base + trim).checked_sub(self.n_fft - 1) {
                    Some(reach) => reach.div_ceil(self.hop_length),
                    None => 0,
                };
                for f in first..num_frames {
                    let start = f * self.hop_length;
                    if start >= base + chunk.len() + trim {
                        break;
                    }
                    let (lo, hi) = match overlap_bounds(start, self.n_fft, base + trim, chunk.len())
                    {
                        Some(bounds) => bounds,
                        None => continue,
                    };
                    let offset = start + lo - trim - base;
                    let frame = &frames[f * self.n_fft..][lo..hi];
                    for (sample, &value) in chunk[offset..offset + frame.len()]
                        .iter_mut()
                        .zip(frame)
                    {
                        *sample = *sample + value;
                    }
                }
            });
        Ok(())
    }

    /// Inverse-transform one frame's spectrum, leaving the windowed and normalised
    /// samples in `frame` ready to be summed into the output.
    fn inverse_frame(
        &self,
        spectrum: &mut [Complex<T>],
        frame: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), String> {
        // A real signal's DC bin — and its Nyquist bin, where there is one — carries no
        // imaginary part. Discarding what is there mirrors `irfft`, and keeps rounding
        // noise in a PyTorch spectrogram from being rejected as malformed.
        spectrum[0].im = T::zero();
        if let Some(nyquist) = self.nyquist {
            spectrum[nyquist].im = T::zero();
        }
        self.inverse.process(spectrum, frame, scratch)?;
        for (sample, &weight) in frame.iter_mut().zip(&self.inverse_window) {
            *sample = *sample * weight;
        }
        Ok(())
    }

    /// Sum of squared windows at every output sample, i.e. what overlap-add multiplied
    /// the signal by and what the result has to be divided back out by.
    fn window_envelope(
        &self,
        num_frames: usize,
        start: usize,
        output_length: usize,
        covered: usize,
    ) -> Result<Vec<T>, String> {
        let mut envelope = vec![T::zero(); output_length];
        for f in 0..num_frames {
            let frame_start = f * self.hop_length;
            let Some((lo, hi)) = overlap_bounds(frame_start, self.n_fft, start, output_length)
            else {
                continue;
            };
            let target = &mut envelope[frame_start + lo - start..frame_start + hi - start];
            for (slot, &weight) in target.iter_mut().zip(&self.window[lo..hi]) {
                *slot = *slot + weight * weight;
            }
        }

        // A `length` reaching past the frames leaves a tail no window covers. torch.istft
        // warns and zero-fills it rather than failing, which a unit envelope reproduces:
        // the samples there are already zero.
        let tail = (start + output_length).saturating_sub(covered.max(start));
        let kept = output_length - tail.min(output_length);
        for slot in envelope[kept..].iter_mut() {
            *slot = T::one();
        }

        let epsilon = T::from(NOLA_EPSILON).expect("NOLA epsilon is representable");
        if let Some(index) = envelope[..kept].iter().position(|w| w.abs() < epsilon) {
            return Err(format!(
                "window overlap-add is zero at sample {index}: the window and hop length \
                 {} do not satisfy NOLA, so the transform is not invertible",
                self.hop_length
            ));
        }
        Ok(envelope)
    }
}

/// Transpose a `rows` x `cols` row-major matrix into a `cols` x `rows` one.
///
/// The spectrogram is frequency-major to match PyTorch but the FFTs produce it
/// frame-major, so one of these runs per channel per direction. It is a large enough
/// share of the total to be worth spreading across cores when the rest of the work is.
fn transpose_into<T>(input: &[T], output: &mut [T], rows: usize, cols: usize, parallel: bool)
where
    T: Copy + Send + Sync,
{
    debug_assert_eq!(input.len(), rows * cols);
    debug_assert_eq!(output.len(), rows * cols);
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    #[cfg(feature = "parallel")]
    if parallel && rows * cols * std::mem::size_of::<T>() >= PARALLEL_TRANSPOSE_BYTES {
        use rayon::prelude::*;

        // Each task owns a band of whole output rows. It first gathers the matching
        // column strip of the input into a contiguous scratch buffer — a stride of
        // `cols` between runs, but whole runs at a time — then transposes that strip,
        // which is small enough to stay in cache.
        const BAND: usize = 64;
        output
            .par_chunks_mut(rows * BAND)
            .enumerate()
            .for_each_init(Vec::new, |scratch: &mut Vec<T>, (index, band)| {
                let first = index * BAND;
                let width = band.len() / rows;
                scratch.clear();
                scratch.reserve(width * rows);
                for r in 0..rows {
                    scratch.extend_from_slice(&input[r * cols + first..r * cols + first + width]);
                }
                transpose::transpose(scratch, band, width, rows);
            });
        return;
    }

    transpose::transpose(input, output, cols, rows);
}

/// Split a shape into its leading axes and its last one, which must be non-empty.
fn split_last_axis<'a>(shape: &'a [usize], what: &str) -> Result<(&'a [usize], usize), String> {
    match shape.split_last() {
        Some((&last, leading)) => Ok((leading, last)),
        None => Err(format!("the {what} has no axes")),
    }
}

/// Re-attach the caller's leading axes to a result computed over flattened rows.
fn reshape<A, D>(array: Array<A, impl Dimension>, shape: Vec<usize>) -> Result<Array<A, D>, String>
where
    D: Dimension,
{
    array
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|e| format!("could not reshape the result to {shape:?}: {e}"))?
        .into_dimensionality::<D>()
        .map_err(|e| format!("could not restore the caller's dimensionality: {e}"))
}

/// Half-open range of a frame's samples that land inside the trimmed output, or `None`
/// when the frame falls entirely outside it — which happens at both ends once `length`
/// stops short of what the frames cover.
fn overlap_bounds(
    start: usize,
    n_fft: usize,
    trim: usize,
    output_length: usize,
) -> Option<(usize, usize)> {
    let lo = trim.saturating_sub(start);
    let hi = n_fft.min((trim + output_length).saturating_sub(start));
    (lo < hi).then_some((lo, hi))
}

fn validate_params(n_fft: usize, hop_length: usize) -> Result<(), String> {
    if n_fft == 0 {
        return Err("n_fft must be greater than zero".to_string());
    }
    if hop_length == 0 {
        return Err("hop_length must be greater than zero".to_string());
    }
    Ok(())
}

/// The bin holding the Nyquist frequency, which is real. An odd-length transform does
/// not reach Nyquist, so every bin above DC keeps its imaginary part.
fn nyquist_bin(n_fft: usize) -> Option<usize> {
    (n_fft % 2 == 0).then_some(n_fft / 2)
}

/// `torch.istft` scales the inverse transform by `1 / n_fft`, or by `1 / sqrt(n_fft)`
/// when `normalized` is on — an orthonormal transform, not a scaled spectrum.
fn inverse_scale<T: Float + FftNum>(n_fft: usize, normalized: bool) -> T {
    let n = T::from(n_fft as f64).expect("n_fft is representable");
    if normalized {
        T::one() / n.sqrt()
    } else {
        T::one() / n
    }
}

/// The window the inverse re-applies. A backend that can scale inside the transform has
/// already done so; one that cannot gets the factor folded in here, which costs nothing
/// because every frame is re-windowed anyway.
fn scaled_window<T: Float + FftNum>(window: &[T], n_fft: usize, normalized: bool) -> Vec<T> {
    if backend::INVERSE_IS_NORMALISED {
        return window.to_vec();
    }
    let scale = inverse_scale::<T>(n_fft, normalized);
    window.iter().map(|&w| w * scale).collect()
}

#[derive(Debug, Clone, Copy)]
pub enum WindowFunction<T>
where
    T: Float + FftNum,
{
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Gaussian(T), // Standard deviation
    Triangular,
    Bartlett,
    FlatTop,
}

impl<T> WindowFunction<T>
where
    T: Float + FftNum,
{
    /// Sample this window over `size` points.
    // Named `new` since 0.0.1 and part of the public API, so it stays despite building a
    // Vec rather than a Self.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(&self, size: usize, periodic: bool) -> Vec<T>
    where
        T: Float + FftNum,
    {
        // A periodic window of length N is the symmetric window of length N + 1 with its
        // last point dropped, so it tiles without a seam under overlap-add.
        let m = if periodic { size + 1 } else { size };
        let denominator = if m > 1 { (m - 1) as f64 } else { 1.0 };

        // PyTorch builds float32 windows entirely in float32, with SLEEF's cosine rather
        // than libm's. Evaluating in f64 and narrowing is more accurate but does not
        // match, so for f32 we follow PyTorch step for step instead.
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            if let Some(samples) = self.samples_f32(size, m, denominator) {
                return samples
                    .into_iter()
                    .map(|v| T::from(v).expect("f32 converts into T = f32"))
                    .collect();
            }
        }

        // PyTorch folds the constant once and multiplies by the index -- `n * (2pi / N)`
        // rather than `2pi * (n / N)`. The two differ by up to 2 ulp whenever `n / N` is
        // inexact, i.e. for every transform length that is not a power of two, and that
        // difference propagates through the whole spectrogram.
        let step = 2.0 * PI / denominator;
        let half_step = PI / denominator;

        let mut window = Vec::with_capacity(size);
        for n in 0..size {
            let k = n as f64;
            let value = match self {
                WindowFunction::Rectangular => T::one(),
                WindowFunction::Hann => T::from(0.5 - 0.5 * (k * step).cos())
                    .expect("Failed to create Hann window"),
                WindowFunction::Hamming => T::from(0.54 - 0.46 * (k * step).cos())
                    .expect("Failed to create Hamming window"),
                WindowFunction::Blackman => {
                    // Summed in PyTorch's order, which is not the textbook one.
                    let w = k * half_step;
                    T::from((4.0 * w).cos() * 0.08 - (2.0 * w).cos() * 0.5 + 0.42)
                        .expect("Failed to create Blackman window")
                }
                WindowFunction::Gaussian(sigma) => {
                    let x = k / denominator;
                    let alpha = T::one() / *sigma;
                    (T::from(-0.5).expect("Failed to create Gaussian window")
                        * (alpha * (T::from(x - 0.5).expect("Failed to create Gaussian window")))
                            .powi(2))
                    .exp()
                }
                WindowFunction::Triangular => T::from(1.0 - (2.0 * (k / denominator) - 1.0).abs())
                    .expect("Failed to create Triangular window"),
                WindowFunction::Bartlett => {
                    // PyTorch splits on the index rather than on the value, and reflects
                    // the upper half as `2 - w`; `1 - |w - 1|` rounds differently.
                    let w = k * (2.0 / denominator);
                    let first_half = ((m - 1) >> 1) + 1;
                    let value = if n < first_half { w } else { 2.0 - w };
                    T::from(value).expect("Failed to create Bartlett window")
                }
                WindowFunction::FlatTop => {
                    let a = k * step;
                    T::from(
                        0.21557895 - 0.41663158 * a.cos() + 0.277263158 * (2.0 * a).cos()
                            - 0.083578947 * (3.0 * a).cos()
                            + 0.006947368 * (4.0 * a).cos(),
                    )
                    .expect("Failed to create FlatTop window")
                }
            };
            window.push(value);
        }
        window
    }

    /// The four windows PyTorch also provides, evaluated the way it evaluates them in
    /// float32: the scale factor computed in `f64` and narrowed once, everything after
    /// that in `f32`, and SLEEF's cosine rather than libm's.
    ///
    /// Returns `None` for the shapes PyTorch has no counterpart for, which fall back to
    /// the `f64` evaluation because there is nothing to match and it is more accurate.
    fn samples_f32(&self, size: usize, m: usize, denominator: f64) -> Option<Vec<f32>> {
        let step = (2.0 * PI / denominator) as f32;
        let half_step = (PI / denominator) as f32;
        let mut window = Vec::with_capacity(size);
        for n in 0..size {
            let k = n as f32;
            window.push(match self {
                WindowFunction::Rectangular => 1.0,
                WindowFunction::Hann => sleef::cosf(k * step) * -0.5 + 0.5,
                WindowFunction::Hamming => sleef::cosf(k * step) * -0.46 + 0.54,
                WindowFunction::Blackman => {
                    let w = k * half_step;
                    sleef::cosf(4.0 * w) * 0.08 - sleef::cosf(2.0 * w) * 0.5 + 0.42
                }
                WindowFunction::Bartlett => {
                    let w = k * (2.0 / denominator) as f32;
                    let first_half = ((m - 1) >> 1) + 1;
                    if n < first_half {
                        w
                    } else {
                        2.0 - w
                    }
                }
                _ => return None,
            });
        }
        Some(window)
    }

    /// Whether two window functions would produce the same samples.
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (WindowFunction::Gaussian(a), WindowFunction::Gaussian(b)) => a == b,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

/// Pad `signal` by `pad` samples on both sides into `out`, matching
/// `torch.nn.functional.pad`: reflection does not repeat the edge sample.
///
/// `out` must be `signal.len() + 2 * pad` long; the callers check `pad` against the
/// signal length for the modes that constrain it.
fn pad_into<T>(signal: ArrayView1<T>, pad: usize, mode: PadMode, out: &mut [T])
where
    T: Float + FftNum,
{
    let length = signal.len();
    debug_assert_eq!(out.len(), length + 2 * pad);

    for (slot, &sample) in out[pad..pad + length].iter_mut().zip(signal.iter()) {
        *slot = sample;
    }
    // Mirror or copy out of the run just written rather than back through the array
    // view, so both sides of every read are contiguous.
    for i in 0..pad {
        out[pad - 1 - i] = match mode {
            PadMode::Reflect => out[pad + i + 1],
            PadMode::Constant => T::zero(),
            PadMode::Replicate => out[pad],
            PadMode::Circular => out[pad + length - 1 - i],
        };
    }
    for i in 0..pad {
        out[pad + length + i] = match mode {
            PadMode::Reflect => out[pad + length - 2 - i],
            PadMode::Constant => T::zero(),
            PadMode::Replicate => out[pad + length - 1],
            PadMode::Circular => out[pad + i],
        };
    }
}

/// Forward FFT of a real signal, returning the full complex spectrum.
///
/// Plans the transform on every call; use [`Stft`] when transforming repeatedly.
pub fn fft<T>(input: ArrayView1<T>) -> Result<Array1<Complex<T>>, String>
where
    T: Float + FftNum,
{
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(input.len());
    let mut buffer: Vec<Complex<T>> = input.iter().map(|&x| Complex::new(x, T::zero())).collect();
    fft.process(&mut buffer);
    Ok(Array1::from_vec(buffer))
}

/// Inverse FFT, normalised by the transform length and reduced to its real part.
///
/// Plans the transform on every call; use [`Stft`] when transforming repeatedly.
pub fn ifft<T>(input: ArrayView1<Complex<T>>) -> Result<Array1<T>, String>
where
    T: Float + FftNum,
{
    let mut planner = FftPlanner::new();
    let len = input.len();
    let ifft = planner.plan_fft_inverse(len);
    let mut buffer = input.to_vec();
    ifft.process(&mut buffer);
    let scale = T::from(len).expect("Failed to convert len to T");
    Ok(buffer.into_iter().map(|c| c.re / scale).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array2, Axis};

    fn assert_send_sync<T: Send + Sync>() {}

    fn signal(channels: usize, length: usize) -> Array2<f64> {
        // Deterministic, broadband and not periodic in n_fft, so frame boundaries and
        // window tails all carry energy.
        Array2::from_shape_fn((channels, length), |(c, i)| {
            let t = i as f64;
            (t * 0.017 + c as f64).sin() + 0.5 * (t * 0.31).cos() + 0.25 * (t * 1.7).sin()
        })
    }

    fn pad_reflect(signal: Array1<f64>, pad: usize) -> Array1<f64> {
        let mut out = vec![0.0; signal.len() + 2 * pad];
        pad_into(signal.view(), pad, PadMode::Reflect, &mut out);
        Array1::from_vec(out)
    }

    #[test]
    fn stft_is_send_and_sync() {
        assert_send_sync::<Stft<f64>>();
    }

    #[test]
    fn test_apply_padding() {
        let padded = pad_reflect(arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]), 2);
        assert_eq!(padded, arr1(&[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]));
    }

    #[test]
    fn padding_of_zero_is_the_identity() {
        let padded = pad_reflect(arr1(&[1.0, 2.0, 3.0]), 0);
        assert_eq!(padded, arr1(&[1.0, 2.0, 3.0]));
    }

    /// The forward transform is checked against a direct DFT of the windowed frames,
    /// which is the definition `torch.stft` implements.
    #[test]
    fn forward_matches_a_direct_dft() {
        let (n_fft, hop) = (16, 4);
        let stft = Stft::new(n_fft, hop, WindowFunction::Hann::<f64>, true);
        let input = signal(1, 64);
        let spectrogram = stft.forward(input.view()).unwrap();

        let pad = n_fft / 2;
        let mut padded = vec![0.0; input.dim().1 + 2 * pad];
        pad_into(input.index_axis(Axis(0), 0), pad, PadMode::Reflect, &mut padded);

        for frame in 0..spectrogram.dim().2 {
            for bin in 0..n_fft / 2 + 1 {
                let mut expected = Complex::new(0.0, 0.0);
                for i in 0..n_fft {
                    let angle = -2.0 * PI * (bin * i) as f64 / n_fft as f64;
                    let sample = padded[frame * hop + i] * stft.window()[i];
                    expected += Complex::new(angle.cos(), angle.sin()) * sample;
                }
                let actual = spectrogram[[0, bin, frame]];
                assert!(
                    (actual - expected).norm() < 1e-10,
                    "bin {bin} of frame {frame}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_is_exact_for_even_and_odd_n_fft() {
        // Odd n_fft has no Nyquist bin; treating bin n_fft / 2 as one, and failing to
        // mirror it into the upper half, used to corrupt the whole inverse.
        for (n_fft, hop) in [(1024, 256), (512, 128), (501, 125), (101, 25), (63, 16), (7, 2)] {
            let stft = Stft::new(n_fft, hop, WindowFunction::Hann::<f64>, true);
            let input = signal(2, 4096);
            let spectrogram = stft.forward(input.view()).unwrap();
            let recovered = stft.inverse(spectrogram.view()).unwrap();

            // Odd n_fft trims n_fft / 2 from each end, so one sample survives in the middle.
            let expected = hop * (spectrogram.dim().2 - 1) + n_fft - 2 * (n_fft / 2);
            assert_eq!(recovered.dim().1, expected);
            let error = input
                .slice(s![.., ..recovered.dim().1])
                .iter()
                .zip(recovered.iter())
                .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
            assert!(error < 1e-12, "n_fft {n_fft}: roundtrip error {error:e}");
        }
    }

    /// Big enough to cross `PARALLEL_THRESHOLD`, so `--features parallel` actually
    /// exercises the blocked transforms and the segmented overlap-add. The shapes vary
    /// how many frames overlap each output sample and whether the output divides evenly
    /// into segments.
    #[test]
    fn large_roundtrip_exercises_the_parallel_path() {
        for (n_fft, hop, length) in [(512, 128, 200_000), (512, 64, 200_003), (1024, 384, 150_000)]
        {
            let stft = Stft::new(n_fft, hop, WindowFunction::Hann::<f64>, true);
            let input = signal(2, length);
            let spectrogram = stft.forward(input.view()).unwrap();
            let recovered = stft.inverse(spectrogram.view()).unwrap();
            let error = input
                .slice(s![.., ..recovered.dim().1])
                .iter()
                .zip(recovered.iter())
                .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
            assert!(error < 1e-12, "n_fft {n_fft} hop {hop}: roundtrip error {error:e}");
        }
    }

    #[test]
    fn roundtrip_holds_for_every_window() {
        let windows = [
            WindowFunction::Rectangular,
            WindowFunction::Hann,
            WindowFunction::Hamming,
            WindowFunction::Blackman,
            WindowFunction::Gaussian(0.3),
            WindowFunction::Triangular,
            WindowFunction::Bartlett,
            WindowFunction::FlatTop,
        ];
        let input = signal(1, 2048);
        for window in windows {
            for periodic in [true, false] {
                let stft = Stft::new(256, 64, window, periodic);
                let spectrogram = stft.forward(input.view()).unwrap();
                let recovered = stft.inverse(spectrogram.view()).unwrap();
                let error = input
                    .slice(s![.., ..recovered.dim().1])
                    .iter()
                    .zip(recovered.iter())
                    .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
                assert!(error < 1e-11, "{window:?} periodic={periodic}: error {error:e}");
            }
        }
    }

    #[test]
    fn f32_roundtrips_too() {
        let stft = Stft::new(256, 64, WindowFunction::Hann::<f32>, true);
        let input = Array2::from_shape_fn((1, 2048), |(_, i)| (i as f32 * 0.017).sin());
        let spectrogram = stft.forward(input.view()).unwrap();
        let recovered = stft.inverse(spectrogram.view()).unwrap();
        let error = input
            .slice(s![.., ..recovered.dim().1])
            .iter()
            .zip(recovered.iter())
            .fold(0.0_f32, |worst, (a, b)| worst.max((a - b).abs()));
        assert!(error < 1e-4, "f32 roundtrip error {error:e}");
    }

    /// PyTorch hands back spectrograms whose numpy views are not always row-major, and
    /// the reversed frequency axis below leaves each channel's slab non-contiguous —
    /// which `to_owned` alone does not fix, because it preserves the source's layout.
    #[test]
    fn a_non_contiguous_spectrogram_inverts_the_same() {
        let stft = Stft::new(128, 32, WindowFunction::Hann::<f64>, true);
        let spectrogram = stft.forward(signal(2, 1024).view()).unwrap();
        for view in [
            spectrogram.slice(s![.., ..;-1, ..]),
            spectrogram.slice(s![..;-1, .., ..]),
            spectrogram.slice(s![.., .., ..;-1]),
        ] {
            let repacked = view.as_standard_layout().into_owned();
            assert_eq!(stft.inverse(view).unwrap(), stft.inverse(repacked.view()).unwrap());
        }
    }

    #[test]
    fn a_signal_shorter_than_the_pad_is_an_error_not_a_panic() {
        let stft = Stft::new(1024, 256, WindowFunction::Hann::<f64>, true);
        let error = stft.forward(signal(1, 300).view()).unwrap_err();
        assert!(error.contains("too short"), "{error}");
    }

    #[test]
    fn a_window_failing_nola_is_an_error_not_silent_nans() {
        // A Hann window advanced by its full length leaves zeros between frames, so the
        // signal there cannot be recovered. torch.istft rejects this case as well.
        let stft = Stft::new(256, 256, WindowFunction::Hann::<f64>, true);
        let spectrogram = stft.forward(signal(1, 4096).view()).unwrap();
        let error = stft.inverse(spectrogram.view()).unwrap_err();
        assert!(error.contains("NOLA"), "{error}");
    }

    #[test]
    fn degenerate_parameters_are_rejected() {
        assert!(Stft::try_new(0, 1, WindowFunction::Hann::<f64>, true).is_err());
        assert!(Stft::try_new(16, 0, WindowFunction::Hann::<f64>, true).is_err());
    }

    #[test]
    fn a_mismatched_bin_count_is_rejected() {
        let stft = Stft::new(128, 32, WindowFunction::Hann::<f64>, true);
        let spectrogram = stft.forward(signal(1, 1024).view()).unwrap();
        let mut wider = Stft::new(256, 32, WindowFunction::Hann::<f64>, true);
        let error = wider.inverse(spectrogram.view()).unwrap_err();
        assert!(error.contains("frequency bins"), "{error}");
        wider.update(Some(128), None, None, None).unwrap();
        assert!(wider.inverse(spectrogram.view()).is_ok());
    }

    /// Changing n_fft alone used to leave the previous window in place, so the next
    /// transform windowed with the wrong length.
    #[test]
    fn update_rebuilds_the_window_for_the_new_length() {
        let mut stft = Stft::new(64, 16, WindowFunction::Hann::<f64>, true);
        stft.update(Some(256), Some(64), None, None).unwrap();
        assert_eq!(stft.window().len(), 256);
        assert_eq!(stft.window(), Stft::new(256, 64, WindowFunction::Hann::<f64>, true).window());

        let input = signal(1, 2048);
        let recovered = stft.inverse(stft.forward(input.view()).unwrap().view()).unwrap();
        let error = input
            .slice(s![.., ..recovered.dim().1])
            .iter()
            .zip(recovered.iter())
            .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
        assert!(error < 1e-12, "roundtrip after update: {error:e}");
    }

    #[test]
    fn a_supplied_window_is_used_verbatim() {
        let window = WindowFunction::Hann::<f64>.new(256, true);
        let custom = Stft::with_window(64, window.clone()).unwrap();
        assert_eq!(custom.window(), &window[..]);
        assert_eq!(custom.n_fft(), 256);
        assert_eq!(custom.hop_length(), 64);

        // Identical to the generated-window transform it was copied from.
        let generated = Stft::new(256, 64, WindowFunction::Hann::<f64>, true);
        let input = signal(2, 4096);
        assert_eq!(
            custom.forward(input.view()).unwrap(),
            generated.forward(input.view()).unwrap()
        );

        // And an arbitrary shape still round-trips, so long as it satisfies NOLA.
        let ramp: Vec<f64> = (0..256).map(|i| 0.25 + i as f64 / 512.0).collect();
        let stft = Stft::with_window(64, ramp).unwrap();
        let recovered = stft.inverse(stft.forward(input.view()).unwrap().view()).unwrap();
        let error = input
            .slice(s![.., ..recovered.dim().1])
            .iter()
            .zip(recovered.iter())
            .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
        assert!(error < 1e-12, "custom window roundtrip error {error:e}");
    }

    #[test]
    fn a_supplied_window_cannot_be_rebuilt_at_a_new_length() {
        let mut stft = Stft::with_window(64, WindowFunction::Hann::<f64>.new(256, true)).unwrap();
        // Hop alone is fine: the window does not depend on it.
        stft.update(None, Some(32), None, None).unwrap();
        assert_eq!(stft.hop_length(), 32);

        let error = stft.update(Some(512), None, None, None).unwrap_err();
        assert!(error.contains("with_window"), "{error}");
        assert_eq!(stft.n_fft(), 256);

        // Supplying a function in the same call gives it something to rebuild from.
        stft.update(Some(512), None, Some(WindowFunction::Hann), Some(true)).unwrap();
        assert_eq!(stft.window(), Stft::new(512, 32, WindowFunction::Hann::<f64>, true).window());
    }

    #[test]
    fn an_empty_window_is_rejected() {
        assert!(Stft::with_window(16, Vec::<f64>::new()).is_err());
        assert!(Stft::with_window(0, vec![1.0_f64; 16]).is_err());
    }

    #[test]
    fn update_rejects_degenerate_parameters_without_changing_state() {
        let mut stft = Stft::new(64, 16, WindowFunction::Hann::<f64>, true);
        assert!(stft.update(Some(0), None, None, None).is_err());
        assert_eq!(stft.n_fft(), 64);
        assert_eq!(stft.hop_length(), 16);
    }

    #[test]
    fn periodic_and_symmetric_windows_differ_as_documented() {
        let periodic = WindowFunction::Hann::<f64>.new(8, true);
        let symmetric = WindowFunction::Hann::<f64>.new(8, false);
        assert!((periodic[0] - 0.0).abs() < 1e-15 && (symmetric[0] - 0.0).abs() < 1e-15);
        // Only the symmetric window returns to zero at its last sample.
        assert!(symmetric[7].abs() < 1e-15);
        assert!(periodic[7] > 0.1);
    }

    /// What this pins is the *evaluation order*: PyTorch folds the constant once as
    /// `n * (2pi / N)`, where rustft used to compute `2pi * (n / N)`. The two differ by
    /// up to 2 ulp whenever `n / N` is inexact — so at every transform length that is
    /// not a power of two — and that drift multiplies through the whole spectrogram.
    /// n = 6 is the smallest periodic size where the two orders actually diverge; at
    /// most sizes, including every power of two, they agree and the bug would hide.
    ///
    /// Written against a locally evaluated reference rather than captured PyTorch bits,
    /// because the order is a property of this crate while the last bit of `cos` is a
    /// property of the platform's libm. `windows_are_close_to_pytorch` covers the values.
    #[test]
    fn windows_use_pytorchs_evaluation_order() {
        const N: usize = 6;
        let k = |n: usize| n as f64;
        let step = 2.0 * PI / N as f64;
        let folded: Vec<f64> = (0..N).map(|n| 0.5 - 0.5 * (k(n) * step).cos()).collect();
        let unfolded: Vec<f64> = (0..N).map(|n| 0.5 * (1.0 - (2.0 * PI * (k(n) / N as f64)).cos())).collect();

        assert_eq!(WindowFunction::Hann::<f64>.new(N, true), folded);
        // The old form really is different here, so the assertion above has teeth.
        assert_ne!(folded, unfolded);
    }

    /// Values captured from PyTorch 2.8: `torch.<name>_window(9, periodic=<p>,
    /// dtype=torch.float64)`. These agree bit for bit on any platform where Rust and
    /// PyTorch resolve `cos` to the same libm — which is the case on macOS/aarch64,
    /// where PyTorch has no vectorised f64 cosine and falls back to `std::cos`. The
    /// tolerance below keeps the test honest elsewhere while still catching a wrong
    /// formula, which would be off by far more than an ulp.
    #[test]
    fn windows_are_close_to_pytorch() {
        #[allow(clippy::type_complexity)]
        let expected: &[(WindowFunction<f64>, bool, &[f64])] = &[
            (WindowFunction::Hann, true, &[0.0, 0.116977778440511, 0.4131759111665348, 0.7499999999999999, 0.9698463103929542, 0.9698463103929542, 0.7500000000000002, 0.413175911166535, 0.1169777784405111]),
            (WindowFunction::Hann, false, &[0.0, 0.1464466094067262, 0.49999999999999994, 0.8535533905932737, 1.0, 0.8535533905932738, 0.5000000000000001, 0.14644660940672632, 0.0]),
            (WindowFunction::Hamming, true, &[0.08000000000000002, 0.18761955616527015, 0.46012183827321207, 0.77, 0.9722586055615179, 0.9722586055615179, 0.7700000000000002, 0.46012183827321224, 0.18761955616527026]),
            (WindowFunction::Hamming, false, &[0.08000000000000002, 0.21473088065418816, 0.54, 0.8652691193458119, 1.0, 0.865269119345812, 0.5400000000000001, 0.21473088065418822, 0.08000000000000002]),
            (WindowFunction::Blackman, true, &[0.0, 0.050869632653865404, 0.2580005015036621, 0.6299999999999999, 0.9511298658424723, 0.9511298658424725, 0.6300000000000003, 0.25800050150366227, 0.05086963265386546]),
            (WindowFunction::Blackman, false, &[0.0, 0.0664466094067262, 0.33999999999999997, 0.7735533905932737, 1.0, 0.7735533905932739, 0.3400000000000001, 0.06644660940672625, 0.0]),
            (WindowFunction::Bartlett, true, &[0.0, 0.2222222222222222, 0.4444444444444444, 0.6666666666666666, 0.8888888888888888, 0.8888888888888888, 0.6666666666666667, 0.44444444444444464, 0.22222222222222232]),
            (WindowFunction::Bartlett, false, &[0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]),
        ];
        for (function, periodic, want) in expected {
            for (got, want) in function.new(want.len(), *periodic).iter().zip(want.iter()) {
                let tolerance = 4.0 * f64::EPSILON;
                assert!(
                    (got - want).abs() <= tolerance,
                    "{function:?} periodic={periodic}: {got} vs PyTorch's {want}"
                );
            }
        }
    }

    /// PyTorch's exact bits for a small f32 case, so a regression in the pocketfft
    /// backend, the framing or the overlap-add order shows up immediately. Generated by
    /// `torch.stft` / `torch.istft` 2.8 with a periodic Hann window, n_fft 16, hop 4.
    ///
    /// The window is passed in verbatim: in f32 PyTorch builds its windows with a
    /// vectorised cosine that Rust's libm does not reproduce, so matching it is only
    /// possible when the caller hands the same samples across.
    #[cfg(feature = "pocketfft")]
    #[test]
    fn f32_output_is_bit_identical_to_pytorch() {
        const SIGNAL: [f32; 48] = [
        2.041_f32, -2.556_f32, 0.418_f32, -0.568_f32, -0.453_f32, -0.216_f32, -2.02_f32,
        -0.232_f32, -0.865_f32, 3.323_f32, 0.226_f32, -0.353_f32, -0.281_f32, -0.668_f32,
        -1.055_f32, -0.391_f32, 0.482_f32, -0.239_f32, 0.958_f32, -0.2_f32, 0.024_f32,
        1.546_f32, 0.545_f32, -0.505_f32, -0.183_f32, 0.541_f32, 1.935_f32, -0.27_f32,
        -0.244_f32, 1.002_f32, -0.886_f32, -0.292_f32, 0.883_f32, 0.58_f32, 0.092_f32,
        0.67_f32, -2.828_f32, 1.021_f32, -0.96_f32, -1.669_f32, 0.276_f32, 0.701_f32,
        -0.445_f32, -1.076_f32, 0.026_f32, -0.053_f32, 1.406_f32, 0.747_f32,
        ];
        const WINDOW: [f32; 16] = [
        0.0_f32, 0.038060248_f32, 0.14644662_f32, 0.3086583_f32, 0.5_f32, 0.69134176_f32,
        0.8535534_f32, 0.9619398_f32, 1.0_f32, 0.96193975_f32, 0.8535533_f32, 0.6913416_f32,
        0.5_f32, 0.30865818_f32, 0.1464465_f32, 0.038060218_f32,
        ];
        const SPECTROGRAM: [f32; 234] = [
        -4.143874_f32, 0.0_f32, -3.6180155_f32, 0.0_f32, -0.7702439_f32, 0.0_f32,
        -0.4519595_f32, 0.0_f32, -0.34700817_f32, 0.0_f32, 2.3762746_f32, 0.0_f32,
        2.9784722_f32, 0.0_f32, 2.0784388_f32, 0.0_f32, 0.43389654_f32, 0.0_f32,
        -2.1871107_f32, 0.0_f32, -2.833099_f32, 0.0_f32, 0.9319134_f32, 0.0_f32, 2.1285114_f32,
        0.0_f32, 1.8123953_f32, 2.4250443e-07_f32, 2.9791532_f32, 0.34921455_f32,
        -0.9642935_f32, 2.5132208_f32, 0.7327849_f32, -2.281183_f32, 0.685923_f32,
        2.2411294_f32, -1.854655_f32, 0.47226197_f32, -1.5998758_f32, 0.08112335_f32,
        -0.952801_f32, -0.983695_f32, -0.952074_f32, -0.55660975_f32, 2.045055_f32,
        -1.9471339_f32, 1.7214844_f32, 1.2032108_f32, -0.2877967_f32, 2.6205273_f32,
        -2.6841967_f32, -1.4694276_f32, -0.34601784_f32, 2.9802322e-07_f32, -0.40353882_f32,
        1.9662272_f32, 2.0730877_f32, -4.4474735_f32, -2.1256917_f32, 3.5582814_f32,
        0.42176408_f32, -1.6922156_f32, 1.0957654_f32, -0.053067148_f32, -0.86161935_f32,
        -1.0192982_f32, -0.17455292_f32, 2.0967712_f32, 1.6032685_f32, -1.4672788_f32,
        -1.8161892_f32, 1.9840039_f32, 0.9465089_f32, -0.9873897_f32, -1.9164436_f32,
        -1.8575933_f32, 2.695263_f32, 2.651798_f32, 0.154598_f32, 1.828998e-07_f32,
        -2.3048518_f32, -1.0272243_f32, -1.3435565_f32, 4.4006186_f32, 2.2674465_f32,
        -0.43423158_f32, -1.2420307_f32, 0.74784386_f32, -0.23511487_f32, 1.9188386_f32,
        2.2376719_f32, 2.0481923_f32, 0.7116309_f32, -1.0906651_f32, -0.73185694_f32,
        2.2150834_f32, 1.0061857_f32, 0.43081468_f32, -1.1145238_f32, 1.7722217_f32,
        2.4814503_f32, 1.2866514_f32, -1.4079585_f32, -1.2562978_f32, 1.4660733_f32, 0.0_f32,
        1.4080806_f32, -0.5383433_f32, 0.3925612_f32, -3.4011433_f32, 0.3906221_f32,
        -2.2618628_f32, 0.32338446_f32, -0.55450344_f32, -1.2382636_f32, -2.1359653_f32,
        -2.4203565_f32, -2.6346898_f32, -0.882664_f32, -2.1677098_f32, -0.118064165_f32,
        -1.5509279_f32, -1.3126955_f32, -2.278055_f32, -0.14513433_f32, -3.5000248_f32,
        0.18351656_f32, -1.5308669_f32, -0.001038909_f32, -0.57637906_f32, -2.3907454_f32,
        -1.828998e-07_f32, 1.1436903_f32, -0.9505336_f32, 1.0399338_f32, 2.1665251_f32,
        -2.4861903_f32, 1.8423083_f32, 0.0012620687_f32, -1.443129_f32, 1.8191428_f32,
        1.133125_f32, 1.107041_f32, 0.4842189_f32, 0.9106974_f32, 3.4744022_f32,
        -2.1945152_f32, -0.9498932_f32, 3.877705_f32, 1.0023568_f32, -1.4437071_f32,
        3.7324238_f32, -1.1799831_f32, -0.02191025_f32, 0.3595366_f32, -0.21081436_f32,
        5.3340178_f32, -2.9802322e-07_f32, -1.678461_f32, -2.1394634_f32, -3.069088_f32,
        -0.18188024_f32, 1.9466918_f32, 0.4992422_f32, 0.7992359_f32, 1.6507572_f32,
        -1.3467653_f32, -1.6338527_f32, 0.7156193_f32, 1.8936747_f32, -1.0134472_f32,
        -2.586296_f32, 3.2347317_f32, 1.0501972_f32, -4.998811_f32, 0.05896175_f32,
        2.4074912_f32, -0.4930909_f32, 0.28644353_f32, 1.0134683_f32, 0.5357369_f32,
        0.08481455_f32, -7.7402477_f32, -1.2329515e-07_f32, -0.0059919357_f32, 6.237905_f32,
        4.7279162_f32, -0.06487286_f32, 0.6099587_f32, -2.6986434_f32, -1.3731543_f32,
        -0.55984354_f32, 0.17462724_f32, 1.0165482_f32, -1.012837_f32, -0.94684994_f32,
        0.30647278_f32, 1.4493723_f32, 0.34644598_f32, 1.4464139_f32, 4.3830547_f32,
        -0.16159189_f32, -0.26725364_f32, -2.5445874_f32, -1.1176704_f32, -0.94803417_f32,
        -1.8913815_f32, 0.5180558_f32, 7.5637274_f32, 0.0_f32, 1.3418539_f32, 0.0_f32,
        -4.9428787_f32, 0.0_f32, -2.2192845_f32, 0.0_f32, 1.1142392_f32, 0.0_f32,
        0.7942525_f32, 0.0_f32, 0.69024074_f32, 0.0_f32, 0.110889316_f32, 0.0_f32,
        -2.8097682_f32, 0.0_f32, -4.181499_f32, 0.0_f32, -1.3766322_f32, 0.0_f32,
        2.1690536_f32, 0.0_f32, 2.6595669_f32, 0.0_f32,
        ];
        const RECOVERED: [f32; 48] = [
        2.041_f32, -2.556_f32, 0.41800007_f32, -0.5680001_f32, -0.45299998_f32,
        -0.21599998_f32, -2.0199997_f32, -0.23199995_f32, -0.86500007_f32, 3.3229997_f32,
        0.22599997_f32, -0.353_f32, -0.28099996_f32, -0.6679999_f32, -1.055_f32,
        -0.39099997_f32, 0.482_f32, -0.23899996_f32, 0.958_f32, -0.2_f32, 0.023999989_f32,
        1.546_f32, 0.545_f32, -0.505_f32, -0.18299998_f32, 0.54099995_f32, 1.935_f32,
        -0.26999998_f32, -0.244_f32, 1.002_f32, -0.8859999_f32, -0.29199997_f32,
        0.88299996_f32, 0.58_f32, 0.092000015_f32, 0.66999996_f32, -2.828_f32, 1.021_f32,
        -0.96_f32, -1.6690001_f32, 0.27600002_f32, 0.7009999_f32, -0.44499993_f32, -1.076_f32,
        0.025999993_f32, -0.05299999_f32, 1.4059999_f32, 0.74699986_f32,
        ];

        let stft = Stft::with_window(4, WINDOW.to_vec()).unwrap();
        let input = Array2::from_shape_vec((1, SIGNAL.len()), SIGNAL.to_vec()).unwrap();

        let spectrogram = stft.forward(input.view()).unwrap();
        let flat: Vec<f32> = spectrogram.iter().flat_map(|v| [v.re, v.im]).collect();
        assert_eq!(flat, SPECTROGRAM, "forward differs from PyTorch");

        let recovered = stft.inverse(spectrogram.view()).unwrap();
        assert_eq!(recovered.as_slice().unwrap(), RECOVERED, "inverse differs from PyTorch");
    }

    /// Every option `torch.stft` exposes, exercised through a roundtrip. Bit-level
    /// agreement with PyTorch is checked by `conformance/`; what this covers is that
    /// each option changes the transform in the expected way and still inverts.
    ///
    /// Some combinations are genuinely not invertible and `torch.istft` rejects them
    /// too: without centring, a tapered or zero-padded window leaves the first samples
    /// with no window energy at all. Those must fail with a NOLA error, not silently.
    #[test]
    fn every_option_roundtrips() {
        let input = signal(2, 8192);
        let (mut inverted, mut refused) = (0, 0);
        for center in [true, false] {
            for pad_mode in [
                PadMode::Reflect,
                PadMode::Constant,
                PadMode::Replicate,
                PadMode::Circular,
            ] {
                for normalized in [false, true] {
                    for onesided in [false, true] {
                        for (window, win_length) in [
                            (WindowFunction::Hann::<f64>, 256),
                            (WindowFunction::Hann::<f64>, 200),
                            (WindowFunction::Rectangular::<f64>, 256),
                        ] {
                            let stft = Stft::builder(256)
                                .hop_length(64)
                                .win_length(win_length)
                                .window(window, true)
                                .center(center)
                                .pad_mode(pad_mode)
                                .normalized(normalized)
                                .onesided(onesided)
                                .build()
                                .unwrap();
                            let label = format!(
                                "center={center} pad={pad_mode:?} norm={normalized} \
                                 one={onesided} {window:?} wl={win_length}"
                            );
                            let spectrogram = stft.forward(input.view()).unwrap();
                            assert_eq!(
                                spectrogram.dim().1,
                                if onesided { 129 } else { 256 },
                                "{label}"
                            );

                            let recovered = match stft.inverse(spectrogram.view()) {
                                Ok(recovered) => recovered,
                                Err(e) => {
                                    assert!(e.contains("NOLA"), "{label}: {e}");
                                    refused += 1;
                                    continue;
                                }
                            };
                            inverted += 1;
                            let error = input
                                .slice(s![.., ..recovered.dim().1])
                                .iter()
                                .zip(recovered.iter())
                                .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
                            assert!(error < 1e-11, "{label}: error {error:e}");
                        }
                    }
                }
            }
        }
        // Guard against the loop quietly turning into a no-op.
        assert!(inverted >= 40 && refused > 0, "{inverted} inverted, {refused} refused");
    }

    /// Samples run along the last axis and anything before it is carried through, so the
    /// same signal transformed at different ranks must give the same numbers.
    #[test]
    fn any_rank_agrees_with_the_two_dimensional_case() {
        let stft = Stft::new(256, 64, WindowFunction::Hann::<f64>, true);
        let flat = signal(6, 4096);
        let reference = stft.forward(flat.view()).unwrap();
        let (_, bins, frames) = reference.dim();

        // One channel, no channel axis at all.
        let mono = flat.index_axis(Axis(0), 0);
        let spectrogram = stft.forward(mono).unwrap();
        assert_eq!(spectrogram.dim(), (bins, frames));
        assert_eq!(spectrogram, reference.index_axis(Axis(0), 0));
        let recovered = stft.inverse(spectrogram.view()).unwrap();
        assert_eq!(recovered.dim(), stft.inverse(reference.view()).unwrap().dim().1);

        // A batch axis in front of the channel axis.
        let batched = flat.clone().into_shape_with_order((3, 2, 4096)).unwrap();
        let spectrogram = stft.forward(batched.view()).unwrap();
        assert_eq!(spectrogram.dim(), (3, 2, bins, frames));
        assert_eq!(
            spectrogram.into_shape_with_order((6, bins, frames)).unwrap(),
            reference
        );

        // And a dynamic-rank view of the same data.
        let dynamic = flat.view().into_dyn();
        let spectrogram = stft.forward(dynamic).unwrap();
        assert_eq!(spectrogram.shape(), [6, bins, frames]);
    }

    #[test]
    fn any_rank_roundtrips() {
        let stft = Stft::new(128, 32, WindowFunction::Hann::<f64>, true);
        let flat = signal(8, 2048);
        for shape in [vec![2048], vec![8, 2048], vec![4, 2, 2048], vec![2, 2, 2, 2048]] {
            let taken: usize = shape.iter().product();
            let input = Array1::from_iter(flat.iter().copied().take(taken))
                .into_shape_with_order(IxDyn(&shape))
                .unwrap();
            let spectrogram = stft.forward(input.view()).unwrap();

            let mut expected = shape.clone();
            expected.pop();
            expected.extend_from_slice(&[65, spectrogram.shape()[spectrogram.ndim() - 1]]);
            assert_eq!(spectrogram.shape(), &expected[..], "forward shape for {shape:?}");

            let recovered = stft.inverse(spectrogram.view()).unwrap();
            let mut expected = shape.clone();
            let samples = recovered.shape()[recovered.ndim() - 1];
            *expected.last_mut().unwrap() = samples;
            assert_eq!(recovered.shape(), &expected[..], "inverse shape for {shape:?}");

            let error = input
                .iter()
                .zip(recovered.iter())
                .take(taken - shape[shape.len() - 1] + samples)
                .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
            assert!(error < 1e-12, "roundtrip for {shape:?}: {error:e}");
        }
    }

    #[test]
    fn a_spectrogram_needs_a_bin_and_a_frame_axis() {
        let stft = Stft::new(128, 32, WindowFunction::Hann::<f64>, true);
        let flat = Array1::<Complex<f64>>::zeros(65);
        let error = stft.inverse(flat.view()).unwrap_err();
        assert!(error.contains("at least a bin and a frame axis"), "{error}");
    }

    #[test]
    fn a_short_window_is_zero_padded_and_centred() {
        let stft = Stft::builder(16)
            .hop_length(4)
            .win_length(10)
            .window(WindowFunction::Rectangular::<f64>, true)
            .build()
            .unwrap();
        // torch.stft centres the window with (n_fft - win_length) / 2 zeros on the left.
        let expected: Vec<f64> = (0..16).map(|i| if (3..13).contains(&i) { 1.0 } else { 0.0 }).collect();
        assert_eq!(stft.window(), &expected[..]);
        assert_eq!(stft.win_length(), 10);
        assert!(Stft::<f64>::builder(16).win_length(17).build().is_err());
        assert!(Stft::<f64>::builder(16).win_length(0).build().is_err());
    }

    #[test]
    fn padding_modes_extend_as_documented() {
        let signal = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let cases = [
            (PadMode::Reflect, [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]),
            (PadMode::Constant, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]),
            (PadMode::Replicate, [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]),
            (PadMode::Circular, [4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]),
        ];
        for (mode, expected) in cases {
            let mut out = vec![0.0; 9];
            pad_into(signal.view(), 2, mode, &mut out);
            assert_eq!(out, expected, "{mode:?}");
        }
    }

    #[test]
    fn length_trims_and_zero_extends() {
        let stft = Stft::new(256, 64, WindowFunction::Hann::<f64>, true);
        let input = signal(1, 4096);
        let spectrogram = stft.forward(input.view()).unwrap();
        let natural = stft.inverse(spectrogram.view()).unwrap().dim().1;

        let short = stft.inverse_with_length(spectrogram.view(), Some(1000)).unwrap();
        assert_eq!(short.dim().1, 1000);

        // Asking for more than the frames cover zero-fills the tail rather than
        // failing, as torch.istft does. The frames reach n_fft / 2 past the natural
        // length — that is the trailing pad, which is real data, not zeros.
        let covered = natural + 256 / 2;
        let long = stft.inverse_with_length(spectrogram.view(), Some(covered + 500)).unwrap();
        assert_eq!(long.dim().1, covered + 500);
        assert!(long.slice(s![.., covered..]).iter().all(|&v| v == 0.0));
        assert_eq!(long.slice(s![.., ..natural]), stft.inverse(spectrogram.view()).unwrap());
    }

    #[test]
    fn a_two_sided_spectrogram_mirrors_the_onesided_one() {
        let one = Stft::builder(64).hop_length(16)
            .window(WindowFunction::Hann::<f64>, true).build().unwrap();
        let two = Stft::builder(64).hop_length(16)
            .window(WindowFunction::Hann::<f64>, true).onesided(false).build().unwrap();
        let input = signal(1, 1024);
        let a = one.forward(input.view()).unwrap();
        let b = two.forward(input.view()).unwrap();
        assert_eq!(b.dim().1, 64);
        assert_eq!(a, b.slice(s![.., ..33, ..]));
        for k in 1..32 {
            assert_eq!(b.slice(s![.., 64 - k, ..]), b.slice(s![.., k, ..]).mapv(|v| v.conj()));
        }
        // And a two-sided spectrogram inverts back to the same signal.
        assert_eq!(one.inverse(a.view()).unwrap(), two.inverse(b.view()).unwrap());
    }

    #[test]
    fn center_off_needs_a_full_frame_and_starts_at_sample_zero() {
        let stft = Stft::builder(256).hop_length(64)
            .window(WindowFunction::Hann::<f64>, true).center(false).build().unwrap();
        assert!(stft.forward(signal(1, 100).view()).unwrap_err().contains("center is off"));
        let input = signal(1, 2048);
        let frames = stft.forward(input.view()).unwrap().dim().2;
        assert_eq!(frames, (2048 - 256) / 64 + 1);
        assert!(!stft.center());
    }

    #[test]
    fn builder_defaults_match_pytorch() {
        let stft = Stft::<f64>::builder(1024).build().unwrap();
        assert_eq!(stft.hop_length(), 256);             // n_fft / 4
        assert_eq!(stft.win_length(), 1024);
        assert_eq!(stft.num_bins(), 513);               // onesided
        assert!(stft.center());
        assert!(stft.window().iter().all(|&w| w == 1.0)); // window=None is all ones
    }

    #[test]
    fn a_single_sample_window_is_finite() {
        for periodic in [true, false] {
            for window in [WindowFunction::<f64>::Hann, WindowFunction::Hamming] {
                assert!(window.new(1, periodic)[0].is_finite());
            }
        }
    }

    #[test]
    fn fft_and_ifft_roundtrip() {
        let input = arr1(&[1.0, -2.0, 3.5, 0.0, 7.25, -1.5, 2.0, 9.0]);
        let recovered = ifft(fft(input.view()).unwrap().view()).unwrap();
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
