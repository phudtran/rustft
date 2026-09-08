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

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{num_complex::Complex, num_traits::Float, FftNum, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

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

/// A planned short-time Fourier transform.
///
/// Planning the FFT, building the window and sizing the scratch buffers all happen
/// once, in [`Stft::new`]. [`Stft::forward`] and [`Stft::inverse`] take `&self`, so a
/// single instance can be shared across threads and reused for every block of audio.
pub struct Stft<T>
where
    T: Float + FftNum + ndarray::ScalarOperand,
{
    n_fft: usize,
    hop_length: usize,
    window: Vec<T>,
    /// `window[i] / n_fft`. The inverse FFT is unnormalised and every output sample is
    /// re-windowed, so folding both into one factor saves a pass over each frame.
    inverse_window: Vec<T>,
    /// Index of the Nyquist bin, which a real signal leaves purely real. Odd transform
    /// lengths have no Nyquist bin, and their top bin keeps its imaginary part.
    nyquist: Option<usize>,
    window_function: WindowFunction<T>,
    window_periodic: bool,
    forward: Arc<dyn RealToComplex<T>>,
    inverse: Arc<dyn ComplexToReal<T>>,
    planner: RealFftPlanner<T>,
}

impl<T> Stft<T>
where
    T: Float + FftNum + ndarray::ScalarOperand,
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
        validate_params(n_fft, hop_length)?;
        let mut planner = RealFftPlanner::new();
        let forward = planner.plan_fft_forward(n_fft);
        let inverse = planner.plan_fft_inverse(n_fft);
        let window = window_function.new(n_fft, window_periodic);
        Ok(Self {
            inverse_window: scaled_window(&window, n_fft),
            nyquist: nyquist_bin(n_fft),
            n_fft,
            hop_length,
            window,
            window_function,
            window_periodic,
            forward,
            inverse,
            planner,
        })
    }

    /// Re-plan in place, keeping the cached FFT plans for sizes already seen.
    ///
    /// Changing `n_fft` or `window_periodic` rebuilds the window, so the window always
    /// matches the transform length even when only `n_fft` is supplied.
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

        let window_function = window_function.unwrap_or(self.window_function);
        let window_periodic = window_periodic.unwrap_or(self.window_periodic);
        let rebuild_window = n_fft != self.n_fft
            || window_periodic != self.window_periodic
            || !window_function.same_as(&self.window_function);

        if n_fft != self.n_fft {
            self.forward = self.planner.plan_fft_forward(n_fft);
            self.inverse = self.planner.plan_fft_inverse(n_fft);
            self.nyquist = nyquist_bin(n_fft);
            self.n_fft = n_fft;
        }
        self.hop_length = hop_length;
        if rebuild_window {
            self.window = window_function.new(n_fft, window_periodic);
            self.inverse_window = scaled_window(&self.window, n_fft);
            self.window_function = window_function;
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

    /// The analysis window, `n_fft` samples long.
    pub fn window(&self) -> &[T] {
        &self.window
    }

    /// Number of frames [`Stft::forward`] will produce for a signal this long.
    pub fn num_frames(&self, signal_length: usize) -> Result<usize, String> {
        let pad = self.n_fft / 2;
        self.check_signal_length(signal_length)?;
        Ok((signal_length + 2 * pad - self.n_fft) / self.hop_length + 1)
    }

    /// Reflect padding cannot mirror more samples than the signal holds; `torch.stft`
    /// rejects the same inputs, so refuse them here rather than panicking on the copy.
    fn check_signal_length(&self, signal_length: usize) -> Result<(), String> {
        let pad = self.n_fft / 2;
        if pad >= signal_length {
            return Err(format!(
                "signal length {signal_length} is too short: reflect padding needs more \
                 than n_fft / 2 = {pad} samples per channel"
            ));
        }
        Ok(())
    }

    /// Forward STFT of a `(channels, samples)` signal.
    ///
    /// Returns `(channels, n_fft / 2 + 1, frames)`, the layout and values of
    /// `torch.stft(..., center=True, onesided=True, return_complex=True)`.
    pub fn forward(&self, input: ArrayView2<T>) -> Result<Array3<Complex<T>>, String> {
        let (num_channels, signal_length) = input.dim();
        self.check_signal_length(signal_length)?;

        let pad = self.n_fft / 2;
        let padded_length = signal_length + 2 * pad;
        let num_frames = (padded_length - self.n_fft) / self.hop_length + 1;
        let n_freqs = self.n_fft / 2 + 1;

        let mut output = Array3::zeros((num_channels, n_freqs, num_frames));
        let mut padded = vec![T::zero(); padded_length];
        // Frames are transformed into `staging` frame-major so each spectrum lands in a
        // contiguous run, then transposed in one blocked pass. Writing straight into the
        // frequency-major output would touch a fresh cache line per bin.
        let mut staging = vec![Complex::new(T::zero(), T::zero()); num_frames * n_freqs];
        let parallel = self.use_parallel(num_frames);

        for (ch, channel) in input.outer_iter().enumerate() {
            pad_reflect_into(channel, pad, &mut padded);
            self.transform_frames(&padded, num_frames, &mut staging)?;
            let mut out_channel = output.index_axis_mut(Axis(0), ch);
            let out_channel = out_channel
                .as_slice_mut()
                .expect("freshly allocated Array3 is contiguous");
            transpose_into(&staging, out_channel, num_frames, n_freqs, parallel);
        }
        Ok(output)
    }

    /// Inverse STFT of a `(channels, n_fft / 2 + 1, frames)` spectrogram.
    ///
    /// Like `torch.istft(..., center=True)`, this trims `n_fft / 2` samples off each
    /// end, leaving `hop_length * (frames - 1)` samples for an even `n_fft` and one
    /// more than that for an odd one.
    ///
    /// Fails if the window and hop length do not satisfy NOLA — that is, if overlap-add
    /// leaves some output sample with no window energy, so nothing there can be
    /// recovered. `torch.istft` rejects the same case.
    pub fn inverse(&self, input: ArrayView3<Complex<T>>) -> Result<Array2<T>, String> {
        let (num_channels, n_freqs, num_frames) = input.dim();
        let expected_freqs = self.n_fft / 2 + 1;
        if n_freqs != expected_freqs {
            return Err(format!(
                "expected {expected_freqs} frequency bins for n_fft = {}, got {n_freqs}",
                self.n_fft
            ));
        }
        if num_frames == 0 {
            return Err("spectrogram has no frames".to_string());
        }

        let trim = self.n_fft / 2;
        let output_length = (num_frames - 1) * self.hop_length + self.n_fft - 2 * trim;
        // The envelope depends only on the window, the hop and the frame count, so it is
        // shared by every channel instead of being rebuilt for each one.
        let envelope = self.window_envelope(num_frames, trim, output_length)?;

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
            let channel = input.index_axis(Axis(0), ch);
            // One blocked transpose turns the frequency-major slab into per-frame
            // contiguous spectra that the inverse FFT can consume directly.
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
            self.overlap_add(&mut spectra, num_frames, trim, row, &mut frames)?;
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
        let mut scratch = self.forward.make_scratch_vec();
        for (f, spectrum) in staging.chunks_exact_mut(self.n_fft / 2 + 1).enumerate() {
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

        let n_freqs = self.n_fft / 2 + 1;
        staging
            .par_chunks_mut(n_freqs * FRAME_BLOCK)
            .enumerate()
            .try_for_each_init(
                || (vec![T::zero(); self.n_fft], self.forward.make_scratch_vec()),
                |(frame, scratch), (block, chunk)| {
                    let base = block * FRAME_BLOCK;
                    for (offset, spectrum) in chunk.chunks_exact_mut(n_freqs).enumerate() {
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
        self.forward
            .process_with_scratch(frame, spectrum, scratch)
            .map_err(|e| format!("forward FFT failed: {e}"))
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
        let mut scratch = self.inverse.make_scratch_vec();

        for (f, spectrum) in spectra.chunks_exact_mut(n_freqs).enumerate() {
            debug_assert!(f < num_frames);
            self.inverse_frame(spectrum, frame, &mut scratch)?;
            let start = f * self.hop_length;
            let (lo, hi) = overlap_bounds(start, self.n_fft, trim, row.len());
            let target = &mut row[start + lo - trim..start + hi - trim];
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
                || self.inverse.make_scratch_vec(),
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
                    let lo = (base + trim).saturating_sub(start);
                    let hi = self
                        .n_fft
                        .min((base + chunk.len() + trim).saturating_sub(start));
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
        self.inverse
            .process_with_scratch(spectrum, frame, scratch)
            .map_err(|e| format!("inverse FFT failed: {e}"))?;
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
        trim: usize,
        output_length: usize,
    ) -> Result<Vec<T>, String> {
        let mut envelope = vec![T::zero(); output_length];
        for f in 0..num_frames {
            let start = f * self.hop_length;
            let (lo, hi) = overlap_bounds(start, self.n_fft, trim, output_length);
            let target = &mut envelope[start + lo - trim..start + hi - trim];
            for (slot, &weight) in target.iter_mut().zip(&self.window[lo..hi]) {
                *slot = *slot + weight * weight;
            }
        }

        let epsilon = T::from(NOLA_EPSILON).expect("NOLA epsilon is representable");
        if let Some(index) = envelope.iter().position(|w| w.abs() < epsilon) {
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

/// Half-open range of a frame's samples that land inside the trimmed output.
fn overlap_bounds(start: usize, n_fft: usize, trim: usize, output_length: usize) -> (usize, usize) {
    let lo = trim.saturating_sub(start);
    let hi = n_fft.min((trim + output_length).saturating_sub(start));
    (lo, hi.max(lo))
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

fn scaled_window<T: Float + FftNum>(window: &[T], n_fft: usize) -> Vec<T> {
    let scale = T::from(1.0 / n_fft as f64).expect("1 / n_fft is representable");
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
        let mut window = Vec::with_capacity(size);
        // A periodic window of length N is the symmetric window of length N + 1 with its
        // last point dropped, so it tiles without a seam under overlap-add.
        let m = if periodic { size + 1 } else { size };
        let denominator = if m > 1 { (m - 1) as f64 } else { 1.0 };

        for n in 0..size {
            let x = n as f64 / denominator;
            let value = match self {
                WindowFunction::Rectangular => T::one(),
                WindowFunction::Hann => T::from(0.5 * (1.0 - (2.0 * PI * x).cos()))
                    .expect("Failed to create Hann window"),
                WindowFunction::Hamming => T::from(0.54 - 0.46 * (2.0 * PI * x).cos())
                    .expect("Failed to create Hamming window"),
                WindowFunction::Blackman => {
                    T::from(0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos())
                        .expect("Failed to create Blackman window")
                }
                WindowFunction::Gaussian(sigma) => {
                    let alpha = T::one() / *sigma;
                    (T::from(-0.5).expect("Failed to create Gaussian window")
                        * (alpha * (T::from(x - 0.5).expect("Failed to create Gaussian window")))
                            .powi(2))
                    .exp()
                }
                WindowFunction::Triangular => T::from(1.0 - (2.0 * x - 1.0).abs())
                    .expect("Failed to create Triangular window"),
                WindowFunction::Bartlett => {
                    if x < 0.5 {
                        T::from(2.0 * x).expect("Failed to create Bartlett window")
                    } else {
                        T::from(2.0 - 2.0 * x).expect("Failed to create Bartlett window")
                    }
                }
                WindowFunction::FlatTop => T::from(
                    0.21557895 - 0.41663158 * (2.0 * PI * x).cos()
                        + 0.277263158 * (4.0 * PI * x).cos()
                        - 0.083578947 * (6.0 * PI * x).cos()
                        + 0.006947368 * (8.0 * PI * x).cos(),
                )
                .expect("Failed to create FlatTop window"),
            };
            window.push(value);
        }
        window
    }

    /// Whether two window functions would produce the same samples.
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (WindowFunction::Gaussian(a), WindowFunction::Gaussian(b)) => a == b,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

/// Reflect-pad `signal` by `pad` samples on both sides into `out`, mirroring
/// `torch.nn.functional.pad(..., mode="reflect")`: the edge sample is not repeated.
///
/// `out` must be `signal.len() + 2 * pad` long and `pad` must be less than the signal
/// length, both of which the callers check.
fn pad_reflect_into<T>(signal: ArrayView1<T>, pad: usize, out: &mut [T])
where
    T: Float + FftNum,
{
    let length = signal.len();
    debug_assert_eq!(out.len(), length + 2 * pad);
    debug_assert!(pad < length);

    for (slot, &sample) in out[pad..pad + length].iter_mut().zip(signal.iter()) {
        *slot = sample;
    }
    // Mirror out of the copy just made rather than back through the array view, so both
    // sides of each read are contiguous.
    for i in 0..pad {
        out[pad - 1 - i] = out[pad + i + 1];
    }
    for i in 0..pad {
        out[pad + length + i] = out[pad + length - 2 - i];
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
    use ndarray::{arr1, s, Array2};

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
        pad_reflect_into(signal.view(), pad, &mut out);
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
        pad_reflect_into(input.index_axis(Axis(0), 0), pad, &mut padded);

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
