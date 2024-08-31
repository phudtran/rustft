use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use rustfft::{num_complex::Complex, num_traits::Float, Fft, FftNum, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

pub struct Stft<T>
where
    T: Float + FftNum + ndarray::ScalarOperand,
{
    n_fft: usize,
    hop_length: usize,
    window: Vec<T>,
    forward: Arc<dyn Fft<T>>,
    inverse: Arc<dyn Fft<T>>,
    planner: FftPlanner<T>,
}

impl<T> Stft<T>
where
    T: Float + FftNum + ndarray::ScalarOperand,
{
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        window_function: WindowFunction<T>,
        window_periodic: bool,
    ) -> Self
    where
        T: Float + FftNum + ndarray::ScalarOperand,
    {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(n_fft);
        let inverse = planner.plan_fft_inverse(n_fft);
        Self {
            n_fft,
            hop_length,
            forward,
            inverse,
            planner,
            window: window_function.new(n_fft, window_periodic),
        }
    }

    pub fn update(
        &mut self,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window_function: Option<WindowFunction<T>>,
        window_periodic: Option<bool>,
    ) {
        match n_fft {
            Some(n_fft) => {
                self.n_fft = n_fft;
                self.forward = self.planner.plan_fft_forward(self.n_fft);
                self.inverse = self.planner.plan_fft_inverse(self.n_fft);
            }
            None => {}
        }
        match hop_length {
            Some(hop_length) => self.hop_length = hop_length,
            None => {}
        }
        match window_function {
            Some(window_function) => {
                self.window = window_function.new(self.n_fft, window_periodic.unwrap_or(false))
            }
            None => {}
        }
    }

    pub fn forward(&self, input: ArrayView2<T>) -> Result<Array3<Complex<T>>, String> {
        let num_channels = input.shape()[0];
        let signal_length = input.shape()[1];

        // For center padding (reflection)
        let pad_length = self.n_fft / 2;
        let padded_length = signal_length + 2 * pad_length;

        // Calculate the number of frames
        let num_frames = (padded_length - self.n_fft) / self.hop_length + 1;
        let n_freqs = self.n_fft / 2 + 1;

        let mut output = Array3::zeros((num_channels, n_freqs, num_frames));
        let mut buffer = Array1::zeros(self.n_fft);
        for (ch, channel) in input.outer_iter().enumerate() {
            let padded_channel = pad_reflect(channel.to_owned(), pad_length);
            for frame in 0..num_frames {
                let start = frame * self.hop_length;
                for i in 0..self.n_fft {
                    buffer[i] = Complex::new(padded_channel[start + i] * self.window[i], T::zero());
                }
                self.forward.process(match buffer.as_slice_mut() {
                    Some(slice) => slice,
                    None => return Err("Failed to get mutable slice".to_string()),
                });
                for (freq, &value) in buffer.iter().take(n_freqs).enumerate() {
                    output[[ch, freq, frame]] = value;
                }
            }
        }
        Ok(output)
    }

    pub fn inverse(&self, input: ArrayView3<Complex<T>>) -> Result<Array2<T>, String> {
        let num_channels = input.shape()[0];
        let num_frames = input.shape()[2];
        let original_length = (num_frames - 1) * self.hop_length + self.n_fft;

        let scale_factor = T::from(1.0 / (self.n_fft as f64)).expect("Division by zero");
        let mut output: Array2<T> = Array2::zeros((num_channels, original_length));
        for (ch, channel) in input.outer_iter().enumerate() {
            let mut wsum: Array1<T> = Array1::zeros(original_length);
            for frame in 0..num_frames {
                let start = frame * self.hop_length;

                let mut full_spectrum = vec![Complex::new(T::zero(), T::zero()); self.n_fft];

                // Reconstruct full spectrum with correct Nyquist handling
                for (i, &value) in channel.slice(s![.., frame]).iter().enumerate() {
                    if i == 0 || i == self.n_fft / 2 {
                        full_spectrum[i] = Complex::new(
                            T::from(value.re).expect("Failed to convert real part to T"),
                            T::zero(),
                        );
                    } else if i < self.n_fft / 2 {
                        full_spectrum[i] = value;
                        full_spectrum[self.n_fft - i] = value.conj();
                    }
                }
                self.inverse.process(&mut full_spectrum);
                // Overlap-add
                for (i, &value) in full_spectrum.iter().enumerate() {
                    if start + i < original_length {
                        output[[ch, start + i]] = output[[ch, start + i]]
                            + T::from(value.re * scale_factor * self.window[i])
                                .expect("Convert overlap sum to T");
                        wsum[start + i] = wsum[start + i] + self.window[i] * self.window[i];
                    }
                }
            }
            let mut temp = output.slice(s![ch, ..]).to_owned();

            temp = temp / wsum;
            output.slice_mut(s![ch, ..]).assign(&temp);
        }

        remove_padding(&mut output, self.n_fft);
        Ok(output)
    }
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
    pub fn new(&self, size: usize, periodic: bool) -> Vec<T>
    where
        T: Float + FftNum,
    {
        let mut window = Vec::with_capacity(size);
        let m = if periodic { size + 1 } else { size };

        for n in 0..size {
            let x = n as f64 / (m - 1) as f64;
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
                        * (alpha
                            * (T::from(x - 0.5).expect(
                                "
                    Failed to create Gaussian window",
                            )))
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
}

fn pad_reflect<T>(signal: Array1<T>, pad_length: usize) -> Array1<T>
where
    T: Float + FftNum,
{
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

fn remove_padding<T>(padded: &mut Array2<T>, n_fft: usize)
where
    T: Float + FftNum,
{
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

pub fn fft<T>(input: ArrayView1<T>) -> Result<Array1<Complex<T>>, String>
where
    T: Float + FftNum,
{
    let mut planner = FftPlanner::new();
    let len = input.len();
    let fft = planner.plan_fft_forward(len);
    let mut buffer: Vec<Complex<T>> = input
        .to_vec()
        .iter()
        .map(|&x| Complex::new(x, T::zero()))
        .collect();
    fft.process(&mut buffer);
    let array = Array1::from_vec(buffer);
    Ok(array)
}

pub fn ifft<T>(input: ArrayView1<Complex<T>>) -> Result<Array1<T>, String>
where
    T: Float + FftNum,
{
    let mut planner = FftPlanner::new();
    let len = input.len();
    let ifft = planner.plan_fft_inverse(len);
    let mut buffer = input.to_vec();
    ifft.process(&mut buffer);
    // Normalize and extract real parts
    let scale = T::from(len).expect("Failed to convert len to T");
    let result: Array1<T> = buffer.into_iter().map(|c| c.re / scale).collect();
    Ok(result)
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
