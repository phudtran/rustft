//! The FFT engine, selected at compile time.
//!
//! Both engines expose the same three types, so the transform above them is written
//! once. They differ in one visible way — whether the inverse already divides by
//! `n_fft` — which [`INVERSE_IS_NORMALISED`] reports.

#[cfg(not(feature = "pocketfft"))]
mod imp {
    use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
    use rustfft::num_complex::Complex;
    use rustfft::FftNum;
    use std::sync::Arc;

    /// realfft's inverse is unnormalised, like the rest of rustfft.
    pub const INVERSE_IS_NORMALISED: bool = false;

    pub struct Planner<T: FftNum>(RealFftPlanner<T>);

    impl<T: FftNum> Planner<T> {
        pub fn new() -> Self {
            Self(RealFftPlanner::new())
        }
        pub fn plan_forward(&mut self, n_fft: usize) -> Forward<T> {
            Forward(self.0.plan_fft_forward(n_fft))
        }
        /// `_scale` is ignored: realfft's inverse is unnormalised and cannot apply a
        /// factor for free, so the caller folds it into the window instead.
        pub fn plan_inverse(&mut self, n_fft: usize, _scale: T) -> Inverse<T> {
            Inverse(self.0.plan_fft_inverse(n_fft))
        }
    }

    #[derive(Clone)]
    pub struct Forward<T: FftNum>(Arc<dyn RealToComplex<T>>);

    impl<T: FftNum> Forward<T> {
        pub fn scratch(&self) -> Vec<Complex<T>> {
            self.0.make_scratch_vec()
        }
        pub fn process(
            &self,
            input: &mut [T],
            output: &mut [Complex<T>],
            scratch: &mut [Complex<T>],
        ) -> Result<(), String> {
            self.0
                .process_with_scratch(input, output, scratch)
                .map_err(|e| format!("forward FFT failed: {e}"))
        }
    }

    #[derive(Clone)]
    pub struct Inverse<T: FftNum>(Arc<dyn ComplexToReal<T>>);

    impl<T: FftNum> Inverse<T> {
        pub fn scratch(&self) -> Vec<Complex<T>> {
            self.0.make_scratch_vec()
        }
        pub fn process(
            &self,
            input: &mut [Complex<T>],
            output: &mut [T],
            scratch: &mut [Complex<T>],
        ) -> Result<(), String> {
            self.0
                .process_with_scratch(input, output, scratch)
                .map_err(|e| format!("inverse FFT failed: {e}"))
        }
    }
}

#[cfg(feature = "pocketfft")]
mod imp {
    use rustfft::num_complex::Complex;
    use rustfft::FftNum;
    use std::any::TypeId;
    use std::marker::PhantomData;

    /// pocketfft applies a scale factor inside the transform, and the inverse is
    /// planned with `1 / n_fft` so the frame comes out of the FFT already normalised —
    /// which is the order `torch.istft` uses.
    pub const INVERSE_IS_NORMALISED: bool = true;

    extern "C" {
        fn rustft_pocketfft_r2c_f32(n: usize, input: *const f32, output: *mut f32, fct: f32);
        fn rustft_pocketfft_c2r_f32(n: usize, input: *const f32, output: *mut f32, fct: f32);
        fn rustft_pocketfft_r2c_f64(n: usize, input: *const f64, output: *mut f64, fct: f64);
        fn rustft_pocketfft_c2r_f64(n: usize, input: *const f64, output: *mut f64, fct: f64);
    }

    /// `T` is checked against `f32` and `f64` by `TypeId` before any pointer is cast,
    /// so the casts below only ever reinterpret a type as itself. `FftNum` is
    /// implemented for exactly these two types, so the fallback is unreachable in
    /// practice and returns an error rather than risking a wrong call.
    fn dispatch<T: FftNum>(
        n: usize,
        input: *const T,
        output: *mut T,
        scale: T,
        f32_fn: unsafe extern "C" fn(usize, *const f32, *mut f32, f32),
        f64_fn: unsafe extern "C" fn(usize, *const f64, *mut f64, f64),
    ) -> Result<(), String> {
        // The scale factor is reinterpreted alongside the pointers, so it keeps the
        // exact bits the caller computed. Narrowing an f64 factor to f32 here would
        // round differently from PyTorch, which forms it in the working precision.
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let scale = unsafe { *(&scale as *const T).cast::<f32>() };
            unsafe { f32_fn(n, input.cast(), output.cast(), scale) };
            Ok(())
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let scale = unsafe { *(&scale as *const T).cast::<f64>() };
            unsafe { f64_fn(n, input.cast(), output.cast(), scale) };
            Ok(())
        } else {
            Err("the pocketfft backend supports only f32 and f64".to_string())
        }
    }

    pub struct Planner<T>(PhantomData<T>);

    impl<T: FftNum> Planner<T> {
        pub fn new() -> Self {
            Self(PhantomData)
        }
        pub fn plan_forward(&mut self, n_fft: usize) -> Forward<T> {
            Forward(n_fft, PhantomData)
        }
        /// pocketfft applies `scale` inside the transform, so `1 / n_fft` — or
        /// `1 / sqrt(n_fft)` for an orthonormal transform — costs nothing extra. That is
        /// also where `torch.istft` applies it.
        pub fn plan_inverse(&mut self, n_fft: usize, scale: T) -> Inverse<T> {
            Inverse(n_fft, scale, PhantomData)
        }
    }

    // pocketfft plans live in a thread-local cache keyed by length, so there is nothing
    // to carry here beyond the length itself.
    #[derive(Clone)]
    pub struct Forward<T>(usize, PhantomData<T>);

    impl<T: FftNum> Forward<T> {
        pub fn scratch(&self) -> Vec<Complex<T>> {
            Vec::new()
        }
        pub fn process(
            &self,
            input: &mut [T],
            output: &mut [Complex<T>],
            _scratch: &mut [Complex<T>],
        ) -> Result<(), String> {
            check_lengths(self.0, input.len(), output.len())?;
            dispatch(
                self.0,
                input.as_ptr(),
                output.as_mut_ptr().cast(),
                T::one(),
                rustft_pocketfft_r2c_f32,
                rustft_pocketfft_r2c_f64,
            )
        }
    }

    #[derive(Clone)]
    pub struct Inverse<T>(usize, T, PhantomData<T>);

    impl<T: FftNum> Inverse<T> {
        pub fn scratch(&self) -> Vec<Complex<T>> {
            Vec::new()
        }
        pub fn process(
            &self,
            input: &mut [Complex<T>],
            output: &mut [T],
            _scratch: &mut [Complex<T>],
        ) -> Result<(), String> {
            check_lengths(self.0, output.len(), input.len())?;
            dispatch(
                self.0,
                input.as_ptr().cast(),
                output.as_mut_ptr(),
                self.1,
                rustft_pocketfft_c2r_f32,
                rustft_pocketfft_c2r_f64,
            )
        }
    }

    fn check_lengths(n_fft: usize, real: usize, complex: usize) -> Result<(), String> {
        if real != n_fft || complex != n_fft / 2 + 1 {
            return Err(format!(
                "pocketfft expected {n_fft} real and {} complex values, got {real} and {complex}",
                n_fft / 2 + 1
            ));
        }
        Ok(())
    }
}

pub(crate) use imp::{Forward, Inverse, Planner, INVERSE_IS_NORMALISED};
