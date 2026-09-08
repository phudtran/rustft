// Minimal C ABI over pocketfft's 1-D real transforms.
//
// PyTorch's CPU FFT is pocketfft (when built without MKL, as it is on Apple
// silicon), so calling the same library with the same scale factor is what makes
// rustft's output bit-identical to torch.stft rather than merely close.
//
// POCKETFFT_CACHE_SIZE is set by build.rs: upstream defaults it to 0, which would
// re-plan on every frame.

#include "pocketfft_hdronly.h"

#include <complex>
#include <cstddef>

namespace {

// pocketfft's entry points take the shape and strides as std::vectors. Building them
// per call would heap-allocate three times per frame, so they are kept alive per thread
// and only the length is written each time.
template <typename T> struct Descriptors {
  pocketfft::shape_t shape{0};
  pocketfft::stride_t real{static_cast<ptrdiff_t>(sizeof(T))};
  pocketfft::stride_t complex{static_cast<ptrdiff_t>(sizeof(std::complex<T>))};
};

template <typename T> Descriptors<T> &descriptors(size_t n) {
  static thread_local Descriptors<T> d;
  d.shape[0] = n;
  return d;
}

template <typename T>
void r2c_1d(size_t n, const T *input, T *output, T fct) {
  auto &d = descriptors<T>(n);
  pocketfft::r2c(d.shape, d.real, d.complex, 0, pocketfft::FORWARD, input,
                 reinterpret_cast<std::complex<T> *>(output), fct, 1);
}

template <typename T>
void c2r_1d(size_t n, const T *input, T *output, T fct) {
  auto &d = descriptors<T>(n);
  pocketfft::c2r(d.shape, d.complex, d.real, 0, pocketfft::BACKWARD,
                 reinterpret_cast<const std::complex<T> *>(input), output, fct,
                 1);
}

} // namespace

extern "C" {

void rustft_pocketfft_r2c_f32(size_t n, const float *in, float *out, float fct) {
  r2c_1d<float>(n, in, out, fct);
}
void rustft_pocketfft_c2r_f32(size_t n, const float *in, float *out, float fct) {
  c2r_1d<float>(n, in, out, fct);
}
void rustft_pocketfft_r2c_f64(size_t n, const double *in, double *out, double fct) {
  r2c_1d<double>(n, in, out, fct);
}
void rustft_pocketfft_c2r_f64(size_t n, const double *in, double *out, double fct) {
  c2r_1d<double>(n, in, out, fct);
}
}
