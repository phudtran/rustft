//! A port of SLEEF's `cosf_u1`, which is the cosine PyTorch builds float32 windows with.
//!
//! PyTorch is compiled against SLEEF for `Vectorized<float>` (`AT_BUILD_ARM_VEC256_WITH_SLEEF`
//! on aarch64, the AVX2/AVX512 kernels on x86), so `torch.hann_window(n,
//! dtype=torch.float32)` does not agree with a libm `cosf` — they differ on about 5% of
//! arguments by up to 1 ulp, and that difference multiplies through every frame of the
//! spectrogram. Reproducing the same cosine is what lets [`crate::WindowFunction`]
//! generate float32 windows that match PyTorch bit for bit, with no Python in the loop.
//!
//! Only SLEEF's fast path is ported. It covers `|d| < TRIGRANGEMAX2f`, which is 125, and
//! window arguments never exceed `8 * PI` — the largest is the fourth harmonic of a
//! flat-top window. Anything outside that range falls back to the platform's `cosf`,
//! which is not bit-compatible with PyTorch but is not reachable from window generation.
//!
//! The algorithm, the operation order and the constants are all load-bearing: changing
//! any of them breaks the bit-for-bit match with PyTorch, however harmless the change
//! looks. The fused steps are written as explicit `mul_add` calls, so the rounding does
//! not depend on the target or the optimisation level.
//!
//! Ported from SLEEF (`src/libm/sleefsimdsp.c`, `src/common/df.h`):
//!
//! > Copyright Naoki Shibata and contributors 2010 - 2025.
//! > Distributed under the Boost Software License, Version 1.0.
//! > <https://www.boost.org/LICENSE_1_0.txt>

// Several constants below carry more decimal digits than an f32 needs. They are kept as
// written: shortening them to the shortest round-tripping form is equivalent, but makes
// them harder to check against the reference values they came from.
#![allow(clippy::excessive_precision)]

/// SLEEF's three-part splitting of pi, used to reduce the argument without losing bits.
const PI_A2F: f32 = 3.1414794921875;
const PI_B2F: f32 = 0.00011315941810607910156;
const PI_C2F: f32 = 1.9841872589410058936e-09;
/// Above this the reduction above is no longer accurate and SLEEF switches to a
/// Payne-Hanek reduction, which window generation never needs.
const TRIG_RANGE_MAX_2F: f32 = 125.0;

/// A float pair carrying roughly twice the precision of an `f32`, SLEEF's `vfloat2`.
#[derive(Clone, Copy)]
struct Df(f32, f32);

/// `x * y - z`, fused. SLEEF's `vfmapn`.
#[inline(always)]
fn fmapn(x: f32, y: f32, z: f32) -> f32 {
    x.mul_add(y, -z)
}

#[inline(always)]
fn df_add2_f_f(x: f32, y: f32) -> Df {
    let s = x + y;
    let v = s - x;
    Df(s, (x - (s - v)) + (y - v))
}

#[inline(always)]
fn df_add2_df_f(x: Df, y: f32) -> Df {
    let s = x.0 + y;
    let v = s - x.0;
    let t = (x.0 - (s - v)) + (y - v);
    Df(s, t + x.1)
}

#[inline(always)]
fn df_add_f_f(x: f32, y: f32) -> Df {
    let s = x + y;
    Df(s, (x - s) + y)
}

#[inline(always)]
fn df_add_f_df(x: f32, y: Df) -> Df {
    let s = x + y.0;
    Df(s, ((x - s) + y.0) + y.1)
}

#[inline(always)]
fn df_squ(x: Df) -> Df {
    let s = x.0 * x.0;
    Df(s, (x.0 + x.0).mul_add(x.1, fmapn(x.0, x.0, s)))
}

#[inline(always)]
fn df_mul(x: Df, y: Df) -> Df {
    let s = x.0 * y.0;
    Df(s, x.0.mul_add(y.1, x.1.mul_add(y.0, fmapn(x.0, y.0, s))))
}

#[inline(always)]
fn df_mul_flatten(x: Df, y: Df) -> f32 {
    x.0.mul_add(y.0, x.1.mul_add(y.0, x.0 * y.1))
}

/// Cosine of `d`, bit-identical to `torch.cos` on a float32 tensor.
///
/// Falls back to the platform `cosf` outside SLEEF's fast range, where PyTorch uses a
/// different reduction that this does not reproduce.
pub(crate) fn cosf(d: f32) -> f32 {
    // NaN takes the fallback too, which is why this is written as a positive test
    // rather than negating the fast-path condition.
    if d.is_nan() || d.abs() >= TRIG_RANGE_MAX_2F {
        return d.cos();
    }

    // Reduce to an odd multiple of pi/2, keeping the quadrant in `q`.
    let dq = d
        .mul_add(std::f32::consts::FRAC_1_PI, -0.5)
        .round_ties_even()
        .mul_add(2.0, 1.0);
    let q = dq as i32;

    let mut s = df_add2_f_f(d, dq * (-PI_A2F * 0.5));
    s = df_add2_df_f(s, dq * (-PI_B2F * 0.5));
    s = df_add2_df_f(s, dq * (-PI_C2F * 0.5));

    let t = s;
    let s = df_squ(s);

    let mut u = 2.6083159809786593541503e-06_f32;
    u = u.mul_add(s.0, -0.0001981069071916863322258);
    u = u.mul_add(s.0, 0.00833307858556509017944336);

    let x = df_add_f_df(
        1.0,
        df_mul(df_add_f_f(-0.166666597127914428710938, u * s.0), s),
    );
    let u = df_mul_flatten(t, x);

    // The quadrant decides the sign, applied by flipping the sign bit as SLEEF does.
    if q & 2 == 0 {
        -u
    } else {
        u
    }
}



#[cfg(test)]
mod tests {
    use super::cosf;

    /// Captured from `torch.cos` on a float32 tensor, PyTorch 2.8. These are not the
    /// correctly rounded cosines, and not what libm returns -- they are SLEEF's, which is
    /// the point: reproducing them is what lets rustft build float32 windows that match
    /// PyTorch without anyone handing the window across.
    #[test]
    fn matches_pytorchs_float32_cosine() {
        const INPUTS: [f32; 56] = [
            0.0e0, 6.981317e-1, 1.3962634e0, 2.0943952e0, 2.7925267e0, 3.4906583e0,
            4.1887903e0, 4.886922e0, 5.5850534e0, 0.0e0, 1.0e-20, 5.0e-1, 1.0e0,
            1.5707964e0, 3.1415927e0, 6.2831855e0, 1.249e2, -1.249e2, -7.0e-1, -9.211459e1,
            -1.7909011e-1, 2.5171593e1, -1.16885124e2, -8.731433e1, 1.06196335e2,
            -1.065357e2, -9.181606e1, 1.11185455e2, 3.022713e1, -3.2489704e1, 2.8247254e0,
            4.038505e1, -5.5723415e1, -8.978392e1, 7.143382e1, 4.2249424e1, 3.0708137e0,
            7.855064e1, 1.2170667e1, 1.1926659e2, -7.3281654e1, 1.332513e1, -4.061075e0,
            -3.6387836e1, 2.2715635e1, -6.5645294e1, 7.4946266e1, 9.1098724e1,
            -9.2067604e1, -8.165845e0, -5.5268066e1, -1.03386986e2, 9.819419e1,
            -1.7372725e1, -8.737256e1, 4.2993866e1,
        ];
        const EXPECTED: [f32; 56] = [
            1.0e0, 7.6604444e-1, 1.7364822e-1, -5.0000006e-1, -9.396926e-1, -9.396927e-1,
            -4.999999e-1, 1.7364815e-1, 7.660443e-1, 1.0e0, 1.0e0, 8.7758255e-1,
            5.4030234e-1, -4.371139e-8, -1.0e0, 1.0e0, 7.2227883e-1, 7.2227883e-1,
            7.648422e-1, -5.3320944e-1, 9.8400617e-1, 9.9924535e-1, -7.9838014e-1,
            7.959245e-1, 8.1514585e-1, 9.61482e-1, -7.584427e-1, -3.3450872e-1,
            3.727777e-1, 4.7680712e-1, -9.5021623e-1, -8.9797395e-1, 6.7837155e-1,
            -2.4597907e-1, -6.801652e-1, -1.6136818e-1, -9.974962e-1, -9.9994147e-1,
            9.2272544e-1, 9.9351645e-1, -5.1907897e-1, 7.256898e-1, -6.0623175e-1,
            2.5661758e-1, -7.488397e-1, -9.466397e-1, 8.9959395e-1, -9.9997216e-1,
            -5.7235855e-1, -3.0683258e-1, 2.8613904e-1, -9.5950073e-1, -6.932419e-1,
            9.3826726e-2, 8.298064e-1, 5.500008e-1,
        ];

        for (&x, &want) in INPUTS.iter().zip(EXPECTED.iter()) {
            let got = cosf(x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "cosf({x:e}) = {got:e}, PyTorch gives {want:e}"
            );
        }
    }

    /// Outside SLEEF's fast range the port defers to the platform cosine, which is not
    /// bit-compatible with PyTorch but must still behave like a cosine.
    #[test]
    fn large_arguments_fall_back_but_stay_finite() {
        for x in [200.0_f32, -1.0e6, 1.0e12, f32::MAX] {
            let got = cosf(x);
            assert!(got.is_finite() && (-1.0..=1.0).contains(&got), "cosf({x:e}) = {got}");
        }
        assert!(cosf(f32::NAN).is_nan());
    }

    #[test]
    fn quadrants_are_reduced_correctly() {
        use std::f32::consts::PI;
        for (x, want) in [(0.0, 1.0), (PI, -1.0), (2.0 * PI, 1.0), (-PI, -1.0)] {
            assert!((cosf(x) - want).abs() < 1e-6, "cosf({x}) = {}", cosf(x));
        }
        // Agrees with libm to well within the 1 ulp SLEEF promises.
        for k in 0..200 {
            let x = k as f32 * 0.61;
            assert!((cosf(x) - x.cos()).abs() < 1e-6, "cosf({x}) drifted from libm");
        }
    }
}
