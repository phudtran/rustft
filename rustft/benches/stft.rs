use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use rustft::{Stft, WindowFunction};

/// The three shapes the project's README benchmarks against PyTorch.
const CASES: [(usize, usize, usize, usize); 3] = [
    (2, 16384, 1024, 512),
    (2, 261120, 6144, 1024),
    (2, 65536, 4096, 2048),
];

fn signal(channels: usize, length: usize) -> Array2<f64> {
    Array2::from_shape_fn((channels, length), |(c, i)| {
        let t = i as f64 / 44100.0;
        (2.0 * std::f64::consts::PI * 440.0 * t).sin() * (1.0 + c as f64 * 0.1)
    })
}

fn bench(c: &mut Criterion) {
    let mut forward = c.benchmark_group("forward");
    for (ch, len, n_fft, hop) in CASES {
        let stft = Stft::new(n_fft, hop, WindowFunction::Hann::<f64>, true);
        let input = signal(ch, len);
        forward.throughput(Throughput::Elements((ch * len) as u64));
        forward.bench_function(
            BenchmarkId::from_parameter(format!("{ch}x{len}_n{n_fft}_h{hop}")),
            |b| b.iter(|| stft.forward(input.view()).unwrap()),
        );
    }
    forward.finish();

    let mut inverse = c.benchmark_group("inverse");
    for (ch, len, n_fft, hop) in CASES {
        let stft = Stft::new(n_fft, hop, WindowFunction::Hann::<f64>, true);
        let spec = stft.forward(signal(ch, len).view()).unwrap();
        inverse.throughput(Throughput::Elements((ch * len) as u64));
        inverse.bench_function(
            BenchmarkId::from_parameter(format!("{ch}x{len}_n{n_fft}_h{hop}")),
            |b| b.iter(|| stft.inverse(spec.view()).unwrap()),
        );
    }
    inverse.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
