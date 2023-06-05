use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use hexl_rs::Ntt;
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};

fn random_values(size: usize, max: u64) -> Vec<u64> {
    Uniform::new(0u64, max)
        .sample_iter(thread_rng())
        .take(size)
        .collect::<Vec<u64>>()
}

fn bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");

    for prime in [1152921504606748673u64, 1125899904679937] {
        let logq = 64 - prime.leading_zeros();
        for degree in [1 << 15] {
            let ntt = Ntt::new(degree, prime);
            let mut a = random_values(degree as usize, prime);

            group.bench_function(
                BenchmarkId::new("forward", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || a.clone(),
                        |mut d| {
                            ntt.forward(&mut d, 1, 1);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new("forward_lazy", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || a.clone(),
                        |mut d| {
                            ntt.forward(&mut d, 1, 2);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new("backward", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || a.clone(),
                        |mut d| {
                            ntt.backward(&mut d, 1, 1);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
}

criterion_group!(ntt, bench_ntt);
criterion_main!(ntt);
