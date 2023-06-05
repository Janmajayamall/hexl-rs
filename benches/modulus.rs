use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use hexl_rs::*;
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

fn bench_modulus(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulus");

    for prime in [1152921504606748673u64, 1125899904679937] {
        let logq = 64 - prime.leading_zeros();
        for degree in [1 << 15] {
            let mut a = random_values(degree as usize, prime);
            let mut a1 = random_values(degree as usize, prime);

            group.bench_function(
                BenchmarkId::new("mul_mod_vec", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || (a.clone(), a1.clone()),
                        |(mut v, v1)| elwise_mult_mod(&mut v, &v1, prime, degree, 1),
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new("scalar_mul_mod_vec", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || a.clone(),
                        |mut v| elwise_mult_scalar_mod(&mut v, prime - 1, prime, degree, 1),
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new("fma_mod_vec", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || (a.clone(), a1.clone()),
                        |(mut v, v1)| elwise_fma_mod(&mut v, prime - 1, &v1, prime, degree, 1),
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
}

criterion_group!(modulus, bench_modulus);
criterion_main!(modulus);
