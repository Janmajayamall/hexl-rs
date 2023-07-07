use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use hexl_rs::*;
use ndarray::Array2;
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

    let batch_size = BatchSize::NumBatches(10000);

    for prime in [1152921504606748673u64, 1125899904679937] {
        let logq = 64 - prime.leading_zeros();
        for degree in [1 << 15] {
            let mut a = random_values(degree, prime);
            let mut a1 = random_values(degree, prime);

            for mod_size in [1, 3, 5, 15] {
                let mut r0 = Array2::from_shape_vec(
                    (mod_size, degree),
                    random_values((degree * mod_size), prime),
                )
                .unwrap();
                let r1 = Array2::from_shape_vec(
                    (mod_size, degree),
                    random_values((degree * mod_size), prime),
                )
                .unwrap();

                group.bench_function(
                    BenchmarkId::new(
                        "elwise_fma_mod_2d",
                        format!("n={degree}/logq={logq}/mod_size={mod_size}"),
                    ),
                    |b| {
                        b.iter(|| {
                            r0.outer_iter_mut()
                                .zip(r1.outer_iter())
                                .for_each(|(mut s0, s1)| {
                                    elwise_fma_mod(
                                        s0.as_slice_mut().unwrap(),
                                        prime - 1,
                                        s1.as_slice().unwrap(),
                                        prime,
                                        degree as u64,
                                        1,
                                    );
                                });
                            // for i in 0..mod_size {
                            //     let mut s0 = r0.row_mut(i);
                            //     let s1 = r1.row(0);
                            //     elwise_fma_mod(
                            //         s0.as_slice_mut().unwrap(),
                            //         prime - 1,
                            //         s1.as_slice().unwrap(),
                            //         prime,
                            //         degree as u64,
                            //         1,
                            //     );
                            // }
                        });
                    },
                );
            }

            group.bench_function(
                BenchmarkId::new("mul_mod_vec", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || (a.clone(), a1.clone()),
                        |(mut v, v1)| elwise_mult_mod(&mut v, &v1, prime, degree as u64, 1),
                        batch_size,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new("scalar_mul_mod_vec", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || a.clone(),
                        |mut v| elwise_mult_scalar_mod(&mut v, prime - 1, prime, degree as u64, 1),
                        batch_size,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new("fma_mod_vec", format!("n={degree}/logq={logq}")),
                |b| {
                    b.iter_batched(
                        || (a.clone(), a1.clone()),
                        |(mut v, v1)| {
                            elwise_fma_mod(&mut v, prime - 1, &v1, prime, degree as u64, 1)
                        },
                        batch_size,
                    );
                },
            );
        }
    }
}

criterion_group!(modulus, bench_modulus);
criterion_main!(modulus);
