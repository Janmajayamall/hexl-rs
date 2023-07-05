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

fn fma_mod_2d() {
    let degree = 1 << 15;
    let mod_size = 2;
    let prime = 1125899904679937; // 50 bits
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

    for _ in 0..1000 {
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
    }
}

fn main() {
    fma_mod_2d();
}
