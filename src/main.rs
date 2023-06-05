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

fn main() {
    let degree = 1 << 15;
    let prime = 1152921504606748673u64;
    let mut a = random_values(degree, prime);
    let mut a1 = random_values(degree, prime);

    let now = std::time::SystemTime::now();
    for _ in 0..100000 {
        elwise_mult_mod(&mut a, &a1, prime, degree as u64, 1);
    }
    println!("Time: {:?}", now.elapsed().unwrap() / 100000);
}
