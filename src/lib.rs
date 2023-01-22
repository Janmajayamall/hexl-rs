use std::{
    ffi::c_void,
    ptr::{null, null_mut},
};

extern crate link_cplusplus;

mod bindgen {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub fn elwise_mult_mod(a: &mut [u64], b: &[u64], q: u64, n: u64, input_mod_factor: u64) {
    unsafe {
        bindgen::Eltwise_MultMod(
            a.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            n,
            q,
            input_mod_factor,
        )
    };
}

pub fn elwise_mult_scalar_mod(a: &mut [u64], b: u64, q: u64, n: u64, input_mod_factor: u64) {
    unsafe {
        bindgen::Eltwise_FMAMod(
            a.as_mut_ptr(),
            a.as_ptr(),
            b,
            null(),
            n,
            q,
            input_mod_factor,
        )
    };
}

pub fn elwise_fma_mod(a: &mut [u64], b: u64, c: &[u64], q: u64, n: u64, input_mod_factor: u64) {
    unsafe {
        bindgen::Eltwise_FMAMod(
            a.as_mut_ptr(),
            c.as_ptr(),
            b,
            a.as_ptr(),
            n,
            q,
            input_mod_factor,
        )
    };
}

pub fn elwise_add_mod(a: &mut [u64], b: &[u64], q: u64, n: u64) {
    unsafe { bindgen::Eltwise_AddMod(a.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n, q) };
}
pub fn elwise_add_scalar_mod(a: &mut [u64], b: u64, q: u64, n: u64) {
    unsafe { bindgen::Eltwise_AddScalarMod(a.as_mut_ptr(), a.as_ptr(), b, n, q) };
}
pub fn elwise_sub_mod(a: &mut [u64], b: &[u64], q: u64, n: u64) {
    unsafe { bindgen::Eltwise_SubMod(a.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n, q) };
}
pub fn elwise_sub_scalar_mod(a: &mut [u64], b: u64, q: u64, n: u64) {
    unsafe { bindgen::Eltwise_SubScalarMod(a.as_mut_ptr(), a.as_ptr(), b, n, q) };
}

pub fn elem_reduce_mod(
    a: &mut [u64],
    q: u64,
    n: u64,
    input_mod_factor: u64,
    output_mod_factor: u64,
) {
    unsafe {
        bindgen::Eltwise_ReduceMod(
            a.as_mut_ptr(),
            a.as_ptr(),
            n,
            q,
            input_mod_factor,
            output_mod_factor,
        )
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ntt {
    handler: *mut c_void,
}

unsafe impl Send for Ntt {}
unsafe impl Sync for Ntt {}

impl Ntt {
    pub fn new(degree: u64, q: u64) -> Ntt {
        let mut handler: *mut c_void = null_mut();
        unsafe {
            bindgen::NTT_Create(degree, q, &mut handler);
        }
        Ntt { handler }
    }
    pub fn forward(&self, a: &mut [u64], input_mod_factor: u64, output_mod_factor: u64) {
        unsafe {
            bindgen::NTT_ComputeForward(
                self.handler,
                a.as_mut_ptr(),
                a.as_ptr(),
                input_mod_factor,
                output_mod_factor,
            )
        }
    }

    pub fn backward(&self, a: &mut [u64], input_mod_factor: u64, output_mod_factor: u64) {
        unsafe {
            bindgen::NTT_ComputeInverse(
                self.handler,
                a.as_mut_ptr(),
                a.as_ptr(),
                input_mod_factor,
                output_mod_factor,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Uniform};
    use rand::thread_rng;
    use rayon::prelude::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};

    #[test]
    fn ntt_round_trip() {
        let degree = 1 << 15;
        let ntt = Ntt::new(degree, 65537);
        for _ in 0..1000 {
            let mut a = Uniform::new(0u64, 65537)
                .sample_iter(thread_rng())
                .take(degree as usize)
                .collect::<Vec<u64>>();
            let a_clone = a.clone();
            ntt.forward(&mut a, 1, 1);
            assert_ne!(a, a_clone);
            ntt.backward(&mut a, 1, 1);
            assert_eq!(a, a_clone);
        }
    }

    #[test]
    fn mul_fails() {
        let mut a = [4454464658814738892];
        let mut b = [13249960090426976534];
        elwise_mult_mod(&mut a, &b, 4478201243008738947, 1, 1);
        dbg!(a);
    }

    #[test]
    fn rayon_ntt() {
        let degree = 1 << 15;
        let ntt = Ntt::new(degree, 65537);
        let mut vals = (0..10000)
            .map(|_| {
                (
                    Ntt::new(degree, 65537),
                    Uniform::new(0u64, 65537)
                        .sample_iter(thread_rng())
                        .take(degree as usize)
                        .collect::<Vec<u64>>(),
                )
            })
            .collect::<Vec<(Ntt, Vec<u64>)>>();
        let now1 = std::time::SystemTime::now();
        vals.par_iter_mut().for_each(|a| {
            // println!("Launched!!");
            // let a_clone = a.clone();
            // let now = std::time::SystemTime::now();
            a.0.forward(&mut a.1, 1, 1);
            // ntt.backward(a, 1, 1);
            // println!("It took: {:?}", now.elapsed());
            // assert_eq!(a_clone, *a);
        });
        println!("Total: {:?}", now1.elapsed());
    }
}
