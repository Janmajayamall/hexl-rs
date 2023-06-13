use std::{
    ffi::c_void,
    ops::Drop,
    ptr::{null, null_mut},
};
use traits::Ntt;

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
/// b - a and stores result in a
pub fn elwise_sub_reversed_mod(a: &mut [u64], b: &[u64], q: u64, n: u64) {
    unsafe { bindgen::Eltwise_SubMod(a.as_mut_ptr(), b.as_ptr(), a.as_ptr(), n, q) };
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
pub struct NttOperator {
    handler: *mut c_void,
}

unsafe impl Send for NttOperator {}
unsafe impl Sync for NttOperator {}

impl Ntt for NttOperator {
    fn new(degree: usize, q: u64) -> NttOperator {
        let mut handler: *mut c_void = null_mut();
        unsafe {
            bindgen::NTT_Create(degree as u64, q, &mut handler);
        }
        NttOperator { handler }
    }
    fn forward(&self, a: &mut [u64]) {
        unsafe { bindgen::NTT_ComputeForward(self.handler, a.as_mut_ptr(), a.as_ptr(), 1, 1) }
    }

    fn forward_lazy(&self, a: &mut [u64]) {
        unsafe { bindgen::NTT_ComputeForward(self.handler, a.as_mut_ptr(), a.as_ptr(), 1, 2) }
    }

    fn backward(&self, a: &mut [u64]) {
        unsafe { bindgen::NTT_ComputeInverse(self.handler, a.as_mut_ptr(), a.as_ptr(), 1, 1) }
    }
}

impl Drop for NttOperator {
    fn drop(&mut self) {
        unsafe {
            bindgen::NTT_Destroy(self.handler);
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
        let ntt = NttOperator::new(degree, 65537);
        for _ in 0..1000 {
            let mut a = Uniform::new(0u64, 65537)
                .sample_iter(thread_rng())
                .take(degree as usize)
                .collect::<Vec<u64>>();
            let a_clone = a.clone();
            ntt.forward(&mut a);
            assert_ne!(a, a_clone);
            ntt.backward(&mut a);
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
}
