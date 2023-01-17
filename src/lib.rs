#![feature(strict_provenance)]

extern crate link_cplusplus;

mod bindgen {
    // include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    extern "C" {
        #[doc = " @brief Initializes an NTT object with degree \\p degree and modulus \\p q.\n @param[in] degree also known as N. Size of the NTT transform. Must be a\n power of\n 2\n @param[in] q Prime modulus. Must satisfy \\f$ q == 1 \\mod 2N \\f$\n @param[in] alloc_ptr Custom memory allocator used for intermediate\n calculations\n @brief Performs pre-computation necessary for forward and inverse\n transforms"]
        #[link_name = "\u{1}_ZN5intel4hexl3NTTC1EmmSt10shared_ptrINS0_13AllocatorBaseEE"]
        pub fn intel_hexl_NTT_NTT(
            this: *mut ::std::os::raw::c_void,
            degree: u64,
            q: u64,
            alloc_ptr: *mut ::std::os::raw::c_void,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};

    #[test]
    fn trial() {
        let handler1: *mut c_void = null_mut();
        let handler2: *mut c_void = null_mut();

        unsafe { bindgen::intel_hexl_NTT_NTT(handler1, 8, 1553, null_mut()) };

        let a = vec![8u64; 8];
        let b = vec![8u64; 8];
        let mut c = vec![0u64; 8];
        // unsafe {
        //     bindgen::intel_hexl_EltwiseMultMod(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 8, 1553, 1);
        // }
        dbg!(c);
        // let res = unsafe { bindgen::intel_hexl_NTT_CheckArguments(1024, 1553) };
    }
}
