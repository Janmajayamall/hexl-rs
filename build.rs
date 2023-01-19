use cmake::Config;
use std::path::PathBuf;

fn main() {
    let dir = Config::new("hexl")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_CXX_FLAGS_RELEASE", "-DNDEBUG -O3")
        .define("CMAKE_C_FLAGS_RELEASE", "-DNDEBUG -O3")
        .build();
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rustc-link-search={}/build/hexl/lib", dir.display());
    println!("cargo:rustc-link-lib=static=hexl");
    println!("-I{}", out_path.join("include").display());

    let bindings = bindgen::builder()
        .clang_arg(format!("-I{}", out_path.join("include/hexl").display()))
        .clang_arg("-Ihexl/hexl/include")
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        .detect_include_paths(true)
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_function("NTT_.*")
        .allowlist_function("Eltwise_.*")
        .generate()
        .unwrap();
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        // .write_to_file("bindings.rs")
        .unwrap();
}
