// Build script for Veritas
//
// Links against libspirix_hip.so (if available) or OpenCL

use std::env;
use std::path::PathBuf;

fn main() {
    // Only link HIP if not using OpenCL feature
    if env::var("CARGO_FEATURE_OPENCL").is_err() {
        // Tell cargo to look for HIP library in gpu/hip directory
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let hip_lib_dir = PathBuf::from(&manifest_dir).join("gpu/hip");

        println!("cargo:rustc-link-search=native={}", hip_lib_dir.display());

        // Try to link against HIP libraries (only if they exist)
        let libs = [
            "spirix_hip",
            "circle_f4e5",
            "ieee_complex",
            "isolated_ops",
            "in_place_ops",
            "ieee_denormal_preserve",
        ];

        for lib in &libs {
            let lib_path = hip_lib_dir.join(format!("lib{}.so", lib));
            if lib_path.exists() {
                println!("cargo:rustc-link-lib=dylib={}", lib);
            }
        }

        // Tell cargo to rerun if the HIP library changes
        println!("cargo:rerun-if-changed=gpu/hip/libspirix_hip.so");
        println!("cargo:rerun-if-changed=gpu/hip/spirix_matmul.hip");
        println!("cargo:rerun-if-changed=gpu/hip/libin_place_ops.so");
    }
}
