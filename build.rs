fn main() {

    // Temporary workaround to find M1 mac location of homebrew libraries
    println!("cargo:rustc-link-search=/opt/homebrew/opt/openblas/lib");

    // Temporary workaround to find python3.9 lib (ie. venv, conda)
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,/path/to/libpython3.9"
    );

    cxx_build::bridge("src/exact/ffi.rs")
        .file("src/exact/fast_transport/EMD_wrapper.cpp")
        .flag_if_supported("-std=c++14")
        .compile("rust-optimal-transport");
}
