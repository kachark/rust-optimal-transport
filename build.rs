

fn main() {

    cxx_build::bridge("src/lib.rs")
        .file("src/fast_transport/EMD_wrapper.cpp")
        .flag_if_supported("-std=c++14")
        .compile("rust-optimal-transport");

}
