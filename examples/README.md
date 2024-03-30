
To run an example:

```bash
cargo run --example emd_2D
```

Some examples will use the matplotlib python library to visualize results.

```bash
pip install matplotlib
```

NOTE: sinkhorn_1D example must be run from sinkhorn_1D directory in order to use the included
python script for visualization. See below:

## M1 Mac + Homebrew OpenBLAS
If OpenBLAS is installed via Homebrew on an M1 mac, you may need to add the following to `build.rs`:
```
println!("cargo:rustc-link-search=/opt/homebrew/opt/openblas/lib");
```

## Anaconda
To link against Anaconda python (for matplotlib visualizations), you may need to add the following to `build.rs`:
```
println!("cargo:rustc-link-arg=-Wl,-rpath,/path/to/anaconda3/lib/");

```

![](https://github.com/kachark/rust-optimal-transport/blob/main/assets/sinkhorn_1D_gaussian.png)
