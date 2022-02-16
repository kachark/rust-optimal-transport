# Rust Optimal Transport

![build](https://github.com/kachark/rust-optimal-transport/actions/workflows/release-packaging.yaml/badge.svg?branch=main)
![Crates.io](https://img.shields.io/crates/v/rust-optimal-transport)

![](https://github.com/kachark/rust-optimal-transport/blob/main/assets/ot_between_samples_2d_gaussian.png)

This library provides solvers for performing regularized and unregularized Optimal Transport in Rust.

Inspired by [Python Optimal Transport](https://pythonot.github.io), this library provides the following solvers: 
- [Network simplex](https://github.com/nbonneel/network_simplex) algorithm for linear program / Earth Movers Distance
- Entropic regularization OT solvers including Sinkhorn Knopp and Greedy Sinkhorn
- Unbalanced Sinkhorn Knopp

## Installation

The library has been tested on macOS. It requires a C++ compiler for building the EMD solver and relies on the following Rust libraries:

- cxx 1.0
- thiserror 1.0
- ndarray 0.15

#### Cargo installation
Edit your Cargo.toml with the following to use rust-optimal-transport in your project.

```toml
[dependencies]
rust-optimal-transport = "0.2"
```

### Features

If you would like to enable LAPACK backend (currently supporting OpenBLAS):

```toml
[dependencies]
rust-optimal-transport = { version = "0.2", features = ["blas"] }
```

This will link against an installed instance of OpenBLAS on your system. For more details see the
[ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) crate.

## Examples

### Short examples

* Import the library

```rust
use rust_optimal_transport as ot;
use ot::prelude::*;

```

* Compute OT matrix as the Earth Mover's Distance

```rust
// Generate data
let n_samples = 100;

// Mean, Covariance of the source distribution
let mu_source = array![0., 0.];
let cov_source = array![[1., 0.], [0., 1.]];

// Mean, Covariance of the target distribution
let mu_target = array![4., 4.];
let cov_target = array![[1., -0.8], [-0.8, 1.]];

// Samples of a 2D gaussian distribution
let source = ot::utils::sample_2D_gauss(n_samples, &mu_source, &cov_source).unwrap();
let target = ot::utils::sample_2D_gauss(n_samples, &mu_target, &cov_target).unwrap();

// Uniform weights on the source and target distributions
let mut source_weights = Array1::<f64>::from_elem(n, 1. / (n as f64));
let mut target_weights = Array1::<f64>::from_elem(n, 1. / (n as f64));

// Compute ground cost matrix - Squared Euclidean distance
let mut cost = dist(&source, &target, SqEuclidean);
let max_cost = cost.max().unwrap();

// Normalize cost matrix for numerical stability
cost = &cost / *max_cost;

// Compute optimal transport matrix as the Earth Mover's Distance
let ot_matrix = match EarthMovers::new(
    &mut source_weights,
    &mut target_weights,
    &mut ground_cost
).solve()?;

```

## Acknowledgements

This library is inspired by Python Optimal Transport. The original authors and contributors of that project are listed at [POT](https://github.com/PythonOT/POT#acknowledgements).

