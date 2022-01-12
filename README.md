# ROT: Rust Optimal Transport

![](https://github.com/kachark/rust-optimal-transport/blob/main/images/ot_between_samples_2d_gaussian.png)

This library provides solvers for performing regularized and unregularized Optimal Transport in Rust.

Heavily inspired by [Python Optimal Transport](https://pythonot.github.io), this library provides the following solvers: 
- [Network simplex](https://github.com/nbonneel/network_simplex) algorithm for linear program / Earth Movers Distance
- Entropic regularization OT solvers including Sinkhorn Knopp and Greedy Sinkhorn
- Unbalanced Sinkhorn Knopp

## Installation

The library has been tested on macOS. It requires a C++ compiler for building the EMD solver and relies on the following Rust libraries:

- cxx 1.0
- thiserror 1.0
- ndarray 0.15

#### Cargo installation
Edit your Cargo.toml with the following to add ROT as a dependency for your project (uses git url pending publishing on Cargo)
NOTE: Update to the latest commit with ```cargo update```.

```toml
[dependencies]
rust-optimal-transport = { git = "https://github.com/kachark/rust-optimal-transport", branch = "main" }
```

## Examples

### Short examples

* Import the library

```rust
use rust_optimal_transport as ot;

use ot::lp::emd;
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
let source = ot::utils::distributions::sample_2D_gauss(n_samples, &mu_source, &cov_source).unwrap();
let target = ot::utils::distributions::sample_2D_gauss(n_samples, &mu_target, &cov_target).unwrap();

// Uniform distribution on the source and target samples
let mut source_mass = Array1::<f64>::from_vec(vec![1f64 / (n_samples as f64); n_samples as usize]);
let mut target_mass = Array1::<f64>::from_vec(vec![1f64 / (n_samples as f64); n_samples as usize]);

// Compute ground cost matrix - Squared Euclidean distance
let mut ground_cost = ot::utils::metrics::dist(&source, &target, ot::utils::metrics::MetricType::SqEuclidean);
let max_cost = ground_cost.max().unwrap();

// Normalize cost matrix for numerical stability
ground_cost = &ground_cost / *max_cost;

// Compute optimal transport matrix as the Earth Mover's Distance
let ot_matrix = match emd(&mut source_mass, &mut target_mass, &mut ground_cost, None, None) {
    Ok(result) => result,
    Err(error) => panic!("{:?}", error)
};

```

## Acknowledgements

This library is inspired by Python Optimal Transport. The original authors and contributors of that project are listed at [POT](https://github.com/PythonOT/POT#acknowledgements).

