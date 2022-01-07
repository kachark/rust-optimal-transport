# ROT: Rust Optimal Transport

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
use rust_optimal_transport as rot;

use rot::lp::emd;
use rot::utils::metrics::{dist, MetricType};
```

* Compute OT matrix

```rust
// a, b are weights for source and target densities
// M is the ground cost matrix

// Generate data
let xs = Array2::<f64>::zeros( (5, 5) );
let xt = Array2::<f64>::from_elem( (5, 5), 5.0 );

// Uniform distribution on the samples
let mut a = Array1::<f64>::from_vec(vec![1f64 / 5f64; 5]);
let mut b = Array1::<f64>::from_vec(vec![1f64 / 5f64; 5]);

// Compute ground cost matrix - Euclidean distance
let mut M = dist(&xs, &xt, MetricType::Euclidean);

// Solve Earth Mover's Distance
let T = emd(&mut a, &mut b, &mut M, None, None)?;
```

## Acknowledgements

This library is inspired by Python Optimal Transport. The original authors and contributors of that project are listed at [POT](https://github.com/PythonOT/POT#acknowledgements).

