# ROT: Rust Optimal Transport

This library provides solvers for performing regularized and unregularized Optimal Transport in Rust.

Heavily inspired by [Python Optimal Transport](https://pythonot.github.io), this library provides the following solvers: 
- [Network simplex](https://github.com/nbonneel/network_simplex) algorithm for linear program / Earth Movers Distance
- Entropic regularization OT solvers including Sinkhorn Knopp and Greedy Sinkhorn
- Unbalanced Sinkhorn Knopp

## Installation

The library has been tested on macOS. It requires a C++ compiler for building the EMD solver and relies on the following Rust libraries:

- cxx 1.0
- nalgebra 0.29.0

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
extern crate rust_optimal_transport as rot;

use rot::ot::emd::emd;
use rot::utils::metrics::{dist, MetricType};
```

* Compute OT matrix

```rust
// a, b are weights for source and target densities
// M is the ground cost matrix

let mut a = DVector::<f64>::from_vec(vec![1f64 / 5f64; 5]);
let mut b = DVector::<f64>::from_vec(vec![1f64 / 5f64; 5]);

// Create row-major matrices where each row is an element/state within the density
let xs = DMatrix::<f64>::zeros(3, 5);
let xt = DMatrix::from_row_slice(3, 5, vec![5.0; 15].as_slice());

// Compute ground cost matrix - Euclidean distance
let M = dist(&xs, Some(&xt), MetricType::Euclidean);

let T = emd(&mut a, &mut b, &mut M) // exact linear program
```

## Acknowledgements

This library is inspired by Python Optimal Transport. The original authors and contributors of that project are listed at [POT](https://github.com/PythonOT/POT#acknowledgements).

