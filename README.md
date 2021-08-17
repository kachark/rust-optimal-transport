# ROT: Rust Optimal Transport

<!-- [![PyPI version](https://badge.fury.io/py/POT.svg)](https://badge.fury.io/py/POT) -->
<!-- [![Anaconda Cloud](https://anaconda.org/conda-forge/pot/badges/version.svg)](https://anaconda.org/conda-forge/pot) -->
<!-- [![Build Status](https://github.com/PythonOT/POT/workflows/build/badge.svg?branch=master&event=push)](https://github.com/PythonOT/POT/actions) -->
<!-- [![Codecov Status](https://codecov.io/gh/PythonOT/POT/branch/master/graph/badge.svg)](https://codecov.io/gh/PythonOT/POT) -->
<!-- [![Downloads](https://pepy.tech/badge/pot)](https://pepy.tech/project/pot) -->
<!-- [![Anaconda downloads](https://anaconda.org/conda-forge/pot/badges/downloads.svg)](https://anaconda.org/conda-forge/pot) -->
<!-- [![License](https://anaconda.org/conda-forge/pot/badges/license.svg)](https://github.com/PythonOT/POT/blob/master/LICENSE) -->

This Rust library provides basic bindings to the C++ fast transport used by the Python Optimal Transport project.

## Installation

The library has been tested on MacOSX. It requires a C++ compiler for building/installing the EMD solver and relies on the following Rust libraries:

- cxx 1.0
- nalgebra 0.29.0

#### Cargo installation
Edit your Cargo.toml with the following to add ROT as a dependency

```toml
[dependencies]
rust-optimal-transport = "0.1.0"
```

## Examples

### Short examples

* Import the toolbox

```rust
use ot::emd;
use ot::utils::metrics::{dist, MetricType};
```

* Compute OT matrix

```rust
# a, b are weights for source and target densities
# M is the ground cost matrix

let mut a = DVector::<f64>::from_vec(vec![1f64 / 5f64; 5]);
let mut b = DVector::<f64>::from_vec(vec![1f64 / 5f64; 5]);

// Create row-major matrices where each row is an element/state within the density
let xs = DMatrix::<f64>::zeros(3, 5);
let xt = DMatrix::from_row_slice(3, 5, vec![5.0; 15].as_slice());

// Compute ground cost matrix - Euclidean distance
let M = dist(&xs, Some(&xt), MetricType::Euclidean);

let T = emd(&mut a, &mut b, &mut M) # exact linear program
```

## Acknowledgements

This library is inspired by Python Optimal Transport. The original authors and contributors are:

* [Rémi Flamary](http://remi.flamary.com/)
* [Nicolas Courty](http://people.irisa.fr/Nicolas.Courty/)

Contributors

* [Alexandre Gramfort](http://alexandre.gramfort.net/) (CI, documentation)
* [Laetitia Chapel](http://people.irisa.fr/Laetitia.Chapel/) (Partial OT)
* [Michael Perrot](http://perso.univ-st-etienne.fr/pem82055/) (Mapping estimation)
* [Léo Gautheron](https://github.com/aje) (GPU implementation)
* [Nathalie Gayraud](https://www.linkedin.com/in/nathalie-t-h-gayraud/?ppe=1) (DA classes)
* [Stanislas Chambon](https://slasnista.github.io/) (DA classes)
* [Antoine Rolet](https://arolet.github.io/) (EMD solver debug)
* Erwan Vautier (Gromov-Wasserstein)
* [Kilian Fatras](https://kilianfatras.github.io/) (Stochastic solvers)
* [Alain Rakotomamonjy](https://sites.google.com/site/alainrakotomamonjy/home)
* [Vayer Titouan](https://tvayer.github.io/) (Gromov-Wasserstein -, Fused-Gromov-Wasserstein)
* [Hicham Janati](https://hichamjanati.github.io/) (Unbalanced OT)
* [Romain Tavenard](https://rtavenar.github.io/) (1d Wasserstein)
* [Mokhtar Z. Alaya](http://mzalaya.github.io/) (Screenkhorn)
* [Ievgen Redko](https://ievred.github.io/) (Laplacian DA, JCPOT)
* [Adrien Corenflos](https://adriencorenflos.github.io/) (Sliced Wasserstein Distance)

