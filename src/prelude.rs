//! rust-optimal-transport prelude
//!
//! This module contains the most used types, traits, and functions
//!
//! ```
//! use rust_optimal_transport::prelude::*;
//!
//! ```

pub use crate::{
    OTSolver, OTError
};

pub use crate::exact::EarthMovers;

pub use crate::regularized::{
    sinkhorn::SinkhornKnopp,
    greenkhorn::Greenkhorn,
};

pub use crate::unbalanced::SinkhornKnoppUnbalanced;

pub use crate::metrics::{
    dist,
    MetricType::SqEuclidean,
    MetricType::Euclidean,
};
