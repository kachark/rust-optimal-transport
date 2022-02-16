//! rust-optimal-transport prelude
//!
//! This module contains the most used types, traits, and functions
//!
//! ```
//! use rust_optimal_transport::prelude::*;
//!
//! ```

pub use crate::OTSolver;

pub use crate::error::OTError;

pub use crate::exact::EarthMovers;

pub use crate::regularized::{greenkhorn::Greenkhorn, sinkhorn::SinkhornKnopp};

pub use crate::unbalanced::SinkhornKnoppUnbalanced;

pub use crate::metrics::{dist, MetricType::Euclidean, MetricType::SqEuclidean};
