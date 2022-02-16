#[cfg(feature = "blas")]
extern crate blas_src;

mod error;
pub mod exact;
pub mod metrics;
pub mod ndarray_logical;
pub mod regularized;
pub mod unbalanced;
pub mod utils;
pub mod prelude;

pub trait OTSolver {
    fn check_shape(&self) -> Result<(), error::OTError>;
    fn solve(&mut self) -> Result<ndarray::Array2<f64>, error::OTError>;
}
