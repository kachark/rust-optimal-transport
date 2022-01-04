
use thiserror::Error;

mod utils;
pub mod lp;

#[derive(Error, Debug)]
pub enum OTError {
    #[error("Histogram weights dimensions, a {dim_a:?} and b {dim_b:?}, do not match loss matrix dimensions, ({dim_m_0:?}, {dim_m_1:?})")]
    DimensionError {
        dim_a: usize,
        dim_b: usize,
        dim_m_0: usize,
        dim_m_1: usize
    },
    #[error("Histogram weights do not sum to zero")]
    HistogramSumError {
        mass_a: f64,
        mass_b: f64
    },
    #[error("Fast transport failed: '{0}'")]
    FastTransportError(String), // TODO: FastTransportError should be in lp.rs
    #[error("Invalid argument: '{0}'")]
    ArgError(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}


