use thiserror::Error;

#[cfg(feature = "blas")]
extern crate blas_src;

pub mod exact;
pub mod metrics;
pub mod ndarray_logical;
pub mod regularized;
pub mod unbalanced;
pub mod utils;
pub mod prelude;

#[derive(Error, Debug)]
pub enum OTError {
    #[error(
        "Sample weight dimensions, source distribution \
            {dim_a:?} and target distribution {dim_b:?}, do \
            not match loss matrix dimensions, ({dim_m_0:?}, {dim_m_1:?})"
    )]
    WeightDimensionError {
        dim_a: usize,
        dim_b: usize,
        dim_m_0: usize,
        dim_m_1: usize,
    },

    // #[error("Histogram weights do not sum to zero")]
    // HistogramSumError {
    //     mass_a: f64,
    //     mass_b: f64
    // },
    #[error("Exact solver failed. ")]
    ExactOTError {
        #[from]
        source: exact::FastTransportErrorCode,
    },

    // #[error("Sinkhorn solver failed. ")]
    // SinkhornError {
    //     #[from]
    //     source: regularized::sinkhorn::SinkhornError,
    // },

    // #[error("Greenkhorn solver failed. ")]
    // GreenkhornError {
    //     #[from]
    //     source: regularized::greenkhorn::GreenkhornError,
    // },
    #[error("Invalid argument: '{0}'")]
    ArgError(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),

    #[error("Non-error")]
    Ok,
}

pub trait OTSolver {
    fn check_shape(&self) -> Result<(), OTError>;
    fn solve(&mut self) -> Result<ndarray::Array2<f64>, OTError>;
}
