use thiserror::Error;
use crate::exact;

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

    #[error("Exact solver failed. ")]
    ExactOTError {
        #[from]
        source: exact::FastTransportErrorCode,
    },

    #[error("Invalid argument: '{0}'")]
    ArgError(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}


