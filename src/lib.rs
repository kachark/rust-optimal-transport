/*
 *  Original Python/C++ implementation written by:
 *
 *  Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer,
 *  POT Python Optimal Transport library,
 *  Journal of Machine Learning Research, 22(78):1−8, 2021.
 *  Website: https://pythonot.github.io/
 *
 *  C++ interface written by:
 *  It was written by Antoine Rolet (2014) and mainly consists of a wrapper
 *  of the code written by Nicolas Bonneel available on this page
 *  https://github.com/nbonneel/network_simplex
 *
*/

use thiserror::Error;

pub mod exact;
pub mod regularized;
pub mod unbalanced;
pub mod metrics;
pub mod ndarray_logical;
pub mod utils;

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
