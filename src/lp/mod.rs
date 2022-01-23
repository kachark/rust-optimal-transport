mod ffi;
mod utils;

use std::fmt;
use std::error::Error;
use ndarray::prelude::*;

use super::OTError;
use ffi::emd_c;
use utils::*;

/// Return codes from the FastTransport network simplex solver
/// FastTransport returns 1 on success
#[derive(Debug)]
pub enum FastTransportErrorCode {
    /// No feasible flow exists for the problem
    IsInfeasible,
    /// The problem is feasible and bounded.
    /// Optimal flow and node potentials (primal and dual solutions) found
    IsOptimal,
    /// Objective function of the problem is unbounded
    /// ie. there is a directed cycle having negative total cost and infinite
    /// upper bound
    IsUnbounded,
    /// Maximum iterations reached by the solver
    IsMaxIterReached,
}

impl Error for FastTransportErrorCode { }

impl fmt::Display for FastTransportErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FastTransportErrorCode::IsInfeasible => write!(f, "Network simplex infeasible!"),
            FastTransportErrorCode::IsOptimal => write!(f, "Optimal solution found!"),
            FastTransportErrorCode::IsUnbounded => write!(f, "Network simplex unbounded!"),
            FastTransportErrorCode::IsMaxIterReached => write!(f, "Max iteration reached!"),
        }
    }
}

impl From<i32> for FastTransportErrorCode {
    fn from(e: i32) -> Self {
        match e {
            0 => FastTransportErrorCode::IsInfeasible,
            1 => FastTransportErrorCode::IsOptimal,
            2 => FastTransportErrorCode::IsUnbounded,
            3 => FastTransportErrorCode::IsMaxIterReached,
            _ => FastTransportErrorCode::IsMaxIterReached,
        }
    }
}

/// a: Source sample weights (defaults to uniform weight if empty)
/// b: Target sample weights (defaults to uniform weight if empty)
/// M: Loss matrix (row-major)
/// num_iter_max: maximum number of iterations before stopping the optimization algorithm if it has
/// not converged (default = 100000)
/// center_dual: If True, centers the dual potential using function (default = true)
#[allow(non_snake_case)]
pub fn emd(
    a: &mut Array1<f64>,
    b: &mut Array1<f64>,
    M: &mut Array2<f64>,
    num_iter_max: Option<i32>,
    center_dual: Option<bool>,
) -> Result<Array2<f64>, OTError> {
    // Defaults
    let mut iterations = 100000;
    if let Some(val) = num_iter_max {
        iterations = val;
    }

    let mut center = true;
    if let Some(val) = center_dual {
        center = val;
    }

    let mshape = M.shape();
    let m0 = mshape[0];
    let m1 = mshape[1];
    let dim_a;
    let dim_b;

    // if a and b empty, default to uniform distribution
    if a.is_empty() {
        *a = Array1::from_vec(vec![1f64 / (m0 as f64); m0]);
        dim_a = m0;
    } else {
        dim_a = a.len();
    }

    if b.is_empty() {
        *b = Array1::from_vec(vec![1f64 / (m1 as f64); m1]);
        dim_b = m1;
    } else {
        dim_b = b.len();
    }

    // Check dimensions
    if dim_a != m0 || dim_b != m1 {
        return Err(OTError::WeightDimensionError {
            dim_a,
            dim_b,
            dim_m_0: m0,
            dim_m_1: m1,
        });
    }

    // TODO: same mass can be lost by summing with machine precision
    // // Ensure the same mass
    // if a.sum() != b.sum() {
    //     return Err( OTError::HistogramSumError{ mass_a: a.sum(), mass_b: b.sum() } )
    // }

    *b *= a.sum() / b.sum();

    // not_asel == ~asel, not_bsel == ~bsel
    // binary indexing of non-zero weights
    let mut not_asel = Array1::<i32>::zeros(dim_a);
    for (i, val) in a.iter().enumerate() {
        if *val == 0f64 {
            not_asel[i] = 1;
        } else {
            not_asel[i] = 0;
        }
    }

    let mut not_bsel = Array1::<i32>::zeros(dim_b);
    for (i, val) in b.iter().enumerate() {
        if *val == 0f64 {
            not_bsel[i] = 1;
        } else {
            not_bsel[i] = 0;
        }
    }

    let (G, _cost, mut u, mut v, result_code) = emd_c(a, b, M, iterations);

    if center {
        let result = center_ot_dual(&u, &v, Some(a), Some(b));
        u = result.0;
        v = result.1;
    }

    if not_asel.sum() > 1 || not_bsel.sum() > 1 {
        let result = estimate_dual_null_weights(&u, &v, a, b, M);
        u = result.0;
        v = result.1;
    }

    // Propogate errors if there are any
    check_result(FastTransportErrorCode::from(result_code))?;

    Ok(G)
}

#[cfg(test)]
mod tests {

    use ndarray::array;

    #[allow(non_snake_case)]
    #[test]
    fn test_emd() {
        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let mut M = array![[0.0, 1.0], [1.0, 0.0]];

        let gamma = match super::emd(&mut a, &mut b, &mut M, None, None) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error),
        };

        let truth = array![[0.5, 0.0], [0.0, 0.5]];

        // println!("{:?}", gamma);

        assert_eq!(gamma, truth);
    }
}
