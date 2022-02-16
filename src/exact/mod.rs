mod ffi;
mod utils;

use ndarray::prelude::*;
use std::error::Error;
use std::fmt;

use super::OTSolver;
use super::error::OTError;
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

impl Error for FastTransportErrorCode {}

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

/// Solves the exact OT Earth Movers Distance using the FastTransport LP solver
/// source_weights: Weights on samples from the source distribution
/// target_weights: Weights on samples from the target distribution
/// cost: Distance between samples in the source and target distributions
/// num_iter_max: maximum number of iterations before stopping the optimization algorithm if it has
/// not converged (default = 100000)
pub struct EarthMovers<'a> {
    source_weights: &'a mut Array1<f64>,
    target_weights: &'a mut Array1<f64>,
    cost: &'a mut Array2<f64>,
    max_iter: i32,
}

impl<'a> EarthMovers<'a> {
    pub fn new(
        source_weights: &'a mut Array1<f64>,
        target_weights: &'a mut Array1<f64>,
        cost: &'a mut Array2<f64>,
    ) -> Self {
        Self {
            source_weights,
            target_weights,
            cost,
            max_iter: 100000,
        }
    }

    pub fn iterations<'b>(&'b mut self, max_iter: i32) -> &'b mut Self {
        self.max_iter = max_iter;
        self
    }
}

impl<'a> OTSolver for EarthMovers<'a> {
    fn check_shape(&self) -> Result<(), OTError> {
        let mshape = self.cost.shape();
        let m0 = mshape[0];
        let m1 = mshape[1];
        let dim_a = self.source_weights.len();
        let dim_b = self.target_weights.len();

        if dim_a != m0 || dim_b != m1 {
            return Err(OTError::WeightDimensionError {
                dim_a,
                dim_b,
                dim_m_0: m0,
                dim_m_1: m1,
            });
        }

        Ok(())
    }

    fn solve(&mut self) -> Result<Array2<f64>, OTError> {
        self.check_shape()?;

        *self.target_weights *= self.source_weights.sum() / self.target_weights.sum();

        emd(
            self.source_weights,
            self.target_weights,
            self.cost,
            Some(self.max_iter),
        )
    }
}

/// a: Source sample weights (defaults to uniform weight if empty)
/// b: Target sample weights (defaults to uniform weight if empty)
/// M: Loss matrix (row-major)
/// num_iter_max: maximum number of iterations before stopping the optimization algorithm if it has
/// not converged (default = 100000)
#[allow(non_snake_case)]
pub(crate) fn emd(
    a: &mut Array1<f64>,
    b: &mut Array1<f64>,
    M: &mut Array2<f64>,
    num_iter_max: Option<i32>,
) -> Result<Array2<f64>, OTError> {
    // Defaults
    let iterations = match num_iter_max {
        Some(val) => val,
        None => 100000,
    };

    // Call FastTransport via wrapper
    let (G, _cost, mut _u, mut _v, result_code) = emd_c(a, b, M, iterations);

    // Propogate errors if there are any
    check_result(FastTransportErrorCode::from(result_code))?;

    Ok(G)
}

#[cfg(test)]
mod tests {

    use crate::OTSolver;
    use ndarray::array;

    #[allow(non_snake_case)]
    #[test]
    fn test_emd() {
        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let mut M = array![[0.0, 1.0], [1.0, 0.0]];

        let gamma = match super::emd(&mut a, &mut b, &mut M, None) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error),
        };

        let truth = array![[0.5, 0.0], [0.0, 0.5]];

        // println!("{:?}", gamma);

        assert_eq!(gamma, truth);
    }

    #[test]
    fn test_earthmovers_builder() {
        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let mut m = array![[0.0, 1.0], [1.0, 0.0]];

        let test = match super::EarthMovers::new(&mut a, &mut b, &mut m).solve() {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error),
        };

        let truth = array![[0.5, 0.0], [0.0, 0.5]];

        assert_eq!(test, truth);
    }
}
