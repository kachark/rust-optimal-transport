mod ffi;
mod utils;

use ndarray::prelude::*;
use std::error::Error;
use std::fmt;

use super::error::OTError;
use super::OTSolver;
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

/// Solves the unregularized Optimal Transport (Earth Movers Distance) between source and target distributions with a given cost matrix.
///
/// ```rust
/// use rust_optimal_transport as ot;
/// use ot::prelude::*;
/// use ndarray::prelude::*;
/// use ndarray_stats::QuantileExt;
///
/// // Generate data
/// let n = 100;
///
/// // Mean, Covariance of the source distribution
/// let mu_source = array![0., 0.];
/// let cov_source = array![[1., 0.], [0., 1.]];
///
/// // Mean, Covariance of the target distribution
/// let mu_target = array![4., 4.];
/// let cov_target = array![[1., -0.8], [-0.8, 1.]];
///
/// // Samples of a 2D gaussian distribution
/// let source = ot::utils::sample_2D_gauss(n, &mu_source, &cov_source).unwrap();
/// let target = ot::utils::sample_2D_gauss(n, &mu_target, &cov_target).unwrap();
///
/// // Uniform weights on the source and target distributions
/// let mut source_weights = Array1::<f64>::from_elem(n, 1. / (n as f64));
/// let mut target_weights = Array1::<f64>::from_elem(n, 1. / (n as f64));
///
/// // Compute the cost between distributions
/// let mut cost = dist(&source, &target, SqEuclidean);
///
/// // Normalize cost matrix for numerical stability
/// let max_cost = cost.max().unwrap();
/// cost = &cost / *max_cost;
///
/// // Compute optimal transport matrix as the Earth Mover's Distance
/// let ot_matrix = match EarthMovers::new(
///     &mut source_weights,
///     &mut target_weights,
///     &mut cost
/// ).solve() {
///     Ok(result) => result,
///     Err(error) => panic!("{:?}", error),
/// };
///
/// ```
///
/// source_weights and target_weights represent histograms of the Source and Target distributions,
/// respectively.
///
pub struct EarthMovers<'a> {
    source_weights: &'a mut Array1<f64>,
    target_weights: &'a mut Array1<f64>,
    cost: &'a mut Array2<f64>,
    iterations: i32,
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
            iterations: 100000,
        }
    }

    pub fn iterations<'b>(&'b mut self, iterations: i32) -> &'b mut Self {
        self.iterations = iterations;
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

        if self.iterations <= 0 {
            return Err(OTError::ArgError(
                "Iterations not a valid value. Must be > 0".to_string(),
            ));
        }

        *self.target_weights *= self.source_weights.sum() / self.target_weights.sum();

        emd(
            self.source_weights,
            self.target_weights,
            self.cost,
            self.iterations,
        )
    }
}

#[allow(non_snake_case)]
fn emd(
    a: &mut Array1<f64>,
    b: &mut Array1<f64>,
    M: &mut Array2<f64>,
    iterations: i32,
) -> Result<Array2<f64>, OTError> {
    // Call FastTransport via wrapper
    let (G, _cost, mut _u, mut _v, result_code) = emd_c(a, b, M, iterations);

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

        let gamma = match super::emd(&mut a, &mut b, &mut M, 100000) {
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
