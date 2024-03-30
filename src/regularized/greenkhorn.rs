use anyhow::anyhow;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use crate::error::OTError;
use crate::OTSolver;

/// Solves the entropic regularization optimal transport problem and return the OT matrix
/// Uses the Greedy Sinkhorn method:
/// Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
/// by Jason Altschuler, Jonathan Weed, Philippe Rigollet
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
/// let regularization = 1E-2;
///
/// // Compute optimal transport matrix as the Earth Mover's Distance
/// let ot_matrix = match Greenkhorn::new(
///     &source_weights,
///     &target_weights,
///     &cost,
///     regularization,
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
pub struct Greenkhorn<'a> {
    source_weights: &'a Array1<f64>,
    target_weights: &'a Array1<f64>,
    cost: &'a Array2<f64>,
    reg: f64,
    iterations: i32,
    threshold: f64,
}

impl<'a> Greenkhorn<'a> {
    pub fn new(
        source_weights: &'a Array1<f64>,
        target_weights: &'a Array1<f64>,
        cost: &'a Array2<f64>,
        reg: f64,
    ) -> Self {
        Self {
            source_weights,
            target_weights,
            cost,
            reg,
            iterations: 1000,
            threshold: 1E-9,
        }
    }

    pub fn iterations<'b>(&'b mut self, iterations: i32) -> &'b mut Self {
        self.iterations = iterations;
        self
    }

    pub fn threshold<'b>(&'b mut self, threshold: f64) -> &'b mut Self {
        self.threshold = threshold;
        self
    }

    pub fn reg<'b>(&'b mut self, reg: f64) -> &'b mut Self {
        self.reg = reg;
        self
    }
}

impl<'a> OTSolver for Greenkhorn<'a> {
    /// Ensures dimensions of the source and target measures are consistent with the
    /// cost matrix dimensions
    fn check_shape(&self) -> Result<(), OTError> {
        let mshape = self.cost.shape();
        let m0 = mshape[0];
        let m1 = mshape[1];
        let dim_a = self.source_weights.len();
        let dim_b = self.target_weights.len();

        // Check dimensions
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

        if self.reg <= 0. {
            return Err(OTError::ArgError("Regularization term <= 0".to_string()));
        }

        if self.iterations <= 0 {
            return Err(OTError::ArgError(
                "Iterations not a valid value. Must be > 0".to_string(),
            ));
        }

        greenkhorn(
            self.source_weights,
            self.target_weights,
            self.cost,
            self.reg,
            self.iterations,
            self.threshold,
        )
    }
}

/// Solves the entropic regularization optimal transport problem and return the OT matrix
/// Uses the Greedy Sinkhorn method:
/// Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
/// by Jason Altschuler, Jonathan Weed, Philippe Rigollet
///
/// a: Source sample weights (defaults to uniform weight if empty)
/// b: Target sample weights (defaults to uniform weight if empty)
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// num_iter_max: Max number of iterations (default = 1000)
/// stop_threshold: Stop threshold on error (> 0) (default = 1E-6)
fn greenkhorn(
    a: &Array1<f64>,
    b: &Array1<f64>,
    M: &Array2<f64>,
    reg: f64,
    iterations: i32,
    threshold: f64,
) -> Result<Array2<f64>, OTError> {
    let dim_a = a.len();
    let dim_b = b.len();
    let mut stop_val;

    let mut u = Array1::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = Array1::<f64>::from_vec(vec![1f64 / (dim_b as f64); dim_b]);

    // K = exp(-M/reg)
    let f = |ele: f64| (-ele / reg).exp();
    let k = M.clone().mapv_into(f);

    let mut G = &u.diag().t() * &k * &v.diag();
    let mut viol = &G.sum_axis(Axis(1)) - a;
    let mut viol_2 = &G.sum_axis(Axis(0)) - b;

    for _ in 0..iterations {
        // Absolute values
        let viol_abs: Array1<f64> = viol.iter().map(|x| x.abs()).collect();
        let viol_2_abs: Array1<f64> = viol_2.iter().map(|x| x.abs()).collect();

        // Argmax
        let i_1 = match viol_abs.argmax() {
            Ok(val) => val,
            // Propagate ndarray-stats error
            Err(err) => return Err(OTError::Other(anyhow!(err))),
        };

        let i_2 = match viol_2_abs.argmax() {
            Ok(val) => val,
            // Propagate ndarray-stats error
            Err(err) => return Err(OTError::Other(anyhow!(err))),
        };

        // Max value
        let m_viol_1 = viol_abs[i_1];
        let m_viol_2 = viol_2_abs[i_2];

        if m_viol_1 >= m_viol_2 {
            stop_val = m_viol_1;
        } else {
            stop_val = m_viol_2;
        }

        if m_viol_1 > m_viol_2 {
            let old_u = u[i_1];
            let k_i1 = &k.row(i_1);
            let denom = k_i1.dot(&v);
            let new_u = a[i_1] / denom;

            // G[i_1, :] = new_u * k[i_1, :] * v
            for (i, mut row) in G.axis_iter_mut(Axis(0)).enumerate() {
                if i != i_1 {
                    continue;
                }

                for (j, g) in row.iter_mut().enumerate() {
                    *g = new_u * k[(i_1, j)] * v[j];
                }
            }

            // let tmp = k_i1.iter().map(|x| new_u*x).collect();
            viol[i_1] = (k_i1.dot(&v) * new_u) - a[i_1];

            // viol_2 += (K[i_1, :].T * (u[i_1] - old_u) * v)
            for (j, ele) in viol_2.iter_mut().enumerate() {
                *ele += k_i1.t()[j] * (new_u - old_u) * v[j];
            }

            u[i_1] = new_u;
        } else {
            let old_v = v[i_2];
            let k_i2 = &k.column(i_2);
            let denom = k_i2.dot(&u);
            let new_v = b[i_2] / denom;

            // G[:, i_2] = u * k[:, i_2] * new_v
            for (i, mut col) in G.axis_iter_mut(Axis(1)).enumerate() {
                if i != i_2 {
                    continue;
                }

                for (j, g) in col.iter_mut().enumerate() {
                    *g = u[j] * k[(j, i_2)] * new_v;
                }
            }

            // viol += (-old_v + v[i_2]) * K[:, i_2] * u
            for (j, ele) in viol.iter_mut().enumerate() {
                *ele += (-old_v + new_v) * k_i2[j] * u[j];
            }

            viol_2[i_2] = new_v * k_i2.dot(&u) - b[i_2];

            v[i_2] = new_v;
        }

        if stop_val <= threshold {
            break;
        }
    }

    Ok(G)
}

#[cfg(test)]
mod tests {

    use crate::OTSolver;
    use ndarray::prelude::*;

    #[test]
    fn test_greenkhorn() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let reg = 1.0;
        let m = array![[0.0, 1.0], [1.0, 0.0]];

        let result = match super::greenkhorn(&a, &b, &m, reg, 1000, 1E-9) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error),
        };

        println!("{:?}", result);

        let truth = array![[0.36552929, 0.13447071], [0.13447071, 0.36552929]];

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));
    }

    #[test]
    fn test_greenkhorn_builder() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let reg = 1.0;
        let m = array![[0.0, 1.0], [1.0, 0.0]];

        let result = match super::Greenkhorn::new(&a, &b, &m, reg).solve() {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error),
        };

        println!("{:?}", result);

        let truth = array![[0.36552929, 0.13447071], [0.13447071, 0.36552929]];

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));
    }
}
