// use crate::ndarray_logical;
use ndarray::prelude::*;
use ndarray_linalg::norm;

use crate::{OTError, OTSolver};

/// Solves the entropic regularization optimal transport problem using the Sinkhorn-Knopp algorithm
/// and returns the OT matrix
/// source_weights: Weights on samples from the source distribution
/// target_weights: Weights on samples from the target distribution
/// cost: Distance between samples in the source and target distributions
/// reg: Entropy regularization term > 0
/// max_iter: Max number of iterations (default = 1000)
/// threshold: Error convergence threshold (> 0) (default = 1E-9)
pub struct SinkhornKnopp<'a> {
    source_weights: &'a Array1<f64>,
    target_weights: &'a Array1<f64>,
    cost: &'a Array2<f64>,
    reg: f64,
    max_iter: i32,
    threshold: f64,
}

impl<'a> SinkhornKnopp<'a> {
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
            max_iter: 1000,
            threshold: 1E-9,
        }
    }

    pub fn iterations<'b>(&'b mut self, max_iter: i32) -> &'b mut Self {
        self.max_iter = max_iter;
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

impl<'a> OTSolver for SinkhornKnopp<'a> {
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

        sinkhorn_knopp(
            self.source_weights,
            self.target_weights,
            self.cost,
            self.reg,
            Some(self.max_iter),
            Some(self.threshold),
        )
    }
}

/// Solves the entropic regularization optimal transport problem and returns the OT matrix
/// a: Source sample weights (defaults to uniform weight if empty)
/// b: Target sample weights (defaults to uniform weight if empty)
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// num_iter_max: Max number of iterations (default = 1000)
/// stop_threshold: Stop threshold on error (> 0) (default = 1E-6)
pub(crate) fn sinkhorn_knopp(
    a: &Array1<f64>,
    b: &Array1<f64>,
    M: &Array2<f64>,
    reg: f64,
    num_iter_max: Option<i32>,
    stop_threshold: Option<f64>,
) -> Result<Array2<f64>, OTError> {
    let mut err: f64;
    let mut ktu;
    let mut v_prev;
    let kp;
    let k_transpose;
    let dim_a = a.len();
    let dim_b = b.len();

    // Defaults
    let iterations = match num_iter_max {
        Some(val) => val,
        None => 1000,
    };

    let stop = match stop_threshold {
        Some(val) => val,
        None => 1E-9,
    };

    // we assume that no distances are null except those of the diagonal distances
    let mut u = Array1::<f64>::from_elem(dim_a, 1. / (dim_a as f64));
    let mut v = Array1::<f64>::from_elem(dim_b, 1. / (dim_b as f64));

    // K = exp(-M/reg)
    let f = |ele: f64| (-ele / reg).exp();
    let k = M.clone().mapv_into(f);

    let a_cache = a.clone();
    let b_cache = b.clone();

    // Kp = (1./a) * K
    let numerator: Array1<f64> = a_cache.mapv_into(|a| 1. / a);
    kp = numerator.into_shape((dim_a, 1)).unwrap() * &k;

    // K.transpose()
    k_transpose = k.t();

    for count in 0..iterations {
        v_prev = v.clone();

        // Update v
        ktu = k_transpose.dot(&u);

        // v = b/ktu
        azip!((v in &mut v, &b in &b_cache, &ktu in &ktu) *v = b / ktu);

        // Update u
        // u = a/kv = 1 / (dot(kp, v)
        azip!((u in &mut u, &kpdotv in &kp.dot(&v)) *u = 1. / kpdotv);

        if count % 10 == 0 {
            err = norm::Norm::norm_l1(&(&v - &v_prev));

            if err < stop {
                break;
            }
        }
    }

    Ok(u.into_shape((dim_a, 1)).unwrap() * k * v.into_shape((1, dim_b)).unwrap())
}

#[cfg(test)]
mod tests {

    use ndarray::prelude::*;

    use crate::OTSolver;

    #[test]
    fn test_sinkhorn_knopp() {
        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let reg = 1.0;
        let mut m = array![[0.0, 1.0], [1.0, 0.0]];

        let result = match super::sinkhorn_knopp(&mut a, &mut b, &mut m, reg, None, None) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error),
        };

        let truth = array![[0.36552929, 0.13447071], [0.13447071, 0.36552929]];

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));
    }

    #[test]
    fn test_sinkhorn_builder() {
        let a = array![0.5, 0.5];
        let b = array![0.5, 0.5];
        let reg = 1.0;
        let m = array![[0.0, 1.0], [1.0, 0.0]];

        let result = match super::SinkhornKnopp::new(&a, &b, &m, reg).solve() {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error),
        };

        let truth = array![[0.36552929, 0.13447071], [0.13447071, 0.36552929]];

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));
    }
}
