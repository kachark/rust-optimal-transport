// use crate::ndarray_logical;
use ndarray::prelude::*;
use ndarray_einsum_beta::einsum;
use ndarray_linalg::norm;

use crate::OTError;

/// Solves the entropic regularization optimal transport problem and returns the OT matrix
/// a: Source sample weights (defaults to uniform weight if empty)
/// b: Target sample weights (defaults to uniform weight if empty)
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// num_iter_max: Max number of iterations (default = 1000)
/// stop_threshold: Stop threshold on error (> 0) (default = 1E-6)
pub fn sinkhorn_knopp(
    a: &mut Array1<f64>,
    b: &mut Array1<f64>,
    M: &Array2<f64>,
    reg: f64,
    num_iter_max: Option<i32>,
    stop_threshold: Option<f64>,
) -> Result<Array2<f64>, OTError> {
    // TODO: check for NaN, inf, etc.

    let mut err;

    // Defaults
    let mut iterations = 1000;
    if let Some(val) = num_iter_max {
        iterations = val;
    }

    let mut stop = 1E-9;
    if let Some(val) = stop_threshold {
        stop = val;
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

    // we assume that no distances are null except those of the diagonal distances
    let mut u = Array1::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = Array1::<f64>::from_vec(vec![1f64 / (dim_b as f64); dim_b]);

    // K = exp(-M/reg)
    let k = Array2::from_shape_fn((mshape[0], mshape[1]), |(i, j)| (-M[[i, j]] / reg).exp());

    let a_cache = a.clone();
    let b_cache = b.clone();

    for count in 0..iterations {

        // Update v
        let ktu = k.t().dot(&u);

        // v = b/ktu
        azip!((v in &mut v, &b in &b_cache, &ktu in &ktu) *v = b / ktu);

        // Update u
        let kv = k.dot(&v);

        // u = a/kv
        azip!((u in &mut u, &a in &a_cache, &kv in &kv) *u = a / kv);

        if count % 10 == 0 {
            let mut tmp = einsum("i,ij,j->j", &[&u, &k, &v]).unwrap();
            tmp -= &b_cache;
            err = norm::Norm::norm_l1(&tmp);

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
}
