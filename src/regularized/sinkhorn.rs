
use na::{DVector, dvector, DMatrix};


/// Solves the entropic regularization optimal transport problem and returns the OT matrix
/// a: Unnormalized histogram of dimension dim_a
/// b: Unnormalized histogram of dimension dim_b
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// num_iter_max: Max number of iterations
/// stop_threshold: Stop threshold on error (> 0)
pub fn sinkhorn_knopp(
    a: &mut DVector<f64>, b: &mut DVector<f64>, M: &mut DMatrix<f64>,
    reg: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>) -> DMatrix<f64> {

    // Defaults
    let mut iterations = 1000;
    if let Some(val) = num_iter_max {
        iterations = val;
    }

    let mut stop = 1E-6;
    if let Some(val) = stop_threshold {
        stop = val;
    }

    let (dim_a, dim_b) = M.shape();

    // if a and b empty, default to uniform distribution
    if a.len() == 0 {
        *a = DVector::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    }

    if b.len() == 0 {
        *b = DVector::from_vec(vec![1f64 / (dim_b as f64); dim_b]);
    }

    // we assume that no distances are null except those of the diagonal distances
    let mut u = DVector::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = DVector::<f64>::from_vec(vec![1f64 / (dim_b as f64); dim_b]);

    // K = exp(-M/reg)
    let mut k = M.clone();
    for m in k.iter_mut() {
        *m = (*m/-reg).exp();
    }

    for _ in 0..iterations {

        let uprev = u.clone();
        let vprev = v.clone();

        // Update u and v
        // u = a/kv
        let kv = &k * &v;
        for (i, ele_u) in u.iter_mut().enumerate() {
            *ele_u = a[i] / kv[i];
        }

        // v = b/ktu
        let ktu = &k.transpose() * &u;
        for (i, ele_v) in v.iter_mut().enumerate() {
            *ele_v = b[i] / ktu[i];
        }

        // Check stop conditions
        let mut ktu_0_flag = false;
        let mut u_nan_flag = false;
        let mut u_inf_flag = false;
        let mut v_nan_flag = false;
        let mut v_inf_flag = false;

        for ele in ktu.iter() {
            if *ele == 0f64 {
                ktu_0_flag = true;
            }
        }

        for ele in u.iter() {
            if (*ele).is_nan() {
                u_nan_flag = true;
            }

            if (*ele).is_infinite() {
                u_inf_flag = true;
            }
        }

        for ele in v.iter() {
            if (*ele).is_nan() {
                v_nan_flag = true;
            }

            if (*ele).is_infinite() {
                v_inf_flag = true;
            }
        }

        // Check stop conditions
        if ktu_0_flag == true || u_nan_flag == true || u_inf_flag == true
            || v_nan_flag == true || v_inf_flag == true {
            u = uprev;
            v = vprev;
            break;
        }

        // check for machine precision
        let err_u = (&u-&uprev).amax() / (dvector![u.amax(), uprev.amax(), 1f64].max());
        let err_v = (&v-&vprev).amax() / (dvector![v.amax(), vprev.amax(), 1f64].max());
        let err = 0.5 * (err_u + err_v);
        if err < stop {
            break;
        }

    }

    // nhists = 1 case only
    // diag(u)*K*diag(v)
    for (i, mut row) in k.row_iter_mut().enumerate() {
        for (j, k) in row.iter_mut().enumerate() {
            *k *= u[i] * v[j];
        }
    }

    k

}

#[cfg(test)]
mod tests {

    use na::{DVector, DMatrix};

    #[test]
    fn test_sinkhorn_knopp() {

        let mut a = DVector::from_vec(vec![0.5, 0.5]);
        let mut b = DVector::from_vec(vec![0.5, 0.5]);
        let reg = 1.0;
        let mut m = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);

        let result = super::sinkhorn_knopp(&mut a, &mut b, &mut m,
                                            reg, None, None);

        let truth = DMatrix::from_row_slice(2,2,
                    &[0.36552929, 0.13447071,
                    0.13447071, 0.36552929]);

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));

    }



}
