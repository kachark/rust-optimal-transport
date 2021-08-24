
pub mod unbalanced {

use na::{DVector, dvector, DMatrix};

pub enum UnbalancedSolverType {
    Sinkhorn,
    SinkhornStabilized,
    SinkhornRegScaling,
}


/// Solves the unbalanced entropic regularization optimal transport problem and return the OT plan
/// a: Unnormalized histogram of dimension dim_a
/// b: One or multiple unnormalized histograms of dimension dim_b. If many, compute all the OT
/// distances (a, b_i)
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// reg_m: Marginal relaxation term > 0
/// method: method used for the solver either 'sinkhorn', 'sinkhorn_stabilized', or
/// 'sinkhorn_reg_scaling'
/// num_iter_max: Max number of iterations
/// stop_threshold: Stop threshold on error (>0)
/// verbose: Print information along iterations
pub fn sinkhorn_unbalanced(
    a: &mut DVector<f64>, b: &mut DMatrix<f64>, M: &mut DMatrix<f64>,
    reg: f64, reg_m: f64, method: UnbalancedSolverType,
    num_iter_max: Option<i32>, stop_threshold: Option<f64>,
    verbose: Option<bool>) -> DMatrix<f64> {

    if b.len() < 2 {
        *b = b.transpose();
    }

    match method {

        UnbalancedSolverType::Sinkhorn => sinkhorn_knopp_unbalanced(
                                              a, b, M, reg, reg_m,
                                              num_iter_max,
                                              stop_threshold,
                                              verbose),

        UnbalancedSolverType::SinkhornStabilized => sinkhorn_stabilized_unbalanced(
                                              a, b, M, reg, reg_m,
                                              num_iter_max,
                                              stop_threshold,
                                              verbose),

        UnbalancedSolverType::SinkhornRegScaling => sinkhorn_knopp_unbalanced(
                                              a, b, M, reg, reg_m,
                                              num_iter_max,
                                              stop_threshold,
                                              verbose),

    }

}


/// Solves the unbalanced entropic regularization optimal transport problem and return the loss
/// a: Unnormalized histogram of dimension dim_a
/// b: One or multiple unnormalized histograms of dimension dim_b. If many, compute all the OT
/// distances (a, b_i)
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// reg_m: Marginal relaxation term > 0
/// num_iter_max: Max number of iterations
/// stop_threshold: Stop threshold on error (>0)
/// verbose: Print information along iterations
fn sinkhorn_knopp_unbalanced(
    a: &mut DVector<f64>, b: &mut DMatrix<f64>, M: &mut DMatrix<f64>,
    reg: f64, reg_m: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>,
    verbose: Option<bool>) -> DMatrix<f64> {

    let (dim_a, dim_b) = M.shape();

    // if a and b empty, default to uniform distribution
    if a.len() == 0 {
        *a = DVector::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    }

    if b.len() == 0 {
        // ensure row-major
        *b = DMatrix::from_row_slice(1, dim_b, vec![1f64 / (dim_b as f64); dim_b].as_slice());
    }

    let n_hists = b.shape().1;

    // we assume that no distances are null except those of the diagonal distances
    let mut u = DVector::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = DMatrix::<f64>::from_row_slice(dim_b, n_hists, vec![1f64 / (dim_b as f64); dim_b].as_slice());

    // K = exp(-M/reg)
    let mut k = M.clone();
    for m in k.iter_mut() {
        *m = (*m/-reg).exp();
    }

    let fi = reg_m / (reg_m + reg);
    let err = 1f64;


    let mut iterations = 100000;
    if let Some(val) = num_iter_max {
        iterations = val;
    }

    let mut stop = 1E-6;
    if let Some(val) = stop_threshold {
        stop = val;
    }

    for _ in 0..iterations {

        let uprev = u.clone();
        let vprev = v.clone();

        let kv = &k * &v; // good
        // if i == 0 {
        //     println!("{:?}", &kv);
        // }

        // Update u and v
        // u = (a/kv) ** fi
        for (i, ele_u) in u.iter_mut().enumerate() {
            *ele_u = (a[i] / kv[i]).powf(fi);
        }
        // if i == 0 {
        //     println!("{:?}", &u);
        // }

        let ktu = &k.transpose() * &u;

        // v = (b/ktu) ** fi
        for (i, ele_v) in v.iter_mut().enumerate() {
            *ele_v = (b[i] / ktu[i]).powf(fi);
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

    // nhists = 1
    let mut result = k.clone();
    for (i, mut row) in result.row_iter_mut().enumerate() {
        for (j, k) in row.iter_mut().enumerate() {
            *k *= u[i] * v[j];
        }
    }

    // nhists > 1 - Don't handle this case
    // res = np.einsum('ik,ij,jk,ij->k', u, K, v, M);
    // einsum('ij,ij->i', Y, Y)
    // let b2 = y.component_mul(y).column_sum();

    result

}


/// Solves the unbalanced entropic regularization optimal transport problem and return the loss
/// This function solves the optimization problem using log-domain stabilization
/// a: Unnormalized histogram of dimension dim_a
/// b: One or multiple unnormalized histograms of dimension dim_b
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// reg_m: Marginal relaxation term > 0
/// num_iter_max: Max number of iterations
/// stop_threshold: Stop threshold on error (>0)
/// verbose: Print information along iterations
fn sinkhorn_stabilized_unbalanced(
    a: &mut DVector<f64>, b: &mut DMatrix<f64>, M: &mut DMatrix<f64>,
    reg: f64, reg_m: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>,
    verbose: Option<bool>) -> DMatrix<f64> {

    DMatrix::<f64>::zeros(2,2)

}


#[cfg(test)]
mod tests {

    use na::{DVector, DMatrix};

    #[test]
    fn test_sinkhorn_unbalanced() {

        let mut a = DVector::from_vec(vec![1./3., 1./3., 1./3.]);
        let mut b = DMatrix::from_vec(3, 1, vec![1./3., 1./3., 1./3.]);
        let reg = 2.0;
        let reg_m = 3.0;
        let mut m = DMatrix::<f64>::from_row_slice(3, 3, &[0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]);

        let result = super::sinkhorn_unbalanced(&mut a, &mut b, &mut m,
                                                reg, reg_m, super::UnbalancedSolverType::Sinkhorn,
                                                None, None, None);

        // println!("{:?}", result);

        // squared = false
        let truth = DMatrix::from_row_slice(3,3,
                    &[0.15874225, 0.20382908, 0.20382908,
                    0.20382908, 0.15874225, 0.20382908,
                    0.20382908, 0.20382908, 0.15874225]);

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));

    }

    #[test]
    fn test_sinkhorn_knopp() {

        let mut a = DVector::from_vec(vec![0.5, 0.5]);
        let mut b = DMatrix::from_vec(2, 1, vec![0.5, 0.5]);
        let reg = 1.0;
        let reg_m = 1.0;
        let mut m = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);

        let result = super::sinkhorn_knopp_unbalanced(&mut a, &mut b, &mut m,
                                                    reg, reg_m,
                                                    None, None, None);

        println!("{:?}", result);

        // squared = false
        let truth = DMatrix::from_row_slice(2,2,
                    &[0.51122823, 0.18807035,
                    0.18807035, 0.51122823]);

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));

    }

}

} // mod unbalanced
