
use na::{DVector, dvector, DMatrix};

use crate::OTError;

/// Solves the unbalanced entropic regularization optimal transport problem and return the OT
/// matrix
/// a: Unnormalized histogram of dimension dim_a
/// b: Unnormalized histogram of dimension dim_b
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// reg_m: Marginal relaxation term > 0
/// num_iter_max: Max number of iterations (default = 1000)
/// stop_threshold: Stop threshold on error (> 0) (default = 1E-6)
pub fn sinkhorn_knopp_unbalanced(
    a: &mut DVector<f64>, b: &mut DVector<f64>, M: &mut DMatrix<f64>,
    reg: f64, reg_m: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>) -> Result<DMatrix<f64>, OTError> {

    // Defaults
    let mut iterations = 1000;
    if let Some(val) = num_iter_max {
        iterations = val;
    }

    let mut stop = 1E-6;
    if let Some(val) = stop_threshold {
        stop = val;
    }

    let (m0, m1) = M.shape();
    let dim_a;
    let dim_b;

    // if a and b empty, default to uniform distribution
    if a.len() == 0 {
        *a = DVector::from_vec(vec![1f64 / (m0 as f64); m0]);
        dim_a = m0;
    } else {
        dim_a = a.len();
    }

    if b.len() == 0 {
        *b = DVector::from_vec(vec![1f64 / (m1 as f64); m1]);
        dim_b = m1;
    } else {
        dim_b = b.len();
    }

    // Check dimensions
    if dim_a != m0 && dim_b != m1 {
        return Err( OTError::DimensionError{ dim_a, dim_b, dim_m_0: m0, dim_m_1: m1 } )
    }

    // Ensure the same mass
    if a.sum() != b.sum() {
        return Err( OTError::HistogramSumError{ mass_a: a.sum(), mass_b: b.sum() } )
    }

    // we assume that no distances are null except those of the diagonal distances
    let mut u = DVector::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = DVector::<f64>::from_vec(vec![1f64 / (dim_b as f64); dim_b]);

    // K = exp(-M/reg)
    let mut k = M.clone();
    for m in k.iter_mut() {
        *m = (*m/-reg).exp();
    }

    let fi = reg_m / (reg_m + reg);

    for _ in 0..iterations {

        let uprev = u.clone();
        let vprev = v.clone();

        // Update u and v
        // u = (a/kv) ** fi
        let kv = &k * &v;
        for (i, ele_u) in u.iter_mut().enumerate() {
            *ele_u = (a[i] / kv[i]).powf(fi);
        }

        // v = (b/ktu) ** fi
        let ktu = &k.transpose() * &u;
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

    // nhists = 1 case only
    // diag(u)*K*diag(v)
    for (i, mut row) in k.row_iter_mut().enumerate() {
        for (j, k) in row.iter_mut().enumerate() {
            *k *= u[i] * v[j];
        }
    }

    Ok(k)

}


#[cfg(test)]
mod tests {

    use na::{DVector, DMatrix};

    #[test]
    fn test_sinkhorn_knopp_unbalanced() {

        let mut a = DVector::from_vec(vec![1./3., 1./3., 1./3.]);
        let mut b = DVector::from_vec(vec![1./4., 1./4., 1./4., 1./4.]);
        let reg = 2.0;
        let reg_m = 3.0;
        let mut m = DMatrix::<f64>::from_row_slice(3, 4,
                    &[0.5, 0.0, 0.0, 0.0,
                    0.0, 0.5, 0.0, 0.0,
                    0.0, 0.0, 0.5, 0.0]);

        let result = match super::sinkhorn_knopp_unbalanced(&mut a, &mut b, &mut m, reg, reg_m, None, None) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error)
        };

        let truth = DMatrix::from_row_slice(3,4, 
                    &[0.1275636 , 0.1637949 , 0.1637949 , 0.15643794,
                    0.1637949 , 0.1275636 , 0.1637949 , 0.15643794,
                    0.1637949 , 0.1637949 , 0.1275636 , 0.15643794]);

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));

    }

}
