
use ndarray::prelude::*;
use ndarray_einsum_beta::einsum;
use ndarray_linalg::norm::Norm;

use crate::OTError;

/// Solves the unbalanced entropic regularization optimal transport problem and return the OT
/// matrix
/// a: Source sample weights (defaults to uniform weight if empty)
/// b: Target sample weights (defaults to uniform weight if empty)
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// reg_m: Marginal relaxation term > 0
/// num_iter_max: Max number of iterations (default = 1000)
/// stop_threshold: Stop threshold on error (> 0) (default = 1E-6)
pub fn sinkhorn_knopp_unbalanced(
    a: &mut Array1<f64>, b: &mut Array1<f64>, M: &mut Array2<f64>,
    reg: f64, reg_m: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>) -> Result<Array2<f64>, OTError> {

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
    let fi = reg_m / (reg_m + reg);

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
        return Err( OTError::WeightDimensionError{ dim_a, dim_b, dim_m_0: m0, dim_m_1: m1 } )
    }

    // we assume that no distances are null except those of the diagonal distances
    let mut u = Array1::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = Array1::<f64>::from_vec(vec![1f64 / (dim_b as f64); dim_b]);

    // K = exp(-M/reg)
    let mut k = Array2::from_shape_fn( (mshape[0], mshape[1]), |(i, j)| (-M[[i,j]] / reg).exp() );

    for count in 0..iterations {

        let uprev = u.clone();
        let vprev = v.clone();

        // Update u and v
        // u = (a/kv) ** fi
        let kv = &k.dot(&v);
        for (i, ele_u) in u.iter_mut().enumerate() {
            *ele_u = (a[i] / kv[i]).powf(fi);
        }

        // v = (b/ktu) ** fi
        let ktu = &k.t().dot(&u);
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

        if count % 10 == 0 {

            let mut tmp = einsum("i,ij,j->j", &[&u,&k,&v]).unwrap();
            tmp -= &b.clone();
            let err = Norm::norm(&tmp);
            if err < stop {
                break;
            }

        }

    }

    // nhists = 1 case only
    // diag(u)*K*diag(v)
    for (i, mut row) in k.axis_iter_mut(Axis(0)).enumerate() {
        for (j, k) in row.iter_mut().enumerate() {
            *k *= u[i] * v[j];
        }
    }

    Ok(k)

}


#[cfg(test)]
mod tests {

    use ndarray::prelude::*;
    use crate::utils::distributions::get_1D_gauss_histogram;

    #[test]
    fn test_sinkhorn_knopp_unbalanced() {

        let reg = 0.1;
        let reg_m = 1.0;

        // Generate data
        let n = 5;
        // let x = Array::range(0.0, n as f64, 1.0);

        let mean_source = 20.0;
        let std_source = 5.0;

        let mean_target = 60.0;
        let std_target = 10.0;

        let mut source_mass = match get_1D_gauss_histogram(n, mean_source, std_source) {
            Ok(val) => val,
            Err(err) => panic!("{:?}", err)
        };

        let mut target_mass = match get_1D_gauss_histogram(n, mean_target, std_target) {
            Ok(val) => val,
            Err(err) => panic!("{:?}", err)
        };

        // unbalance the source and target mass distribution
        target_mass *= 5.0;

        // ot.dist(x.reshape(n,1), x.reshape(n,1))
        let mut m = array![[ 0.,  1.,  4.,  9., 16.],
                        [ 1.,  0.,  1.,  4.,  9.],
                        [ 4.,  1.,  0.,  1.,  4.],
                        [ 9.,  4.,  1.,  0.,  1.],
                        [16.,  9.,  4.,  1.,  0.]];

        let result = match super::sinkhorn_knopp_unbalanced(&mut source_mass, &mut target_mass, &mut m, reg, reg_m, None, None) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error)
        };

        let truth = array![[9.10685822e-02, 2.27553653e-06, 1.34495713e-19, 1.88110798e-41, 6.22729686e-72],
                            [1.44579778e-05, 1.75271954e-01, 5.02604874e-06, 3.41052799e-19, 5.47768558e-41],
                            [4.02566983e-18, 2.36773266e-05, 3.29409842e-01, 1.08447896e-05, 8.45057483e-19],
                            [1.96502178e-39, 5.60727063e-18, 3.78481853e-05, 6.04531953e-01, 2.28546207e-05],
                            [1.68120836e-69, 2.32753023e-39, 7.62216791e-18, 5.90666512e-05, 1.08339480e+00]];

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));

    }

}
