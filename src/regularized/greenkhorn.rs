
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use anyhow::anyhow;

use crate::OTError;

/// Solves the entropic regularization optimal transport problem and return the OT matrix
/// Uses the Greedy Sinkhorn method:
/// Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
/// by Jason Altschuler, Jonathan Weed, Philippe Rigollet
///
/// a: Unnormalized histogram of dimension dim_a
/// b: Unnormalized histogram of dimension dim_b
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// num_iter_max: Max number of iterations (default = 1000)
/// stop_threshold: Stop threshold on error (> 0) (default = 1E-6)
pub fn greenkhorn(
    a: &mut Array1<f64>, b: &mut Array1<f64>, M: &mut Array2<f64>,
    reg: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>) -> Result<Array2<f64>, OTError>{

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
    let mut stop_val;

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
        return Err( OTError::DimensionError{ dim_a, dim_b, dim_m_0: m0, dim_m_1: m1 } )
    }

    // Ensure the same mass
    if a.sum() != b.sum() {
        return Err( OTError::HistogramSumError{ mass_a: a.sum(), mass_b: b.sum() } )
    }

    let mut u = Array1::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = Array1::<f64>::from_vec(vec![1f64 / (dim_b as f64); dim_b]);

    // K = exp(-M/reg)
    let k = Array2::from_shape_fn( (mshape[0], mshape[1]), |(i, j)| (-M[[i,j]] / reg).exp() );

    let mut G = &u.diag().t() * &k * &v.diag();

    // G.sum(1) - a
    let mut viol: Array1<f64> = G.sum_axis(Axis(1)).iter()
        .enumerate()
        .map(|(i, sum_i)| sum_i - a[i])
        .collect();

    // G.sum(0) - b
    let mut viol_2: Array1<f64> = G.sum_axis(Axis(0)).iter()
        .enumerate()
        .map(|(i, sum_i)| sum_i - b[i])
        .collect();

    for _ in 0..iterations {

        // Absolute values
        let viol_abs: Array1<f64> = viol.iter().map(|x| x.abs()).collect(); // NOTE: messes up somehow
        let viol_2_abs: Array1<f64> = viol_2.iter().map(|x| x.abs()).collect();

        // Argmax
        let i_1 = match viol_abs.argmax() {
            Ok(val) => val,
            // Propagate ndarray-stats error
            Err(err) => return Err( OTError::Other(anyhow!(err)) )
        };

        let i_2 = match viol_2_abs.argmax() {
            Ok(val) => val,
            // Propagate ndarray-stats error
            Err(err) => return Err( OTError::Other(anyhow!(err)) )
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

        if stop_val <= stop {
            break;
        }

    }

    Ok(G)

}

#[cfg(test)]
mod tests {

    use ndarray::prelude::*;

    #[test]
    fn test_greenkhorn() {

        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let reg = 1.0;
        let mut m = array![[0.0, 1.0], [1.0, 0.0]];

        let result = match super::greenkhorn(&mut a, &mut b, &mut m, reg, None, None) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error)
        };

        println!("{:?}", result);

        let truth = array![[0.36552929, 0.13447071], [0.13447071, 0.36552929]];

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));

    }

}

