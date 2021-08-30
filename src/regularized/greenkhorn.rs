
use na::{DVector, DMatrix};

pub fn greenkhorn(
    a: &mut DVector<f64>, b: &mut DMatrix<f64>, M: &mut DMatrix<f64>,
    reg: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>,
    verbose: Option<bool>) -> DMatrix<f64> {

    // Defaults
    let mut iterations = 1000;
    if let Some(val) = num_iter_max {
        iterations = val;
    }

    let mut stop = 1E-6;
    if let Some(val) = stop_threshold {
        stop = val;
    }

    let mut _verbose_mode = false;
    if let Some(val) = verbose {
        _verbose_mode = val;
    }

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
    let mut u = DVector::<f64>::from_vec(vec![1f64 / (dim_a as f64); dim_a]);
    let mut v = DMatrix::<f64>::from_row_slice(dim_b, n_hists, vec![1f64 / (dim_b as f64); dim_b].as_slice());

    // K = exp(-M/reg)
    let mut k = M.clone();
    for m in k.iter_mut() {
        *m = (*m/-reg).exp();
    }

    let mut G = k.clone();
    // diag(u)*K*diag(v)
    for (i, mut row) in G.row_iter_mut().enumerate() {
        for (j, g) in row.iter_mut().enumerate() {
            *g *= u[i] * v[j];
        }
    }

    let mut viol = DVector::<f64>::zeros(dim_a);
    let mut viol_2 = DVector::<f64>::zeros(dim_b);
    let mut stop_val;

    // G.sum(1) - a
    for (i, x) in G.row_sum().iter().enumerate() {
        viol[i] = x - a[i];
    }

    // G.sum(0) - b
    for (i, x) in G.column_sum().iter().enumerate() {
        viol_2[i] = x - b[i];
    }

    for _ in 0..iterations {

        let (i_1, val_1) = viol.abs().argmax();
        let (i_2, val_2) = viol_2.abs().argmax();
        let m_viol_1 = val_1.abs();
        let m_viol_2 = val_2.abs();

        if m_viol_1 >= m_viol_2 {
            stop_val = m_viol_1;
        } else {
            stop_val = m_viol_2;
        }

        if m_viol_1 > m_viol_2 {

            let old_u = u[i_1];
            let k_i1_vec: Vec<f64> = k.row(i_1).iter().map(|x| *x).collect();
            let k_i1 = DVector::<f64>::from_vec(k_i1_vec);
            let denom = k_i1.dot(&v);
            u[i_1] = a[i_1] / denom;

            // G[i_1, :] = u[i_1] * k[i_1, :] * v
            for (i, mut row) in G.row_iter_mut().enumerate() {

                if i != i_1 {
                    continue;
                }

                for (j, g) in row.iter_mut().enumerate() {
                    *g = u[i_1] * k[(i_1, j)] * v[j];
                }

            }

            viol[i_1] = u[i_1] * denom - a[i_1];

            // viol_2 += (K[i_1, :].T * (u[i_1] - old_u) * v)
            for (j, ele) in viol_2.iter_mut().enumerate() {
                *ele += &k_i1.transpose()[j] * (u[i_1] - old_u) * v[j];
            }

        } else {

            let old_v = v[i_2];
            let k_i2_vec: Vec<f64> = k.column(i_2).iter().map(|x| *x).collect();
            let k_i2 = DVector::<f64>::from_vec(k_i2_vec);
            let denom = k_i2.transpose().dot(&u.transpose());
            v[i_2] = b[i_2] / denom;

            // G[i_1, :] = u[i_1] * k[i_1, :] * v
            for (i, mut col) in G.column_iter_mut().enumerate() {

                if i != i_2 {
                    continue;
                }

                for (j, g) in col.iter_mut().enumerate() {
                    *g = u[j] * k[(j, i_2)] * v[i_2];
                }

            }

            // viol += (-old_v + v[i_2]) * K[:, i_2] * u
            for (j, ele) in viol.iter_mut().enumerate() {
                *ele += (-old_v + v[i_2]) * k_i2[j] * u[j];
            }

            viol_2[i_2] = v[i_2] * k_i2.dot(&u) - b[i_2];

        }

        if stop_val <= stop {
            break;
        }

    }

    G

}

#[cfg(test)]
mod tests {

    use na::{DVector, DMatrix};

    #[test]
    fn test_greenkhorn() {

        let mut a = DVector::from_vec(vec![0.5, 0.5]);
        let mut b = DMatrix::from_vec(2, 1, vec![0.5, 0.5]);
        let reg = 1.0;
        let mut m = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);

        let result = super::greenkhorn(&mut a, &mut b, &mut m,
                                            reg, None, None, None);

        println!("{:?}", result);

        let truth = DMatrix::from_row_slice(2,2,
                    &[0.36552929, 0.13447071,
                    0.13447071, 0.36552929]);

        assert!(result.relative_eq(&truth, 1E-6, 1E-2));

    }

}

