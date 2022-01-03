
use ndarray::prelude::*;
use anyhow::anyhow;

use crate::OTError;

enum FastTransportResult {
    Infeasible=0,
    Optimal=1,
    Unbounded=2,
    MaxIterReached=3
}

/// Solves the Earth Movers distance problem and returns the OT matrix
/// a: Source histogram (uniform weight if empty)
/// b: Target histogram (uniform weight if empty)
/// M: Loss matrix (row-major)
/// num_iter_max: maximum number of iterations before stopping the optimization algorithm if it has
/// not converged (default = 100000)
/// center_dual: If True, centers the dual potential using function (default = true)
#[allow(non_snake_case)]
pub fn emd(a: &mut Array1<f64>, b: &mut Array1<f64>,
       M: &mut Array2<f64>, num_iter_max: Option<i32>,
       center_dual: Option<bool>) -> Result<Array2<f64>, OTError> {

    // Defaults
    let mut iterations = 100000;
    if let Some(val) = num_iter_max {
        iterations = val;
    }

    let mut center = true;
    if let Some(val) = center_dual {
        center = val;
    }

    let mshape = M.shape();
    let m0 = mshape[0];
    let m1 = mshape[1];
    let dim_a;
    let dim_b;

    // if a and b empty, default to uniform distribution
    if a.is_empty() {
        *a = Array1::from_vec(vec![1f64/(m0 as f64); m0]);
        dim_a = m0;
    } else {
        dim_a = a.len();
    }

    if b.is_empty() {
        *b = Array1::from_vec(vec![1f64/(m1 as f64); m1]);
        dim_b = m1;
    } else {
        dim_b = b.len();
    }

    // Check dimensions
    if dim_a != m0 || dim_b != m1 {
        return Err( OTError::DimensionError{ dim_a, dim_b, dim_m_0: m0, dim_m_1: m1 } )
    }

//     // Ensure the same mass
//     if a.sum() != b.sum() {
//         return Err( OTError::HistogramSumError{ mass_a: a.sum(), mass_b: b.sum() } )
//     }

    // b = b * a.sum/b.sum
    // b.scale_mut(a.sum()/b.sum());
    *b *= a.sum() / b.sum();

    // not_asel == ~asel, not_bsel == ~bsel
    // binary indexing of non-zero weights
    let mut not_asel = Array1::<i32>::zeros(dim_a);
    for (i, val) in a.iter().enumerate() {
        if *val == 0f64 {
            not_asel[i] = 1;
        } else {
            not_asel[i] = 0;
        }
    }

    let mut not_bsel = Array1::<i32>::zeros(dim_b);
    for (i, val) in b.iter().enumerate() {
        if *val == 0f64 {
            not_bsel[i] = 1;
        } else {
            not_bsel[i] = 0;
        }
    }

    let (G, _cost, mut u, mut v, result_code) = emd_c(a, b, M, iterations);

    if center {
        let result = center_ot_dual(&u, &v, Some(a), Some(b));
        u = result.0;
        v = result.1;
    }

    if not_asel.sum() > 1 || not_bsel.sum() > 1 {
        let result = estimate_dual_null_weights(&u, &v, a, b, M);
        u = result.0;
        v = result.1;
    }

    // Propogate errors if there are any
    check_result(result_code)?;

    Ok(G)

}


/// Wrapper of C++ FastTransport OT Network Simplex solver
#[allow(non_snake_case)]
fn emd_c(a: &mut Array1<f64>, b: &mut Array1<f64>, M: &mut Array2<f64>, max_iter: i32)
    -> (Array2<f64>, f64, Array1<f64>, Array1<f64>, i32) {

    let mshape = M.shape();
    let n1 = mshape[0];
    let n2 = mshape[1];

    let _nmax = n1 + n2 - 1;
    let _nG = 0i32;
    let mut cost = 0f64;
    let mut alpha = Array1::<f64>::zeros(n1);
    let mut beta = Array1::<f64>::zeros(n2);
    let mut G = Array2::<f64>::zeros( (n1, n2) );

    if a.is_empty() {
        *a = Array1::from_vec(vec![1f64/(n1 as f64); n1]);
    }

    if b.is_empty() {
        *b = Array1::from_vec(vec![1f64/(n2 as f64); n2]);
    }

    unsafe {
    let result_code = crate::ffi::EMD_wrap(n1 as i32,
                           n2 as i32,
                           a.as_mut_ptr(),
                           b.as_mut_ptr(),
                           M.as_mut_ptr(),
                           G.as_mut_ptr(),
                           alpha.as_mut_ptr(),
                           beta.as_mut_ptr(),
                           &mut cost,
                           max_iter);

    (G, cost, alpha, beta, result_code)

    }

}

/// Finds a unique dual potential such that the same objective value is achieved for both
/// source and target potentials. Helps ensure stability of the linear program solver
/// when calling multiple times with minor changes.
fn center_ot_dual(
    alpha0: &Array1<f64>, beta0: &Array1<f64>,
    a: Option<&Array1<f64>>, b: Option<&Array1<f64>>)
    -> (Array1<f64>, Array1<f64>)
{

    let a_vec: Array1<f64>;
    let b_vec: Array1<f64>;

    if a == None {
        let ns = alpha0.len();
        a_vec = Array1::from_vec(vec![1f64/(ns as f64); ns]);
    } else {
        a_vec = a.unwrap().clone();
    }

    if b == None {
        let nt = beta0.len();
        b_vec = Array1::from_vec(vec![1f64/(nt as f64); nt]);
    } else {
        b_vec = b.unwrap().clone();
    }


    let c = (b_vec.dot(beta0) - a_vec.dot(alpha0)) / (a_vec.sum() + b_vec.sum());

    (alpha0 + c, beta0 -c)

}

/// Estimate feasible values for 0-weighted dual potentials
///
/// The feasible values are computed efficiently but rather coarsely
///
/// This function is necessary because the C++ solver in emd_c
/// discards all samples in the distributions with 
/// zeros weights. This means that while the primal variable (transport 
/// matrix) is exact, the solver only returns feasible dual potentials
/// on the samples with weights different from zero.
///
/// alpha0: Source dual potential
/// beta0: Target dual potential
/// a: Source distribution (uniform weights if empty)
/// b: Target distribution (uniform weights if empty)
/// M: Loss matrix (row-major)
#[allow(non_snake_case)]
fn estimate_dual_null_weights(
    alpha0: &Array1<f64>, beta0: &Array1<f64>,
    a: &Array1<f64>, b: &Array1<f64>,
    M: &Array2<f64>)
    -> (Array1<f64>, Array1<f64>)
{

    // binary indexing of non-zero weights
    let mut asel = Array1::<i32>::zeros(a.len());
    for (i, val) in a.iter().enumerate() {
        if *val == 0f64 {
            asel[i] = 0;
        } else {
            asel[i] = 1;
        }
    }

    let mut bsel = Array1::<i32>::zeros(b.len());
    for (i, val) in b.iter().enumerate() {
        if *val == 0f64 {
            bsel[i] = 0;
        } else {
            bsel[i] = 1;
        }
    }

    // compute dual constraints violation
    // NOTE: alpha0 as a col vec added to each col of row vec beta0
    // to make a matrix
    let mut tmp = Array2::<f64>::zeros( (alpha0.len(), beta0.len()) );
    for (i, valx) in alpha0.iter().enumerate() {
       for (j, valy) in beta0.iter().enumerate() {
            tmp[(i,j)] = valx + valy;
        }
    }

    let constraint_violation = tmp - M;

    // compute largest violation per line and columns
    // NOTE: we want the max in col dimension for aviol and max in row dimension for bviol
    let mut aviol = Array1::<f64>::zeros(alpha0.len());
    for (j, row) in constraint_violation.axis_iter(Axis(0)).enumerate() {
        aviol[j] = row.iter().fold(0f64, |a, &b| a.max(b));
    }

    let mut bviol = Array1::<f64>::zeros(beta0.len());
    for (i, col) in constraint_violation.axis_iter(Axis(1)).enumerate() {
        bviol[i] = col.iter().fold(0f64, |a, &b| a.max(b));
    }

    // update
    let max_aviol = aviol.iter().fold(0f64, |a, &b| a.max(b));
    let mut alpha_up = Array1::<f64>::zeros(alpha0.len());
    for (i, selection) in asel.iter().enumerate() {
        alpha_up[i] = -1f64 * (!selection as f64) * (max_aviol as f64);
    }

    let max_bviol = bviol.iter().fold(0f64, |a, &b| a.max(b));
    let mut beta_up = Array1::<f64>::zeros(beta0.len());
    for (i, selection) in bsel.iter().enumerate() {
        beta_up[i] = -1f64 * (!selection as f64) * (max_bviol as f64);
    }

    let alpha = alpha0 + alpha_up;
    let beta = beta0 + beta_up;

    center_ot_dual(&alpha, &beta, Some(a), Some(b))

}

/// Convert FastTransport error codes to EMDErrors
fn check_result(result_code: i32) -> Result<(), OTError> {

    if result_code == FastTransportResult::Optimal as i32 {
        Ok(())
    } else if result_code ==FastTransportResult::Unbounded as i32 {
        Err( OTError::FastTransportError(String::from("Problem unbounded")) )
    } else if result_code ==FastTransportResult::MaxIterReached as i32 {
        Err( OTError::FastTransportError(String::from("numItermax reached before optimality. Try to increase numItermax")) )
    } else if result_code ==FastTransportResult::Infeasible as i32 {
        Err( OTError::FastTransportError(String::from("Problem infeasible. Check that a and b are in the simplex")) )
    } else {
        Err( OTError::Other(anyhow!("oops!")) )
    }

}


#[cfg(test)]
mod tests {

    use ndarray::array;

    #[allow(non_snake_case)]
    #[test]
    fn test_emd_c() {

        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let mut M = array![[0.0, 1.0],[1.0, 0.0]];

        let (G, _cost, _u, _v, _result_code) = super::emd_c(&mut a, &mut b, &mut M, 10000);

        let truth = array![[0.5, 0.0],[0.0, 0.5]];

        // println!("{:?}", G);

        assert_eq!(G, truth);

    }

    #[allow(non_snake_case)]
    #[test]
    fn test_emd() {

        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let mut M = array![[0.0, 1.0],[1.0, 0.0]];

        let gamma = match super::emd(&mut a, &mut b, &mut M, None, None) {
            Ok(result) => result,
            Err(error) => panic!("{:?}", error)
        };

        let truth = array![[0.5, 0.0],[0.0, 0.5]];

        // println!("{:?}", gamma);

        assert_eq!(gamma, truth);

    }

}
