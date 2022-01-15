use super::FastTransportResult;
use crate::OTError;
use anyhow::anyhow;
use ndarray::prelude::*;

/// Finds a unique dual potential such that the same objective value is achieved for both
/// source and target potentials. Helps ensure stability of the linear program solver
/// when calling multiple times with minor changes.
pub fn center_ot_dual(
    alpha0: &Array1<f64>,
    beta0: &Array1<f64>,
    a: Option<&Array1<f64>>,
    b: Option<&Array1<f64>>,
) -> (Array1<f64>, Array1<f64>) {
    let a_vec: Array1<f64>;
    let b_vec: Array1<f64>;

    if a == None {
        let ns = alpha0.len();
        a_vec = Array1::from_vec(vec![1f64 / (ns as f64); ns]);
    } else {
        a_vec = a.unwrap().clone();
    }

    if b == None {
        let nt = beta0.len();
        b_vec = Array1::from_vec(vec![1f64 / (nt as f64); nt]);
    } else {
        b_vec = b.unwrap().clone();
    }

    let c = (b_vec.dot(beta0) - a_vec.dot(alpha0)) / (a_vec.sum() + b_vec.sum());

    (alpha0 + c, beta0 - c)
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
pub fn estimate_dual_null_weights(
    alpha0: &Array1<f64>,
    beta0: &Array1<f64>,
    a: &Array1<f64>,
    b: &Array1<f64>,
    M: &Array2<f64>,
) -> (Array1<f64>, Array1<f64>) {
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
    let mut tmp = Array2::<f64>::zeros((alpha0.len(), beta0.len()));
    for (i, valx) in alpha0.iter().enumerate() {
        for (j, valy) in beta0.iter().enumerate() {
            tmp[(i, j)] = valx + valy;
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
pub fn check_result(result_code: i32) -> Result<(), OTError> {
    if result_code == FastTransportResult::Optimal as i32 {
        Ok(())
    } else if result_code == FastTransportResult::Unbounded as i32 {
        Err(OTError::FastTransportError(String::from(
            "Problem unbounded",
        )))
    } else if result_code == FastTransportResult::MaxIterReached as i32 {
        Err(OTError::FastTransportError(String::from(
            "numItermax reached before optimality. Try to increase numItermax",
        )))
    } else if result_code == FastTransportResult::Infeasible as i32 {
        Err(OTError::FastTransportError(String::from(
            "Problem infeasible. Check that a and b are in the simplex",
        )))
    } else {
        Err(OTError::Other(anyhow!("oops!")))
    }
}
