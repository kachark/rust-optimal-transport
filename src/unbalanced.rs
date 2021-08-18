
pub mod unbalanced {

use na::{DVector, DMatrix};

pub enum UnbalancedSolverType {
    Sinkhorn,
    SinkhornStabilized,
    SinkhornRegScaling
}


/// Solves the unbalanced entropic regularization optimal transport problem and return the OT plan
/// a: Unnormalized histogram of dimension dim_a
/// b: One or multiple unnormalized histograms of dimension dim_b
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// reg_m: Marginal relaxation term > 0
/// method: method used for the solver either 'sinkhorn', 'sinkhorn_stabilized', or
/// 'sinkhorn_reg_scaling'
/// num_iter_max: Max number of iterations
/// stop_threshold: Stop threshold on error (>0)
/// verbose: Print information along iterations
pub fn sinkhorn_unbalanced(
    a: &mut DVector<f64>, b: &mut DVector<f64>, M: &mut DMatrix<f64>,
    reg: f64, reg_m: f64, method: UnbalancedSolverType,
    num_iter_max: Option<i32>, stop_threshold: Option<f64>,
    verbose: Option<bool>) -> DMatrix<f64> {

    if b.len() < 2 {
        b.transpose();
    }

    match method {
        Sinkhorn => sinkhorn_knopp_unbalanced(
                                              a, b, M, reg, reg_m,
                                              num_iter_max,
                                              stop_threshold,
                                              verbose),
        SinkhornStabilized => sinkhorn_stabilized_unbalanced(
                                              a, b, M, reg, reg_m,
                                              num_iter_max,
                                              stop_threshold,
                                              verbose),
        SinkhornRegScaling => sinkhorn_knopp_unbalanced(
                                              a, b, M, reg, reg_m,
                                              num_iter_max,
                                              stop_threshold,
                                              verbose),
    }

}


/// Solves the unbalanced entropic regularization optimal transport problem and return the loss
/// a: Unnormalized histogram of dimension dim_a
/// b: One or multiple unnormalized histograms of dimension dim_b
/// M: Loss matrix
/// reg: Entropy regularization term > 0
/// reg_m: Marginal relaxation term > 0
/// num_iter_max: Max number of iterations
/// stop_threshold: Stop threshold on error (>0)
/// verbose: Print information along iterations
fn sinkhorn_knopp_unbalanced(
    a: &mut DVector<f64>, b: &mut DVector<f64>, M: &mut DMatrix<f64>,
    reg: f64, reg_m: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>,
    verbose: Option<bool>) -> DMatrix<f64> {

    DMatrix::<f64>::zeros(2,2)

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
    a: &mut DVector<f64>, b: &mut DVector<f64>, M: &mut DMatrix<f64>,
    reg: f64, reg_m: f64, num_iter_max: Option<i32>, stop_threshold: Option<f64>,
    verbose: Option<bool>) -> DMatrix<f64> {

    DMatrix::<f64>::zeros(2,2)

}

}