

/*
 * Original Python/C++ implementation written by:
 *
 *  Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer,
 *  POT Python Optimal Transport library,
 *  Journal of Machine Learning Research, 22(78):1−8, 2021.
 *  Website: https://pythonot.github.io/
 *
 *  C++ interface written by:
 *  It was written by Antoine Rolet (2014) and mainly consists of a wrapper
 *  of the code written by Nicolas Bonneel available on this page
 *  http://people.seas.harvard.edu/~nbonneel/FastTransport/
 *
*/


extern crate nalgebra as na;

use std::error::Error;
use std::fmt;
use na::{dvector, DVector, DMatrix};

#[cxx::bridge]
mod ffi {

    extern "C++" {
        include!("rust-optimal-transport/src/EMD.h");

        unsafe fn EMD_wrap(n1: i32, n2: i32, X: *mut f64, Y: *mut f64, D: *mut f64, G: *mut f64,
                    alpha: *mut f64, beta: *mut f64, cost: *mut f64, maxIter: i32) -> i32;
    }

}

enum ProblemType {
    INFEASIBLE=0,
    OPTIMAL=1,
    UNBOUNDED=2,
    MAX_ITER_REACHED=3
}

#[derive(Debug)]
struct ProblemError {
    details: String
}

impl fmt::Display for ProblemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for ProblemError {
    fn description(&self) -> &str {
        &self.details
    }
}


// TODO: return ProblemType/ResultCode instead of i32
#[allow(non_snake_case)]
fn emd_c(a: &mut DVector<f64>, b: &mut DVector<f64>, M: &mut DMatrix<f64>, max_iter: i32)
    -> (DMatrix<f64>, f64, DVector<f64>, DVector<f64>, i32) {

    let (n1, n2) = M.shape();
    let _nmax = n1 + n2 - 1;
    let mut result_code = 0i32;
    let _nG = 0i32;
    let mut cost = 0f64;
    let mut alpha = dvector![0f64];
    let mut beta = dvector![0f64];
    let mut G = DMatrix::<f64>::zeros(2, 2);
    let Gv = dvector![0];

    if a.len() != 0 {
        *a = DVector::from_vec(vec![1f64; n1]).scale(1f64/n1 as f64);
    }

    if b.len() != 0 {
        *b = DVector::from_vec(vec![1f64; n2]).scale(1f64/n2 as f64);
    }

    G = DMatrix::<f64>::zeros(n1, n2);

    unsafe {
    result_code = ffi::EMD_wrap(n1 as i32,
                           n2 as i32,
                           a.as_mut_ptr(),
                           b.as_mut_ptr(),
                           M.as_mut_ptr(),
                           G.as_mut_ptr(),
                           alpha.as_mut_ptr(),
                           beta.as_mut_ptr(),
                           &mut cost,
                           max_iter);
    }

    (G, cost, alpha, beta, result_code)

}

fn center_ot_dual(
    alpha0: &DVector<f64>, beta0: &DVector<f64>,
    a: Option<&DVector<f64>>, b: Option<&DVector<f64>>)
    -> (DVector<f64>, DVector<f64>)
{

    let a_vec: DVector<f64>;
    let b_vec: DVector<f64>;

    if a == None {
        let ns = alpha0.len();
        a_vec = DVector::from_vec(vec![1f64; ns]).scale(1f64/ (ns as f64))
    } else {
        a_vec = a.unwrap().clone();
    }

    if b == None {
        let nt = beta0.len();
        b_vec = DVector::from_vec(vec![1f64; nt]).scale(1f64/ (nt as f64))
    } else {
        b_vec = b.unwrap().clone();
    }

    let c = (b_vec.dot(beta0) - a_vec.dot(alpha0)) / (a_vec.sum() + b_vec.sum());

    (alpha0.add_scalar(c), beta0.add_scalar(-c))

}

// TODO: implement estimate_dual_null_weights()
fn estimate_dual_null_weights(
    alpha0: &DVector<f64>, beta0: &DVector<f64>,
    a: &DVector<f64>, b: &DVector<f64>,
    M: &DMatrix<f64>)
    -> (DVector<f64>, DVector<f64>)
{

    // binary indexing of non-zero weights
    let mut asel = DVector::<i32>::zeros(a.len());
    for (i, val) in a.iter().enumerate() {
        if *val == 0f64 {
            asel[i] = 0;
        } else {
            asel[i] = 1;
        }
    }

    let mut bsel = DVector::<i32>::zeros(b.len());
    for (i, val) in b.iter().enumerate() {
        if *val == 0f64 {
            bsel[i] = 0;
        } else {
            bsel[i] = 1;
        }
    }

    // compute dual constraints violation
    // TODO: alpha0 as a col vec added to each col of row vec beta0
    // to make a matrix
    let mut tmp = DMatrix::<f64>::zeros(alpha0.len(), beta0.len());
    for (i, valx) in alpha0.iter().enumerate() {
       for (j, valy) in beta0.iter().enumerate() {
            tmp[(i,j)] = valx + valy;
        }
    }

    let constraint_violation = tmp - M;

    // compute largest violation per line and columns
    // TODO: we want the max in col dimension for aviol and max in row dimension for bviol
    let mut aviol = DVector::<f64>::zeros(alpha0.len());
    for (j, row) in constraint_violation.row_iter().enumerate() {
        aviol[j] = row.max();
    }

    let mut bviol = DVector::<f64>::zeros(beta0.len());
    for (i, col) in constraint_violation.column_iter().enumerate() {
        bviol[i] = col.max();
    }

    // update
    let max_aviol = aviol.max();
    let mut alpha_up = DVector::<f64>::zeros(alpha0.len());
    for (i, selection) in asel.iter().enumerate() {
        alpha_up[i] = -1f64 * (!selection as f64) * (max_aviol as f64);
    }

    let max_bviol = bviol.max();
    let mut beta_up = DVector::<f64>::zeros(beta0.len());
    for (i, selection) in bsel.iter().enumerate() {
        beta_up[i] = -1f64 * (!selection as f64) * (max_bviol as f64);
    }

    let alpha = alpha0 + alpha_up;
    let beta = beta0 + beta_up;

    center_ot_dual(&alpha, &beta, Some(a), Some(b))

}


fn check_result(result_code: i32) -> Result<(), ProblemError> {

    if result_code == ProblemType::OPTIMAL as i32 {
        Ok(())
    } else if result_code == ProblemType::UNBOUNDED as i32 {
        Err( ProblemError{details: String::from("Problem unbounded")} )
    } else if result_code == ProblemType::MAX_ITER_REACHED as i32 {
        Err( ProblemError{details: String::from("numItermax reached before optimality. Try to increase numItermax")} )
    } else {
        Err( ProblemError{details: String::from("Problem infeasible. Check that a and b are in the simplex")} )
    }

}


#[cfg(test)]
mod tests {

    use na::{dvector, DVector, DMatrix};
    use crate::ffi::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_emd_c() {

        let mut a = dvector![0.5, 0.5];
        let mut b = dvector![0.5, 0.5];
        let mut M = DMatrix::<f64>::from_row_slice(2, 2,
                                &[0.0, 1.0,
                                1.0, 0.0]);

        let (G, cost, u, v, result_code) = super::emd_c(&mut a, &mut b, &mut M, 10000);

        println!("{:?}", G);

    }
}
