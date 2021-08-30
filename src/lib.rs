

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

pub mod ot;
pub mod regularized;
pub mod unbalanced;
pub mod utils;

#[cxx::bridge]
mod ffi {

    extern "C++" {
        include!("rust-optimal-transport/src/fast_transport/EMD.h");

        unsafe fn EMD_wrap(n1: i32, n2: i32, X: *mut f64, Y: *mut f64, D: *mut f64, G: *mut f64,
                    alpha: *mut f64, beta: *mut f64, cost: *mut f64, maxIter: i32) -> i32;
    }

}

enum ProblemType {
    Infeasible=0,
    Optimal=1,
    Unbounded=2,
    MaxIterReached=3
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


