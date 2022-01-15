
use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::StandardNormal;

use anyhow::anyhow;
use thiserror::Error;

// TODO: Add additional error cases for DistributionError enum
#[derive(Error, Debug)]
pub enum DistributionError {
    #[error("Oops!")]
    Oops(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Returns a 1D histogram for a gaussian distribution
/// n: number of bins in histogram
/// mean: mean value of distribution
/// std: standard distribution of distribution
pub fn get_1D_gauss_histogram(
    n: i32,
    mean: f64,
    std: f64,
) -> Result<Array1<f64>, DistributionError> {
    let x = Array1::<f64>::range(0.0, n as f64, 1.0);
    let var = std.powf(2.0);
    let denom = 2.0 * var;
    let diff = &x - mean;
    let numerator = -&diff * &diff;
    let mut result: Array1<f64> = numerator.iter().map(|val| (val / denom).exp()).collect();
    let summed_val = result.sum();

    result /= summed_val;

    // TODO: add error handling

    Ok(result)
}

/// Returns n samples drawn from a 2D gaussian distribution
/// n: number of samples to take
/// mean: mean values (x,y) of distribution
/// cov: covariance matrix of the distribution
pub fn sample_2D_gauss(
    n: i32,
    mean: &Array1<f64>,
    cov: &Array2<f64>,
) -> Result<Array2<f64>, DistributionError> {
    let cov_shape = cov.shape();

    if n <= 0 {
        return Err(DistributionError::Oops(
            "n is not greater than zero".to_string(),
        ));
    }

    if mean.is_empty() || cov.is_empty() {
        return Err(DistributionError::Oops(
            "zero length mean or covariance".to_string(),
        ));
    }

    if cov_shape[0] != mean.len() && cov_shape[1] != mean.len() {
        return Err(DistributionError::Oops(
            "covariance dimensions do not match mean dimensions".to_string(),
        ));
    }

    let mut rng = thread_rng();
    let mut samples = Array2::<f64>::zeros((n as usize, 2));
    for mut row in samples.axis_iter_mut(Axis(0)) {
        row[0] = rng.sample(StandardNormal);
        row[1] = rng.sample(StandardNormal);
    }

    // add small perturbation to covariance matrix for numerical stability
    let epsilon = 0.0001;
    let cov_perturbed = cov + Array2::<f64>::eye(cov_shape[0]) * epsilon;

    // Compute cholesky decomposition
    let lower = match cov_perturbed.cholesky(UPLO::Lower) {
        Ok(val) => val,
        Err(_) => return Err(DistributionError::Other(anyhow!("oops!"))),
    };

    Ok(mean + samples.dot(&lower))
}


#[cfg(test)]
mod tests {

    use ndarray::array;

    #[test]
    fn test_get_1D_gauss_hist() {
        let n = 50;
        let mean = 20.0;
        let std = 5.0;

        let result = match super::get_1D_gauss_histogram(n, mean, std) {
            Ok(val) => val,
            Err(err) => panic!("{:?}", err),
        };

        // TODO: assert correctness get_1D_gauss_histogram()
        // println!("{:?}", result);
    }

    #[test]
    fn test_sample_2D_gauss() {
        let n = 50;
        let mean = array![0.0, 0.0];
        let covariance = array![[1.0, 0.0], [0.0, 1.0]];

        let result = match super::sample_2D_gauss(n, &mean, &covariance) {
            Ok(val) => val,
            Err(err) => panic!("{:?}", err),
        };

        // TODO: assert correctness of sample_2D_gauss()
        // println!("{:?}", result);
    }

}
