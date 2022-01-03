

pub mod distributions {

use ndarray::prelude::*;
use ndarray_linalg::error::LinalgError;
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_linalg::cholesky::*;

use thiserror::Error;
use anyhow::anyhow;

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
pub fn get_1D_gauss_histogram(n: i32, mean: f64, std: f64) -> Result<Array1<f64>, DistributionError> {

    let x = Array1::<f64>::range(0.0, n as f64, 1.0);
    let var = std.powf(2.0);
    let denom = 2.0 * var;
    let diff = &x - mean;
    let numerator = -&diff * &diff;
    let mut result: Array1<f64> = numerator.iter().map(|val| (val/denom).exp()).collect();
    let summed_val = result.sum();

    result /= summed_val;

    // TODO: add error handling

    Ok(result)

}

/// Returns n samples drawn from a 2D gaussian distribution
/// n: number of samples to take
/// mean: mean values (x,y) of distribution
/// cov: covariance matrix of the distribution
pub fn sample_2D_gauss(n: i32, mean: &Array1<f64>, cov: &Array2<f64>) -> Result<Array2<f64>, DistributionError> {

    let cov_shape = cov.shape();

    if n <= 0 {
        return Err(DistributionError::Oops("n is not greater than zero".to_string()));
    }

    if mean.is_empty() || cov.is_empty() {
        return Err(DistributionError::Oops("zero length mean or covariance".to_string()));
    }

    if cov_shape[0] != mean.len() && cov_shape[1] != mean.len() {
        return Err(DistributionError::Oops("covariance dimensions do not match mean dimensions".to_string()));
    }

    let mut rng = thread_rng();
    let mut samples = Array2::<f64>::zeros( (n as usize, 2) );
    for mut row in samples.axis_iter_mut(Axis(0)) {
        row[0] = rng.sample(StandardNormal);
        row[1] = rng.sample(StandardNormal);
    }

    // add small perturbation to covariance matrix for numerical stability
    let epsilon = 0.0001;
    let cov_perturbed = cov + Array2::<f64>::eye(cov_shape[0])*epsilon;

    // Compute cholesky decomposition
    let lower = match cov_perturbed.cholesky(UPLO::Lower) {
        Ok(val) => val,
        Err(_) => return Err( DistributionError::Other(anyhow!("oops!")) )
    };

    Ok(mean + samples.dot(&lower))

}

}


pub mod metrics {

use ndarray::prelude::*;
use ndarray_einsum_beta::*;


pub enum MetricType {
    SqEuclidean,
    Euclidean
}

/// Compute distance between samples in x1 and x2
/// x1: matrix with n1 samples of size d
/// x2: matrix with n2 samples of size d
/// metric: choice of distance metric
pub fn dist(x1: &Array2<f64>, x2: &Array2<f64>, metric: MetricType) -> Array2<f64> {

    match metric {

        MetricType::SqEuclidean => euclidean_distances(x1, x2, true),
        MetricType::Euclidean => euclidean_distances(x1, x2, false)

    }

}

/// Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair
/// of vectors
/// X: matrix of nsamples x nfeatures
/// Y: matrix of nsamples x nfeatures
/// squared: Return squared Euclidean distances
fn euclidean_distances(x: &Array2<f64>, y: &Array2<f64>, squared: bool) -> Array2<f64> {

    // einsum('ij,ij->i', X, X)
    // repeated i and j for both x and y inout matrices : multiply those components
    // - element-wise multiplication
    // ommitted j in output : sum along j axis
    // - summation in j
    let a2 = einsum("ij,ij->i", &[x, x]).unwrap();
    // einsum('ij,ij->i', Y, Y)
    let b2 = einsum("ij,ij->i", &[y, y]).unwrap();

    let mut c = (x.dot(&y.t())) * -2f64;

    // c += a2[:, None]
    for (mut row, a2val) in c.axis_iter_mut(Axis(0)).zip(&a2.t()) {
        for ele in row.iter_mut() {
            *ele += a2val;
        }
    }

    // c += b2[None, :]
    for (mut col, b2val) in c.axis_iter_mut(Axis(1)).zip(&b2) {
        for ele in col.iter_mut() {
            *ele += b2val;
        }
    }

    // c = nx.maximum(c, 0)
    for val in c.iter_mut() {
        if *val <= 0f64 {
            *val = 0f64;
        }
    }

    if !squared {

        // np.sqrt(c)
        for val in c.iter_mut() {
            *val = val.powf(0.5);
        }

    }


    if x == y {

        // ones matrix with diagonals set to zero
        let mut anti_diag = Array2::<f64>::ones( (a2.len(), b2.len()) );
        for ele in anti_diag.diag_mut().iter_mut() {
            *ele = 0f64;
        }

        c = c * anti_diag;

    }

    c

}


#[cfg(test)]
mod tests {

    use ndarray::prelude::*;
    use crate::utils::distributions;

    #[test]
    fn test_euclidean_distances() {

        let x = Array2::<f64>::zeros( (3, 5) );
        let y = Array2::from_elem((3, 5), 5.0);
        // let y = DMatrix::from_element(3, 5, 5.0);

        let distance = super::euclidean_distances(&x, &y, false);

        // println!("euclidean_distances: {:?}", distance);

        // squared = true
        // let truth = array![
        //             [125.0, 125.0, 125.0],
        //             [125.0, 125.0, 125.0],
        //             [125.0, 125.0, 125.0]];

        // squared = false
        let truth = array![
                    [11.180339887498949, 11.180339887498949, 11.180339887498949],
                    [11.180339887498949, 11.180339887498949, 11.180339887498949],
                    [11.180339887498949, 11.180339887498949, 11.180339887498949]];

        assert_eq!(distance, truth);

    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dist() {

        let x = Array2::<f64>::zeros( (3, 5) );
        let y = Array2::from_elem((3, 5), 5.0);

        let M = super::dist(&x, &y, super::MetricType::Euclidean);

        // println!("dist: {:?}", M);

        // squared = false
        let truth = array![
                    [11.180339887498949, 11.180339887498949, 11.180339887498949],
                    [11.180339887498949, 11.180339887498949, 11.180339887498949],
                    [11.180339887498949, 11.180339887498949, 11.180339887498949]];

        assert_eq!(M, truth);

    }

    #[test]
    fn test_get_1D_gauss_hist() {

        let n = 50;
        let mean = 20.0;
        let std = 5.0;

        let result = match distributions::get_1D_gauss_histogram(n, mean, std) {
            Ok(val) => val,
            Err(err) => panic!("{:?}", err)
        };

        // TODO: assert correctness get_1D_gauss_histogram()
        // println!("{:?}", result);

    }

    #[test]
    fn test_sample_2D_gauss() {

        let n = 50;
        let mean = array![0.0, 0.0];
        let covariance = array![[1.0, 0.0], [0.0, 1.0]];

        let result = match distributions::sample_2D_gauss(n, &mean, &covariance) {
            Ok(val) => val,
            Err(err) => panic!("{:?}", err)
        };

        // TODO: assert correctness of sample_2D_gauss()
        // println!("{:?}", result);

    }

}

} // mod metric


