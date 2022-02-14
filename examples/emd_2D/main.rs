use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use ot::exact::EarthMovers;
use ot::OTSolver;
use ot::metrics::MetricType::SqEuclidean;
use rust_optimal_transport as ot;

mod plot;

fn main() {
    // Generate data
    let n_samples = 100;

    // Mean, Covariance of the source distribution
    let mu_source = array![0., 0.];
    let cov_source = array![[1., 0.], [0., 1.]];

    // Mean, Covariance of the target distribution
    let mu_target = array![4., 4.];
    let cov_target = array![[1., -0.8], [-0.8, 1.]];

    // Samples of a 2D gaussian distribution
    let source = ot::utils::sample_2D_gauss(n_samples, &mu_source, &cov_source).unwrap();
    let target = ot::utils::sample_2D_gauss(n_samples, &mu_target, &cov_target).unwrap();

    // Uniform distribution on the source and target samples
    let mut source_mass =
        Array1::<f64>::from_vec(vec![1f64 / (n_samples as f64); n_samples as usize]);
    let mut target_mass =
        Array1::<f64>::from_vec(vec![1f64 / (n_samples as f64); n_samples as usize]);

    // Compute ground cost matrix - Squared Euclidean distance
    let mut ground_cost = ot::metrics::dist(&source, &target, SqEuclidean);
    let max_cost = ground_cost.max().unwrap();

    // Normalize cost matrix for numerical stability
    ground_cost = &ground_cost / *max_cost;

    // Compute optimal transport matrix as the Earth Mover's Distance
    let ot_matrix = match EarthMovers::new(
        &mut source_mass,
        &mut target_mass,
        &mut ground_cost,
    ).solve() {
        Ok(result) => result,
        Err(error) => panic!("{:?}", error),
    };

    // Plot using matplotlib
    match plot::plot_py(&source, &target, &ot_matrix) {
        Ok(_) => (),
        Err(error) => panic!("{:?}", error),
    };
}
