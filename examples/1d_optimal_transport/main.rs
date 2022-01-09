
use ndarray::{prelude::*, stack};
use ndarray_stats::QuantileExt;

use rust_optimal_transport as rot;

mod plot;

fn main() {

    // Generate data
    let n_samples = 100;
    let x = Array::range(0.0, n_samples as f64, 1.0);

    let mean_source = 20.0;
    let std_source = 5.0;

    let mean_target = 60.0;
    let std_target = 10.0;

    let mut source_mass = match rot::utils::distributions::get_1D_gauss_histogram(n_samples, mean_source, std_source) {
        Ok(val) => val,
        Err(err) => panic!("{:?}", err)
    };


    let mut target_mass = match rot::utils::distributions::get_1D_gauss_histogram(n_samples, mean_target, std_target) {
        Ok(val) => val,
        Err(err) => panic!("{:?}", err)
    };

    let source_samples = stack![Axis(1), x, source_mass];
    let target_samples = stack![Axis(1), x, target_mass];

    let x_reshaped: Array2<f64> = x.into_shape((n_samples as usize, 1)).unwrap();

    // Compute ground cost matrix - Squared Euclidean distance
    let mut ground_cost = rot::utils::metrics::dist(&x_reshaped, &x_reshaped, rot::utils::metrics::MetricType::SqEuclidean);
    let max_cost = ground_cost.max().unwrap();

    // Normalize cost matrix for numerical stability
    ground_cost = &ground_cost / *max_cost;

    // Compute optimal transport matrix as the Earth Mover's Distance
    let ot_matrix = match rot::lp::emd(&mut source_mass, &mut target_mass, &mut ground_cost, None, None) {
        Ok(result) => result,
        Err(error) => panic!("{:?}", error)
    };

    // Plot using matplotlib
    match plot::plot_py(&source_samples, &target_samples, &ot_matrix) {
        Ok(_) => (),
        Err(error) => panic!("{:?}", error)
    };

}
