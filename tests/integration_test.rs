use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use rust_optimal_transport as ot;

#[test]
fn emd_integration_test() {
    // Generate data
    let n = 5;
    let x = Array::range(0.0, n as f64, 1.0);

    let mean_source = 20.0;
    let std_source = 5.0;

    let mean_target = 60.0;
    let std_target = 10.0;

    let mut source_mass = match ot::utils::get_1D_gauss_histogram(n, mean_source, std_source) {
        Ok(val) => val,
        Err(err) => panic!("{:?}", err),
    };

    let mut target_mass = match ot::utils::get_1D_gauss_histogram(n, mean_target, std_target) {
        Ok(val) => val,
        Err(err) => panic!("{:?}", err),
    };

    // Compute ground cost matrix - Squared Euclidean distance
    let x_reshaped: Array2<f64> = x.into_shape((n as usize, 1)).unwrap();
    let mut ground_cost = ot::metrics::dist(
        &x_reshaped,
        &x_reshaped,
        ot::metrics::MetricType::SqEuclidean,
    );
    ground_cost = &ground_cost / *ground_cost.max().unwrap();

    let result = match ot::lp::emd(
        &mut source_mass,
        &mut target_mass,
        &mut ground_cost,
        None,
        None,
    ) {
        Ok(result) => result,
        Err(error) => panic!("{:?}", error),
    };

    println!("result: {:?}", result);

    let truth = array![
        [0.02875604, 0., 0., 0., 0.],
        [0.01664376, 0.04608674, 0., 0., 0.],
        [0., 0.0362245, 0.09525459, 0., 0.],
        [0., 0., 0.05249336, 0.21227303, 0.],
        [0., 0., 0., 0.05029436, 0.46197361]
    ];

    assert!(result.relative_eq(&truth, 1E-6, 1E-2));
}

#[test]
fn sinkhorn_integration_test() {
    let gamma = 1E-1;

    // Generate data
    let n = 5;
    let _mu_source = array![0., 0.];
    let _cov_source = array![[1., 0.], [0., 1.]];

    let _mu_target = array![4., 4.];
    let _cov_target = array![[1., -0.8], [-0.8, 1.]];

    // Samples of a 2D gaussian distribution
    let source = array![
        [-0.33422316, -1.40157595],
        [1.01640207, 1.58920135],
        [0.45938047, -0.59832115],
        [-0.90015176, -0.0695026],
        [0.24890721, 0.25353813]
    ];

    let target = array![
        [6.18308211, 2.38144413],
        [4.01974517, 3.3010811],
        [4.99330784, 3.29090987],
        [1.07482414, 6.19599718],
        [2.62013006, 5.61165631]
    ];

    // Uniform distribution on the source and target densities
    let mut source_mass = Array1::<f64>::from_vec(vec![1f64 / (n as f64); n as usize]);
    let mut target_mass = Array1::<f64>::from_vec(vec![1f64 / (n as f64); n as usize]);

    // Compute ground cost matrix - Euclidean distance
    let mut ground_cost = ot::metrics::dist(&source, &target, ot::metrics::MetricType::SqEuclidean);
    ground_cost = &ground_cost / *ground_cost.max().unwrap();

    // Solve Sinkhorn Distance
    let result = match ot::regularized::sinkhorn::sinkhorn_knopp(
        &mut source_mass,
        &mut target_mass,
        &mut ground_cost,
        gamma,
        None,
        None,
    ) {
        Ok(result) => result,
        Err(err) => panic!("{:?}", err),
    };

    let truth = array![
        [0.05553532, 0.05473157, 0.0475888, 0.0197709, 0.02237341],
        [0.02927879, 0.02724481, 0.03642575, 0.04720207, 0.05984858],
        [0.06377359, 0.0452918, 0.0508736, 0.01629669, 0.02376431],
        [0.01499525, 0.03356821, 0.02415873, 0.07711427, 0.05016355],
        [0.03641705, 0.03916361, 0.04095311, 0.03961608, 0.04385015]
    ];

    println!("result: {:?}", result);

    assert!(result.relative_eq(&truth, 1E-6, 1E-2));
}

#[test]
fn greenkhorn_integration_test() {
    let gamma = 1E-1;

    // Generate data
    let n = 5;
    let _mu_source = array![0., 0.];
    let _cov_source = array![[1., 0.], [0., 1.]];

    let _mu_target = array![4., 4.];
    let _cov_target = array![[1., -0.8], [-0.8, 1.]];

    // Samples of a 2D gaussian distribution
    let source = array![
        [-0.33422316, -1.40157595],
        [1.01640207, 1.58920135],
        [0.45938047, -0.59832115],
        [-0.90015176, -0.0695026],
        [0.24890721, 0.25353813]
    ];

    let target = array![
        [6.18308211, 2.38144413],
        [4.01974517, 3.3010811],
        [4.99330784, 3.29090987],
        [1.07482414, 6.19599718],
        [2.62013006, 5.61165631]
    ];

    // Uniform distribution on the source and target densities
    let mut source_mass = Array1::<f64>::from_vec(vec![1f64 / (n as f64); n as usize]);
    let mut target_mass = Array1::<f64>::from_vec(vec![1f64 / (n as f64); n as usize]);

    // Compute ground cost matrix - Euclidean distance
    let mut ground_cost = ot::metrics::dist(&source, &target, ot::metrics::MetricType::SqEuclidean);
    ground_cost = &ground_cost / *ground_cost.max().unwrap();

    let result = match ot::regularized::greenkhorn::greenkhorn(
        &mut source_mass,
        &mut target_mass,
        &mut ground_cost,
        gamma,
        None,
        None,
    ) {
        Ok(result) => result,
        Err(err) => panic!("{:?}", err),
    };

    let truth = array![
        [0.05553532, 0.05473157, 0.0475888, 0.0197709, 0.02237341],
        [0.02927879, 0.02724481, 0.03642576, 0.04720207, 0.05984858],
        [0.06377359, 0.0452918, 0.05087361, 0.01629669, 0.02376431],
        [0.01499525, 0.03356821, 0.02415873, 0.07711427, 0.05016355],
        [0.03641705, 0.03916361, 0.04095311, 0.03961608, 0.04385015]
    ];

    println!("result: {:?}", result);

    assert!(result.relative_eq(&truth, 1E-6, 1E-2));
}

#[test]
fn unbalanced_sinkhorn_integration_test() {
    let epsilon = 0.1;
    let alpha = 1.;

    // Generate data
    let n = 5;
    let x = Array::range(0.0, n as f64, 1.0);

    let mean_source = 20.0;
    let std_source = 5.0;

    let mean_target = 60.0;
    let std_target = 10.0;

    let mut source_mass = match ot::utils::get_1D_gauss_histogram(n, mean_source, std_source) {
        Ok(val) => val,
        Err(err) => panic!("{:?}", err),
    };

    let mut target_mass = match ot::utils::get_1D_gauss_histogram(n, mean_target, std_target) {
        Ok(val) => val,
        Err(err) => panic!("{:?}", err),
    };

    // unbalance the source and target mass distribution
    target_mass *= 5.0;

    // Compute ground cost matrix - Squared Euclidean distance
    let x_reshaped: Array2<f64> = x.into_shape((n as usize, 1)).unwrap();
    let mut ground_cost = ot::metrics::dist(
        &x_reshaped,
        &x_reshaped,
        ot::metrics::MetricType::SqEuclidean,
    );
    ground_cost = &ground_cost / *ground_cost.max().unwrap();

    let result = match ot::unbalanced::sinkhorn_knopp_unbalanced(
        &mut source_mass,
        &mut target_mass,
        &mut ground_cost,
        epsilon,
        alpha,
        None,
        None,
    ) {
        Ok(result) => result,
        Err(err) => panic!("{:?}", err),
    };

    let truth = array![
        [
            5.12755466e-02,
            2.53950152e-02,
            3.90657040e-03,
            1.84359745e-04,
            3.63935236e-06
        ],
        [
            4.39939919e-02,
            7.60500758e-02,
            4.08333425e-02,
            6.72594758e-03,
            4.63425137e-04
        ],
        [
            1.25388468e-02,
            7.56540148e-02,
            1.41780127e-01,
            8.15120868e-02,
            1.96027149e-02
        ],
        [
            9.45281829e-04,
            1.99069074e-02,
            1.30213280e-01,
            2.61294358e-01,
            2.19327125e-01
        ],
        [
            2.37871304e-05,
            1.74844768e-03,
            3.99183176e-02,
            2.79585782e-01,
            8.19116169e-01
        ]
    ];

    println!("result: {:?}", result);

    assert!(result.relative_eq(&truth, 1E-6, 1E-2));
}
