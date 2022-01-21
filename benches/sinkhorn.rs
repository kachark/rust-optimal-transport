use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use rust_optimal_transport as ot;
use ot::regularized::sinkhorn::sinkhorn_knopp;

fn sinkhorn_benchmark(c: &mut Criterion) {

    let sinkhorn_test = |_| {

        let gamma = 1.;

        // Generate data
        let n = 5;

        // TODO: baseline this benchmark against the Python Optimal Transport sinkhorn benchmark
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
        sinkhorn_knopp(
            &mut source_mass,
            &mut target_mass,
            &ground_cost,
            gamma,
            None,
            None,
        ).unwrap()

    };

    c.bench_function("sinkhorn", |b| b.iter(|| sinkhorn_test(black_box(20))));

}

criterion_group!(benches, sinkhorn_benchmark);
criterion_main!(benches);
