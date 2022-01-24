use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use ndarray_rand::rand_distr::uniform::Uniform;
use rand::prelude::*;
use rand::{Rng, SeedableRng};

use rust_optimal_transport as ot;
use ot::regularized::sinkhorn::sinkhorn_knopp;

fn sinkhorn_benchmark(c: &mut Criterion) {

    let gamma = 1.;

    // Generate data
    let n_samples = 50;

    let mut rng = StdRng::seed_from_u64(123456789);
    let distribution = Uniform::<f64>::new(0.0, 1.0);

    let mut source = Array2::<f64>::zeros((n_samples/4, 100));
    for ele in source.iter_mut() {
        *ele = rng.sample(distribution);
    }
    let source = source; // remove mutability

    let mut target = Array2::<f64>::zeros((n_samples, 100));
    for ele in target.iter_mut() {
        *ele = rng.sample(distribution);
    }
    let target = target;

    // Uniform distribution on the source and target densities
    let mut source_mass = Array1::<f64>::from_vec(vec![1f64 / (n_samples as f64); (n_samples/4) as usize]);
    let mut target_mass = Array1::<f64>::from_vec(vec![1f64 / (n_samples as f64); n_samples as usize]);

    // Compute ground cost matrix - Euclidean distance
    let mut ground_cost = ot::metrics::dist(&source, &target, ot::metrics::MetricType::SqEuclidean);
    ground_cost = &ground_cost / *ground_cost.max().unwrap();

//     let mut sinkhorn_test = || {

//         // Solve Sinkhorn Distance
//         sinkhorn_knopp(
//             &mut source_mass,
//             &mut target_mass,
//             &ground_cost,
//             gamma,
//             None,
//             Some(1E-7),
//         ).unwrap()

//     };

    // c.bench_function("sinkhorn", |b| b.iter(|| sinkhorn_test()));


    c.bench_function("sinkhorn", |b| {
        // per-sample
        b.iter(|| {
            sinkhorn_knopp(&mut source_mass, &mut target_mass, &mut ground_cost, gamma, None, Some(1E-7))
        })});

}

criterion_group!(benches, sinkhorn_benchmark);
criterion_main!(benches);
