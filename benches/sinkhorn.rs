use std::f64;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use ndarray::prelude::*;
use ndarray_rand::rand_distr::uniform::Uniform;
use rand::prelude::*;
use rand::{Rng, SeedableRng};

use rust_optimal_transport as ot;
use ot::regularized::sinkhorn::sinkhorn_knopp;

struct SinkhornInput {

    n_samples: usize,
    source_mass: Array1<f64>,
    target_mass: Array1<f64>,
    cost: Array2<f64>,
    reg: f64,
    num_iter_max: u32,
    threshold: f64,

}

impl SinkhornInput {

    fn new(n_samples: usize) -> Self {

        // parametrize sinkhorn_knopp(n_samples)
        // Need this to be static across each benchmark
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
        let cost = ot::metrics::dist(&source.clone(), &target.clone(), ot::metrics::MetricType::SqEuclidean);

        Self {
            n_samples,
            source_mass,
            target_mass,
            cost,
            reg: 1.,
            num_iter_max: 1000,
            threshold: 1E-7,
        }

    }

}

fn sinkhorn_benchmark(c: &mut Criterion) {

    let inputs_50 = SinkhornInput::new(50);
    let inputs_100 = SinkhornInput::new(100);
    let inputs_500 = SinkhornInput::new(500);
    let inputs_1000 = SinkhornInput::new(1000);
    let inputs_2000 = SinkhornInput::new(2000);
    let inputs_5000 = SinkhornInput::new(5000);

    // Done setup

    let mut group = c.benchmark_group("sinkhorn_group");

    group.bench_function("sinkhorn 50", |b| {
        // per-sample
        b.iter(|| {
            sinkhorn_knopp(&mut inputs_50.source_mass.clone(), &mut inputs_50.target_mass.clone(), &inputs_50.cost.clone(), inputs_50.reg, None, Some(inputs_50.threshold)).unwrap()
        })
    });

    group.bench_function("sinkhorn 100", |b| {
        // per-sample
        b.iter(|| {
            sinkhorn_knopp(&mut inputs_100.source_mass.clone(), &mut inputs_100.target_mass.clone(), &inputs_100.cost.clone(), inputs_100.reg, None, Some(inputs_100.threshold)).unwrap()
        })
    });

    group.bench_function("sinkhorn 500", |b| {
        // per-sample
        b.iter(|| {
            sinkhorn_knopp(&mut inputs_500.source_mass.clone(), &mut inputs_500.target_mass.clone(), &inputs_500.cost.clone(), inputs_500.reg, None, Some(inputs_500.threshold)).unwrap()
        })
    });

    group.bench_function("sinkhorn 1000", |b| {
        // per-sample
        b.iter(|| {
            sinkhorn_knopp(&mut inputs_1000.source_mass.clone(), &mut inputs_1000.target_mass.clone(), &inputs_1000.cost.clone(), inputs_1000.reg, None, Some(inputs_1000.threshold)).unwrap()
        })
    });

    group.bench_function("sinkhorn 2000", |b| {
        // per-sample
        b.iter(|| {
            sinkhorn_knopp(&mut inputs_2000.source_mass.clone(), &mut inputs_2000.target_mass.clone(), &inputs_2000.cost.clone(), inputs_2000.reg, None, Some(inputs_2000.threshold)).unwrap()
        })
    });

    group.bench_function("sinkhorn 5000", |b| {
        // per-sample
        b.iter(|| {
            sinkhorn_knopp(&mut inputs_5000.source_mass.clone(), &mut inputs_5000.target_mass.clone(), &inputs_5000.cost.clone(), inputs_5000.reg, None, Some(inputs_5000.threshold)).unwrap()
        })
    });

    group.finish();

}

criterion_group!(benches, sinkhorn_benchmark);
criterion_main!(benches);
