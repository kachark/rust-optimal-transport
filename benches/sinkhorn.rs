use std::f64;
use std::fmt;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use ndarray::prelude::*;
use ndarray_rand::rand_distr::uniform::Uniform;
use rand::prelude::*;
use rand::{Rng, SeedableRng};

use rust_optimal_transport as ot;
use ot::regularized::sinkhorn::SinkhornKnopp;
use ot::OTSolver;

#[derive(Clone)]
struct SinkhornParams {

    n_samples: usize,
    source_mass: Array1<f64>,
    target_mass: Array1<f64>,
    cost: Array2<f64>,
    reg: f64,
    num_iter_max: i32,
    threshold: f64,

}

impl SinkhornParams {

    fn new(n_samples: usize) -> Self {

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
        let source_mass = Array1::<f64>::from_elem(n_samples/4, 1. / ((n_samples/4) as f64));
        let target_mass = Array1::<f64>::from_elem(n_samples, 1. / (n_samples as f64));

        // Compute ground cost matrix - Euclidean distance
        let cost = ot::metrics::dist(&source, &target, ot::metrics::MetricType::SqEuclidean);

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

impl fmt::Display for SinkhornParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Num samples: {}", self.n_samples)
    }
}

fn sinkhorn_benchmark(c: &mut Criterion) {

    let inputs_50 = SinkhornParams::new(50);
    let inputs_100 = SinkhornParams::new(100);
    let inputs_500 = SinkhornParams::new(500);
    let inputs_1000 = SinkhornParams::new(1000);
    let inputs_2000 = SinkhornParams::new(2000);
    let inputs_5000 = SinkhornParams::new(5000);

    // Done setup

    let n_runs = 100;
    let mut group = c.benchmark_group("sinkhorn_group");

    for input in [inputs_50, inputs_100, inputs_500, inputs_1000, inputs_2000, inputs_5000].iter() {

        group.bench_with_input(
            BenchmarkId::new("sinkhorn", input.n_samples), input,
            move |b, i| b.iter_with_large_drop(|| {
                SinkhornKnopp::new(&i.source_mass, &i.target_mass, &i.cost, i.reg)
                    .iterations(i.num_iter_max)
                    .threshold(i.threshold)
                    .solve()
                    .unwrap();
            }),
        ).sample_size(n_runs);

    }

    group.finish();

}


fn sinkhorn_benchmark_single(c: &mut Criterion) {

    let inputs_500 = SinkhornParams::new(500);

    // Done setup

    c.bench_with_input(
        BenchmarkId::new("sinkhorn_single", inputs_500.n_samples), &inputs_500, |b, i| b.iter_with_large_drop(|| {
            SinkhornKnopp::new(&i.source_mass, &i.target_mass, &i.cost, i.reg)
                .iterations(i.num_iter_max)
                .threshold(i.threshold)
                .solve()
                .unwrap();
        }),
    );


}

criterion_group!(benches, sinkhorn_benchmark_single, sinkhorn_benchmark);
criterion_main!(benches);
