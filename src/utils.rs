

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

}

} // mod metric


