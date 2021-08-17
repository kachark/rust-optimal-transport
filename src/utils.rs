

pub mod metrics {

use na::DMatrix;


pub enum MetricType {
    SqEuclidean,
    Euclidean
}

/// Compute distance between samples in x1 and x2
/// x1: matrix with n1 samples of size d
/// x2: matrix with n2 samples of size d (if None then x2=x1)
/// metric: choice of distance metric
pub fn dist(x1: &DMatrix<f64>, x2: Option<&DMatrix<f64>>, metric: MetricType) -> DMatrix<f64> {

    let x2 = match x2 {

        Some(matrix) => matrix,
        None => x1

    };

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
fn euclidean_distances(x: &DMatrix<f64>, y: &DMatrix<f64>, squared: bool) -> DMatrix<f64> {

    // einsum('ij,ij->i', X, X)
    // repeated i and j for both x and y inout matrices : multiply those components
    // - element-wise multiplication
    // ommitted j in output : sum along j axis
    // - summation in j
    let a2 = x.component_mul(x).column_sum();
    // einsum('ij,ij->i', Y, Y)
    let b2 = y.component_mul(y).column_sum();

    // for y = 3x3 mtx of 5.0, b2 = [75, 75, 75]

    // println!("a2: {:?}", a2);
    // println!("b2: {:?}", b2);

    // let tmpscalar = -2f64 * x.dot(&y.transpose());
    // let mut tmp = DMatrix::<f64>::zeros(a2.len(), b2.len());
    let mut c = (x * &y.transpose()).scale(-2f64);

    // c += a2[:, None]
    for (mut row, a2val) in c.row_iter_mut().zip(&a2.transpose()) {
        for ele in row.iter_mut() {
            *ele += a2val;
        }
    }

    // c += b2[None, :]
    for (mut col, b2val) in c.column_iter_mut().zip(&b2) {
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

    if squared == false {

        // np.sqrt(c)
        for val in c.iter_mut() {
            *val = val.powf(0.5);
        }

    }


    if x == y {

        // ones matrix with diagonals set to zero
        let anti_diag = DMatrix::from_element(a2.len(), b2.len(), 1f64) - DMatrix::from_diagonal_element(a2.len(), b2.len(), 1f64);

        c = c * anti_diag;

    }

    c

}


#[cfg(test)]
mod tests {

    use na::DMatrix;

    #[test]
    fn test_euclidean_distances() {

        let x = DMatrix::<f64>::zeros(3, 5);
        let y = DMatrix::from_row_slice(3, 5, vec![5.0; 15].as_slice());
        // let y = DMatrix::from_element(3, 5, 5.0);

        println!("euclidean_distances: {:?}", super::euclidean_distances(&x, &y, false));

        // squared = true
        // let truth = DMatrix::from_row_slice(3,3,
        //             &[125.0, 125.0, 125.0,
        //             125.0, 125.0, 125.0,
        //             125.0, 125.0, 125.0]);

        // squared = false
        let _truth = DMatrix::from_row_slice(3,3,
                    &[11.180339887498949, 11.180339887498949, 11.180339887498949,
                    11.180339887498949, 11.180339887498949, 11.180339887498949,
                    11.180339887498949, 11.180339887498949, 11.180339887498949]);

    }

    #[test]
    fn test_dist() {

        let x = DMatrix::<f64>::zeros(3,3);
        let y = DMatrix::from_element(3, 3, 5.0);

        println!("dist: {:?}", super::dist(&x, Some(&y), super::MetricType::SqEuclidean));

        let _truth = DMatrix::from_row_slice(3,3,
                    &[8.660254037844387, 8.660254037844387, 8.660254037844387,
                    8.660254037844387, 8.660254037844387, 8.660254037844387,
                    8.660254037844387, 8.660254037844387, 8.660254037844387]);

    }

}

} // mod metric


