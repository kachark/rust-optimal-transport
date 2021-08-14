

pub mod metrics {

use na::DMatrix;


pub enum MetricType {
    SqEuclidean,
    Euclidean
}

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

fn euclidean_distances(x: &DMatrix<f64>, y: &DMatrix<f64>, squared: bool) -> DMatrix<f64> {

    // einsum('ij,ij->i', X, X)
    // repeated i and j for both x and y inout matrices : multiply those components
    // - element-wise multiplication
    // ommitted j in output : sum along j axis
    // - summation in j
    let a2 = x.component_mul(x).row_sum();
    // einsum('ij,ij->i', Y, Y)
    let b2 = y.component_mul(y).row_sum();

    // for y = 3x3 mtx of 5.0, b2 = [75, 75, 75]

    println!("b2: {:?}", b2);

    let tmpscalar = -2f64 * x.dot(&y.transpose());
    let mut tmp = DMatrix::<f64>::zeros(a2.len(), b2.len());

    for (i, valx) in a2.iter().enumerate() {
        for (j, valy) in b2.iter().enumerate() {

            tmp[(i,j)] = valx + valy + tmpscalar;

        }
    }

    // c = nx.maximum(c, 0)
    let mut c = DMatrix::<f64>::zeros(a2.len(), b2.len());
    for (i, row) in tmp.row_iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if *val > 0f64 {
                c[(i,j)] = *val;
            } else {
                c[(i,j)] = 0f64;
            }
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

        let x = DMatrix::<f64>::zeros(3,3);
        let y = DMatrix::from_element(3, 3, 5.0);

        println!("{:?}", super::euclidean_distances(&x, &y, false));

        let truth = DMatrix::from_row_slice(3,3,
                    &[75.0, 75.0, 75.0,
                    75.0, 75.0, 75.0,
                    75.0, 75.0, 75.0]);

    }

    #[test]
    fn test_dist() {

        let x = DMatrix::<f64>::zeros(3,3);
        let y = DMatrix::from_element(3, 3, 5.0);

        println!("{:?}", super::dist(&x, Some(&y), super::MetricType::SqEuclidean));

        let truth = DMatrix::from_row_slice(3,3,
                    &[8.660254037844387, 8.660254037844387, 8.660254037844387,
                    8.660254037844387, 8.660254037844387, 8.660254037844387,
                    8.660254037844387, 8.660254037844387, 8.660254037844387]);

    }

}

} // mod metric


