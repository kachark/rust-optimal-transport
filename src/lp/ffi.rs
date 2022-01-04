
use ndarray::prelude::*;

#[cxx::bridge]
mod ffi {

    extern "C++" {
        include!("rust-optimal-transport/src/fast_transport/EMD.h");

        unsafe fn EMD_wrap(n1: i32, n2: i32, X: *mut f64, Y: *mut f64, D: *mut f64, G: *mut f64,
                    alpha: *mut f64, beta: *mut f64, cost: *mut f64, maxIter: i32) -> i32;
    }

}


/// Wrapper of C++ FastTransport OT Network Simplex solver
#[allow(non_snake_case)]
pub fn emd_c(a: &mut Array1<f64>, b: &mut Array1<f64>, M: &mut Array2<f64>, max_iter: i32)
    -> (Array2<f64>, f64, Array1<f64>, Array1<f64>, i32) {

    let mshape = M.shape();
    let n1 = mshape[0];
    let n2 = mshape[1];

    let _nmax = n1 + n2 - 1;
    let _nG = 0i32;
    let mut cost = 0f64;
    let mut alpha = Array1::<f64>::zeros(n1);
    let mut beta = Array1::<f64>::zeros(n2);
    let mut G = Array2::<f64>::zeros( (n1, n2) );

    if a.is_empty() {
        *a = Array1::from_vec(vec![1f64/(n1 as f64); n1]);
    }

    if b.is_empty() {
        *b = Array1::from_vec(vec![1f64/(n2 as f64); n2]);
    }

    unsafe {
    let result_code = ffi::EMD_wrap(n1 as i32,
                           n2 as i32,
                           a.as_mut_ptr(),
                           b.as_mut_ptr(),
                           M.as_mut_ptr(),
                           G.as_mut_ptr(),
                           alpha.as_mut_ptr(),
                           beta.as_mut_ptr(),
                           &mut cost,
                           max_iter);

    (G, cost, alpha, beta, result_code)

    }

}

#[cfg(test)]
mod tests {

    use ndarray::array;

    #[allow(non_snake_case)]
    #[test]
    fn test_emd_c() {

        let mut a = array![0.5, 0.5];
        let mut b = array![0.5, 0.5];
        let mut M = array![[0.0, 1.0],[1.0, 0.0]];

        let (G, _cost, _u, _v, _result_code) = super::emd_c(&mut a, &mut b, &mut M, 10000);

        let truth = array![[0.5, 0.0],[0.0, 0.5]];

        // println!("{:?}", G);

        assert_eq!(G, truth);

    }

}
