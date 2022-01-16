use thiserror::Error;
use num::Float;
use ndarray::prelude::*;
use ndarray::Data;

#[derive(Error, Debug)]
pub enum LogicalError {


}


/// Tests whether any array element evaluates to True
/// Returns true if the number is neither zero, infinite, subnormal, or NaN
/// subnormal are values between '0' and 'f32/f64::MIN_POSITIVE'
pub fn any<S, D, A>(arr: &ArrayBase<S, D>) -> Result<bool, LogicalError>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{

    let result: bool = arr.iter().any(|ele| ele.is_normal());

    Ok(result)

}


/// Tests element-wise for NaN and returns boolean indicating
/// presence of NaN values (true) or lack of NaN values (false).
pub fn is_nan<S, D, A>(arr: &ArrayBase<S, D>) -> Result<bool, LogicalError>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{

    // TODO: test if arr is valid

    let result: bool = arr.iter().any(|ele| ele.is_nan());

    Ok(result)

}


/// Tests element-wise for inf and returns boolean indicating
/// presence of inf values (true) or lack of inf values (false).
pub fn is_inf<S, D, A>(arr: &ArrayBase<S, D>) -> Result<bool, LogicalError>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{

    let result: bool = arr.iter().any(|ele| ele.is_infinite());

    Ok(result)

}

#[cfg(test)]
mod tests {

    use super::{is_inf, is_nan};
    use ndarray::array;
    use std::f64;
    use num::Float;

    #[test]
    fn test_is_nan() {

        let arr = array![1., 2., f64::NAN];
        let result = match is_nan(&arr) {
            Ok(val) => val,
            Err(error) => panic!("{:?}", error)
        };

        assert_eq!(result, true);

    }

    #[test]
    fn test_is_inf() {

        let arr = array![1f32, 2f32, Float::infinity()];
        let result = match is_inf(&arr) {
            Ok(val) => val,
            Err(error) => panic!("{:?}", error)
        };

        assert_eq!(result, true);

    }

    #[test]
    fn test_any() {

        let arr = array![1., 2., Float::infinity()];
        let result = match is_inf(&arr) {
            Ok(val) => val,
            Err(error) => panic!("{:?}", error)
        };

        assert_eq!(result, true);

    }

}
