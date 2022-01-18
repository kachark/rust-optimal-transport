use std::usize;

use thiserror::Error;
use num::Float;
use ndarray::{prelude::*, RemoveAxis};
use ndarray::{Axis, Data};

#[derive(Error, Debug)]
pub enum LogicalError {

    #[error("axis {axis:?} is greater than array dimension {bound:?}")]
    AxisOutOfBoundsError {
        axis: usize,
        bound: usize,
    },

}

/// Checks if a given ndarray Axis is valid for a set of dimensions
fn check_axis(axis: Axis, shape: &[usize]) -> Result<(), LogicalError> {

    for dim in shape.iter() {

        let bound = *dim - 1;

        if axis.0 > bound {
            return Err(LogicalError::AxisOutOfBoundsError{
                axis: axis.0,
                bound,
            })
        }

    }

    Ok(())

}


/// Returns True if all array elements are neither zero, 
/// infinite, subnormal, or NaN. Subnormal values are those 
/// between '0' and f32 or f64::MIN_POSITIVE
/// Returns False otherwise
pub fn all<S, D, A>(arr: &ArrayBase<S, D>) -> bool
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{

    let result: bool = arr.iter().all(|ele| ele.is_normal() );

    result

}


/// Tests whether all array elements along a given axis evaluates to True
///
/// Returns an array of booleans
///
/// Example:
///
/// ```rust
///
/// use rust_optimal_transport as rot;
/// use rot::ndarray_logical::axis_all;
/// use ndarray::{prelude::*, Axis};
///
/// let arr = array![[f32::INFINITY, 42.], [2., 11.]];
/// assert_eq!(axis_all(&arr, Axis(0)).unwrap(), array![false, true]);
/// ```
///
pub fn axis_all<S, D, A>(arr: &ArrayBase<S, D>, axis: Axis) -> Result<Array1<bool>, LogicalError>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{

    check_axis(axis, arr.shape())?;

    let result: Array1<bool> = arr.axis_iter(axis)
        .map(|axis_view| self::all(&axis_view))
        .collect();

    Ok(result)

}




/// Tests whether any array element evaluates to True
/// Returns true if the number is neither zero, infinite, subnormal, or NaN.
/// Subnormal values are those between '0' and 'f32 or f64::MIN_POSITIVE'
/// Returns false for empty arrays
pub fn any<S, D, A>(arr: &ArrayBase<S, D>) -> bool
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{

    let result: bool = arr.iter().any(|ele| ele.is_normal());

    result

}


/// Tests whether any array element along a given axis evaluates to True
///
/// Returns an array of booleans
///
/// Example:
///
/// ```rust
///
/// use rust_optimal_transport as rot;
/// use rot::ndarray_logical::axis_any;
/// use ndarray::{prelude::*, Axis};
///
/// let arr = array![[f32::INFINITY, f32::INFINITY], [f32::NAN, 11.]];
/// assert_eq!(axis_any(&arr, Axis(0)).unwrap(), array![false, true]);
/// ```
///
pub fn axis_any<S, D, A>(arr: &ArrayBase<S, D>, axis: Axis) -> Result<Array1<bool>, LogicalError>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{

    check_axis(axis, arr.shape())?;

    let result: Array1<bool> = arr.axis_iter(axis)
        .map(|axis_view| self::any(&axis_view))
        .collect();

    Ok(result)

}


/// Tests element-wise for NaN elements in an array.
/// Returns True if there are NaN, False otherwise
pub fn is_nan<S, D, A>(arr: &ArrayBase<S, D>) -> bool
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{

    let result: bool = arr.iter().any(|ele| ele.is_nan());

    result

}


/// Tests whether any array element along a given axis is NaN
///
/// Returns an array of booleans
///
/// Example:
///
/// ```rust
///
/// use rust_optimal_transport as rot;
/// use rot::ndarray_logical::axis_is_nan;
/// use ndarray::{prelude::*, Axis};
///
/// let arr = array![[f64::NAN, 0.], [2., 11.]];
/// assert_eq!(axis_is_nan(&arr, Axis(0)).unwrap(), array![true, false]);
/// ```
///
pub fn axis_is_nan<S, D, A>(arr: &ArrayBase<S, D>, axis: Axis) -> Result<Array1<bool>, LogicalError>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{

    check_axis(axis, arr.shape())?;

    let result: Array1<bool> = arr.axis_iter(axis)
        .map(|axis_view| self::is_nan(&axis_view))
        .collect();

    Ok(result)

}


/// Tests element-wise for inf elements in an array.
/// Returns True if there are NaN, False otherwise
pub fn is_inf<S, D, A>(arr: &ArrayBase<S, D>) -> bool
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension,
{

    let result: bool = arr.iter().any(|ele| ele.is_infinite());

    result

}


/// Tests whether any array element along a given axis is inf
///
/// Returns an array of booleans
///
/// Example:
///
/// ```rust
///
/// use rust_optimal_transport as rot;
/// use rot::ndarray_logical::axis_is_inf;
/// use ndarray::{prelude::*, Axis};
///
/// let arr = array![[f64::INFINITY, 0.], [2., 11.]];
/// assert_eq!(axis_is_inf(&arr, Axis(0)).unwrap(), array![true, false]);
/// ```
///
pub fn axis_is_inf<S, D, A>(arr: &ArrayBase<S, D>, axis: Axis) -> Result<Array1<bool>, LogicalError>
where
    A: Float,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{

    check_axis(axis, arr.shape())?;

    let result: Array1<bool> = arr.axis_iter(axis)
        .map(|axis_view| self::is_inf(&axis_view))
        .collect();

    Ok(result)

}



#[cfg(test)]
mod tests {

    use super::{is_inf, is_nan, any, axis_all, axis_any, axis_is_nan, axis_is_inf};
    use ndarray::{array, Axis};
    use num::Float;

    #[test]
    fn test_is_nan() {

        let arr = array![1., 2., f64::NAN];

        assert_eq!(is_nan(&arr), true);

    }

    #[test]
    fn test_is_inf() {

        let arr = array![1f32, 2f32, Float::infinity()];

        assert_eq!(is_inf(&arr), true);

    }

    #[test]
    fn test_any() {

        let arr = array![1., 2., Float::infinity()];

        assert_eq!(any(&arr), true);

    }

    #[test]
    fn test_axis_all() {

        let arr = array![[f32::INFINITY, 42.], [2., 11.]];
        assert_eq!(axis_all(&arr, Axis(0)).unwrap(), array![false, true]);

    }

    #[test]
    fn test_axis_any() {

        let arr = array![[f32::INFINITY, f32::INFINITY], [f32::NAN, 11.]];
        let result = match axis_any(&arr, Axis(0)) {
            Ok(val) => val,
            Err(error) => panic!("{:?}", error)
        };

        assert_eq!(result, array![false, true]);

    }

    #[test]
    fn test_axis_is_nan() {

        let arr = array![[f64::NAN, 0.], [2., 11.]];
        assert_eq!(axis_is_nan(&arr, Axis(0)).unwrap(), array![true, false]);

    }

    #[test]
    fn test_axis_is_inf() {

        let arr = array![[f64::INFINITY, 0.], [2., 11.]];
        assert_eq!(axis_is_inf(&arr, Axis(0)).unwrap(), array![true, false]);

    }

}
