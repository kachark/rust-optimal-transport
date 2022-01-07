
use ndarray::prelude::*;
use pyo3::{prelude::*, types::IntoPyDict};
use numpy::ToPyArray;

pub fn plot_py(source_samples: &Array2<f64>, target_samples: &Array2<f64>, coupling_matrix: &Array2<f64>) -> PyResult<()> {

    let source_x = source_samples.slice(s![.., 0]);
    let source_y = source_samples.slice(s![.., 1]);

    let target_x = target_samples.slice(s![.., 0]);
    let target_y = target_samples.slice(s![.., 1]);

    // Start the python interpreter
    let gil = Python::acquire_gil();
    let py = gil.python();

    // Import matplotlib
    let plt = py.import("matplotlib.pyplot")?;

    // Translate to numpy array
    let source_x_py = source_x.to_pyarray(py);
    let source_y_py = source_y.to_pyarray(py);

    let target_x_py = target_x.to_pyarray(py);
    let target_y_py = target_y.to_pyarray(py);

    let result_py = coupling_matrix.to_pyarray(py);

    // Plot data by calling into matplotlib
    plt.getattr("figure")?.call1((1,))?;
    plt.call_method( "imshow", (result_py,), Some(vec![("interpolation", "nearest")].into_py_dict(py)) )?;
    plt.call_method1( "title", ("OT matrix",) )?;

    plt.getattr("figure")?.call1((2,))?;
    plt.call_method("plot", (source_x_py, source_y_py, "+b"), Some(vec![("label", "Source samples")].into_py_dict(py)) )?;
    plt.call_method("plot", (target_x_py, target_y_py, "xr"), Some(vec![("label", "Target samples")].into_py_dict(py)) )?;
    plt.getattr("legend")?.call0()?;
    plt.getattr("show")?.call0()?;

    Ok(())

}
