use ndarray::prelude::*;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::IntoPyDict, types::PyList};
use std::env;

pub fn plot_py(
    source_samples: &Array2<f64>,
    target_samples: &Array2<f64>,
    ot_matrix: &Array2<f64>,
    title: &str,
) -> PyResult<()> {
    let source_y = source_samples.slice(s![.., 1]);
    let target_y = target_samples.slice(s![.., 1]);

    // Start the python interpreter
    let gil = Python::acquire_gil();
    let py = gil.python();

    // Import matplotlib
    let plt = py.import("matplotlib.pyplot")?;

    // Import plotting function by adding python script to path
    let pwd = env::current_dir()?;
    let syspath: &PyList = py
        .import("sys")
        .unwrap()
        .getattr("path")
        .unwrap()
        .try_into()
        .unwrap();

    syspath.insert(0, pwd.display().to_string()).unwrap();
    let plot_mod = py.import("plot_1d_mat")?;

    // Translate to numpy array
    let source_y_py = source_y.to_pyarray(py);
    let target_y_py = target_y.to_pyarray(py);
    let ot_matrix_py = ot_matrix.to_pyarray(py);

    // Plot by calling into matplotlib via python script
    plt.call_method(
        "figure",
        (4,),
        Some(vec![("figsize", (5, 5))].into_py_dict(py)),
    )?;
    plot_mod
        .getattr("plot1D_mat")?
        .call1((source_y_py, target_y_py, ot_matrix_py, title))?;

    plt.getattr("show")?.call0()?;

    Ok(())
}
