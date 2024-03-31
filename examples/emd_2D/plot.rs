use ndarray::prelude::*;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::IntoPyDict};

pub fn plot_py(
    source_samples: &Array2<f64>,
    target_samples: &Array2<f64>,
    ot_matrix: &Array2<f64>,
) -> PyResult<()> {
    let source_x = source_samples.slice(s![.., 0]);
    let source_y = source_samples.slice(s![.., 1]);

    let target_x = target_samples.slice(s![.., 0]);
    let target_y = target_samples.slice(s![.., 1]);

    // Start the python interpreter
    Python::with_gil(|py| {
        // Import matplotlib
        let plt = py.import("matplotlib.pyplot")?;

        // Translate to numpy array
        let source_x_py = source_x.to_pyarray(py);
        let source_y_py = source_y.to_pyarray(py);

        let target_x_py = target_x.to_pyarray(py);
        let target_y_py = target_y.to_pyarray(py);

        let ot_matrix_py = ot_matrix.to_pyarray(py);

        // Plot by calling into matplotlib

        // plot ot matrix
        plt.getattr("figure")?.call1((1,))?;
        plt.call_method(
            "imshow",
            (ot_matrix_py,),
            Some(vec![("interpolation", "nearest")].into_py_dict(py)),
        )?;
        plt.call_method1("title", ("OT matrix",))?;

        // plot data with coupling between source and target distributions
        plt.getattr("figure")?.call1((2,))?;

        let threshold = 1E-8;
        for i in 0..ot_matrix.shape()[0] {
            for j in 0..ot_matrix.shape()[1] {
                if ot_matrix[[i, j]] > threshold {
                    let args = (
                        array![source_x[i], target_x[j]].to_pyarray(py),
                        array![source_y[i], target_y[j]].to_pyarray(py),
                    );
                    let kwargs = Some(vec![("color", "0.8")].into_py_dict(py));

                    plt.call_method("plot", args, kwargs)?;
                }
            }
        }

        plt.call_method(
            "plot",
            (source_x_py, source_y_py, "+b"),
            Some(vec![("label", "Source samples")].into_py_dict(py)),
        )?;
        plt.call_method(
            "plot",
            (target_x_py, target_y_py, "xr"),
            Some(vec![("label", "Target samples")].into_py_dict(py)),
        )?;
        plt.getattr("legend")?.call0()?;

        plt.getattr("show")?.call0()?;

        Ok(())
    })
}
