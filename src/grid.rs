use flucoma_sys::grid_process;

use crate::matrix::{AsMatrixView, Matrix, MatrixView};

// -------------------------------------------------------------------------------------------------

/// Redistribute a 2D point set onto a regular grid.
///
/// Given an arbitrary set of 2D points, `Grid` finds an assignment that places
/// each point on a grid cell while minimising the total displacement — useful
/// for generating ordered visual layouts from unordered data.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{Grid, Matrix};
///
/// let points = Matrix::from_vec(
///     vec![0.1, 0.2,  0.8, 0.9,  0.5, 0.5,  0.0, 1.0], 4, 2
/// ).unwrap();
/// let out = Grid::process(&points, 1, 0, 0).unwrap();
/// // `out` contains 4 (x, y) grid-cell coordinates
/// ```
///
/// See <https://learn.flucoma.org/reference/grid>
pub struct Grid(Matrix);

impl Grid {
    /// Redistribute 2D points to grid coordinates.
    ///
    /// Returns a `Grid` (an Nx2 matrix) of grid-cell coordinates, one row per
    /// input point.
    ///
    /// # Arguments
    /// * `input` — row-major Nx2 matrix of `[x0,y0, x1,y1, ...]` points.
    /// * `over_sample` — oversampling factor for the assignment solver (must be > 0).
    /// * `extent` — neighbourhood extent used during assignment (0 = automatic).
    /// * `axis` — primary sort axis: `0` = x, `1` = y.
    ///
    /// # Errors
    /// Returns an error if `input.cols() != 2`, arguments are out of range, or
    /// the underlying assignment fails.
    pub fn process(
        input: impl AsMatrixView,
        over_sample: usize,
        extent: usize,
        axis: usize,
    ) -> Result<Grid, &'static str> {
        let input = input.as_matrix_view();
        let rows = input.rows();
        if input.cols() != 2 {
            return Err("input must have exactly 2 columns (x, y)");
        }
        if rows == 0 {
            return Err("rows must be > 0");
        }
        if over_sample == 0 {
            return Err("over_sample must be > 0");
        }
        if axis > 1 {
            return Err("axis must be 0 or 1");
        }
        let mut out = vec![0.0; rows * 2];
        let ok = grid_process(
            input.data().as_ptr(),
            rows as isize,
            over_sample as isize,
            extent as isize,
            axis as isize,
            out.as_mut_ptr(),
        );
        if !ok {
            return Err("grid process failed");
        }
        Ok(Grid(Matrix::from_vec(out, rows, 2).unwrap()))
    }

    /// Consume the `Grid` and return the inner [`Matrix`].
    pub fn into_inner(self) -> Matrix {
        self.0
    }

    /// Return a borrowed [`MatrixView`] over the grid data.
    pub fn view(&self) -> MatrixView<'_> {
        self.0.view()
    }
}

impl AsMatrixView for Grid {
    fn as_matrix_view(&self) -> MatrixView<'_> {
        self.0.view()
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_process_basic() {
        let input = Matrix::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.2, 0.8, 0.8, 0.2], 4, 2).unwrap();
        let out = Grid::process(&input, 1, 0, 0).unwrap();
        let m = out.into_inner();
        assert_eq!(m.rows(), 4);
        assert_eq!(m.cols(), 2);
        assert!(m.data().iter().all(|&v| v >= 0.0));
    }
}
