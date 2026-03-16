use flucoma_sys::grid_process;

// -------------------------------------------------------------------------------------------------

/// Redistribute a 2D point set onto a regular grid.
///
/// Given an arbitrary set of 2D points, `Grid` finds an assignment that places
/// each point on a grid cell while minimising the total displacement — useful
/// for generating ordered visual layouts from unordered data.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::Grid;
///
/// let points = vec![0.1, 0.2,  0.8, 0.9,  0.5, 0.5,  0.0, 1.0];
/// let out = Grid::process(&points, 4, 1, 0, 0).unwrap();
/// // `out` contains 4 (x, y) grid-cell coordinates
/// ```
///
/// See <https://learn.flucoma.org/reference/grid>
pub struct Grid;

impl Grid {
    /// Redistribute 2D points to grid coordinates.
    ///
    /// Returns a flat row-major `[x0,y0, x1,y1, ...]` vector of grid-cell
    /// coordinates, one per input point.
    ///
    /// # Arguments
    /// * `input` — row-major `[x0,y0, x1,y1, ...]`; length must be `rows * 2`.
    /// * `rows` — number of points (must be > 0).
    /// * `over_sample` — oversampling factor for the assignment solver (must be > 0).
    /// * `extent` — neighbourhood extent used during assignment (0 = automatic).
    /// * `axis` — primary sort axis: `0` = x, `1` = y.
    ///
    /// # Errors
    /// Returns an error if arguments are out of range or the underlying
    /// assignment fails.
    pub fn process(
        input: &[f64],
        rows: usize,
        over_sample: usize,
        extent: usize,
        axis: usize,
    ) -> Result<Vec<f64>, &'static str> {
        if rows == 0 {
            return Err("rows must be > 0");
        }
        if input.len() != rows * 2 {
            return Err("input length must be rows * 2");
        }
        if over_sample == 0 {
            return Err("over_sample must be > 0");
        }
        if axis > 1 {
            return Err("axis must be 0 or 1");
        }
        let mut out = vec![0.0; rows * 2];
        let ok = grid_process(
            input.as_ptr(),
            rows as isize,
            over_sample as isize,
            extent as isize,
            axis as isize,
            out.as_mut_ptr(),
        );
        if !ok {
            return Err("grid process failed");
        }
        Ok(out)
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_process_basic() {
        let input = vec![0.0, 0.0, 1.0, 1.0, 0.2, 0.8, 0.8, 0.2];
        let out = Grid::process(&input, 4, 1, 0, 0).unwrap();
        assert_eq!(out.len(), 8);
        // Grid coordinates are non-negative
        assert!(out.iter().all(|&v| v >= 0.0));
    }
}
