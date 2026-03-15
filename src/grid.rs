use flucoma_sys::grid_process;

/// Grid redistribution for 2D point sets.
pub struct Grid;

impl Grid {
    /// Redistribute 2D points to grid coordinates.
    ///
    /// `input` must be row-major `[x0,y0, x1,y1, ...]`.
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
