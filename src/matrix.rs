// -------------------------------------------------------------------------------------------------

/// Row-major `f64` matrix.
///
/// Used to pass and receive 2-D data to/from e.g. NMF algorithms.
///
/// # Layout
/// Data is stored in **row-major** (C) order: element `(r, c)` is at index
/// `r * cols + c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create a zero-initialised matrix of shape `rows × cols`.
    ///
    /// # Panics
    /// Panics if either dimension is zero.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "rows must be > 0");
        assert!(cols > 0, "cols must be > 0");
        Self {
            data: vec![0.0f64; rows * cols],
            rows,
            cols,
        }
    }

    /// Wrap an existing `Vec<f64>` as a `rows × cols` matrix.
    ///
    /// # Errors
    /// Returns an error if `data.len() != rows * cols`, or if either dimension
    /// is zero.
    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, &'static str> {
        if rows == 0 {
            return Err("rows must be > 0");
        }
        if cols == 0 {
            return Err("cols must be > 0");
        }
        if data.len() != rows * cols {
            return Err("data length does not match rows * cols");
        }
        Ok(Self { data, rows, cols })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Flat row-major data slice.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable flat row-major data slice.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Return the transpose of this matrix as a new `Matrix`.
    ///
    /// The original is `rows × cols`; the result is `cols × rows`.
    pub fn transpose(&self) -> Self {
        let mut out = vec![0.0f64; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Self {
            data: out,
            rows: self.cols,
            cols: self.rows,
        }
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zeros() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data().len(), 6);
        assert!(m.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn from_vec_ok() {
        let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(m.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn from_vec_length_mismatch() {
        assert!(Matrix::from_vec(vec![1.0, 2.0, 3.0], 2, 2).is_err());
    }

    #[test]
    fn transpose_2x3() {
        // [[1, 2, 3],   transposed   [[1, 4],
        //  [4, 5, 6]]      =>         [2, 5],
        //                             [3, 6]]
        let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
