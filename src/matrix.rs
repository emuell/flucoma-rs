// -------------------------------------------------------------------------------------------------

/// Row-major `f64` matrix with owned data.  See also [`MatrixView`].
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
    pub fn transposed(&self) -> Self {
        self.view().transposed()
    }

    /// Return a borrowed [`MatrixView`] over this matrix's data.
    pub fn view(&self) -> MatrixView<'_> {
        MatrixView {
            data: &self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Return a mutably borrowed [`MatrixViewMut`] over this matrix's data.
    pub fn view_mut(&mut self) -> MatrixViewMut<'_> {
        MatrixViewMut {
            data: &mut self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// -------------------------------------------------------------------------------------------------

/// A borrowed, non-owning view of a row-major `f64` matrix. See also [`Matrix`].
///
/// `MatrixView` is `Copy` and cheap to pass around. Use it with algorithms
/// that accept [`AsMatrixView`] to avoid unnecessary allocations.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
}

impl<'a> MatrixView<'a> {
    /// Create a matrix from a flat slice with the given rows and cols.
    ///
    /// # Errors
    /// Returns an error if either dimension is zero or if `data.len() != rows * cols`.
    pub fn from_slice(data: &'a [f64], rows: usize, cols: usize) -> Result<Self, &'static str> {
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
        self.data
    }

    /// Return the transpose as a new owned [`Matrix`].
    pub fn transposed(&self) -> Matrix {
        let mut out = vec![0.0f64; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Matrix {
            data: out,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Clone the viewed data into an owned [`Matrix`].
    pub fn to_owned(&self) -> Matrix {
        Matrix::from_vec(self.data.to_vec(), self.rows, self.cols).unwrap()
    }
}

// -------------------------------------------------------------------------------------------------

/// A mutably borrowed, non-owning view of a row-major `f64` matrix.
///
/// Use with algorithms that accept [`AsMatrixViewMut`] to write results into a
/// pre-allocated buffer, avoiding per-call heap allocation.
#[derive(Debug)]
pub struct MatrixViewMut<'a> {
    data: &'a mut [f64],
    rows: usize,
    cols: usize,
}

impl<'a> MatrixViewMut<'a> {
    /// Create a mutable matrix view from a flat slice with the given dimensions.
    ///
    /// # Errors
    /// Returns an error if either dimension is zero or if `data.len() != rows * cols`.
    pub fn from_slice(data: &'a mut [f64], rows: usize, cols: usize) -> Result<Self, &'static str> {
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
        self.data
    }

    /// Mutable flat row-major data slice.
    pub fn data_mut(&mut self) -> &mut [f64] {
        self.data
    }

    /// Clone the viewed data into an owned [`Matrix`].
    pub fn to_owned(&self) -> Matrix {
        Matrix::from_vec(self.data.to_vec(), self.rows, self.cols).unwrap()
    }
}

// -------------------------------------------------------------------------------------------------

/// Conversion to a mutable borrowed [`MatrixViewMut`].
pub trait AsMatrixViewMut {
    fn as_matrix_view_mut(&mut self) -> MatrixViewMut<'_>;
}

impl AsMatrixViewMut for Matrix {
    fn as_matrix_view_mut(&mut self) -> MatrixViewMut<'_> {
        self.view_mut()
    }
}

impl AsMatrixViewMut for MatrixViewMut<'_> {
    fn as_matrix_view_mut(&mut self) -> MatrixViewMut<'_> {
        MatrixViewMut {
            data: &mut *self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T: AsMatrixViewMut> AsMatrixViewMut for &mut T {
    fn as_matrix_view_mut(&mut self) -> MatrixViewMut<'_> {
        (*self).as_matrix_view_mut()
    }
}

impl AsMatrixView for MatrixViewMut<'_> {
    fn as_matrix_view(&self) -> MatrixView<'_> {
        MatrixView {
            data: self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// -------------------------------------------------------------------------------------------------

/// Conversion to a borrowed [`MatrixView`].
pub trait AsMatrixView {
    fn as_matrix_view(&self) -> MatrixView<'_>;
}

impl AsMatrixView for Matrix {
    fn as_matrix_view(&self) -> MatrixView<'_> {
        self.view()
    }
}

impl AsMatrixView for MatrixView<'_> {
    fn as_matrix_view(&self) -> MatrixView<'_> {
        *self
    }
}

impl<T: AsMatrixView> AsMatrixView for &T {
    fn as_matrix_view(&self) -> MatrixView<'_> {
        (*self).as_matrix_view()
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
    fn matrix_view_mut_from_slice() {
        let mut buf = vec![1.0f64, 2.0, 3.0, 4.0];
        let v = MatrixViewMut::from_slice(&mut buf, 2, 2).unwrap();
        assert_eq!(v.rows(), 2);
        assert_eq!(v.cols(), 2);
        assert!(MatrixViewMut::from_slice(&mut buf, 2, 3).is_err());
    }

    #[test]
    fn matrix_view_mut_read_write() {
        let mut buf = vec![0.0f64; 4];
        let mut v = MatrixViewMut::from_slice(&mut buf, 2, 2).unwrap();
        v.data_mut()[0] = 42.0;
        assert_eq!(v.data()[0], 42.0);
    }

    #[test]
    fn matrix_view_mut_as_matrix_view() {
        let mut m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let vm = m.view_mut();
        let view = vm.as_matrix_view();
        assert_eq!(view.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn transpose_2x3() {
        // [[1, 2, 3],   transposed   [[1, 4],
        //  [4, 5, 6]]      =>         [2, 5],
        //                             [3, 6]]
        let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let t = m.transposed();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
