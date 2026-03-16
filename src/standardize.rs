use flucoma_sys::{
    standardization_create, standardization_destroy, standardization_fit,
    standardization_initialized, standardization_process, FlucomaIndex,
};

use crate::matrix::{AsMatrixView, AsMatrixViewMut, Matrix, MatrixView, MatrixViewMut};

// -------------------------------------------------------------------------------------------------

/// Z-score standardizer for dataset-style matrices.
///
/// Learns a per-feature mean and standard deviation from a dataset and maps
/// each column to zero mean and unit variance. Fit once on training data,
/// then call [`transform`](Standardize::transform) on any same-width matrix,
/// or [`inverse_transform`](Standardize::inverse_transform) to recover the
/// original scale.
///
/// Input/output layout is row-major: `[row0_col0, row0_col1, …, rowN_colM]`.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{Matrix, Standardize};
///
/// let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
/// let mut s = Standardize::new().unwrap();
/// let scaled = s.fit_transform(&data).unwrap();
/// let restored = s.inverse_transform(&scaled).unwrap();
/// ```
///
/// See <https://learn.flucoma.org/reference/standardize>
pub struct Standardize {
    inner: *mut u8,
    cols: Option<usize>,
}

unsafe impl Send for Standardize {}

impl Standardize {
    /// Create a new standardizer.
    ///
    /// # Errors
    /// Returns an error if the underlying C++ allocation fails.
    pub fn new() -> Result<Self, &'static str> {
        let inner = standardization_create();
        if inner.is_null() {
            return Err("failed to create Standardization instance");
        }
        Ok(Self { inner, cols: None })
    }

    /// Fit the standardizer to a row-major matrix.
    ///
    /// Computes per-column mean and standard deviation. Calling `fit` again
    /// on new data overwrites the previously learned statistics.
    pub fn fit(&mut self, data: impl AsMatrixView) -> Result<(), &'static str> {
        let data = data.as_matrix_view();
        standardization_fit(
            self.inner,
            data.data().as_ptr(),
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
        );
        self.cols = Some(data.cols());
        Ok(())
    }

    /// Map a matrix to zero mean and unit variance using the fitted statistics.
    ///
    /// # Errors
    /// Returns an error if the standardizer has not been fitted yet, or if the
    /// matrix column count differs from the fitted feature dimension.
    pub fn transform(&self, data: impl AsMatrixView) -> Result<Matrix, &'static str> {
        self.process_internal(data.as_matrix_view(), false)
    }

    /// Map a matrix to zero mean and unit variance, writing results into a pre-allocated buffer.
    ///
    /// # Errors
    /// Returns an error if the standardizer is not fitted, column counts differ, or
    /// output dimensions do not match input dimensions.
    pub fn transform_into(
        &self,
        data: impl AsMatrixView,
        mut output: impl AsMatrixViewMut,
    ) -> Result<(), &'static str> {
        self.process_internal_into(data.as_matrix_view(), output.as_matrix_view_mut(), false)
    }

    /// Recover the original scale by reversing a previous [`transform`](Self::transform).
    ///
    /// # Errors
    /// Returns an error if the standardizer has not been fitted yet, or if the
    /// matrix column count differs from the fitted feature dimension.
    pub fn inverse_transform(&self, data: impl AsMatrixView) -> Result<Matrix, &'static str> {
        self.process_internal(data.as_matrix_view(), true)
    }

    /// Reverse a previous [`transform`](Self::transform), writing results into a pre-allocated buffer.
    ///
    /// # Errors
    /// Returns an error if the standardizer is not fitted, column counts differ, or
    /// output dimensions do not match input dimensions.
    pub fn inverse_transform_into(
        &self,
        data: impl AsMatrixView,
        mut output: impl AsMatrixViewMut,
    ) -> Result<(), &'static str> {
        self.process_internal_into(data.as_matrix_view(), output.as_matrix_view_mut(), true)
    }

    /// Fit the standardizer and transform the same matrix in one step.
    ///
    /// Equivalent to calling [`fit`](Self::fit) followed by
    /// [`transform`](Self::transform) on the same data.
    ///
    /// # Errors
    /// Propagates errors from [`fit`](Self::fit) or [`transform`](Self::transform).
    pub fn fit_transform(&mut self, data: impl AsMatrixView) -> Result<Matrix, &'static str> {
        let data = data.as_matrix_view();
        self.fit(data)?;
        self.transform(data)
    }

    pub fn is_fitted(&self) -> bool {
        standardization_initialized(self.inner)
    }

    fn process_internal_into(
        &self,
        data: MatrixView<'_>,
        mut output: MatrixViewMut<'_>,
        inverse: bool,
    ) -> Result<(), &'static str> {
        if !self.is_fitted() {
            return Err("standardizer is not fitted");
        }
        if self.cols != Some(data.cols()) {
            return Err("cols must match fitted feature dimension");
        }
        if output.rows() != data.rows() || output.cols() != data.cols() {
            return Err("output dimensions must match input dimensions");
        }
        standardization_process(
            self.inner,
            data.data().as_ptr(),
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
            output.data_mut().as_mut_ptr(),
            inverse,
        );
        Ok(())
    }

    fn process_internal(
        &self,
        data: MatrixView<'_>,
        inverse: bool,
    ) -> Result<Matrix, &'static str> {
        let mut out = Matrix::new(data.rows(), data.cols());
        self.process_internal_into(data, out.view_mut(), inverse)?;
        Ok(out)
    }
}

impl Drop for Standardize {
    fn drop(&mut self) {
        standardization_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardize_then_inverse_returns_input() {
        let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
        let mut s = Standardize::new().unwrap();
        let z = s.fit_transform(&data).unwrap();
        let inv = s.inverse_transform(&z).unwrap();
        for (a, b) in data.data().iter().zip(inv.data().iter()) {
            assert!((a - b).abs() < 1e-9, "expected {a}, got {b}");
        }
    }

    #[test]
    fn transform_before_fit_fails() {
        let s = Standardize::new().unwrap();
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
        let err = s.transform(&data).unwrap_err();
        assert_eq!(err, "standardizer is not fitted");
    }

    #[test]
    fn transform_into_matches_transform() {
        let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
        let mut s = Standardize::new().unwrap();
        s.fit(&data).unwrap();
        let expected = s.transform(&data).unwrap();
        let mut out = Matrix::new(3, 2);
        s.transform_into(&data, &mut out).unwrap();
        assert_eq!(expected.data(), out.data());
    }

    #[test]
    fn transform_into_wrong_dims_fails() {
        let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
        let mut s = Standardize::new().unwrap();
        s.fit(&data).unwrap();
        let mut out = Matrix::new(2, 2);
        let err = s.transform_into(&data, &mut out).unwrap_err();
        assert_eq!(err, "output dimensions must match input dimensions");
    }
}
