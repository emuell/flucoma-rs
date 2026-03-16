use flucoma_sys::{
    normalization_create, normalization_destroy, normalization_fit, normalization_initialized,
    normalization_process, FlucomaIndex,
};

use crate::matrix::{AsMatrixView, AsMatrixViewMut, Matrix, MatrixView, MatrixViewMut};

// -------------------------------------------------------------------------------------------------

/// Min-max normalizer for dataset-style matrices.
///
/// Learns per-column minimum and maximum values from a dataset and maps each
/// feature into a configured output range `[min, max]`. Fit once on training
/// data, then call [`transform`](Normalize::transform) on any same-width
/// matrix, or [`inverse_transform`](Normalize::inverse_transform) to recover
/// the original scale.
///
/// Input/output layout is row-major: `[row0_col0, row0_col1, …, rowN_colM]`.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{Matrix, Normalize};
///
/// let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
/// let mut n = Normalize::new(0.0, 1.0).unwrap();
/// let scaled = n.fit_transform(&data).unwrap();
/// let restored = n.inverse_transform(&scaled).unwrap();
/// ```
///
/// See <https://learn.flucoma.org/reference/normalize>
pub struct Normalize {
    inner: *mut u8,
    min: f64,
    max: f64,
    cols: Option<usize>,
}

unsafe impl Send for Normalize {}

impl Normalize {
    /// Create a new min-max normalizer targeting the inclusive range `[min, max]`.
    ///
    /// # Arguments
    /// * `min` - Lower bound of the output range.
    /// * `max` - Upper bound of the output range. Must differ from `min`.
    ///
    /// # Errors
    /// Returns an error if `min == max`.
    pub fn new(min: f64, max: f64) -> Result<Self, &'static str> {
        if min == max {
            return Err("min and max must be different");
        }
        let inner = normalization_create();
        if inner.is_null() {
            return Err("failed to create Normalization instance");
        }
        Ok(Self {
            inner,
            min,
            max,
            cols: None,
        })
    }

    /// Fit the normalizer to a row-major matrix.
    ///
    /// Computes per-column minimum and maximum values. Calling `fit` again
    /// on new data overwrites the previously learned statistics.
    pub fn fit(&mut self, data: impl AsMatrixView) -> Result<(), &'static str> {
        let data = data.as_matrix_view();
        normalization_fit(
            self.inner,
            self.min,
            self.max,
            data.data().as_ptr(),
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
        );
        self.cols = Some(data.cols());
        Ok(())
    }

    /// Map a matrix into the fitted output range.
    ///
    /// # Errors
    /// Returns an error if the normalizer has not been fitted yet, or if the
    /// matrix column count differs from the fitted feature dimension.
    pub fn transform(&self, data: impl AsMatrixView) -> Result<Matrix, &'static str> {
        self.process_internal(data.as_matrix_view(), false)
    }

    /// Map a matrix into the fitted output range, writing results into a pre-allocated buffer.
    ///
    /// # Errors
    /// Returns an error if the normalizer is not fitted, column counts differ, or
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
    /// Returns an error if the normalizer has not been fitted yet, or if the
    /// matrix column count differs from the fitted feature dimension.
    pub fn inverse_transform(&self, data: impl AsMatrixView) -> Result<Matrix, &'static str> {
        self.process_internal(data.as_matrix_view(), true)
    }

    /// Reverse a previous [`transform`](Self::transform), writing results into a pre-allocated buffer.
    ///
    /// # Errors
    /// Returns an error if the normalizer is not fitted, column counts differ, or
    /// output dimensions do not match input dimensions.
    pub fn inverse_transform_into(
        &self,
        data: impl AsMatrixView,
        mut output: impl AsMatrixViewMut,
    ) -> Result<(), &'static str> {
        self.process_internal_into(data.as_matrix_view(), output.as_matrix_view_mut(), true)
    }

    /// Fit the normalizer and transform the same matrix in one step.
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
        normalization_initialized(self.inner)
    }

    fn process_internal_into(
        &self,
        data: MatrixView<'_>,
        mut output: MatrixViewMut<'_>,
        inverse: bool,
    ) -> Result<(), &'static str> {
        if !self.is_fitted() {
            return Err("normalizer is not fitted");
        }
        if self.cols != Some(data.cols()) {
            return Err("cols must match fitted feature dimension");
        }
        if output.rows() != data.rows() || output.cols() != data.cols() {
            return Err("output dimensions must match input dimensions");
        }
        normalization_process(
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

impl Drop for Normalize {
    fn drop(&mut self) {
        normalization_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_then_inverse_returns_input() {
        // 3 points x 2 dims (row-major)
        let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
        let mut n = Normalize::new(0.0, 1.0).unwrap();
        let norm = n.fit_transform(&data).unwrap();
        let inv = n.inverse_transform(&norm).unwrap();
        for (a, b) in data.data().iter().zip(inv.data().iter()) {
            assert!((a - b).abs() < 1e-9, "expected {a}, got {b}");
        }
    }

    #[test]
    fn transform_before_fit_fails() {
        let n = Normalize::new(0.0, 1.0).unwrap();
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
        let err = n.transform(&data).unwrap_err();
        assert_eq!(err, "normalizer is not fitted");
    }

    #[test]
    fn transform_into_matches_transform() {
        let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
        let mut n = Normalize::new(0.0, 1.0).unwrap();
        n.fit(&data).unwrap();
        let expected = n.transform(&data).unwrap();
        let mut out = Matrix::new(3, 2);
        n.transform_into(&data, &mut out).unwrap();
        assert_eq!(expected.data(), out.data());
    }

    #[test]
    fn transform_into_wrong_dims_fails() {
        let data = Matrix::from_vec(vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0], 3, 2).unwrap();
        let mut n = Normalize::new(0.0, 1.0).unwrap();
        n.fit(&data).unwrap();
        let mut out = Matrix::new(2, 2);
        let err = n.transform_into(&data, &mut out).unwrap_err();
        assert_eq!(err, "output dimensions must match input dimensions");
    }
}
