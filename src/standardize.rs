use flucoma_sys::{
    standardization_create, standardization_destroy, standardization_fit,
    standardization_initialized, standardization_process, FlucomaIndex,
};

use crate::matrix::Matrix;

/// Z-score standardizer for dataset-style matrices.
///
/// Learns a per-feature mean and standard deviation from a dataset and then
/// maps each column to zero mean and unit variance.
///
/// See <https://learn.flucoma.org/reference/standardize>
///
/// Input/output layout is row-major over points:
/// `[row0_cols..., row1_cols..., ...]`.
pub struct Standardize {
    inner: *mut u8,
    cols: Option<usize>,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for Standardize {}

impl Standardize {
    /// Create a new standardizer.
    pub fn new() -> Result<Self, &'static str> {
        let inner = standardization_create();
        if inner.is_null() {
            return Err("failed to create Standardization instance");
        }
        Ok(Self { inner, cols: None })
    }

    /// Fit the standardizer from a row-major matrix.
    pub fn fit(&mut self, data: &Matrix) -> Result<(), &'static str> {
        standardization_fit(
            self.inner,
            data.data().as_ptr(),
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
        );
        self.cols = Some(data.cols());
        Ok(())
    }

    /// Transform a matrix using the fitted statistics.
    pub fn transform(&self, data: &Matrix) -> Result<Matrix, &'static str> {
        self.process_internal(data, false)
    }

    /// Undo a previous standardization step.
    pub fn inverse_transform(&self, data: &Matrix) -> Result<Matrix, &'static str> {
        self.process_internal(data, true)
    }

    /// Fit the standardizer and transform the same matrix in one step.
    pub fn fit_transform(&mut self, data: &Matrix) -> Result<Matrix, &'static str> {
        self.fit(data)?;
        self.transform(data)
    }

    pub fn is_fitted(&self) -> bool {
        standardization_initialized(self.inner)
    }

    fn process_internal(&self, data: &Matrix, inverse: bool) -> Result<Matrix, &'static str> {
        if !self.is_fitted() {
            return Err("standardizer is not fitted");
        }
        if self.cols != Some(data.cols()) {
            return Err("cols must match fitted feature dimension");
        }
        let mut out = Matrix::new(data.rows(), data.cols());
        standardization_process(
            self.inner,
            data.data().as_ptr(),
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
            out.data_mut().as_mut_ptr(),
            inverse,
        );
        Ok(out)
    }
}

impl Drop for Standardize {
    fn drop(&mut self) {
        standardization_destroy(self.inner);
    }
}

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
}
