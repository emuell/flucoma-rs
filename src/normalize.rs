use flucoma_sys::{
    normalization_create, normalization_destroy, normalization_fit, normalization_initialized,
    normalization_process, FlucomaIndex,
};

use crate::matrix::Matrix;

/// Min-max normalizer for dataset-style matrices.
///
/// Learns per-column minimum and maximum values from a dataset and maps each
/// feature into the configured range.
///
/// See <https://learn.flucoma.org/reference/normalize>
///
/// Input/output layout is row-major over points:
/// `[row0_cols..., row1_cols..., ...]`.
pub struct Normalize {
    inner: *mut u8,
    min: f64,
    max: f64,
    cols: Option<usize>,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for Normalize {}

impl Normalize {
    /// Create a new min-max normalizer targeting the inclusive range `[min, max]`.
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

    /// Fit the normalizer from a row-major matrix.
    pub fn fit(&mut self, data: &Matrix) -> Result<(), &'static str> {
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

    /// Transform a matrix into the fitted output range.
    ///
    /// # Errors
    /// Returns an error if the normalizer has not been fitted yet, or if the
    /// matrix column count differs from the fitted feature dimension.
    pub fn transform(&self, data: &Matrix) -> Result<Matrix, &'static str> {
        self.process_internal(data, false)
    }

    /// Undo a previous min-max transform.
    pub fn inverse_transform(&self, data: &Matrix) -> Result<Matrix, &'static str> {
        self.process_internal(data, true)
    }

    /// Fit the normalizer and transform the same matrix in one step.
    pub fn fit_transform(&mut self, data: &Matrix) -> Result<Matrix, &'static str> {
        self.fit(data)?;
        self.transform(data)
    }

    pub fn is_fitted(&self) -> bool {
        normalization_initialized(self.inner)
    }

    fn process_internal(&self, data: &Matrix, inverse: bool) -> Result<Matrix, &'static str> {
        if !self.is_fitted() {
            return Err("normalizer is not fitted");
        }
        if self.cols != Some(data.cols()) {
            return Err("cols must match fitted feature dimension");
        }
        let mut out = Matrix::new(data.rows(), data.cols());
        normalization_process(
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

impl Drop for Normalize {
    fn drop(&mut self) {
        normalization_destroy(self.inner);
    }
}

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
}
