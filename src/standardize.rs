use flucoma_sys::{
    standardization_create, standardization_destroy, standardization_fit,
    standardization_initialized, standardization_process, FlucomaIndex,
};

/// Z-score standardizer for dataset-style matrices.
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
    pub fn new() -> Result<Self, &'static str> {
        let inner = standardization_create();
        if inner.is_null() {
            return Err("failed to create Standardization instance");
        }
        Ok(Self { inner, cols: None })
    }

    pub fn fit(&mut self, data: &[f64], rows: usize, cols: usize) -> Result<(), &'static str> {
        if rows == 0 {
            return Err("rows must be > 0");
        }
        if cols == 0 {
            return Err("cols must be > 0");
        }
        if data.len() != rows * cols {
            return Err("data length does not match rows * cols");
        }
        standardization_fit(
            self.inner,
            data.as_ptr(),
            rows as FlucomaIndex,
            cols as FlucomaIndex,
        );
        self.cols = Some(cols);
        Ok(())
    }

    pub fn transform(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>, &'static str> {
        self.process_internal(data, rows, cols, false)
    }

    pub fn inverse_transform(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>, &'static str> {
        self.process_internal(data, rows, cols, true)
    }

    pub fn fit_transform(
        &mut self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>, &'static str> {
        self.fit(data, rows, cols)?;
        self.transform(data, rows, cols)
    }

    pub fn is_fitted(&self) -> bool {
        standardization_initialized(self.inner)
    }

    fn process_internal(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
        inverse: bool,
    ) -> Result<Vec<f64>, &'static str> {
        if !self.is_fitted() {
            return Err("standardizer is not fitted");
        }
        if rows == 0 {
            return Err("rows must be > 0");
        }
        if cols == 0 {
            return Err("cols must be > 0");
        }
        if self.cols != Some(cols) {
            return Err("cols must match fitted feature dimension");
        }
        if data.len() != rows * cols {
            return Err("data length does not match rows * cols");
        }
        let mut out = vec![0.0; data.len()];
        standardization_process(
            self.inner,
            data.as_ptr(),
            rows as FlucomaIndex,
            cols as FlucomaIndex,
            out.as_mut_ptr(),
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
        let data = vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0];
        let mut s = Standardize::new().unwrap();
        let z = s.fit_transform(&data, 3, 2).unwrap();
        let inv = s.inverse_transform(&z, 3, 2).unwrap();
        for (a, b) in data.iter().zip(inv.iter()) {
            assert!((a - b).abs() < 1e-9, "expected {a}, got {b}");
        }
    }

    #[test]
    fn transform_before_fit_fails() {
        let s = Standardize::new().unwrap();
        let err = s.transform(&[1.0, 2.0], 1, 2).unwrap_err();
        assert_eq!(err, "standardizer is not fitted");
    }
}
