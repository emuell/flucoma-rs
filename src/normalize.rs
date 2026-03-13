use flucoma_sys::{
    normalization_create, normalization_destroy, normalization_fit, normalization_initialized,
    normalization_process, FlucomaIndex,
};

/// Min-max normalizer for dataset-style matrices.
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
        normalization_fit(
            self.inner,
            self.min,
            self.max,
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
        normalization_initialized(self.inner)
    }

    fn process_internal(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
        inverse: bool,
    ) -> Result<Vec<f64>, &'static str> {
        if !self.is_fitted() {
            return Err("normalizer is not fitted");
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
        normalization_process(
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
        let data = vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0];
        let mut n = Normalize::new(0.0, 1.0).unwrap();
        let norm = n.fit_transform(&data, 3, 2).unwrap();
        let inv = n.inverse_transform(&norm, 3, 2).unwrap();
        for (a, b) in data.iter().zip(inv.iter()) {
            assert!((a - b).abs() < 1e-9, "expected {a}, got {b}");
        }
    }

    #[test]
    fn transform_before_fit_fails() {
        let n = Normalize::new(0.0, 1.0).unwrap();
        let err = n.transform(&[1.0, 2.0], 1, 2).unwrap_err();
        assert_eq!(err, "normalizer is not fitted");
    }
}
