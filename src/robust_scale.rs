use flucoma_sys::{
    robust_scaling_create, robust_scaling_destroy, robust_scaling_fit, robust_scaling_initialized,
    robust_scaling_process, FlucomaIndex,
};

/// Percentile-based robust scaler for dataset-style matrices.
///
/// Uses `(x - median) / (high_percentile - low_percentile)` per feature,
/// which is less sensitive to outliers than min-max or z-score scaling.
///
/// Input/output layout is row-major over points:
/// `[row0_cols..., row1_cols..., ...]`.
pub struct RobustScale {
    inner: *mut u8,
    low_percentile: f64,
    high_percentile: f64,
    cols: Option<usize>,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for RobustScale {}

impl RobustScale {
    pub fn new(low_percentile: f64, high_percentile: f64) -> Result<Self, &'static str> {
        if !(0.0..=100.0).contains(&low_percentile) {
            return Err("low_percentile must be in [0, 100]");
        }
        if !(0.0..=100.0).contains(&high_percentile) {
            return Err("high_percentile must be in [0, 100]");
        }
        if low_percentile > high_percentile {
            return Err("low_percentile must be <= high_percentile");
        }
        let inner = robust_scaling_create();
        if inner.is_null() {
            return Err("failed to create RobustScaling instance");
        }
        Ok(Self {
            inner,
            low_percentile,
            high_percentile,
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
        robust_scaling_fit(
            self.inner,
            self.low_percentile,
            self.high_percentile,
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
        robust_scaling_initialized(self.inner)
    }

    fn process_internal(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
        inverse: bool,
    ) -> Result<Vec<f64>, &'static str> {
        if !self.is_fitted() {
            return Err("robust scaler is not fitted");
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
        robust_scaling_process(
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

impl Drop for RobustScale {
    fn drop(&mut self) {
        robust_scaling_destroy(self.inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_scale_then_inverse_returns_input() {
        let data = vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0, 1000.0, -999.0];
        let mut r = RobustScale::new(25.0, 75.0).unwrap();
        let scaled = r.fit_transform(&data, 4, 2).unwrap();
        let inv = r.inverse_transform(&scaled, 4, 2).unwrap();
        for (a, b) in data.iter().zip(inv.iter()) {
            assert!((a - b).abs() < 1e-8, "expected {a}, got {b}");
        }
    }

    #[test]
    fn transform_before_fit_fails() {
        let r = RobustScale::new(25.0, 75.0).unwrap();
        let err = r.transform(&[1.0, 2.0], 1, 2).unwrap_err();
        assert_eq!(err, "robust scaler is not fitted");
    }
}
