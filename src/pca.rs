use flucoma_sys::{
    pca_create, pca_destroy, pca_dims, pca_fit, pca_initialized, pca_inverse_transform,
    pca_transform, FlucomaIndex,
};

use crate::normalize::Normalize;
use crate::robust_scale::RobustScale;
use crate::standardize::Standardize;

/// Optional preprocessing scaler applied before PCA fit/transform.
#[derive(Debug, Clone, Copy)]
pub enum PcaScaler {
    None,
    Normalize {
        min: f64,
        max: f64,
    },
    Standardize,
    RobustScale {
        low_percentile: f64,
        high_percentile: f64,
    },
}

impl Default for PcaScaler {
    fn default() -> Self {
        Self::None
    }
}

/// PCA settings.
#[derive(Debug, Clone, Copy)]
pub struct PcaConfig {
    pub whiten: bool,
    pub scaler: PcaScaler,
}

impl Default for PcaConfig {
    fn default() -> Self {
        Self {
            whiten: false,
            scaler: PcaScaler::None,
        }
    }
}

enum FittedScaler {
    None,
    Normalize(Normalize),
    Standardize(Standardize),
    RobustScale(RobustScale),
}

/// Principal Component Analysis with optional scaler preprocessing.
pub struct Pca {
    inner: *mut u8,
    config: PcaConfig,
    dims: Option<usize>,
    fitted_scaler: Option<FittedScaler>,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for Pca {}

impl Pca {
    pub fn new(config: PcaConfig) -> Result<Self, &'static str> {
        validate_scaler_config(config.scaler)?;
        let inner = pca_create();
        if inner.is_null() {
            return Err("failed to create PCA instance");
        }
        Ok(Self {
            inner,
            config,
            dims: None,
            fitted_scaler: None,
        })
    }

    pub fn config(&self) -> PcaConfig {
        self.config
    }

    pub fn fit(&mut self, data: &[f64], rows: usize, cols: usize) -> Result<(), &'static str> {
        validate_matrix(data, rows, cols)?;

        let (scaled_data, fitted_scaler) = self.fit_scaler_and_transform(data, rows, cols)?;
        pca_fit(
            self.inner,
            scaled_data.as_ptr(),
            rows as FlucomaIndex,
            cols as FlucomaIndex,
        );
        self.dims = Some(cols);
        self.fitted_scaler = Some(fitted_scaler);
        Ok(())
    }

    pub fn fit_transform(
        &mut self,
        data: &[f64],
        rows: usize,
        cols: usize,
        target_dims: usize,
    ) -> Result<(Vec<f64>, f64), &'static str> {
        self.fit(data, rows, cols)?;
        self.transform(data, rows, cols, target_dims)
    }

    /// Transform to `target_dims`; returns `(projected_data, explained_variance_ratio)`.
    pub fn transform(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
        target_dims: usize,
    ) -> Result<(Vec<f64>, f64), &'static str> {
        validate_matrix(data, rows, cols)?;
        self.ensure_fitted(cols)?;
        if target_dims == 0 {
            return Err("target_dims must be > 0");
        }
        if target_dims > cols {
            return Err("target_dims must be <= input cols");
        }

        let scaled_data = self.apply_scaler_transform(data, rows, cols)?;
        let mut out = vec![0.0; rows * target_dims];
        let explained = pca_transform(
            self.inner,
            scaled_data.as_ptr(),
            rows as FlucomaIndex,
            cols as FlucomaIndex,
            out.as_mut_ptr(),
            target_dims as FlucomaIndex,
            self.config.whiten,
        );
        Ok((out, explained))
    }

    /// Inverse-transform projected PCA data with `projected_cols` back to original feature space.
    pub fn inverse_transform(
        &self,
        projected: &[f64],
        rows: usize,
        projected_cols: usize,
    ) -> Result<Vec<f64>, &'static str> {
        if rows == 0 {
            return Err("rows must be > 0");
        }
        if projected_cols == 0 {
            return Err("projected_cols must be > 0");
        }
        if projected.len() != rows * projected_cols {
            return Err("projected length does not match rows * projected_cols");
        }
        let cols = self.dims.ok_or("PCA is not fitted")?;
        if projected_cols > cols {
            return Err("projected_cols must be <= fitted dims");
        }

        // Upstream PCA inverse expects an input matrix with full `dims` columns,
        // with projected data occupying the leading columns.
        let mut padded = vec![0.0; rows * cols];
        for r in 0..rows {
            let src_start = r * projected_cols;
            let src_end = src_start + projected_cols;
            let dst_start = r * cols;
            let dst_end = dst_start + projected_cols;
            padded[dst_start..dst_end].copy_from_slice(&projected[src_start..src_end]);
        }

        let mut recon_scaled = vec![0.0; rows * cols];
        pca_inverse_transform(
            self.inner,
            padded.as_ptr(),
            rows as FlucomaIndex,
            cols as FlucomaIndex,
            recon_scaled.as_mut_ptr(),
            cols as FlucomaIndex,
            self.config.whiten,
        );
        self.apply_scaler_inverse_transform(&recon_scaled, rows, cols)
    }

    pub fn is_fitted(&self) -> bool {
        pca_initialized(self.inner)
    }

    pub fn dims(&self) -> Option<usize> {
        if !self.is_fitted() {
            return None;
        }
        Some(pca_dims(self.inner) as usize)
    }

    fn ensure_fitted(&self, cols: usize) -> Result<(), &'static str> {
        if !self.is_fitted() {
            return Err("PCA is not fitted");
        }
        if self.dims != Some(cols) {
            return Err("cols must match fitted feature dimension");
        }
        Ok(())
    }

    fn fit_scaler_and_transform(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f64>, FittedScaler), &'static str> {
        match self.config.scaler {
            PcaScaler::None => Ok((data.to_vec(), FittedScaler::None)),
            PcaScaler::Normalize { min, max } => {
                let mut n = Normalize::new(min, max)?;
                let out = n.fit_transform(data, rows, cols)?;
                Ok((out, FittedScaler::Normalize(n)))
            }
            PcaScaler::Standardize => {
                let mut s = Standardize::new()?;
                let out = s.fit_transform(data, rows, cols)?;
                Ok((out, FittedScaler::Standardize(s)))
            }
            PcaScaler::RobustScale {
                low_percentile,
                high_percentile,
            } => {
                let mut r = RobustScale::new(low_percentile, high_percentile)?;
                let out = r.fit_transform(data, rows, cols)?;
                Ok((out, FittedScaler::RobustScale(r)))
            }
        }
    }

    fn apply_scaler_transform(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>, &'static str> {
        match self.fitted_scaler.as_ref().ok_or("PCA is not fitted")? {
            FittedScaler::None => Ok(data.to_vec()),
            FittedScaler::Normalize(n) => n.transform(data, rows, cols),
            FittedScaler::Standardize(s) => s.transform(data, rows, cols),
            FittedScaler::RobustScale(r) => r.transform(data, rows, cols),
        }
    }

    fn apply_scaler_inverse_transform(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>, &'static str> {
        match self.fitted_scaler.as_ref().ok_or("PCA is not fitted")? {
            FittedScaler::None => Ok(data.to_vec()),
            FittedScaler::Normalize(n) => n.inverse_transform(data, rows, cols),
            FittedScaler::Standardize(s) => s.inverse_transform(data, rows, cols),
            FittedScaler::RobustScale(r) => r.inverse_transform(data, rows, cols),
        }
    }
}

impl Drop for Pca {
    fn drop(&mut self) {
        pca_destroy(self.inner);
    }
}

fn validate_matrix(data: &[f64], rows: usize, cols: usize) -> Result<(), &'static str> {
    if rows == 0 {
        return Err("rows must be > 0");
    }
    if cols == 0 {
        return Err("cols must be > 0");
    }
    if data.len() != rows * cols {
        return Err("data length does not match rows * cols");
    }
    Ok(())
}

fn validate_scaler_config(scaler: PcaScaler) -> Result<(), &'static str> {
    match scaler {
        PcaScaler::None | PcaScaler::Standardize => Ok(()),
        PcaScaler::Normalize { min, max } => {
            if min == max {
                return Err("Normalize scaler requires min != max");
            }
            Ok(())
        }
        PcaScaler::RobustScale {
            low_percentile,
            high_percentile,
        } => {
            if !(0.0..=100.0).contains(&low_percentile) {
                return Err("RobustScale low_percentile must be in [0, 100]");
            }
            if !(0.0..=100.0).contains(&high_percentile) {
                return Err("RobustScale high_percentile must be in [0, 100]");
            }
            if low_percentile > high_percentile {
                return Err("RobustScale low_percentile must be <= high_percentile");
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        // 8 x 3 row-major
        vec![
            1.0, 2.0, 0.9, //
            1.2, 2.2, 1.1, //
            0.8, 1.7, 0.7, //
            3.0, 3.2, 2.9, //
            2.8, 3.0, 2.6, //
            10.0, -8.0, 9.0, //
            2.9, 3.1, 2.7, //
            1.1, 2.1, 1.0,
        ]
    }

    #[test]
    fn pca_fit_transform_works_without_scaler() {
        let data = sample_data();
        let mut p = Pca::new(PcaConfig::default()).unwrap();
        let (proj, explained) = p.fit_transform(&data, 8, 3, 2).unwrap();
        assert_eq!(proj.len(), 16);
        assert!((0.0..=1.0).contains(&explained));
    }

    #[test]
    fn pca_with_robust_scaler_inverse_roundtrip_shape() {
        let data = sample_data();
        let mut p = Pca::new(PcaConfig {
            whiten: false,
            scaler: PcaScaler::RobustScale {
                low_percentile: 25.0,
                high_percentile: 75.0,
            },
        })
        .unwrap();
        let (proj, _) = p.fit_transform(&data, 8, 3, 2).unwrap();
        let inv = p.inverse_transform(&proj, 8, 2).unwrap();
        assert_eq!(inv.len(), data.len());
    }

    #[test]
    fn pca_with_standardize_scaler_runs() {
        let data = sample_data();
        let mut p = Pca::new(PcaConfig {
            whiten: true,
            scaler: PcaScaler::Standardize,
        })
        .unwrap();
        let (proj, _) = p.fit_transform(&data, 8, 3, 2).unwrap();
        assert_eq!(proj.len(), 16);
    }
}
