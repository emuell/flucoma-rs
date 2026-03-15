use flucoma_sys::{
    pca_create, pca_destroy, pca_dims, pca_fit, pca_initialized, pca_inverse_transform,
    pca_transform, FlucomaIndex,
};

use crate::matrix::Matrix;
use crate::normalize::Normalize;
use crate::robust_scale::RobustScale;
use crate::standardize::Standardize;

/// Optional preprocessing scaler applied before PCA fit/transform.
#[derive(Debug, Clone, Copy)]
#[derive(Default)]
pub enum PcaScaler {
    #[default]
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
///
/// Learns a linear projection from a row-major matrix and can project new
/// matrices into a lower-dimensional space, optionally applying a preprocessing
/// scaler first.
///
/// See <https://learn.flucoma.org/reference/pca>
pub struct Pca {
    inner: *mut u8,
    config: PcaConfig,
    dims: Option<usize>,
    fitted_scaler: Option<FittedScaler>,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for Pca {}

impl Pca {
    /// Create a new PCA processor.
    ///
    /// # Errors
    /// Returns an error if the scaler configuration is invalid.
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

    /// Fit the PCA model from a row-major matrix.
    pub fn fit(&mut self, data: &Matrix) -> Result<(), &'static str> {
        let (scaled_data, fitted_scaler) = self.fit_scaler_and_transform(data)?;
        pca_fit(
            self.inner,
            scaled_data.data().as_ptr(),
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
        );
        self.dims = Some(data.cols());
        self.fitted_scaler = Some(fitted_scaler);
        Ok(())
    }

    /// Fit the model and project the same matrix in one step.
    pub fn fit_transform(
        &mut self,
        data: &Matrix,
        target_dims: usize,
    ) -> Result<(Matrix, f64), &'static str> {
        self.fit(data)?;
        self.transform(data, target_dims)
    }

    /// Project a matrix to `target_dims`; returns
    /// `(projected_matrix, explained_variance_ratio)`.
    pub fn transform(&self, data: &Matrix, target_dims: usize) -> Result<(Matrix, f64), &'static str> {
        self.ensure_fitted(data.cols())?;
        if target_dims == 0 {
            return Err("target_dims must be > 0");
        }
        if target_dims > data.cols() {
            return Err("target_dims must be <= input cols");
        }

        let scaled_data = self.apply_scaler_transform(data)?;
        let mut out = Matrix::new(data.rows(), target_dims);
        let explained = pca_transform(
            self.inner,
            scaled_data.data().as_ptr(),
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
            out.data_mut().as_mut_ptr(),
            target_dims as FlucomaIndex,
            self.config.whiten,
        );
        Ok((out, explained))
    }

    /// Inverse-transform projected PCA data back to the original feature space.
    pub fn inverse_transform(&self, projected: &Matrix) -> Result<Matrix, &'static str> {
        let cols = self.dims.ok_or("PCA is not fitted")?;
        if projected.cols() > cols {
            return Err("projected_cols must be <= fitted dims");
        }

        // Upstream PCA inverse expects an input matrix with full `dims` columns,
        // with projected data occupying the leading columns.
        let mut padded = Matrix::new(projected.rows(), cols);
        for r in 0..projected.rows() {
            let src_start = r * projected.cols();
            let src_end = src_start + projected.cols();
            let dst_start = r * cols;
            let dst_end = dst_start + projected.cols();
            padded.data_mut()[dst_start..dst_end].copy_from_slice(&projected.data()[src_start..src_end]);
        }

        let mut recon_scaled = Matrix::new(projected.rows(), cols);
        pca_inverse_transform(
            self.inner,
            padded.data().as_ptr(),
            projected.rows() as FlucomaIndex,
            cols as FlucomaIndex,
            recon_scaled.data_mut().as_mut_ptr(),
            cols as FlucomaIndex,
            self.config.whiten,
        );
        self.apply_scaler_inverse_transform(&recon_scaled)
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
        data: &Matrix,
    ) -> Result<(Matrix, FittedScaler), &'static str> {
        match self.config.scaler {
            PcaScaler::None => Ok((data.clone(), FittedScaler::None)),
            PcaScaler::Normalize { min, max } => {
                let mut n = Normalize::new(min, max)?;
                let out = n.fit_transform(data)?;
                Ok((out, FittedScaler::Normalize(n)))
            }
            PcaScaler::Standardize => {
                let mut s = Standardize::new()?;
                let out = s.fit_transform(data)?;
                Ok((out, FittedScaler::Standardize(s)))
            }
            PcaScaler::RobustScale {
                low_percentile,
                high_percentile,
            } => {
                let mut r = RobustScale::new(low_percentile, high_percentile)?;
                let out = r.fit_transform(data)?;
                Ok((out, FittedScaler::RobustScale(r)))
            }
        }
    }

    fn apply_scaler_transform(&self, data: &Matrix) -> Result<Matrix, &'static str> {
        match self.fitted_scaler.as_ref().ok_or("PCA is not fitted")? {
            FittedScaler::None => Ok(data.clone()),
            FittedScaler::Normalize(n) => n.transform(data),
            FittedScaler::Standardize(s) => s.transform(data),
            FittedScaler::RobustScale(r) => r.transform(data),
        }
    }

    fn apply_scaler_inverse_transform(&self, data: &Matrix) -> Result<Matrix, &'static str> {
        match self.fitted_scaler.as_ref().ok_or("PCA is not fitted")? {
            FittedScaler::None => Ok(data.clone()),
            FittedScaler::Normalize(n) => n.inverse_transform(data),
            FittedScaler::Standardize(s) => s.inverse_transform(data),
            FittedScaler::RobustScale(r) => r.inverse_transform(data),
        }
    }
}

impl Drop for Pca {
    fn drop(&mut self) {
        pca_destroy(self.inner);
    }
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

    fn sample_data() -> Matrix {
        // 8 x 3 row-major
        Matrix::from_vec(vec![
            1.0, 2.0, 0.9, //
            1.2, 2.2, 1.1, //
            0.8, 1.7, 0.7, //
            3.0, 3.2, 2.9, //
            2.8, 3.0, 2.6, //
            10.0, -8.0, 9.0, //
            2.9, 3.1, 2.7, //
            1.1, 2.1, 1.0,
        ], 8, 3)
        .unwrap()
    }

    #[test]
    fn pca_fit_transform_works_without_scaler() {
        let data = sample_data();
        let mut p = Pca::new(PcaConfig::default()).unwrap();
        let (proj, explained) = p.fit_transform(&data, 2).unwrap();
        assert_eq!(proj.rows(), 8);
        assert_eq!(proj.cols(), 2);
        assert_eq!(proj.data().len(), 16);
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
        let (proj, _) = p.fit_transform(&data, 2).unwrap();
        let inv = p.inverse_transform(&proj).unwrap();
        assert_eq!(inv.rows(), data.rows());
        assert_eq!(inv.cols(), data.cols());
        assert_eq!(inv.data().len(), data.data().len());
    }

    #[test]
    fn pca_with_standardize_scaler_runs() {
        let data = sample_data();
        let mut p = Pca::new(PcaConfig {
            whiten: true,
            scaler: PcaScaler::Standardize,
        })
        .unwrap();
        let (proj, _) = p.fit_transform(&data, 2).unwrap();
        assert_eq!(proj.rows(), 8);
        assert_eq!(proj.cols(), 2);
    }
}
