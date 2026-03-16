use flucoma_sys::{
    pca_create, pca_destroy, pca_dims, pca_fit, pca_initialized, pca_inverse_transform,
    pca_transform, FlucomaIndex,
};

use crate::matrix::{AsMatrixView, Matrix, MatrixView};
use crate::normalize::Normalize;
use crate::robust_scale::RobustScale;
use crate::standardize::Standardize;

// -------------------------------------------------------------------------------------------------

/// Optional preprocessing scaler applied to data before PCA fit and transform.
///
/// Scaling often improves PCA quality when features have different units or
/// dynamic ranges. The same scaler is automatically applied (and inverted)
/// during [`Pca::transform`] and [`Pca::inverse_transform`].
#[derive(Debug, Clone, Copy, Default)]
pub enum PcaScaler {
    /// No preprocessing -- data is passed to PCA as-is.
    #[default]
    None,
    /// Min-max normalization into `[min, max]` per feature. Requires `min != max`.
    Normalize { min: f64, max: f64 },
    /// Z-score standardization to zero mean and unit variance per feature.
    Standardize,
    /// Percentile-based robust scaling: `(x − median) / (high − low)` per feature.
    RobustScale {
        low_percentile: f64,
        high_percentile: f64,
    },
}

impl PcaScaler {
    fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::None | Self::Standardize => Ok(()),
            Self::Normalize { min, max } => {
                if min == max {
                    return Err("Normalize scaler requires min != max");
                }
                Ok(())
            }
            Self::RobustScale {
                low_percentile,
                high_percentile,
            } => {
                if !(0.0..=100.0).contains(low_percentile) {
                    return Err("RobustScale low_percentile must be in [0, 100]");
                }
                if !(0.0..=100.0).contains(high_percentile) {
                    return Err("RobustScale high_percentile must be in [0, 100]");
                }
                if low_percentile > high_percentile {
                    return Err("RobustScale low_percentile must be <= high_percentile");
                }
                Ok(())
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------

/// Configuration for a [`Pca`] processor.
#[derive(Debug, Clone, Copy)]
pub struct PcaConfig {
    /// Divide projected components by their standard deviation so each component has unit variance.
    /// Useful when feeding PCA output into a distance-based algorithm.
    pub whiten: bool,
    /// Optional preprocessing scaler applied before fitting and transforming. See [`PcaScaler`].
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

// -------------------------------------------------------------------------------------------------

enum FittedScaler {
    None,
    Normalize(Normalize),
    Standardize(Standardize),
    RobustScale(RobustScale),
}

// -------------------------------------------------------------------------------------------------

/// Principal Component Analysis with optional scaler preprocessing.
///
/// Learns a linear projection from a row-major matrix and can project new
/// matrices into a lower-dimensional space, optionally whitening the output
/// and/or applying a preprocessing scaler first.
///
/// Typical workflow:
/// - call [`fit`](Pca::fit) (or [`fit_transform`](Pca::fit_transform)) on a training set
/// - call [`transform`](Pca::transform) on new data to project it
/// - optionally call [`inverse_transform`](Pca::inverse_transform) to reconstruct the original space
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{Matrix, Pca, PcaConfig, PcaScaler};
///
/// let data = Matrix::from_vec(
///     vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  7.0, 8.0, 9.0], 3, 3,
/// ).unwrap();
///
/// let mut pca = Pca::new(PcaConfig {
///     whiten: false,
///     scaler: PcaScaler::Standardize,
/// }).unwrap();
///
/// let (projected, explained) = pca.fit_transform(&data, 2).unwrap();
/// println!("{:.1}% variance explained", explained * 100.0);
///
/// let reconstructed = pca.inverse_transform(&projected).unwrap();
/// ```
///
/// See <https://learn.flucoma.org/reference/pca>
pub struct Pca {
    inner: *mut u8,
    config: PcaConfig,
    dims: Option<usize>,
    fitted_scaler: Option<FittedScaler>,
}

unsafe impl Send for Pca {}

impl Pca {
    /// Create a new PCA processor with the given configuration.
    ///
    /// # Errors
    /// Returns an error if the scaler configuration is invalid (e.g. `min == max`
    /// for [`PcaScaler::Normalize`], or invalid percentile range for
    /// [`PcaScaler::RobustScale`]), or if the underlying C++ allocation fails.
    pub fn new(config: PcaConfig) -> Result<Self, &'static str> {
        config.scaler.validate()?;
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

    /// Return the configuration this processor was created with.
    pub fn config(&self) -> PcaConfig {
        self.config
    }

    /// Fit the PCA model on a row-major matrix.
    ///
    /// If a scaler is configured, it is fitted and applied to `data` before
    /// the PCA decomposition. Calling `fit` again overwrites the model.
    ///
    /// # Errors
    /// Propagates errors from the configured scaler's fit step.
    pub fn fit(&mut self, data: impl AsMatrixView) -> Result<(), &'static str> {
        let data = data.as_matrix_view();
        let fitted_scaler;
        let _scaled_data; // keep owned Matrix alive through pca_fit
        let data_ptr = if matches!(self.config.scaler, PcaScaler::None) {
            fitted_scaler = FittedScaler::None;
            _scaled_data = None::<Matrix>;
            data.data().as_ptr()
        } else {
            let (scaled, scaler) = self.fit_scaler_and_transform(data)?;
            fitted_scaler = scaler;
            let ptr = scaled.data().as_ptr();
            _scaled_data = Some(scaled);
            ptr
        };
        pca_fit(
            self.inner,
            data_ptr,
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
        );
        self.dims = Some(data.cols());
        self.fitted_scaler = Some(fitted_scaler);
        Ok(())
    }

    /// Fit the model and project the training matrix in one step.
    ///
    /// Equivalent to calling [`fit`](Self::fit) followed by
    /// [`transform`](Self::transform) on the same data. Returns
    /// `(projected, explained_variance_ratio)`. See [`transform`](Self::transform)
    /// for details on the return value.
    ///
    /// # Arguments
    /// * `data` - Row-major training matrix.
    /// * `target_dims` - Number of principal components to keep.
    ///
    /// # Errors
    /// Propagates errors from [`fit`](Self::fit) or [`transform`](Self::transform).
    pub fn fit_transform(
        &mut self,
        data: impl AsMatrixView,
        target_dims: usize,
    ) -> Result<(Matrix, f64), &'static str> {
        let data = data.as_matrix_view();
        self.fit(data)?;
        self.transform(data, target_dims)
    }

    /// Project a matrix into a lower-dimensional space.
    ///
    /// Applies the same preprocessing scaler used during [`fit`](Self::fit),
    /// then projects the data onto the leading `target_dims` principal
    /// components. Returns `(projected, explained_variance_ratio)` where
    /// `projected` has shape `(data.rows(), target_dims)` and
    /// `explained_variance_ratio` is in `[0, 1]`.
    ///
    /// # Arguments
    /// * `data` - Row-major input matrix. Column count must match the
    ///   dimension the model was fitted on.
    /// * `target_dims` - Number of principal components to keep. Must be in
    ///   `[1, data.cols()]`.
    ///
    /// # Errors
    /// Returns an error if the model is not fitted, if `data.cols()` does not
    /// match the fitted dimension, or if `target_dims` is out of range.
    pub fn transform(
        &self,
        data: impl AsMatrixView,
        target_dims: usize,
    ) -> Result<(Matrix, f64), &'static str> {
        let data = data.as_matrix_view();
        self.ensure_fitted(data.cols())?;
        if target_dims == 0 {
            return Err("target_dims must be > 0");
        }
        if target_dims > data.cols() {
            return Err("target_dims must be <= input cols");
        }

        let _scaled_data; // keep owned Matrix alive through pca_transform
        let data_ptr = match self.fitted_scaler.as_ref().ok_or("PCA is not fitted")? {
            FittedScaler::None => {
                _scaled_data = None::<Matrix>;
                data.data().as_ptr()
            }
            _ => {
                let scaled = self.apply_scaler_transform(data)?;
                let ptr = scaled.data().as_ptr();
                _scaled_data = Some(scaled);
                ptr
            }
        };
        let mut out = Matrix::new(data.rows(), target_dims);
        let explained = pca_transform(
            self.inner,
            data_ptr,
            data.rows() as FlucomaIndex,
            data.cols() as FlucomaIndex,
            out.data_mut().as_mut_ptr(),
            target_dims as FlucomaIndex,
            self.config.whiten,
        );
        Ok((out, explained))
    }

    /// Reconstruct data in the original feature space from a projected matrix.
    ///
    /// Reverses the PCA projection and, if a scaler was configured, the
    /// preprocessing step as well. The reconstruction is exact only when
    /// `projected.cols() == fitted_dims`; with fewer components it is a
    /// low-rank approximation.
    ///
    /// Returns a matrix with shape `(projected.rows(), fitted_dims)`.
    ///
    /// # Errors
    /// Returns an error if the model is not fitted, or if
    /// `projected.cols() > fitted_dims`.
    pub fn inverse_transform(&self, projected: impl AsMatrixView) -> Result<Matrix, &'static str> {
        let projected = projected.as_matrix_view();
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
            padded.data_mut()[dst_start..dst_end]
                .copy_from_slice(&projected.data()[src_start..src_end]);
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
        if matches!(self.fitted_scaler, Some(FittedScaler::None)) {
            Ok(recon_scaled)
        } else {
            self.apply_scaler_inverse_transform(recon_scaled)
        }
    }

    /// Return `true` if the model has been fitted at least once.
    pub fn is_fitted(&self) -> bool {
        pca_initialized(self.inner)
    }

    /// Return the number of features (columns) the model was fitted on, or
    /// `None` if the model has not been fitted yet.
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
        data: MatrixView<'_>,
    ) -> Result<(Matrix, FittedScaler), &'static str> {
        match self.config.scaler {
            PcaScaler::None => unreachable!("No scaling necessary"),
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

    fn apply_scaler_transform(&self, data: MatrixView<'_>) -> Result<Matrix, &'static str> {
        match self.fitted_scaler.as_ref().ok_or("PCA is not fitted")? {
            FittedScaler::None => unreachable!("No scaling necessary"),
            FittedScaler::Normalize(n) => n.transform(data),
            FittedScaler::Standardize(s) => s.transform(data),
            FittedScaler::RobustScale(r) => r.transform(data),
        }
    }

    fn apply_scaler_inverse_transform(&self, data: Matrix) -> Result<Matrix, &'static str> {
        match self.fitted_scaler.as_ref().ok_or("PCA is not fitted")? {
            FittedScaler::None => unreachable!("No scaling necessary"),
            FittedScaler::Normalize(n) => n.inverse_transform(&data),
            FittedScaler::Standardize(s) => s.inverse_transform(&data),
            FittedScaler::RobustScale(r) => r.inverse_transform(&data),
        }
    }
}

impl Drop for Pca {
    fn drop(&mut self) {
        pca_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Matrix {
        // 8 x 3 row-major
        Matrix::from_vec(
            vec![
                1.0, 2.0, 0.9, //
                1.2, 2.2, 1.1, //
                0.8, 1.7, 0.7, //
                3.0, 3.2, 2.9, //
                2.8, 3.0, 2.6, //
                10.0, -8.0, 9.0, //
                2.9, 3.1, 2.7, //
                1.1, 2.1, 1.0,
            ],
            8,
            3,
        )
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
