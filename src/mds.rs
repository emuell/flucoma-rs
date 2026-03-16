use crate::matrix::Matrix;
use flucoma_sys::{mds_create, mds_destroy, mds_process, FlucomaIndex};

// -------------------------------------------------------------------------------------------------

/// Distance metric used by [`Mds`] when computing the pairwise distance matrix.
#[derive(Debug, Clone, Copy)]
#[repr(isize)]
pub enum MdsDistance {
    /// L1 (sum of absolute differences).
    Manhattan = 0,
    /// L2 (square root of sum of squared differences).
    Euclidean = 1,
    /// Squared L2 (sum of squared differences).
    SquaredEuclidean = 2,
    /// Chebyshev / L∞ (maximum absolute difference).
    Max = 3,
    /// Minimum absolute difference across dimensions.
    Min = 4,
    /// Kullback–Leibler divergence (requires non-negative inputs).
    KullbackLeibler = 5,
    /// Cosine dissimilarity (`1 − cos θ`).
    Cosine = 6,
    /// Jensen–Shannon divergence.
    JensenShannon = 7,
}

// -------------------------------------------------------------------------------------------------

/// Multidimensional scaling (MDS) dimensionality reduction.
///
/// Projects a row-major dataset into a lower-dimensional space by preserving
/// pairwise distances as well as possible. MDS is useful for visualisation
/// and for feeding distance-based representations into downstream algorithms.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{Mds, MdsDistance, Matrix};
///
/// let data = Matrix::from_vec(vec![
///     1.0, 0.0,
///     0.0, 1.0,
///     -1.0, 0.0,
///     0.0, -1.0,
/// ], 4, 2).unwrap();
///
/// let mut mds = Mds::new().unwrap();
/// let out = mds.project(&data, 2, MdsDistance::Euclidean).unwrap();
/// assert_eq!(out.rows(), 4);
/// assert_eq!(out.cols(), 2);
/// ```
///
/// See <https://learn.flucoma.org/reference/mds>
pub struct Mds {
    inner: *mut u8,
}

unsafe impl Send for Mds {}

impl Mds {
    /// Create a new MDS instance.
    ///
    /// # Errors
    /// Returns an error if the underlying C++ allocation fails.
    pub fn new() -> Result<Self, &'static str> {
        let inner = mds_create();
        if inner.is_null() {
            return Err("failed to create MDS instance");
        }
        Ok(Self { inner })
    }

    /// Project a row-major dataset into a lower-dimensional space.
    ///
    /// Returns a matrix of shape `data.rows() × target_dims`.
    ///
    /// # Arguments
    /// * `data` — row-major input matrix.
    /// * `target_dims` — output dimensionality (must be in `[1, data.rows()]`).
    /// * `distance` — distance metric used to build the pairwise distance matrix.
    ///
    /// # Errors
    /// Returns an error if `target_dims` is out of range.
    pub fn project(
        &mut self,
        data: &Matrix,
        target_dims: usize,
        distance: MdsDistance,
    ) -> Result<Matrix, &'static str> {
        let rows = data.rows();
        let cols = data.cols();

        if target_dims == 0 {
            return Err("target_dims must be > 0");
        }
        if target_dims > rows {
            return Err("target_dims must be <= rows");
        }

        let mut out = vec![0.0; rows * target_dims];
        mds_process(
            self.inner,
            data.data().as_ptr(),
            rows as FlucomaIndex,
            cols as FlucomaIndex,
            out.as_mut_ptr(),
            target_dims as FlucomaIndex,
            distance as FlucomaIndex,
        );
        Ok(Matrix::from_vec(out, rows, target_dims).unwrap())
    }
}

impl Drop for Mds {
    fn drop(&mut self) {
        mds_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mds_projection_shape_and_finite_values() {
        use crate::matrix::Matrix;
        let data = Matrix::from_vec(
            vec![
                0.0, 0.0, //
                0.0, 1.0, //
                1.0, 0.0, //
                1.0, 1.0,
            ],
            4,
            2,
        )
        .unwrap();
        let mut mds = Mds::new().unwrap();
        let out = mds.project(&data, 2, MdsDistance::Euclidean).unwrap();
        assert_eq!(out.rows(), 4);
        assert_eq!(out.cols(), 2);
        assert!(out.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mds_rejects_invalid_target_dims() {
        use crate::matrix::Matrix;
        let data = Matrix::from_vec(vec![0.0, 0.0, 1.0, 1.0], 2, 2).unwrap();
        let mut mds = Mds::new().unwrap();
        let err = mds.project(&data, 3, MdsDistance::Euclidean).unwrap_err();
        assert_eq!(err, "target_dims must be <= rows");
    }
}
