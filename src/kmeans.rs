use flucoma_sys::{
    kmeans_create, kmeans_destroy, kmeans_fit, skmeans_create, skmeans_destroy, skmeans_encode,
    skmeans_fit, FlucomaIndex,
};

use crate::matrix::Matrix;

// -------------------------------------------------------------------------------------------------

/// Centroid initialisation strategy for [`KMeans`] and [`SKMeans`].
#[derive(Debug, Clone, Copy)]
#[repr(isize)]
pub enum KMeansInit {
    /// Assign each point to a random cluster, then compute centroids.
    RandomPartition = 0,
    /// Choose k random data points as initial centroids.
    RandomPoint = 1,
    /// KMeans++ probabilistic seeding for better convergence.
    RandomSampling = 2,
}

// -------------------------------------------------------------------------------------------------

/// Configuration passed to [`KMeans::fit`] and [`SKMeans::fit`].
#[derive(Debug, Clone, Copy)]
pub struct KMeansConfig {
    /// Number of clusters (must be > 0 and ≤ the number of data rows).
    pub k: usize,
    /// Maximum number of Lloyd iterations (must be > 0).
    pub max_iter: usize,
    /// Centroid initialisation strategy.
    pub init: KMeansInit,
    /// Random seed; use `-1` for a non-deterministic run.
    pub seed: isize,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 8,
            max_iter: 64,
            init: KMeansInit::RandomPoint,
            seed: -1,
        }
    }
}

impl KMeansConfig {
    fn validate_input(&self, data: &Matrix) -> Result<(), &'static str> {
        if self.k == 0 {
            return Err("k must be > 0");
        }
        if self.k > data.rows() {
            return Err("k must be <= rows");
        }
        if self.max_iter == 0 {
            return Err("max_iter must be > 0");
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------------

/// Result returned by [`KMeans::fit`] and [`SKMeans::fit`].
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Centroid matrix, shape `k × dims`.
    pub means: Matrix,
    /// Cluster index for each input row, length `rows`.
    pub assignments: Vec<usize>,
    /// Number of clusters.
    pub k: usize,
    /// Feature dimension.
    pub dims: usize,
}

// -------------------------------------------------------------------------------------------------

/// Hard K-Means clustering.
///
/// Partitions a row-major dataset into `k` clusters using Lloyd's algorithm.
/// Each call to [`fit`](KMeans::fit) re-fits the model from scratch.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{KMeans, KMeansConfig, Matrix};
///
/// let data = Matrix::from_vec(
///     vec![0.0, 0.0,  0.1, 0.0,  10.0, 10.0,  10.1, 10.0],
///     4, 2,
/// ).unwrap();
///
/// let mut km = KMeans::new().unwrap();
/// let result = km.fit(&data, KMeansConfig { k: 2, ..Default::default() }).unwrap();
/// println!("assignments: {:?}", result.assignments);
/// ```
///
/// See <https://learn.flucoma.org/reference/kmeans>
pub struct KMeans {
    inner: *mut u8,
}

unsafe impl Send for KMeans {}

impl KMeans {
    /// Create a new KMeans instance.
    ///
    /// # Errors
    /// Returns an error if the underlying C++ allocation fails.
    pub fn new() -> Result<Self, &'static str> {
        let inner = kmeans_create();
        if inner.is_null() {
            return Err("failed to create KMeans instance");
        }
        Ok(Self { inner })
    }

    /// Fit the K-Means model on a row-major dataset.
    ///
    /// # Arguments
    /// * `data` — input matrix; rows = number of data points, cols = feature dimension.
    /// * `config` — clustering configuration.
    ///
    /// # Errors
    /// Returns an error if `data` dimensions are invalid or `config` is invalid.
    pub fn fit(
        &mut self,
        data: &Matrix,
        config: KMeansConfig,
    ) -> Result<KMeansResult, &'static str> {
        config.validate_input(data)?;
        let rows = data.rows();
        let dims = data.cols();
        let k = config.k;
        let mut means = vec![0.0; k * dims];
        let mut assignments = vec![0isize; rows];
        kmeans_fit(
            self.inner,
            data.data().as_ptr(),
            rows as FlucomaIndex,
            dims as FlucomaIndex,
            k as FlucomaIndex,
            config.max_iter as FlucomaIndex,
            config.init as FlucomaIndex,
            config.seed as FlucomaIndex,
            means.as_mut_ptr(),
            assignments.as_mut_ptr(),
        );
        Ok(KMeansResult {
            means: Matrix::from_vec(means, k, dims).unwrap(),
            assignments: assignments.into_iter().map(|x| x as usize).collect(),
            k,
            dims,
        })
    }
}

impl Drop for KMeans {
    fn drop(&mut self) {
        kmeans_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

/// Soft (spherical) K-Means with soft assignment encoding.
///
/// Like [`KMeans`] but can also encode new data as soft cluster membership
/// weights via [`encode`](SKMeans::encode).
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{SKMeans, KMeansConfig, Matrix};
///
/// let data = Matrix::from_vec(vec![1.0, 0.0,  0.0, 1.0,  -1.0, 0.0], 3, 2).unwrap();
/// let mut sk = SKMeans::new().unwrap();
/// let result = sk.fit(&data, KMeansConfig { k: 2, ..Default::default() }).unwrap();
/// let weights = sk.encode(&data, 0.5).unwrap();
/// ```
///
/// See <https://learn.flucoma.org/reference/kmeans>
pub struct SKMeans {
    inner: *mut u8,
    k: usize,
}

unsafe impl Send for SKMeans {}

impl SKMeans {
    /// Create a new SKMeans instance.
    ///
    /// # Errors
    /// Returns an error if the underlying C++ allocation fails.
    pub fn new() -> Result<Self, &'static str> {
        let inner = skmeans_create();
        if inner.is_null() {
            return Err("failed to create SKMeans instance");
        }
        Ok(Self { inner, k: 0 })
    }

    /// Fit the SKMeans model on a row-major dataset.
    ///
    /// # Arguments
    /// * `data` — input matrix; rows = number of data points, cols = feature dimension.
    /// * `config` — clustering configuration.
    ///
    /// # Errors
    /// Returns an error if `data` dimensions are invalid or `config` is invalid.
    pub fn fit(
        &mut self,
        data: &Matrix,
        config: KMeansConfig,
    ) -> Result<KMeansResult, &'static str> {
        config.validate_input(data)?;
        let rows = data.rows();
        let dims = data.cols();
        let k = config.k;
        let mut means = vec![0.0; k * dims];
        let mut assignments = vec![0isize; rows];
        skmeans_fit(
            self.inner,
            data.data().as_ptr(),
            rows as FlucomaIndex,
            dims as FlucomaIndex,
            k as FlucomaIndex,
            config.max_iter as FlucomaIndex,
            config.init as FlucomaIndex,
            config.seed as FlucomaIndex,
            means.as_mut_ptr(),
            assignments.as_mut_ptr(),
        );
        self.k = k;
        Ok(KMeansResult {
            means: Matrix::from_vec(means, k, dims).unwrap(),
            assignments: assignments.into_iter().map(|x| x as usize).collect(),
            k,
            dims,
        })
    }

    /// Encode data as soft cluster membership weights.
    ///
    /// Returns a matrix of shape `rows × k`, where each row sums to 1.0 and
    /// represents the soft assignment of the corresponding input point to each centroid.
    ///
    /// # Arguments
    /// * `data` — input matrix; rows = number of data points, cols = feature dimension.
    /// * `alpha` — softmax temperature; lower values produce softer assignments.
    ///
    /// # Errors
    /// Returns an error if the model has not been fitted or dimensions are invalid.
    pub fn encode(&self, data: &Matrix, alpha: f64) -> Result<Matrix, &'static str> {
        if self.k == 0 {
            return Err("SKMeans is not fitted");
        }
        let rows = data.rows();
        let dims = data.cols();
        let mut out = vec![0.0; rows * self.k];
        skmeans_encode(
            self.inner,
            data.data().as_ptr(),
            rows as FlucomaIndex,
            dims as FlucomaIndex,
            alpha,
            out.as_mut_ptr(),
            self.k as FlucomaIndex,
        );
        Ok(Matrix::from_vec(out, rows, self.k).unwrap())
    }
}

impl Drop for SKMeans {
    fn drop(&mut self) {
        skmeans_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_fit_basic() {
        let data = Matrix::from_vec(
            vec![
                0.0, 0.0, 0.1, 0.0, -0.1, 0.0, //
                10.0, 10.0, 10.1, 10.0, 9.9, 10.0,
            ],
            6,
            2,
        )
        .unwrap();
        let mut km = KMeans::new().unwrap();
        let cfg = KMeansConfig {
            k: 2,
            max_iter: 64,
            init: KMeansInit::RandomPoint,
            seed: 1234,
        };
        let res = km.fit(&data, cfg).unwrap();
        assert_eq!(res.means.rows(), 2);
        assert_eq!(res.means.cols(), 2);
        assert_eq!(res.assignments.len(), 6);
        assert!(res.assignments.iter().all(|&a| a < 2));
    }

    #[test]
    fn skmeans_fit_and_encode() {
        let data = Matrix::from_vec(
            vec![
                1.0, 0.0, 0.9, 0.1, 0.0, 1.0, //
                -1.0, 0.0, -0.9, -0.1, 0.0, -1.0,
            ],
            6,
            2,
        )
        .unwrap();
        let mut sk = SKMeans::new().unwrap();
        let cfg = KMeansConfig {
            k: 2,
            max_iter: 64,
            init: KMeansInit::RandomPoint,
            seed: 1234,
        };
        let res = sk.fit(&data, cfg).unwrap();
        assert_eq!(res.assignments.len(), 6);
        let enc = sk.encode(&data, 0.25).unwrap();
        assert_eq!(enc.rows(), 6);
        assert_eq!(enc.cols(), 2);
    }
}
