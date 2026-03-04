use flucoma_sys::{
    kmeans_create, kmeans_destroy, kmeans_fit, skmeans_create, skmeans_destroy, skmeans_encode,
    skmeans_fit, FlucomaIndex,
};

#[derive(Debug, Clone, Copy)]
#[repr(isize)]
pub enum KMeansInit {
    RandomPartition = 0,
    RandomPoint = 1,
    RandomSampling = 2,
}

#[derive(Debug, Clone, Copy)]
pub struct KMeansConfig {
    pub k: usize,
    pub max_iter: usize,
    pub init: KMeansInit,
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

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub means: Vec<f64>,
    pub assignments: Vec<usize>,
    pub k: usize,
    pub dims: usize,
}

pub struct KMeans {
    inner: *mut u8,
}

pub struct SKMeans {
    inner: *mut u8,
    k: usize,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for KMeans {}
// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for SKMeans {}

impl KMeans {
    pub fn new() -> Result<Self, &'static str> {
        let inner = kmeans_create();
        if inner.is_null() {
            return Err("failed to create KMeans instance");
        }
        Ok(Self { inner })
    }

    pub fn fit(
        &mut self,
        data: &[f64],
        rows: usize,
        dims: usize,
        config: KMeansConfig,
    ) -> Result<KMeansResult, &'static str> {
        validate_kmeans_input(data, rows, dims, config)?;
        let k = config.k;
        let mut means = vec![0.0; k * dims];
        let mut assignments = vec![0isize; rows];
        kmeans_fit(
            self.inner,
            data.as_ptr(),
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
            means,
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

impl SKMeans {
    pub fn new() -> Result<Self, &'static str> {
        let inner = skmeans_create();
        if inner.is_null() {
            return Err("failed to create SKMeans instance");
        }
        Ok(Self { inner, k: 0 })
    }

    pub fn fit(
        &mut self,
        data: &[f64],
        rows: usize,
        dims: usize,
        config: KMeansConfig,
    ) -> Result<KMeansResult, &'static str> {
        validate_kmeans_input(data, rows, dims, config)?;
        let k = config.k;
        let mut means = vec![0.0; k * dims];
        let mut assignments = vec![0isize; rows];
        skmeans_fit(
            self.inner,
            data.as_ptr(),
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
            means,
            assignments: assignments.into_iter().map(|x| x as usize).collect(),
            k,
            dims,
        })
    }

    pub fn encode(
        &self,
        data: &[f64],
        rows: usize,
        dims: usize,
        alpha: f64,
    ) -> Result<Vec<f64>, &'static str> {
        if self.k == 0 {
            return Err("SKMeans is not fitted");
        }
        if rows == 0 || dims == 0 {
            return Err("rows and dims must be > 0");
        }
        if data.len() != rows * dims {
            return Err("data length does not match rows * dims");
        }
        let mut out = vec![0.0; rows * self.k];
        skmeans_encode(
            self.inner,
            data.as_ptr(),
            rows as FlucomaIndex,
            dims as FlucomaIndex,
            alpha,
            out.as_mut_ptr(),
            self.k as FlucomaIndex,
        );
        Ok(out)
    }
}

impl Drop for SKMeans {
    fn drop(&mut self) {
        skmeans_destroy(self.inner);
    }
}

fn validate_kmeans_input(
    data: &[f64],
    rows: usize,
    dims: usize,
    config: KMeansConfig,
) -> Result<(), &'static str> {
    if rows == 0 || dims == 0 {
        return Err("rows and dims must be > 0");
    }
    if data.len() != rows * dims {
        return Err("data length does not match rows * dims");
    }
    if config.k == 0 {
        return Err("k must be > 0");
    }
    if config.k > rows {
        return Err("k must be <= rows");
    }
    if config.max_iter == 0 {
        return Err("max_iter must be > 0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_fit_basic() {
        let data = vec![
            0.0, 0.0, 0.1, 0.0, -0.1, 0.0, //
            10.0, 10.0, 10.1, 10.0, 9.9, 10.0,
        ];
        let mut km = KMeans::new().unwrap();
        let cfg = KMeansConfig {
            k: 2,
            max_iter: 64,
            init: KMeansInit::RandomPoint,
            seed: 1234,
        };
        let res = km.fit(&data, 6, 2, cfg).unwrap();
        assert_eq!(res.means.len(), 4);
        assert_eq!(res.assignments.len(), 6);
        assert!(res.assignments.iter().all(|&a| a < 2));
    }

    #[test]
    fn skmeans_fit_and_encode() {
        let data = vec![
            1.0, 0.0, 0.9, 0.1, 0.0, 1.0, //
            -1.0, 0.0, -0.9, -0.1, 0.0, -1.0,
        ];
        let mut sk = SKMeans::new().unwrap();
        let cfg = KMeansConfig {
            k: 2,
            max_iter: 64,
            init: KMeansInit::RandomPoint,
            seed: 1234,
        };
        let res = sk.fit(&data, 6, 2, cfg).unwrap();
        assert_eq!(res.assignments.len(), 6);
        let enc = sk.encode(&data, 6, 2, 0.25).unwrap();
        assert_eq!(enc.len(), 12);
    }
}
