use flucoma_sys::{mds_create, mds_destroy, mds_process, FlucomaIndex};

#[derive(Debug, Clone, Copy)]
#[repr(isize)]
pub enum MdsDistance {
    Manhattan = 0,
    Euclidean = 1,
    SquaredEuclidean = 2,
    Max = 3,
    Min = 4,
    KullbackLeibler = 5,
    Cosine = 6,
    JensenShannon = 7,
}

/// Multidimensional scaling projection for row-major datasets.
pub struct Mds {
    inner: *mut u8,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for Mds {}

impl Mds {
    pub fn new() -> Result<Self, &'static str> {
        let inner = mds_create();
        if inner.is_null() {
            return Err("failed to create MDS instance");
        }
        Ok(Self { inner })
    }

    pub fn project(
        &mut self,
        data: &[f64],
        rows: usize,
        cols: usize,
        target_dims: usize,
        distance: MdsDistance,
    ) -> Result<Vec<f64>, &'static str> {
        if rows == 0 || cols == 0 {
            return Err("rows and cols must be > 0");
        }
        if data.len() != rows * cols {
            return Err("data length does not match rows * cols");
        }
        if target_dims == 0 {
            return Err("target_dims must be > 0");
        }
        if target_dims > rows {
            return Err("target_dims must be <= rows");
        }

        let mut out = vec![0.0; rows * target_dims];
        mds_process(
            self.inner,
            data.as_ptr(),
            rows as FlucomaIndex,
            cols as FlucomaIndex,
            out.as_mut_ptr(),
            target_dims as FlucomaIndex,
            distance as FlucomaIndex,
        );
        Ok(out)
    }
}

impl Drop for Mds {
    fn drop(&mut self) {
        mds_destroy(self.inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mds_projection_shape_and_finite_values() {
        let data = vec![
            0.0, 0.0, //
            0.0, 1.0, //
            1.0, 0.0, //
            1.0, 1.0,
        ];
        let mut mds = Mds::new().unwrap();
        let out = mds
            .project(&data, 4, 2, 2, MdsDistance::Euclidean)
            .unwrap();
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mds_rejects_invalid_target_dims() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let mut mds = Mds::new().unwrap();
        let err = mds
            .project(&data, 2, 2, 3, MdsDistance::Euclidean)
            .unwrap_err();
        assert_eq!(err, "target_dims must be <= rows");
    }
}
