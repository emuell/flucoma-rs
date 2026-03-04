use flucoma_sys::{
    novelty_seg_create, novelty_seg_destroy, novelty_seg_init, novelty_seg_process_frame,
};

// -------------------------------------------------------------------------------------------------

/// Novelty-curve segmenter for feature streams.
///
/// Two-phase setup:
/// 1. [`NoveltySegmentation::new`] -- allocates buffers and initialises the detector.
/// 2. Call [`NoveltySegmentation::process_frame`] per feature frame.
///
/// Each call to `process_frame` takes a feature vector of length `n_dims` (e.g.
/// mel bands or MFCCs). The algorithm computes a self-similarity novelty curve
/// internally and returns 1.0 when a peak above `threshold` is detected.
pub struct NoveltySegmentation {
    inner: *mut u8,
    n_dims: usize,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for NoveltySegmentation {}

impl NoveltySegmentation {
    /// Create and initialise a novelty segmenter.
    ///
    /// # Arguments
    /// * `kernel_size`  - Size of the checkerboard kernel (must be odd, >= 1).
    /// * `n_dims`       - Dimensionality of the input feature vectors.
    /// * `filter_size`  - Median filter size applied to the novelty curve (must be odd, >= 1)
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(
        kernel_size: usize,
        n_dims: usize,
        filter_size: usize,
    ) -> Result<Self, &'static str> {
        if kernel_size == 0 || kernel_size.is_multiple_of(2) {
            return Err("kernel_size must be odd and > 0");
        }
        if n_dims == 0 {
            return Err("n_dims must be > 0");
        }
        if filter_size == 0 || filter_size.is_multiple_of(2) {
            return Err("filter_size must be odd and > 0");
        }

        let inner = novelty_seg_create(kernel_size as isize, n_dims as isize, filter_size as isize);
        if inner.is_null() {
            return Err("failed to create NoveltySegmentation instance");
        }
        novelty_seg_init(
            inner,
            kernel_size as isize,
            filter_size as isize,
            n_dims as isize,
        );
        Ok(Self { inner, n_dims })
    }

    /// Process one feature frame.
    ///
    /// # Arguments
    /// * `input`           - Feature vector. Length must equal `n_dims`.
    /// * `threshold`       - Novelty value above which a slice point is declared.
    /// * `min_slice_length`- Minimum frames between successive slice points.
    ///
    /// Returns 1.0 at novelty slice points, 0.0 otherwise.
    ///
    /// # Panics
    /// Panics if `input.len() != n_dims`.
    pub fn process_frame(&mut self, input: &[f64], threshold: f64, min_slice_length: usize) -> f64 {
        assert_eq!(
            input.len(),
            self.n_dims,
            "input length ({}) must equal n_dims ({})",
            input.len(),
            self.n_dims
        );
        novelty_seg_process_frame(
            self.inner,
            input.as_ptr(),
            input.len() as isize,
            threshold,
            min_slice_length as isize,
        )
    }

    /// Dimensionality of the input feature vectors.
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }
}

impl Drop for NoveltySegmentation {
    fn drop(&mut self) {
        novelty_seg_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn novelty_seg_zero_input_returns_zero_or_one() {
        let mut seg = NoveltySegmentation::new(3, 13, 1).unwrap();
        let frame = vec![0.0f64; 13];
        let val = seg.process_frame(&frame, 0.5, 2);
        assert!(val == 0.0 || val == 1.0, "expected 0.0 or 1.0, got {val}");
    }

    #[test]
    fn novelty_seg_changing_input_can_trigger() {
        const N_DIMS: usize = 13;
        let mut seg = NoveltySegmentation::new(3, N_DIMS, 7).unwrap();
        let silence = vec![0.0f64; N_DIMS];
        let loud: Vec<f64> = (0..N_DIMS).map(|i| (i as f64) * 0.1).collect();
        let mut triggered = false;
        for i in 0..100 {
            let frame = if i % 20 < 10 { &silence } else { &loud };
            let val = seg.process_frame(frame, 0.01, 1);
            if val == 1.0 {
                triggered = true;
                break;
            }
        }
        assert!(
            triggered,
            "alternating signal should trigger at least one novelty slice"
        );
    }
}
