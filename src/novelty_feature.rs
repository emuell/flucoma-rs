use flucoma_sys::{
    novelty_feature_create, novelty_feature_destroy, novelty_feature_init,
    novelty_feature_process_frame,
};

// -------------------------------------------------------------------------------------------------

/// Novelty-curve feature extractor for feature streams.
///
/// Two-phase setup:
/// 1. [`Novelty::new`] -- allocates buffers and initialises the extractor.
/// 2. Call [`Novelty::process_frame`] per feature frame.
///
/// Each call to `process_frame` takes a feature vector of length `n_dims` (e.g.
/// mel bands or MFCCs) and returns the smoothed novelty value for that frame.
///
/// See <https://learn.flucoma.org/reference/noveltyfeature>
pub struct Novelty {
    inner: *mut u8,
    n_dims: usize,
    kernel_size: usize,
    filter_size: usize,
}

unsafe impl Send for Novelty {}

impl Novelty {
    /// Create and initialise a novelty feature extractor.
    ///
    /// # Arguments
    /// * `kernel_size`  - Size of the checkerboard kernel (must be odd, >= 1).
    /// * `n_dims`       - Dimensionality of the input feature vectors.
    /// * `filter_size`  - Median filter size applied to the novelty curve (must be odd, >= 1).
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

        let inner =
            novelty_feature_create(kernel_size as isize, n_dims as isize, filter_size as isize);
        if inner.is_null() {
            return Err("failed to create Novelty instance");
        }
        novelty_feature_init(
            inner,
            kernel_size as isize,
            filter_size as isize,
            n_dims as isize,
        );
        Ok(Self {
            inner,
            n_dims,
            kernel_size,
            filter_size,
        })
    }

    /// Reset internal history and median filter without reallocating.
    pub fn reset(&mut self) {
        novelty_feature_init(
            self.inner,
            self.kernel_size as isize,
            self.filter_size as isize,
            self.n_dims as isize,
        );
    }

    /// Process one feature frame and return the novelty value.
    ///
    /// # Arguments
    /// * `input` - Feature vector. Length must equal `n_dims`.
    ///
    /// Returns the smoothed self-similarity novelty value for this frame.
    ///
    /// # Panics
    /// Panics if `input.len() != n_dims`.
    pub fn process_frame(&mut self, input: &[f64]) -> f64 {
        assert_eq!(
            input.len(),
            self.n_dims,
            "input length ({}) must equal n_dims ({})",
            input.len(),
            self.n_dims
        );
        novelty_feature_process_frame(self.inner, input.as_ptr(), input.len() as isize)
    }

    /// Dimensionality of the input feature vectors.
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }
}

impl Drop for Novelty {
    fn drop(&mut self) {
        novelty_feature_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn novelty_feature_zero_input_returns_value() {
        let mut nf = Novelty::new(3, 13, 1).unwrap();
        let frame = vec![0.0f64; 13];
        let val = nf.process_frame(&frame);
        assert!(
            val.is_finite(),
            "expected finite value for zero input, got {val}"
        );
    }

    #[test]
    fn novelty_feature_reset_clears_history() {
        const N_DIMS: usize = 13;
        let mut nf = Novelty::new(3, N_DIMS, 1).unwrap();
        let frame = vec![0.0f64; N_DIMS];
        let first = nf.process_frame(&frame);
        // Advance state with varying input
        let loud: Vec<f64> = (0..N_DIMS).map(|i| i as f64 * 0.1).collect();
        for _ in 0..10 {
            nf.process_frame(&loud);
        }
        nf.reset();
        let after = nf.process_frame(&frame);
        assert_eq!(first, after, "reset should restore output to initial state");
    }

    #[test]
    fn novelty_feature_changing_input_varies() {
        const N_DIMS: usize = 13;
        let mut nf = Novelty::new(3, N_DIMS, 1).unwrap();
        let silence = vec![0.0f64; N_DIMS];
        let loud: Vec<f64> = (0..N_DIMS).map(|i| (i as f64) * 0.1).collect();
        let mut values = Vec::new();
        for i in 0..20 {
            let frame = if i % 4 < 2 { &silence } else { &loud };
            values.push(nf.process_frame(frame));
        }
        let all_same = values.windows(2).all(|w| w[0] == w[1]);
        assert!(
            !all_same,
            "alternating input should produce varying novelty values"
        );
    }
}
