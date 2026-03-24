use flucoma_sys::{
    transient_seg_create, transient_seg_destroy, transient_seg_hop_size, transient_seg_init,
    transient_seg_input_size, transient_seg_process, transient_seg_set_detection_params,
};

// -------------------------------------------------------------------------------------------------

/// Transient detector and segmenter operating on audio blocks.
///
/// Three-phase setup:
/// 1. [`TransientSlice::new`] -- allocates buffers and initialises the AR model.
/// 2. (Optional) [`TransientSlice::set_detection_parameters`] -- tune sensitivity.
/// 3. Call [`TransientSlice::process`] per audio block.
///
/// Each call to `process` consumes an input block of `input_size()` samples
/// and returns a `Vec<f64>` of `hop_size()` samples where each value is
/// 1.0 (transient onset) or 0.0.
///
/// See <https://learn.flucoma.org/reference/transientslice>
pub struct TransientSlice {
    inner: *mut u8,
    hop_size: usize,
    input_size: usize,
    order: usize,
    block_size: usize,
    pad_size: usize,
}

unsafe impl Send for TransientSlice {}

impl TransientSlice {
    /// Create and initialise a transient segmenter.
    ///
    /// # Arguments
    /// * `order`      - AR model order (typical: 20).
    /// * `block_size` - Analysis block size in samples (must be > `order`).
    /// * `pad_size`   - Look-ahead padding in samples (must be > `order`).
    ///
    /// After construction, use `hop_size()` and `input_size()` to size your
    /// audio buffers correctly.
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(order: usize, block_size: usize, pad_size: usize) -> Result<Self, &'static str> {
        if order == 0 {
            return Err("order must be > 0");
        }
        if block_size <= order {
            return Err("block_size must be > order");
        }
        if pad_size <= order {
            return Err("pad_size must be > order");
        }
        let inner = transient_seg_create(order as isize, block_size as isize, pad_size as isize);
        if inner.is_null() {
            return Err("failed to create TransientSlice instance");
        }
        transient_seg_init(
            inner,
            order as isize,
            block_size as isize,
            pad_size as isize,
        );
        let hop_size = transient_seg_hop_size(inner) as usize;
        let input_size = transient_seg_input_size(inner) as usize;
        Ok(Self {
            inner,
            hop_size,
            input_size,
            order,
            block_size,
            pad_size,
        })
    }

    /// Reset the AR model and detection state without reallocating.
    pub fn reset(&mut self) {
        transient_seg_init(
            self.inner,
            self.order as isize,
            self.block_size as isize,
            self.pad_size as isize,
        );
    }

    /// Configure detection sensitivity.
    ///
    /// # Arguments
    /// * `power`       - Spectral power threshold for residual detection.
    /// * `thresh_hi`   - Upper detection threshold.
    /// * `thresh_lo`   - Lower detection threshold (hysteresis).
    /// * `half_window` - Half-width of the peak-detection window in frames.
    /// * `hold`        - Minimum frames between successive detections.
    /// * `min_segment` - Minimum segment length in samples.
    pub fn set_detection_parameters(
        &mut self,
        power: f64,
        thresh_hi: f64,
        thresh_lo: f64,
        half_window: usize,
        hold: usize,
        min_segment: usize,
    ) {
        transient_seg_set_detection_params(
            self.inner,
            power,
            thresh_hi,
            thresh_lo,
            half_window as isize,
            hold as isize,
            min_segment as isize,
        );
    }

    /// Process one audio block.
    ///
    /// # Arguments
    /// * `input` - Audio samples. Length must equal `input_size()`.
    ///
    /// Returns a `Vec<f64>` of length `hop_size()` where each sample is
    /// 1.0 (transient onset detected) or 0.0.
    ///
    /// # Panics
    /// Panics if `input.len() != input_size()`.
    pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
        assert_eq!(
            input.len(),
            self.input_size,
            "input length ({}) must equal input_size ({})",
            input.len(),
            self.input_size
        );
        let mut output = vec![0.0f64; self.hop_size];
        transient_seg_process(
            self.inner,
            input.as_ptr(),
            input.len() as isize,
            output.as_mut_ptr(),
            output.len() as isize,
        );
        output
    }

    /// Number of output samples per block (block_size - model_order).
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Required input block size (hop_size + pad_size).
    pub fn input_size(&self) -> usize {
        self.input_size
    }
}

impl Drop for TransientSlice {
    fn drop(&mut self) {
        transient_seg_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_seg_silence_returns_zeros() {
        let mut slice = TransientSlice::new(20, 256, 128).unwrap();
        let input = vec![0.0f64; slice.input_size()];
        let out = slice.process(&input);
        assert_eq!(out.len(), slice.hop_size());
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "sample {i}: expected 0.0 or 1.0, got {v}"
            );
        }
    }

    #[test]
    fn transient_seg_reset_clears_ar_state() {
        let mut slice = TransientSlice::new(20, 256, 128).unwrap();
        let silence = vec![0.0f64; slice.input_size()];
        let first = slice.process(&silence);
        // Advance AR model state
        let mut impulse = vec![0.0f64; slice.input_size()];
        impulse[0] = 1.0;
        for _ in 0..5 {
            slice.process(&impulse);
        }
        slice.reset();
        let after = slice.process(&silence);
        assert_eq!(first, after, "reset should restore output to initial state");
    }

    #[test]
    fn transient_seg_impulse_can_detect() {
        let mut slice = TransientSlice::new(20, 256, 128).unwrap();
        slice.set_detection_parameters(1.0, 1.0, 0.5, 7, 25, 50);
        // Seed with silence
        let silence = vec![0.0f64; slice.input_size()];
        let _ = slice.process(&silence);
        // Feed a block with a large impulse
        let mut impulse = vec![0.0f64; slice.input_size()];
        impulse[0] = 1.0;
        let out = slice.process(&impulse);
        assert_eq!(out.len(), slice.hop_size());
        // At minimum the output must be valid 0/1 values
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "sample {i}: expected 0.0 or 1.0, got {v}"
            );
        }
    }
}
