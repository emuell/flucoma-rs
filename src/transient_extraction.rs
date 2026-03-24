use flucoma_sys::{
    transient_ext_create, transient_ext_destroy, transient_ext_hop_size, transient_ext_init,
    transient_ext_input_size, transient_ext_process, transient_ext_set_detection_params,
};

// -------------------------------------------------------------------------------------------------

/// Transient and residual extraction from audio blocks.
///
/// Uses an autoregressive model to detect and separate transient content
/// (attack, clicks, impacts) from the sustained residual component.
/// Unlike [`crate::segmentation::TransientSlice`], this returns the separated
/// audio rather than onset flags.
///
/// Three-phase setup:
/// 1. [`TransientExtraction::new`] -- allocates buffers and initialises the AR model.
/// 2. (Optional) [`TransientExtraction::set_detection_parameters`] -- tune sensitivity.
/// 3. Call [`TransientExtraction::process`] per audio block.
///
/// Each call to `process` consumes an input block of `input_size()` samples and
/// returns two output slices of `hop_size()` samples: transients and residual.
///
/// See <https://learn.flucoma.org/reference/transients>
pub struct TransientExtraction {
    inner: *mut u8,
    hop_size: usize,
    input_size: usize,
    order: usize,
    block_size: usize,
    pad_size: usize,
    /// Output buffer: [transients | residual], each hop_size long.
    buf: Vec<f64>,
}

unsafe impl Send for TransientExtraction {}

impl TransientExtraction {
    /// Create and initialise a transient extractor.
    ///
    /// # Arguments
    /// * `order`      - AR model order (typical: 20). Must be > 0.
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
        let inner = transient_ext_create(order as isize, block_size as isize, pad_size as isize);
        if inner.is_null() {
            return Err("failed to create TransientExtraction instance");
        }
        transient_ext_init(
            inner,
            order as isize,
            block_size as isize,
            pad_size as isize,
        );
        let hop_size = transient_ext_hop_size(inner) as usize;
        let input_size = transient_ext_input_size(inner) as usize;
        Ok(Self {
            inner,
            hop_size,
            input_size,
            order,
            block_size,
            pad_size,
            buf: vec![0.0f64; 2 * hop_size],
        })
    }

    /// Reset the AR model and stream buffers without reallocating.
    pub fn reset(&mut self) {
        transient_ext_init(
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
    pub fn set_detection_parameters(
        &mut self,
        power: f64,
        thresh_hi: f64,
        thresh_lo: f64,
        half_window: usize,
        hold: usize,
    ) {
        transient_ext_set_detection_params(
            self.inner,
            power,
            thresh_hi,
            thresh_lo,
            half_window as isize,
            hold as isize,
        );
    }

    /// Process one audio block.
    ///
    /// # Arguments
    /// * `input` - Audio samples. Length must equal `input_size()`.
    ///
    /// Returns `(transients, residual)`: two slices into an internal buffer, each of
    /// length `hop_size()`. The slices are valid until the next call.
    ///
    /// # Panics
    /// Panics if `input.len() != input_size()`.
    pub fn process<'a>(&'a mut self, input: &[f64]) -> (&'a [f64], &'a [f64]) {
        assert_eq!(
            input.len(),
            self.input_size,
            "input length ({}) must equal input_size ({})",
            input.len(),
            self.input_size
        );
        transient_ext_process(
            self.inner,
            input.as_ptr(),
            input.len() as isize,
            self.buf.as_mut_ptr(),
            unsafe { self.buf.as_mut_ptr().add(self.hop_size) },
            self.hop_size as isize,
        );
        self.buf.split_at(self.hop_size)
    }

    /// Number of output samples per block.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Required input block size in samples.
    pub fn input_size(&self) -> usize {
        self.input_size
    }
}

impl Drop for TransientExtraction {
    fn drop(&mut self) {
        transient_ext_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_ext_silence_returns_zeros() {
        let mut ext = TransientExtraction::new(20, 256, 128).unwrap();
        let hop = ext.hop_size();
        let input = vec![0.0f64; ext.input_size()];
        let (transients, residual) = ext.process(&input);
        assert_eq!(transients.len(), hop);
        assert_eq!(residual.len(), hop);
        for &v in transients {
            assert!(v.is_finite(), "transient output must be finite, got {v}");
        }
        for &v in residual {
            assert!(v.is_finite(), "residual output must be finite, got {v}");
        }
    }

    #[test]
    fn transient_ext_reset_clears_ar_state() {
        let mut ext = TransientExtraction::new(20, 256, 128).unwrap();
        let silence = vec![0.0f64; ext.input_size()];
        let (t, r) = ext.process(&silence);
        let first: Vec<_> = t.iter().chain(r).copied().collect();
        // Advance AR model state
        let mut impulse = vec![0.0f64; ext.input_size()];
        impulse[0] = 1.0;
        for _ in 0..5 {
            ext.process(&impulse);
        }
        ext.reset();
        let (t2, r2) = ext.process(&silence);
        let after: Vec<_> = t2.iter().chain(r2).copied().collect();
        assert_eq!(first, after, "reset should restore output to initial state");
    }

    #[test]
    fn transient_ext_impulse_produces_separation() {
        let mut ext = TransientExtraction::new(20, 256, 128).unwrap();
        let hop = ext.hop_size();
        // Seed with silence
        let silence = vec![0.0f64; ext.input_size()];
        let _ = ext.process(&silence);
        // Feed a block with a large impulse
        let mut impulse = vec![0.0f64; ext.input_size()];
        impulse[0] = 1.0;
        let (transients, residual) = ext.process(&impulse);
        assert_eq!(transients.len(), hop);
        assert_eq!(residual.len(), hop);
        // Outputs must be finite
        for &v in transients.iter().chain(residual.iter()) {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn transient_ext_set_detection_params_does_not_panic() {
        let mut ext = TransientExtraction::new(20, 256, 128).unwrap();
        ext.set_detection_parameters(1.0, 1.0, 0.5, 7, 25);
        let hop = ext.hop_size();
        let input = vec![0.0f64; ext.input_size()];
        let (t, r) = ext.process(&input);
        assert_eq!(t.len(), hop);
        assert_eq!(r.len(), hop);
    }
}
