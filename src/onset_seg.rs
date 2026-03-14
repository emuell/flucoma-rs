use flucoma_sys::{onset_seg_create, onset_seg_destroy, onset_seg_init, onset_seg_process_frame};

pub use crate::onset::OnsetFunction;

// -------------------------------------------------------------------------------------------------

/// Detects onsets in an audio stream and returns a binary decision per frame.
///
/// Two-phase setup:
/// 1. [`OnsetSlice::new`] -- allocates buffers and initialises the detector.
/// 2. Call [`OnsetSlice::process_frame`] per frame.
///
/// Unlike [`crate::analyzation::Onset`], which returns a continuous
/// detection value, this algorithm applies a threshold and debounce internally
/// and returns 1.0 (onset detected) or 0.0 (no onset).
///
/// See <https://learn.flucoma.org/reference/onsetslice>
pub struct OnsetSlice {
    inner: *mut u8,
    window_size: usize,
    fft_size: usize,
    max_filter_size: usize,
}

unsafe impl Send for OnsetSlice {}

impl OnsetSlice {
    /// Create and initialise an onset segmenter.
    ///
    /// # Arguments
    /// * `window_size`  - Analysis window size in samples.
    /// * `fft_size`     - FFT size (must be >= `window_size`).
    /// * `filter_size`  - Median filter size for background subtraction
    ///   (use 0 or 1 to disable, minimum effective value is 3).
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(
        window_size: usize,
        fft_size: usize,
        filter_size: usize,
    ) -> Result<Self, &'static str> {
        if window_size == 0 {
            return Err("window_size must be > 0");
        }
        if fft_size < window_size {
            return Err("fft_size must be >= window_size");
        }
        let max_filter = filter_size.max(3);
        let inner = onset_seg_create(fft_size as isize, max_filter as isize);
        if inner.is_null() {
            return Err("failed to create OnsetSlice instance");
        }
        onset_seg_init(
            inner,
            window_size as isize,
            fft_size as isize,
            filter_size as isize,
        );
        Ok(Self {
            inner,
            window_size,
            fft_size,
            max_filter_size: max_filter,
        })
    }

    /// Process one audio frame.
    ///
    /// # Arguments
    /// * `input`       - Audio samples. Length must be at least `window_size`
    ///   (or `window_size + frame_delta` when `frame_delta > 0`).
    /// * `function`    - Onset detection function to use.
    /// * `filter_size` - Median filter size for this frame (0 to disable).
    /// * `threshold`   - Detection threshold. Values above trigger an onset.
    /// * `debounce`    - Minimum number of frames between successive onsets.
    /// * `frame_delta` - History offset in samples (0 for most functions).
    ///
    /// Returns 1.0 if an onset is detected this frame, 0.0 otherwise.
    ///
    /// # Panics
    /// Panics if `input.len() < window_size + frame_delta`.
    pub fn process_frame(
        &mut self,
        input: &[f64],
        function: OnsetFunction,
        filter_size: usize,
        threshold: f64,
        debounce: usize,
        frame_delta: usize,
    ) -> f64 {
        let min_len = self.window_size + frame_delta;
        assert!(
            input.len() >= min_len,
            "input length ({}) must be >= window_size + frame_delta ({})",
            input.len(),
            min_len
        );
        assert!(
            filter_size <= self.max_filter_size,
            "filter_size ({}) must be <= max_filter_size ({})",
            filter_size,
            self.max_filter_size
        );
        onset_seg_process_frame(
            self.inner,
            input.as_ptr(),
            input.len() as isize,
            function as isize,
            filter_size as isize,
            threshold,
            debounce as isize,
            frame_delta as isize,
        )
    }

    /// Analysis window size in samples.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// FFT size in samples.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}

impl Drop for OnsetSlice {
    fn drop(&mut self) {
        onset_seg_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onset_seg_silent_frame_returns_zero() {
        let mut slice = OnsetSlice::new(1024, 1024, 5).unwrap();
        let silence = vec![0.0f64; 1024];
        let val = slice.process_frame(&silence, OnsetFunction::PowerSpectrum, 5, 0.5, 0, 0);
        assert_eq!(val, 0.0, "silence should not trigger an onset, got {val}");
    }

    #[test]
    fn onset_seg_impulse_after_silence_triggers() {
        let mut slice = OnsetSlice::new(1024, 1024, 0).unwrap();
        let silence = vec![0.0f64; 1024];
        // Seed the history with silence
        let _ = slice.process_frame(&silence, OnsetFunction::PowerSpectrum, 0, 0.01, 0, 0);
        // Feed a loud impulse (away from sample 0 which has zero Hann weight)
        let mut impulse = vec![0.0f64; 1024];
        impulse[512] = 1.0;
        let val = slice.process_frame(&impulse, OnsetFunction::PowerSpectrum, 0, 0.01, 0, 0);
        assert!(val == 1.0 || val == 0.0, "expected 0.0 or 1.0, got {val}");
    }
}
