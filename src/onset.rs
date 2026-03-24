use flucoma_sys::{onset_create, onset_destroy, onset_init, onset_process_frame};

// -------------------------------------------------------------------------------------------------

/// Onset detection function selector.
///
/// Each variant computes a different measure of spectral change between frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(isize)]
pub enum OnsetFunction {
    /// Power of the difference between successive spectra.
    #[default]
    PowerSpectrum = 0,
    /// High-frequency content.
    HighFrequency = 1,
    /// Complex domain difference.
    ComplexDomain = 2,
    /// Rectified complex domain difference.
    RectifiedComplex = 3,
    /// Phase deviation.
    PhaseDev = 4,
    /// Weighted phase deviation.
    WeightedPhaseDev = 5,
    /// Modified Kullback-Leibler divergence.
    ModKL = 6,
    /// Itakura-Saito distance.
    ItakuraSaito = 7,
    /// Cosine similarity.
    Cosine = 8,
    /// Normalised power spectrum.
    NormPower = 9,
}

// -------------------------------------------------------------------------------------------------

/// Computes frame-by-frame onset detection values from audio.
///
/// Two-phase setup:
/// 1. [`Onset::new`] -- allocates buffers.
/// 2. Call [`Onset::process_frame`] per frame.
///
/// The algorithm maintains internal frame history for differential functions.
///
/// See <https://learn.flucoma.org/reference/onsetfeature>
pub struct Onset {
    inner: *mut u8,
    window_size: usize,
    fft_size: usize,
    filter_size: usize,
    max_filter_size: usize,
}

unsafe impl Send for Onset {}

impl Onset {
    /// Create and initialise an onset detector.
    ///
    /// # Arguments
    /// * `window_size`  - Analysis window size in samples.
    /// * `fft_size`     - FFT size (must be >= `window_size`).
    /// * `filter_size`  - Median filter size for background subtraction
    ///   (use 0 or 1 to disable, minimum effective value is 3).
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid.
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
        let inner = onset_create(fft_size as isize, max_filter as isize);
        if inner.is_null() {
            return Err("failed to create OnsetDetectionFunctions instance");
        }
        onset_init(
            inner,
            window_size as isize,
            fft_size as isize,
            filter_size as isize,
        );
        Ok(Self {
            inner,
            window_size,
            fft_size,
            filter_size,
            max_filter_size: max_filter,
        })
    }

    /// Reset internal frame history and median filter without reallocating.
    pub fn reset(&mut self) {
        onset_init(
            self.inner,
            self.window_size as isize,
            self.fft_size as isize,
            self.filter_size as isize,
        );
    }

    /// Process one audio frame and return an onset detection value.
    ///
    /// # Arguments
    /// * `input`       - Audio samples. For `frame_delta == 0`, length must be
    ///   at least `window_size`. For `frame_delta > 0`, length
    ///   must be at least `window_size + frame_delta`.
    /// * `function`    - Detection function to compute.
    /// * `filter_size` - Median filter size for this frame (0 to disable).
    /// * `frame_delta` - History offset in samples (0 for most functions).
    ///
    /// Returns a positive value where larger values indicate more likely onsets.
    ///
    /// # Panics
    /// Panics if `input.len() < window_size` (or `< window_size + frame_delta`
    /// when `frame_delta > 0`).
    pub fn process_frame(
        &mut self,
        input: &[f64],
        function: OnsetFunction,
        filter_size: usize,
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
        onset_process_frame(
            self.inner,
            input.as_ptr(),
            input.len() as isize,
            function as isize,
            filter_size as isize,
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

impl Drop for Onset {
    fn drop(&mut self) {
        onset_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onset_silent_frame_returns_value() {
        let mut odf = Onset::new(1024, 1024, 5).unwrap();
        let silence = vec![0.0f64; 1024];
        let val = odf.process_frame(&silence, OnsetFunction::PowerSpectrum, 5, 0);
        // Silence should give a near-zero detection value
        assert!(
            val.abs() < 1.0,
            "expected small value for silence, got {}",
            val
        );
    }

    #[test]
    fn onset_reset_clears_history() {
        let mut odf = Onset::new(1024, 1024, 0).unwrap();
        let silence = vec![0.0f64; 1024];
        let first = odf.process_frame(&silence, OnsetFunction::PowerSpectrum, 0, 0);
        // Advance state
        let mut impulse = vec![0.0f64; 1024];
        impulse[512] = 1.0;
        odf.process_frame(&impulse, OnsetFunction::PowerSpectrum, 0, 0);
        // After reset the first frame should match the original first output
        odf.reset();
        let after = odf.process_frame(&silence, OnsetFunction::PowerSpectrum, 0, 0);
        assert_eq!(first, after, "reset should restore output to initial state");
    }

    #[test]
    fn onset_impulse_gives_larger_value() {
        let mut odf = Onset::new(1024, 1024, 0).unwrap();
        // First frame: silence (seeds history)
        let silence = vec![0.0f64; 1024];
        let _ = odf.process_frame(&silence, OnsetFunction::PowerSpectrum, 0, 0);
        // Second frame: impulse in the middle of the frame (Hann window is
        // non-zero there; w[0] = 0 so an impulse at sample 0 would vanish).
        let mut impulse = vec![0.0f64; 1024];
        impulse[512] = 1.0;
        let val = odf.process_frame(&impulse, OnsetFunction::PowerSpectrum, 0, 0);
        assert!(
            val != 0.0,
            "expected non-zero onset value for impulse, got {}",
            val
        );
    }
}
