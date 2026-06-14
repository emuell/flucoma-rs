use flucoma_sys::{yinfft_create, yinfft_destroy, yinfft_process_frame};

// -------------------------------------------------------------------------------------------------

/// The result of a pitch estimation frame.
#[derive(Debug, Clone, Copy)]
pub struct PitchResult {
    /// Estimated fundamental frequency in Hz (0 when no pitch is detected).
    pub pitch_hz: f64,
    /// Confidence of the pitch estimate, in the range 0–1.
    pub confidence: f64,
}

// -------------------------------------------------------------------------------------------------

/// YIN pitch estimator operating in the spectral domain (YINFFT).
///
/// Estimates the fundamental frequency of a magnitude spectrum frame using the
/// YIN algorithm adapted for the frequency domain. Takes the magnitude spectrum
/// produced by an FFT as input and returns a [`PitchResult`] with the
/// fundamental frequency in Hz and a confidence value.
///
/// See <https://learn.flucoma.org/reference/pitch>
pub struct YinFft {
    inner: *mut u8,
    n_bins: usize,
}

unsafe impl Send for YinFft {}

impl YinFft {
    /// Create a new YINFFT pitch estimator.
    ///
    /// # Arguments
    /// * `n_bins` - Number of magnitude spectrum bins (`fft_size / 2 + 1`).
    ///   This is also the maximum accepted input length.
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(n_bins: usize) -> Result<Self, &'static str> {
        if n_bins == 0 {
            return Err("n_bins must be > 0");
        }
        let inner = yinfft_create(n_bins as isize);
        if inner.is_null() {
            return Err("failed to create YINFFT instance");
        }
        Ok(Self { inner, n_bins })
    }

    /// Estimate the fundamental frequency of a magnitude spectrum frame.
    ///
    /// # Arguments
    /// * `magnitudes`  - Magnitude spectrum; must have exactly `n_bins` elements
    ///   (i.e. `fft_size / 2 + 1`).
    /// * `min_freq`    - Minimum frequency to search for (Hz). Must be > 0.
    /// * `max_freq`    - Maximum frequency to search for (Hz). Must be > `min_freq`.
    /// * `sample_rate` - Audio sample rate in Hz. Must be > 0.
    ///
    /// Returns a [`PitchResult`] with `pitch_hz` and `confidence`.
    ///
    /// # Panics
    /// Panics if `magnitudes.len() != n_bins`.
    pub fn process_frame(
        &mut self,
        magnitudes: &[f64],
        min_freq: f64,
        max_freq: f64,
        sample_rate: f64,
    ) -> PitchResult {
        assert_eq!(
            magnitudes.len(),
            self.n_bins,
            "magnitudes length ({}) must equal n_bins ({})",
            magnitudes.len(),
            self.n_bins
        );
        let mut out = [0.0f64; 2];
        yinfft_process_frame(
            self.inner,
            magnitudes.as_ptr(),
            magnitudes.len() as isize,
            out.as_mut_ptr(),
            min_freq,
            max_freq,
            sample_rate,
        );
        PitchResult {
            pitch_hz: out[0],
            confidence: out[1],
        }
    }

    /// Number of magnitude spectrum bins expected as input (`fft_size / 2 + 1`).
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}

impl Drop for YinFft {
    fn drop(&mut self) {
        yinfft_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yinfft_silent_spectrum_returns_no_pitch() {
        let fft_size = 1024usize;
        let n_bins = fft_size / 2 + 1;
        let mut yin = YinFft::new(n_bins).unwrap();
        let silence = vec![0.0f64; n_bins];
        let result = yin.process_frame(&silence, 20.0, 20000.0, 44100.0);
        // A silent/zero spectrum contains no pitch information
        assert_eq!(
            result.pitch_hz, 0.0,
            "expected 0 Hz pitch for silent spectrum, got {}",
            result.pitch_hz
        );
    }

    #[test]
    fn yinfft_harmonic_spectrum_detects_pitch() {
        // Build a synthetic magnitude spectrum with peaks at harmonics of 440 Hz.
        let sample_rate = 44100.0f64;
        let fft_size = 4096usize;
        let n_bins = fft_size / 2 + 1;
        let mut mags = vec![0.0f64; n_bins];
        let bin_hz = sample_rate / fft_size as f64;
        // Place energy at harmonics of 440 Hz
        for k in 1..=8 {
            let freq = 440.0 * k as f64;
            let bin = (freq / bin_hz).round() as usize;
            if bin < n_bins {
                mags[bin] = 1.0 / k as f64;
            }
        }
        let mut yin = YinFft::new(n_bins).unwrap();
        let result = yin.process_frame(&mags, 100.0, 2000.0, sample_rate);
        // Expect a pitch reasonably close to 440 Hz
        assert!(
            result.pitch_hz > 200.0 && result.pitch_hz < 900.0,
            "expected pitch near 440 Hz, got {} Hz",
            result.pitch_hz
        );
        assert!(
            result.confidence > 0.0,
            "expected positive confidence, got {}",
            result.confidence
        );
    }
}
