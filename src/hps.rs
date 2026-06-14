use flucoma_sys::{hps_create, hps_destroy, hps_process_frame};

use super::yin_fft::PitchResult;

// -------------------------------------------------------------------------------------------------

/// Harmonic Product Spectrum (HPS) pitch estimator.
///
/// Estimates the fundamental frequency of a magnitude spectrum frame by
/// multiplying successive decimations of the spectrum, reinforcing bins that
/// have energy at harmonic multiples. Takes the magnitude spectrum produced by
/// an FFT as input and returns a [`PitchResult`] with the fundamental
/// frequency in Hz and a confidence value.
///
/// See <https://learn.flucoma.org/reference/pitch>
pub struct Hps {
    inner: *mut u8,
    n_bins: usize,
}

unsafe impl Send for Hps {}

impl Hps {
    /// Create a new HPS pitch estimator.
    ///
    /// # Arguments
    /// * `n_bins` - Number of magnitude spectrum bins (`fft_size / 2 + 1`).
    ///
    /// # Errors
    /// Returns an error string if `n_bins` is 0 or allocation fails.
    pub fn new(n_bins: usize) -> Result<Self, &'static str> {
        if n_bins == 0 {
            return Err("n_bins must be > 0");
        }
        let inner = hps_create();
        if inner.is_null() {
            return Err("failed to create HPS instance");
        }
        Ok(Self { inner, n_bins })
    }

    /// Estimate the fundamental frequency of a magnitude spectrum frame.
    ///
    /// # Arguments
    /// * `magnitudes`   - Magnitude spectrum; must have exactly `n_bins` elements
    ///   (i.e. `fft_size / 2 + 1`).
    /// * `n_harmonics`  - Number of harmonic products to accumulate (>= 2;
    ///   the FluCoMa default is 4).
    /// * `min_freq`     - Minimum frequency to search for (Hz). Must be > 0.
    /// * `max_freq`     - Maximum frequency to search for (Hz). Must be > `min_freq`.
    /// * `sample_rate`  - Audio sample rate in Hz. Must be > 0.
    ///
    /// Returns a [`PitchResult`] with `pitch_hz` and `confidence`.
    ///
    /// # Panics
    /// Panics if `magnitudes.len() != n_bins`.
    pub fn process_frame(
        &mut self,
        magnitudes: &[f64],
        n_harmonics: usize,
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
        hps_process_frame(
            self.inner,
            magnitudes.as_ptr(),
            magnitudes.len() as isize,
            out.as_mut_ptr(),
            n_harmonics as isize,
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

impl Drop for Hps {
    fn drop(&mut self) {
        hps_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hps_silent_spectrum_returns_no_pitch() {
        let fft_size = 1024usize;
        let n_bins = fft_size / 2 + 1;
        let mut hps = Hps::new(n_bins).unwrap();
        let silence = vec![0.0f64; n_bins];
        let result = hps.process_frame(&silence, 4, 20.0, 20000.0, 44100.0);
        // hpsSum == 0 for silence, so pitch and confidence stay at 0
        assert_eq!(
            result.pitch_hz, 0.0,
            "expected 0 Hz pitch for silent spectrum, got {}",
            result.pitch_hz
        );
        assert_eq!(
            result.confidence, 0.0,
            "expected 0 confidence for silent spectrum, got {}",
            result.confidence
        );
    }

    #[test]
    fn hps_harmonic_spectrum_detects_pitch() {
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
        let mut hps = Hps::new(n_bins).unwrap();
        let result = hps.process_frame(&mags, 4, 100.0, 2000.0, sample_rate);
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
