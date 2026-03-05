use flucoma_sys::{sine_create, sine_destroy, sine_init, sine_process_frame};
use num_complex::Complex64 as Complex;

// -------------------------------------------------------------------------------------------------

/// Peak sort order for [`Sine::process_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(isize)]
pub enum SortBy {
    /// Sort detected peaks by frequency (bin index), ascending.
    #[default]
    Frequency = 0,
    /// Sort detected peaks by magnitude, descending.
    Magnitude = 1,
}

// -------------------------------------------------------------------------------------------------

/// Sinusoidal peak detector for complex spectral frames.
///
/// Two-phase setup:
/// 1. [`Sine::new`] -- allocates and initialises the detector.
/// 2. Call [`Sine::process_frame`] per spectral frame.
///
/// Takes the complex FFT output from [`crate::fourier::Stft`] and returns the
/// frequencies (Hz) and magnitudes (dB) of detected sinusoidal peaks.
///
/// See <https://learn.flucoma.org/reference/sinefeature>
pub struct Sine {
    inner: *mut u8,
    num_bins: usize, // fft_size / 2 + 1
}

unsafe impl Send for Sine {}

impl Sine {
    /// Create and initialise a sinusoidal peak detector.
    ///
    /// # Arguments
    /// * `window_size` - Analysis window size in samples.
    /// * `fft_size`    - FFT size (must be >= `window_size`).
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(window_size: usize, fft_size: usize) -> Result<Self, &'static str> {
        if window_size == 0 {
            return Err("window_size must be > 0");
        }
        if fft_size < window_size {
            return Err("fft_size must be >= window_size");
        }
        let inner = sine_create();
        if inner.is_null() {
            return Err("failed to create SineFeature instance");
        }
        sine_init(inner, window_size as isize, fft_size as isize);
        Ok(Self {
            inner,
            num_bins: fft_size / 2 + 1,
        })
    }

    /// Detect sinusoidal peaks in one complex spectral frame.
    ///
    /// # Arguments
    /// * `input`     - Complex FFT bins, DC to Nyquist. Length must equal `num_bins`.
    /// * `freq_out`  - Buffer to receive peak frequencies in Hz. Length sets the maximum
    ///   number of peaks returned.
    /// * `mag_out`   - Buffer to receive peak magnitudes in dB. Must be the same length
    ///   as `freq_out`.
    /// * `sample_rate`         - Sample rate in Hz.
    /// * `threshold`           - Detection threshold in dB (peaks below this are ignored).
    /// * `sort_by`             - Whether to sort output by frequency or magnitude.
    ///
    /// Returns the number of peaks actually written to `freq_out`/`mag_out`.
    ///
    /// # Panics
    /// Panics if `input.len() != num_bins` or `freq_out.len() != mag_out.len()`.
    pub fn process_frame(
        &mut self,
        input: &[Complex],
        freq_out: &mut [f64],
        mag_out: &mut [f64],
        sample_rate: f64,
        threshold: f64,
        sort_by: SortBy,
    ) -> usize {
        assert_eq!(
            input.len(),
            self.num_bins,
            "input length ({}) must equal num_bins ({})",
            input.len(),
            self.num_bins
        );
        assert_eq!(
            freq_out.len(),
            mag_out.len(),
            "freq_out and mag_out must have the same length"
        );
        let n = sine_process_frame(
            self.inner,
            input.as_ptr() as *const f64,
            input.len() as isize,
            freq_out.as_mut_ptr(),
            mag_out.as_mut_ptr(),
            freq_out.len() as isize,
            sample_rate,
            threshold,
            sort_by as isize,
        );
        n as usize
    }

    /// Number of complex bins expected as input (`fft_size / 2 + 1`).
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

impl Drop for Sine {
    fn drop(&mut self) {
        sine_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_zero_spectrum_finds_no_peaks_above_threshold() {
        let fft_size = 1024;
        let mut sine = Sine::new(1024, fft_size).unwrap();
        let input = vec![Complex::default(); fft_size / 2 + 1];
        let mut freqs = vec![0.0f64; 32];
        let mut mags = vec![0.0f64; 32];
        let count = sine.process_frame(&input, &mut freqs, &mut mags, 44100.0, -60.0, SortBy::Frequency);
        // Zero input produces -inf dB, so nothing should exceed -60 dB threshold
        assert!(count == 0, "expected no peaks for zero input, got {count}");
    }

    #[test]
    fn sine_single_bin_impulse_finds_a_peak() {
        let fft_size = 1024;
        let mut sine = Sine::new(1024, fft_size).unwrap();
        let mut input = vec![Complex::default(); fft_size / 2 + 1];
        // Strong single bin at index 100
        input[100] = Complex::new(100.0, 0.0);
        let mut freqs = vec![0.0f64; 32];
        let mut mags = vec![0.0f64; 32];
        let count = sine.process_frame(&input, &mut freqs, &mut mags, 44100.0, -60.0, SortBy::Frequency);
        assert!(count >= 1, "expected at least one peak for impulse input");
    }
}
