use flucoma_sys::{melbands_create, melbands_destroy, melbands_init, melbands_process_frame};

// -------------------------------------------------------------------------------------------------

/// Mel-scaled filter bank -- converts a magnitude spectrum into mel band energies.
///
/// Call [`MelBands::process_frame`] with magnitude spectra (not raw complex).
///
/// See <https://learn.flucoma.org/reference/melbands>
pub struct MelBands {
    inner: *mut u8,
    n_bins: usize,
    n_bands: usize,
}

unsafe impl Send for MelBands {}

impl MelBands {
    /// Create and fully initialise a mel filter bank.
    ///
    /// # Arguments
    /// * `n_bands`     - Number of mel bands (must be >= 2).
    /// * `n_bins`      - Number of FFT magnitude bins (`fft_size / 2 + 1`).
    /// * `lo_hz`       - Low-frequency edge of the filter bank in Hz.
    /// * `hi_hz`       - High-frequency edge of the filter bank in Hz.
    /// * `sample_rate` - Audio sample rate in Hz.
    /// * `window_size` - Analysis window size (for amplitude normalisation).
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid.
    pub fn new(
        n_bands: usize,
        n_bins: usize,
        lo_hz: f64,
        hi_hz: f64,
        sample_rate: f64,
        window_size: usize,
    ) -> Result<Self, &'static str> {
        if n_bands < 2 {
            return Err("n_bands must be >= 2");
        }
        if n_bins == 0 {
            return Err("n_bins must be > 0");
        }
        if lo_hz >= hi_hz {
            return Err("lo_hz must be < hi_hz");
        }
        if sample_rate <= 0.0 {
            return Err("sample_rate must be > 0");
        }
        if window_size == 0 {
            return Err("window_size must be > 0");
        }
        let inner = melbands_create(n_bands as isize, ((n_bins - 1) * 2) as isize);
        if inner.is_null() {
            return Err("failed to create MelBands instance");
        }
        melbands_init(
            inner,
            lo_hz,
            hi_hz,
            n_bands as isize,
            n_bins as isize,
            sample_rate,
            window_size as isize,
        );
        Ok(Self {
            inner,
            n_bins,
            n_bands,
        })
    }

    /// Process a magnitude spectrum frame and return mel band energies.
    ///
    /// # Arguments
    /// * `magnitudes` - Magnitude spectrum; must have exactly `n_bins` values.
    /// * `mag_norm`   - Normalise by magnitude (area-normalised filters).
    /// * `use_power`  - Square the magnitudes (power spectrum input).
    /// * `log_output` - Return output in dB (20*log10).
    ///
    /// # Panics
    /// Panics if `magnitudes.len() != n_bins`.
    pub fn process_frame(
        &mut self,
        magnitudes: &[f64],
        mag_norm: bool,
        use_power: bool,
        log_output: bool,
    ) -> Vec<f64> {
        assert_eq!(
            magnitudes.len(),
            self.n_bins,
            "magnitudes length ({}) must equal n_bins ({})",
            magnitudes.len(),
            self.n_bins
        );
        let mut output = vec![0.0f64; self.n_bands];
        melbands_process_frame(
            self.inner,
            magnitudes.as_ptr(),
            magnitudes.len() as isize,
            output.as_mut_ptr(),
            output.len() as isize,
            mag_norm,
            use_power,
            log_output,
        );
        output
    }

    /// Number of mel bands in each output frame.
    pub fn n_bands(&self) -> usize {
        self.n_bands
    }

    /// Number of FFT magnitude bins expected as input (`fft_size / 2 + 1`).
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}

impl Drop for MelBands {
    fn drop(&mut self) {
        melbands_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn melbands_output_count() {
        let n_bands = 40usize;
        let fft_size = 1024usize;
        let n_bins = fft_size / 2 + 1;
        let mut mel = MelBands::new(n_bands, n_bins, 80.0, 8000.0, 44100.0, fft_size).unwrap();
        let magnitudes = vec![1.0f64; n_bins];
        let bands = mel.process_frame(&magnitudes, false, false, false);
        assert_eq!(bands.len(), n_bands);
    }

    #[test]
    fn melbands_silent_spectrum() {
        let n_bands = 40usize;
        let fft_size = 1024usize;
        let n_bins = fft_size / 2 + 1;
        let mut mel = MelBands::new(n_bands, n_bins, 80.0, 8000.0, 44100.0, fft_size).unwrap();
        let magnitudes = vec![0.0f64; n_bins];
        let bands = mel.process_frame(&magnitudes, false, false, false);
        for &v in &bands {
            assert!(v.abs() < 1e-10, "expected zero band, got {}", v);
        }
    }
}
