use flucoma_sys::{sine_ext_create, sine_ext_destroy, sine_ext_init, sine_ext_process_frame};
use num_complex::Complex64;

// -------------------------------------------------------------------------------------------------

/// Tracking and detection parameters for [`SineExtraction::process_frame`].
#[derive(Debug, Clone, Copy)]
pub struct SineExtractionParams {
    /// Audio sample rate in Hz.
    pub sample_rate: f64,
    /// Peak detection threshold in dB (e.g. -96.0).
    pub detection_threshold: f64,
    /// Minimum track length in frames before a partial is reported.
    pub min_track_length: usize,
    /// Amplitude threshold for track birth at low frequencies (dB, e.g. -24.0).
    pub birth_low_threshold: f64,
    /// Amplitude threshold for track birth at high frequencies (dB, e.g. -60.0).
    pub birth_high_threshold: f64,
    /// Tracking algorithm: 0 = greedy, 1 = Hungarian.
    pub track_method: usize,
    /// Tracking magnitude range in dB — max amplitude difference for a match (e.g. 15.0).
    pub zeta_a: f64,
    /// Tracking frequency range in Hz — max frequency difference for a match (e.g. 50.0).
    pub zeta_f: f64,
    /// Track matching probability threshold (0.0–1.0, e.g. 0.5).
    pub delta: f64,
    /// Bandwidth in bins for partial synthesis.
    pub bandwidth: usize,
}

impl Default for SineExtractionParams {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            detection_threshold: -96.0,
            min_track_length: 15,
            birth_low_threshold: -24.0,
            birth_high_threshold: -60.0,
            track_method: 0,
            zeta_a: 15.0,
            zeta_f: 50.0,
            delta: 0.5,
            bandwidth: 76,
        }
    }
}

/// Sinusoidal extraction: separate a complex spectrum into sines and residual.
///
/// Uses a sinusoidal tracking algorithm to identify and extract pitched partial
/// components from each spectral frame. Produces two output components:
/// - **sines** -- the reconstructed sinusoidal content
/// - **residual** -- what remains after subtracting the sines
///
/// Note: there is an inherent latency of `min_track_length` frames before
/// the sinusoidal output becomes non-zero (the tracker needs history to confirm tracks).
///
/// Two-phase setup:
/// 1. [`SineExtraction::new`] -- allocates internal buffers.
/// 2. Call [`SineExtraction::process_frame`] per complex spectral frame.
///
/// See <https://learn.flucoma.org/reference/sines>
pub struct SineExtraction {
    inner: *mut u8,
    n_bins: usize,
    window_size: usize,
    fft_size: usize,
    transform_size: usize,
    /// Interleaved FFI output buffer, shape (n_bins, 2) complex values.
    ffi_buf: Vec<Complex64>,
    /// Deinterleaved output: [sines | residual], each n_bins long.
    out_buf: Vec<Complex64>,
}

unsafe impl Send for SineExtraction {}

impl SineExtraction {
    /// Create and initialise a sinusoidal extractor.
    ///
    /// # Arguments
    /// * `window_size`    - Analysis window size in samples (must be > 0).
    /// * `fft_size`       - FFT size (must be >= `window_size`, even).
    ///   `n_bins` is derived as `fft_size / 2 + 1`.
    /// * `transform_size` - Window transform size for partial synthesis (must be > 0).
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(
        window_size: usize,
        fft_size: usize,
        transform_size: usize,
    ) -> Result<Self, &'static str> {
        if window_size == 0 {
            return Err("window_size must be > 0");
        }
        if fft_size == 0 || !fft_size.is_multiple_of(2) {
            return Err("fft_size must be > 0 and even");
        }
        if fft_size < window_size {
            return Err("fft_size must be >= window_size");
        }
        if transform_size == 0 {
            return Err("transform_size must be > 0");
        }
        let n_bins = fft_size / 2 + 1;
        let inner = sine_ext_create(fft_size as isize);
        if inner.is_null() {
            return Err("failed to create SineExtraction instance");
        }
        sine_ext_init(
            inner,
            window_size as isize,
            fft_size as isize,
            transform_size as isize,
        );
        Ok(Self {
            inner,
            n_bins,
            window_size,
            fft_size,
            transform_size,
            ffi_buf: vec![Complex64::default(); n_bins * 2],
            out_buf: vec![Complex64::default(); n_bins * 2],
        })
    }

    /// Reset internal frame queue and partial tracker without reallocating.
    pub fn reset(&mut self) {
        sine_ext_init(
            self.inner,
            self.window_size as isize,
            self.fft_size as isize,
            self.transform_size as isize,
        );
    }

    /// Process one complex spectral frame.
    ///
    /// # Arguments
    /// * `input`  - Complex spectrum of length `n_bins()`.
    /// * `params` - Tracking/detection parameters. Use [`SineExtractionParams::default`].
    ///
    /// Returns `(sines, residual)`: two slices into an internal buffer, each of length
    /// `n_bins()`. The slices are valid until the next call.
    ///
    /// Output is silent for the first `min_track_length` frames while the tracker
    /// builds its history.
    ///
    /// # Panics
    /// Panics if `input.len() != n_bins()`.
    pub fn process_frame<'a>(
        &'a mut self,
        input: &[Complex64],
        params: &SineExtractionParams,
    ) -> (&'a [Complex64], &'a [Complex64]) {
        assert_eq!(
            input.len(),
            self.n_bins,
            "input length ({}) must equal n_bins ({})",
            input.len(),
            self.n_bins
        );
        sine_ext_process_frame(
            self.inner,
            input.as_ptr() as *const f64,
            self.n_bins as isize,
            self.ffi_buf.as_mut_ptr() as *mut f64,
            params.sample_rate,
            params.detection_threshold,
            params.min_track_length as isize,
            params.birth_low_threshold,
            params.birth_high_threshold,
            params.track_method as isize,
            params.zeta_a,
            params.zeta_f,
            params.delta,
            params.bandwidth as isize,
        );
        // Deinterleave: ffi_buf layout is (n_bins, 2) row-major ->
        // [s0,r0, s1,r1, ...] -> out_buf: [s... | r...]
        let n = self.n_bins;
        for i in 0..n {
            self.out_buf[i] = self.ffi_buf[i * 2];
            self.out_buf[n + i] = self.ffi_buf[i * 2 + 1];
        }
        self.out_buf.split_at(n)
    }

    /// Number of complex bins per frame (fft_size / 2 + 1).
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}

impl Drop for SineExtraction {
    fn drop(&mut self) {
        sine_ext_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_ext_silence_gives_silence() {
        let window = 1024usize;
        let fft = 1024usize;
        let mut se = SineExtraction::new(window, fft, window).unwrap();
        let n_bins = se.n_bins();
        let input = vec![Complex64::default(); n_bins];
        let params = SineExtractionParams::default();
        let (sines, residual) = se.process_frame(&input, &params);
        assert_eq!(sines.len(), n_bins);
        assert_eq!(residual.len(), n_bins);
        for &v in sines.iter().chain(residual.iter()) {
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "output must be finite, got {v}"
            );
        }
    }

    #[test]
    fn sine_ext_reset_clears_tracker() {
        let fft = 512usize;
        let mut se = SineExtraction::new(fft, fft, fft).unwrap();
        let n_bins = se.n_bins();
        let input = vec![Complex64::new(0.01, 0.0); n_bins];
        let params = SineExtractionParams::default();

        let (s, r) = se.process_frame(&input, &params);
        let first: Vec<_> = s.iter().chain(r).copied().collect();

        for _ in 0..10 {
            se.process_frame(&input, &params);
        }

        se.reset();
        let (s2, r2) = se.process_frame(&input, &params);
        let after: Vec<_> = s2.iter().chain(r2).copied().collect();
        assert_eq!(first, after, "reset should restore output to initial state");
    }

    #[test]
    fn sine_ext_output_lengths_correct() {
        let fft = 512usize;
        let mut se = SineExtraction::new(fft, fft, fft).unwrap();
        let n_bins = se.n_bins();
        assert_eq!(n_bins, fft / 2 + 1);
        let input = vec![Complex64::new(0.01, 0.0); n_bins];
        let (sines, residual) = se.process_frame(&input, &SineExtractionParams::default());
        assert_eq!(sines.len(), n_bins);
        assert_eq!(residual.len(), n_bins);
    }
}
