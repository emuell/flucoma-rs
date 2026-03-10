use flucoma_sys::{hpss_create, hpss_destroy, hpss_init, hpss_process_frame};
use num_complex::Complex64;

// -------------------------------------------------------------------------------------------------

/// HPSS separation mode.
///
/// Controls how harmonic and percussive masks are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(isize)]
pub enum HpssMode {
    /// Binary masking: energy goes to whichever component dominates. No residual.
    #[default]
    Classic = 0,
    /// Wiener-style binary mask with a frequency-varying threshold. No residual.
    Coupled = 1,
    /// Independent masks for both harmonic and percussive; remainder is residual.
    Advanced = 2,
}

/// Per-frame parameters for [`Hpss::process_frame`].
///
/// `h_size` and `v_size` are fixed at construction time and therefore not included here;
/// see [`Hpss::new`].
#[derive(Debug, Clone, Copy)]
pub struct HpssParams {
    /// Separation mode.
    pub mode: HpssMode,
    /// Harmonic mask low-frequency point (normalised 0–1).
    pub h_threshold_x1: f64,
    /// Harmonic mask threshold at `h_threshold_x1`.
    pub h_threshold_y1: f64,
    /// Harmonic mask high-frequency point (normalised 0–1).
    pub h_threshold_x2: f64,
    /// Harmonic mask threshold at `h_threshold_x2`.
    pub h_threshold_y2: f64,
    /// Percussive mask low-frequency point (used in [`HpssMode::Advanced`]).
    pub p_threshold_x1: f64,
    /// Percussive mask threshold at `p_threshold_x1`.
    pub p_threshold_y1: f64,
    /// Percussive mask high-frequency point (used in [`HpssMode::Advanced`]).
    pub p_threshold_x2: f64,
    /// Percussive mask threshold at `p_threshold_x2`.
    pub p_threshold_y2: f64,
}

impl Default for HpssParams {
    fn default() -> Self {
        Self {
            mode: HpssMode::Classic,
            h_threshold_x1: 0.0,
            h_threshold_y1: 1.0,
            h_threshold_x2: 1.0,
            h_threshold_y2: 1.0,
            p_threshold_x1: 0.0,
            p_threshold_y1: 1.0,
            p_threshold_x2: 1.0,
            p_threshold_y2: 1.0,
        }
    }
}

/// Harmonic-Percussive Source Separation operating on complex spectral frames.
///
/// Separates a complex spectrum into up to three components per frame:
/// - **harmonic** -- spectrally smooth, time-varying content (pitched sounds)
/// - **percussive** -- spectrally noisy, time-impulsive content (drums, transients)
/// - **residual** -- everything else (only non-zero in [`HpssMode::Advanced`])
///
/// Two-phase setup:
/// 1. [`Hpss::new`] -- allocates internal median-filter history buffers.
/// 2. Call [`Hpss::process_frame`] per complex spectral frame.
///
/// The input is typically the output of an STFT. Input and output frames
/// are complex spectra of `n_bins` bins.
///
/// See <https://learn.flucoma.org/reference/hpss>
pub struct Hpss {
    inner: *mut u8,
    n_bins: usize,
    /// Horizontal (time) filter size, fixed at construction.
    h_size: usize,
    /// Vertical (frequency) filter size, fixed at construction.
    v_size: usize,
    /// Interleaved FFI output buffer, layout (n_bins × 3) complex values.
    ffi_buf: Vec<Complex64>,
    /// Deinterleaved output: [harmonic | percussive | residual], each n_bins long.
    out_buf: Vec<Complex64>,
}

unsafe impl Send for Hpss {}

impl Hpss {
    /// Create and initialise an HPSS processor.
    ///
    /// # Arguments
    /// * `fft_size` - FFT size used by the upstream STFT (must be > 0 and even).
    ///   `n_bins` is derived as `fft_size / 2 + 1`.
    /// * `h_size`   - Horizontal (time) median filter length in frames. Must be odd and ≥ 1.
    /// * `v_size`   - Vertical (frequency) median filter length in bins. Must be odd and ≥ 1.
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(fft_size: usize, h_size: usize, v_size: usize) -> Result<Self, &'static str> {
        if fft_size == 0 || !fft_size.is_multiple_of(2) {
            return Err("fft_size must be > 0 and even");
        }
        if h_size == 0 || h_size.is_multiple_of(2) {
            return Err("h_size must be odd and >= 1");
        }
        if v_size == 0 || v_size.is_multiple_of(2) {
            return Err("v_size must be odd and >= 1");
        }
        let n_bins = fft_size / 2 + 1;
        let inner = hpss_create(fft_size as isize, h_size as isize);
        if inner.is_null() {
            return Err("failed to create HPSS instance");
        }
        hpss_init(inner, n_bins as isize, h_size as isize);
        Ok(Self {
            inner,
            n_bins,
            h_size,
            v_size,
            ffi_buf: vec![Complex64::default(); n_bins * 3],
            out_buf: vec![Complex64::default(); n_bins * 3],
        })
    }

    /// Process one complex spectral frame.
    ///
    /// # Arguments
    /// * `input`  - Complex spectrum of length `n_bins()`.
    /// * `params` - Mask parameters. Use [`HpssParams::default`] for typical settings.
    ///
    /// Returns `(harmonic, percussive, residual)`: three slices into an internal buffer,
    /// each of length `n_bins()`, valid until the next call.
    ///
    /// In [`HpssMode::Classic`] and [`HpssMode::Coupled`] the `residual` slice is all zeros.
    ///
    /// # Panics
    /// Panics if `input.len() != n_bins()`.
    pub fn process_frame<'a>(
        &'a mut self,
        input: &[Complex64],
        params: &HpssParams,
    ) -> (&'a [Complex64], &'a [Complex64], &'a [Complex64]) {
        assert_eq!(
            input.len(),
            self.n_bins,
            "input length ({}) must equal n_bins ({})",
            input.len(),
            self.n_bins
        );
        hpss_process_frame(
            self.inner,
            input.as_ptr() as *const f64,
            self.n_bins as isize,
            self.ffi_buf.as_mut_ptr() as *mut f64,
            self.v_size as isize,
            self.h_size as isize,
            params.mode as isize,
            params.h_threshold_x1,
            params.h_threshold_y1,
            params.h_threshold_x2,
            params.h_threshold_y2,
            params.p_threshold_x1,
            params.p_threshold_y1,
            params.p_threshold_x2,
            params.p_threshold_y2,
        );
        // Deinterleave: ffi_buf layout is (n_bins × 3) row-major ->
        // [h0,p0,r0, h1,p1,r1, ...] -> out_buf: [h... | p... | r...]
        let n = self.n_bins;
        for i in 0..n {
            self.out_buf[i] = self.ffi_buf[i * 3];
            self.out_buf[n + i] = self.ffi_buf[i * 3 + 1];
            self.out_buf[2 * n + i] = self.ffi_buf[i * 3 + 2];
        }
        let (harmonic, rest) = self.out_buf.split_at(n);
        let (percussive, residual) = rest.split_at(n);
        (harmonic, percussive, residual)
    }

    /// Number of complex bins per frame (`fft_size / 2 + 1`).
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Horizontal (time) median filter size in frames.
    pub fn h_size(&self) -> usize {
        self.h_size
    }

    /// Vertical (frequency) median filter size in bins.
    pub fn v_size(&self) -> usize {
        self.v_size
    }
}

impl Drop for Hpss {
    fn drop(&mut self) {
        hpss_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hpss_silence_gives_silence() {
        let fft_size = 1024usize;
        let mut hpss = Hpss::new(fft_size, 17, 17).unwrap();
        let n_bins = hpss.n_bins();
        let input = vec![Complex64::default(); n_bins];
        let (harmonic, percussive, residual) = hpss.process_frame(&input, &HpssParams::default());
        assert_eq!(harmonic.len(), n_bins);
        assert_eq!(percussive.len(), n_bins);
        assert_eq!(residual.len(), n_bins);
        for &v in harmonic
            .iter()
            .chain(percussive.iter())
            .chain(residual.iter())
        {
            assert!(
                v.norm() < 1e-9,
                "silence input should give near-zero output, got {v}"
            );
        }
    }

    #[test]
    fn hpss_output_lengths_correct() {
        let fft_size = 512usize;
        let mut hpss = Hpss::new(fft_size, 9, 9).unwrap();
        let n_bins = hpss.n_bins();
        assert_eq!(n_bins, fft_size / 2 + 1);
        let input = vec![Complex64::new(0.1, 0.0); n_bins];
        let (h, p, r) = hpss.process_frame(&input, &HpssParams::default());
        assert_eq!(h.len(), n_bins);
        assert_eq!(p.len(), n_bins);
        assert_eq!(r.len(), n_bins);
    }

    #[test]
    fn hpss_advanced_mode_runs() {
        let fft_size = 512usize;
        let mut hpss = Hpss::new(fft_size, 9, 9).unwrap();
        let n_bins = hpss.n_bins();
        let input = vec![Complex64::new(0.5, 0.1); n_bins];
        let params = HpssParams {
            mode: HpssMode::Advanced,
            ..HpssParams::default()
        };
        let (h, p, r) = hpss.process_frame(&input, &params);
        for &v in h.iter().chain(p.iter()).chain(r.iter()) {
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "output must be finite"
            );
        }
    }
}
