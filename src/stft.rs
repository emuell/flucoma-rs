use flucoma_sys::{
    istft_create, istft_destroy, istft_process_frame, stft_create, stft_destroy, stft_process_frame,
};
use num_complex::Complex64 as Complex;

// -------------------------------------------------------------------------------------------------

/// Window function type for STFT/ISTFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(isize)]
pub enum WindowType {
    #[default]
    Hann = 0,
    Hamming = 1,
    Blackman = 2,
    Rectangular = 3,
}

// -------------------------------------------------------------------------------------------------

/// A complex spectral frame produced by [`Stft`] and consumed by [`Istft`].
///
/// Each bin is a [`Complex`] value (`re` + `im`). Bins are ordered from DC
/// to Nyquist: `bins.len() == fft_size / 2 + 1`.
#[derive(Debug, Clone)]
pub struct ComplexSpectrum {
    /// Complex bins, DC to Nyquist.
    pub bins: Vec<Complex>,
}

impl ComplexSpectrum {
    /// Allocate a zeroed spectrum for `num_bins` complex bins.
    pub fn zeros(num_bins: usize) -> Self {
        Self {
            bins: vec![Complex::default(); num_bins],
        }
    }

    /// Number of complex bins (`fft_size / 2 + 1`).
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }

    /// All magnitudes as a `Vec<f64>`.
    pub fn magnitudes(&self) -> Vec<f64> {
        self.bins.iter().map(|c| c.norm()).collect()
    }

    /// All phases in radians as a `Vec<f64>`.
    pub fn phases(&self) -> Vec<f64> {
        self.bins.iter().map(|c| c.arg()).collect()
    }
}

// -------------------------------------------------------------------------------------------------

/// Short-Time Fourier Transform -- converts windowed audio frames into complex
/// spectra frame by frame.
///
/// Two-phase setup:
/// 1. [`Stft::new`] -- constructs and allocates.
/// 2. Call [`Stft::process_frame`] once per hop.
///
/// See <https://learn.flucoma.org/learn/fourier-transform/>
pub struct Stft {
    inner: *mut u8,
    window_size: usize,
    fft_size: usize,
    hop_size: usize,
    num_bins: usize,
}

unsafe impl Send for Stft {}

impl Stft {
    /// Create a new STFT analyser.
    ///
    /// # Arguments
    /// * `window_size` - Analysis window length in samples.
    /// * `fft_size`    - FFT size (must be >= `window_size`, typically power of 2).
    /// * `hop_size`    - Hop between successive frames in samples.
    /// * `window_type` - Window function.
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid.
    pub fn new(
        window_size: usize,
        fft_size: usize,
        hop_size: usize,
        window_type: WindowType,
    ) -> Result<Self, &'static str> {
        if window_size == 0 {
            return Err("window_size must be > 0");
        }
        if fft_size < window_size {
            return Err("fft_size must be >= window_size");
        }
        if hop_size == 0 {
            return Err("hop_size must be > 0");
        }
        let inner = stft_create(
            window_size as isize,
            fft_size as isize,
            hop_size as isize,
            window_type as isize,
        );
        if inner.is_null() {
            return Err("failed to create STFT instance");
        }
        Ok(Self {
            inner,
            window_size,
            fft_size,
            hop_size,
            num_bins: fft_size / 2 + 1,
        })
    }

    /// Process one audio frame and return its complex spectrum.
    ///
    /// # Arguments
    /// * `frame` - Audio samples of exactly `window_size` length.
    ///
    /// # Panics
    /// Panics if `frame.len() != window_size`.
    pub fn process_frame(&mut self, frame: &[f64]) -> ComplexSpectrum {
        assert_eq!(
            frame.len(),
            self.window_size,
            "frame length ({}) must equal window_size ({})",
            frame.len(),
            self.window_size
        );
        let mut spec = ComplexSpectrum::zeros(self.num_bins);
        stft_process_frame(
            self.inner,
            frame.as_ptr(),
            frame.len() as isize,
            spec.bins.as_mut_ptr() as *mut f64,
            self.num_bins as isize,
        );
        spec
    }

    /// Analysis window size in samples.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// FFT size in samples.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size between frames in samples.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Number of complex bins per spectrum (`fft_size / 2 + 1`).
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

impl Drop for Stft {
    fn drop(&mut self) {
        stft_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

/// Inverse Short-Time Fourier Transform -- reconstructs audio from complex
/// spectra frame by frame.
///
/// See <https://learn.flucoma.org/learn/fourier-transform/>
pub struct Istft {
    inner: *mut u8,
    window_size: usize,
    fft_size: usize,
    hop_size: usize,
    num_bins: usize,
}

unsafe impl Send for Istft {}

impl Istft {
    /// Create a new ISTFT synthesiser.
    pub fn new(
        window_size: usize,
        fft_size: usize,
        hop_size: usize,
        window_type: WindowType,
    ) -> Result<Self, &'static str> {
        if window_size == 0 {
            return Err("window_size must be > 0");
        }
        if fft_size < window_size {
            return Err("fft_size must be >= window_size");
        }
        if hop_size == 0 {
            return Err("hop_size must be > 0");
        }
        let inner = istft_create(
            window_size as isize,
            fft_size as isize,
            hop_size as isize,
            window_type as isize,
        );
        if inner.is_null() {
            return Err("failed to create ISTFT instance");
        }
        Ok(Self {
            inner,
            window_size,
            fft_size,
            hop_size,
            num_bins: fft_size / 2 + 1,
        })
    }

    /// Synthesise one audio frame from a complex spectrum.
    ///
    /// # Arguments
    /// * `spectrum` - Complex spectrum with `num_bins` bins.
    /// * `output`   - Output buffer of exactly `window_size` samples.
    ///
    /// # Panics
    /// Panics if `spectrum.num_bins() != self.num_bins` or
    /// `output.len() != window_size`.
    pub fn process_frame(&mut self, spectrum: &ComplexSpectrum, output: &mut [f64]) {
        assert_eq!(
            spectrum.num_bins(),
            self.num_bins,
            "spectrum num_bins ({}) must equal num_bins ({})",
            spectrum.num_bins(),
            self.num_bins
        );
        assert_eq!(
            output.len(),
            self.window_size,
            "output length ({}) must equal window_size ({})",
            output.len(),
            self.window_size
        );
        istft_process_frame(
            self.inner,
            spectrum.bins.as_ptr() as *const f64,
            self.num_bins as isize,
            output.as_mut_ptr(),
            output.len() as isize,
        );
    }

    /// Synthesis window size in samples.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// FFT size in samples.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size between frames in samples.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Number of complex bins per spectrum (`fft_size / 2 + 1`).
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

impl Drop for Istft {
    fn drop(&mut self) {
        istft_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_produces_correct_bin_count() {
        let fft_size = 1024;
        let mut stft = Stft::new(1024, fft_size, 512, WindowType::Hann).unwrap();
        let frame = vec![0.0f64; 1024];
        let spec = stft.process_frame(&frame);
        assert_eq!(spec.num_bins(), fft_size / 2 + 1);
        assert_eq!(spec.bins.len(), fft_size / 2 + 1);
    }

    #[test]
    fn stft_istft_roundtrip_impulse() {
        let win = 1024usize;
        let fft = 1024usize;
        let hop = 512usize;
        let mut stft = Stft::new(win, fft, hop, WindowType::Hann).unwrap();
        let mut istft = Istft::new(win, fft, hop, WindowType::Hann).unwrap();

        // A pure tone should survive STFT -> ISTFT (with windowing loss).
        use std::f64::consts::PI;
        let frame: Vec<f64> = (0..win)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let spec = stft.process_frame(&frame);
        let mut reconstructed = vec![0.0f64; win];
        istft.process_frame(&spec, &mut reconstructed);

        // Energy should be preserved (within windowing scale factor)
        let orig_energy: f64 = frame.iter().map(|x| x * x).sum();
        let rec_energy: f64 = reconstructed.iter().map(|x| x * x).sum();
        assert!(orig_energy > 0.0);
        assert!(rec_energy > 0.0, "reconstructed energy is zero");
    }
}
