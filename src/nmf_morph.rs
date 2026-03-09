use num_complex::Complex64 as Complex;

use flucoma_sys::{nmf_morph_create, nmf_morph_destroy, nmf_morph_init, nmf_morph_process_frame};

use crate::matrix::Matrix;

// -------------------------------------------------------------------------------------------------

/// NMF-based real-time spectral morphing with phase reconstruction (RTPGHI).
///
/// Morphs between two sets of NMF spectral bases using optimal transport, modulated
/// by a set of NMF activations.  Outputs a complex spectrum per frame that can be
/// fed directly to [`crate::fourier::Istft`].
///
/// # Two-phase setup
/// 1. [`NMFMorph::new`] -- allocates buffers for a given maximum FFT size.
/// 2. [`NMFMorph::init`] -- provide two trained bases matrices (W1, W2) and an
///    activation matrix H.  Re-call `init` any time the matrices change.
/// 3. [`NMFMorph::process_frame`] -- call once per STFT hop to get the next
///    interpolated complex spectrum.
///
/// See <https://learn.flucoma.org/reference/nmfmorph>
pub struct NMFMorph {
    inner: *mut u8,
    max_fft_size: usize,
    num_bins: usize,
    buf: Vec<Complex>,
}

unsafe impl Send for NMFMorph {}

impl NMFMorph {
    /// Allocate a NMFMorph instance.
    ///
    /// # Arguments
    /// * `max_fft_size` - Maximum FFT size that will be used. Must be > 0.
    pub fn new(max_fft_size: usize) -> Result<Self, &'static str> {
        if max_fft_size == 0 {
            return Err("max_fft_size must be > 0");
        }
        let inner = nmf_morph_create(max_fft_size as isize);
        if inner.is_null() {
            return Err("failed to create NMFMorph instance");
        }
        Ok(Self {
            inner,
            max_fft_size,
            num_bins: 0,
            buf: Vec::new(),
        })
    }

    /// Initialise with two bases matrices and an activations matrix.
    ///
    /// All matrices are **row-major**:
    /// - `w1`, `w2`: shape `rank × n_bins` where `n_bins = fft_size / 2 + 1`.
    /// - `h`:        shape `rank × n_frames`.
    ///
    /// `w1` and `w2` must have the same number of rows (rank).
    /// `h.rows()` must equal that rank.
    ///
    /// # Arguments
    /// * `w1`       - Source bases matrix (`rank × n_bins`).
    /// * `w2`       - Target bases matrix (`rank × n_bins`).
    /// * `h`        - Activations matrix (`rank × n_frames`).
    /// * `win_size` - Analysis window size in samples.
    /// * `fft_size` - FFT size (>= `win_size`, power of 2).
    /// * `hop_size` - Hop size in samples (> 0).
    /// * `assign`   - Use Hungarian assignment to match W1/W2 components optimally.
    ///
    /// # Errors
    /// Returns an error string if dimension constraints are violated.
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        &mut self,
        w1: &Matrix,
        w2: &Matrix,
        h: &Matrix,
        window_size: usize,
        fft_size: usize,
        hop_size: usize,
        assign: bool,
    ) -> Result<(), &'static str> {
        if w1.rows() != w2.rows() {
            return Err("w1 and w2 must have the same number of rows (rank)");
        }
        if h.rows() != w1.rows() {
            return Err("h.rows() must equal rank (w1.rows())");
        }
        if window_size == 0 {
            return Err("win_size must be > 0");
        }
        if fft_size < window_size {
            return Err("fft_size must be >= win_size");
        }
        if fft_size > self.max_fft_size {
            return Err("fft_size must be <= max_fft_size");
        }
        if hop_size == 0 {
            return Err("hop_size must be > 0");
        }
        let n_bins = fft_size / 2 + 1;
        if w1.cols() != n_bins {
            return Err("w1.cols() must equal fft_size / 2 + 1");
        }
        if w2.cols() != n_bins {
            return Err("w2.cols() must equal fft_size / 2 + 1");
        }
        nmf_morph_init(
            self.inner,
            w1.data().as_ptr(),
            w1.rows() as isize,
            w1.cols() as isize,
            w2.data().as_ptr(),
            w2.rows() as isize,
            w2.cols() as isize,
            h.data().as_ptr(),
            h.rows() as isize,
            h.cols() as isize,
            window_size as isize,
            fft_size as isize,
            hop_size as isize,
            assign,
        );
        self.num_bins = fft_size / 2 + 1;
        self.buf = vec![Complex::default(); self.num_bins];
        Ok(())
    }

    /// Generate one morphed complex-spectrum frame.
    ///
    /// Must be called after [`NMFMorph::init`].  Each call advances the internal
    /// activation column pointer by one hop.
    ///
    /// # Arguments
    /// * `interpolation` - Morph weight in `[0.0, 1.0]`. 0.0 = W1, 1.0 = W2.
    /// * `seed`          - Random seed for phase reconstruction (-1 = random).
    ///
    /// Returns a slice of [`Complex`] bins of length `num_bins`.
    /// Valid until the next call to this method.
    ///
    /// # Panics
    /// Panics if `init` has not been called yet.
    pub fn process_frame(&mut self, interpolation: f64, seed: i64) -> &[Complex] {
        assert!(
            self.num_bins > 0,
            "NMFMorph::init must be called before process_frame"
        );
        let interpolation = interpolation.clamp(0.0, 1.0);
        nmf_morph_process_frame(
            self.inner,
            self.buf.as_mut_ptr() as *mut f64,
            self.num_bins as isize,
            interpolation,
            seed as isize,
        );
        &self.buf
    }

    /// Number of complex bins output per frame (`fft_size / 2 + 1`), or 0 before `init`.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

impl Drop for NMFMorph {
    fn drop(&mut self) {
        nmf_morph_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_bases(rank: usize, n_bins: usize) -> Matrix {
        let mut data = vec![0.0f64; rank * n_bins];
        for r in 0..rank {
            for c in 0..n_bins {
                data[r * n_bins + c] = (r + 1) as f64 / n_bins as f64;
            }
        }
        Matrix::from_vec(data, rank, n_bins).unwrap()
    }

    #[test]
    fn nmf_morph_process_returns_correct_length() {
        let fft_size = 1024usize;
        let win_size = 1024usize;
        let hop_size = 512usize;
        let rank = 4usize;
        let n_frames = 8usize;
        let n_bins = fft_size / 2 + 1;

        let mut m = NMFMorph::new(fft_size).unwrap();

        let w1 = make_identity_bases(rank, n_bins);
        let w2 = make_identity_bases(rank, n_bins);
        let h = Matrix::from_vec(vec![0.5f64; rank * n_frames], rank, n_frames).unwrap();

        m.init(&w1, &w2, &h, win_size, fft_size, hop_size, false)
            .unwrap();

        assert_eq!(m.num_bins(), n_bins);

        let out = m.process_frame(0.5, -1);
        assert_eq!(out.len(), n_bins);
        for c in out {
            assert!(
                c.re.is_finite() && c.im.is_finite(),
                "output must be finite, got {c}"
            );
        }
    }

    #[test]
    fn nmf_morph_cycles_through_frames() {
        let fft_size = 512usize;
        let n_bins = fft_size / 2 + 1;
        let rank = 2usize;
        let n_frames = 4usize;

        let mut m = NMFMorph::new(fft_size).unwrap();
        let w = make_identity_bases(rank, n_bins);
        let h = Matrix::from_vec(vec![1.0f64; rank * n_frames], rank, n_frames).unwrap();

        m.init(&w, &w, &h, fft_size, fft_size, 256, false).unwrap();

        // Should be able to call more frames than n_frames (cycles modulo)
        for _ in 0..(n_frames * 2) {
            let out = m.process_frame(0.0, 0);
            assert_eq!(out.len(), n_bins);
        }
    }
}
