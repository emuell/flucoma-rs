use num_complex::Complex64 as Complex;

use flucoma_sys::{nmf_morph_create, nmf_morph_destroy, nmf_morph_init, nmf_morph_process_frame};

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
    /// FFT bins = fft_size / 2 + 1; 0 until `init` is called.
    num_bins: usize,
    buf: Vec<Complex>,
}

unsafe impl Send for NMFMorph {}

impl NMFMorph {
    /// Allocate an NMFMorph instance.
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
            num_bins: 0,
            buf: Vec::new(),
        })
    }

    /// Initialise with two bases matrices and an activations matrix.
    ///
    /// All matrices are **row-major flat slices**:
    /// - `w1`, `w2`: shape `rank × n_bins`.
    /// - `h`:        shape `rank × n_frames`.
    ///
    /// `w1_cols`, `w2_cols` must equal `fft_size / 2 + 1`.
    /// `w1_rows` and `w2_rows` must be equal (same rank).
    ///
    /// # Arguments
    /// * `w1` / `w1_rows` / `w1_cols` - Source bases matrix.
    /// * `w2` / `w2_rows` / `w2_cols` - Target bases matrix.
    /// * `h`  / `h_rows`  / `h_cols`  - Activations matrix (rank × n_frames).
    /// * `win_size`  - Analysis window size in samples.
    /// * `fft_size`  - FFT size (>= `win_size`, power of 2).
    /// * `hop_size`  - Hop size in samples (> 0).
    /// * `assign`    - Use Hungarian assignment to match W1/W2 components optimally.
    ///
    /// # Errors
    /// Returns an error string if dimension constraints are violated.
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        &mut self,
        w1: &[f64],
        w1_rows: usize,
        w1_cols: usize,
        w2: &[f64],
        w2_rows: usize,
        w2_cols: usize,
        h: &[f64],
        h_rows: usize,
        h_cols: usize,
        win_size: usize,
        fft_size: usize,
        hop_size: usize,
        assign: bool,
    ) -> Result<(), &'static str> {
        if w1.len() != w1_rows * w1_cols {
            return Err("w1 slice length does not match w1_rows * w1_cols");
        }
        if w2.len() != w2_rows * w2_cols {
            return Err("w2 slice length does not match w2_rows * w2_cols");
        }
        if h.len() != h_rows * h_cols {
            return Err("h slice length does not match h_rows * h_cols");
        }
        if w1_rows != w2_rows {
            return Err("w1 and w2 must have the same number of rows (rank)");
        }
        if h_rows != w1_rows {
            return Err("h_rows must equal rank (w1_rows)");
        }
        if win_size == 0 {
            return Err("win_size must be > 0");
        }
        if fft_size < win_size {
            return Err("fft_size must be >= win_size");
        }
        if hop_size == 0 {
            return Err("hop_size must be > 0");
        }
        nmf_morph_init(
            self.inner,
            w1.as_ptr(),
            w1_rows as isize,
            w1_cols as isize,
            w2.as_ptr(),
            w2_rows as isize,
            w2_cols as isize,
            h.as_ptr(),
            h_rows as isize,
            h_cols as isize,
            win_size as isize,
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
    /// Generate one morphed complex-spectrum frame.
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

    fn make_identity_bases(rank: usize, n_bins: usize) -> Vec<f64> {
        // Simple bases: each row is a uniform spectrum scaled by row index+1
        let mut w = vec![0.0f64; rank * n_bins];
        for r in 0..rank {
            for c in 0..n_bins {
                w[r * n_bins + c] = (r + 1) as f64 / n_bins as f64;
            }
        }
        w
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
        let h = vec![0.5f64; rank * n_frames];

        m.init(
            &w1, rank, n_bins, &w2, rank, n_bins, &h, rank, n_frames, win_size, fft_size, hop_size,
            false,
        )
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
        let h = vec![1.0f64; rank * n_frames];

        m.init(
            &w, rank, n_bins, &w, rank, n_bins, &h, rank, n_frames, fft_size, fft_size, 256, false,
        )
        .unwrap();

        // Should be able to call more frames than n_frames (cycles modulo)
        for _ in 0..(n_frames * 2) {
            let out = m.process_frame(0.0, 0);
            assert_eq!(out.len(), n_bins);
        }
    }
}
