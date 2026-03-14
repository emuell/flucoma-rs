use flucoma_sys::{nmf_create, nmf_destroy, nmf_process};

use crate::matrix::Matrix;
use crate::nmf_filter::NmfResult;

// -------------------------------------------------------------------------------------------------

/// Offline Non-negative Matrix Factorization decomposition.
///
/// Factorises a magnitude spectrogram `V` (shape `n_frames × n_bins`) into:
/// - **W** (bases, shape `rank × n_bins`) -- spectral dictionary templates
/// - **H** (activations, shape `n_frames × rank`) -- per-frame component weights
/// - **V̂** (estimate, shape `n_frames × n_bins`) -- reconstruction ≈ H · W
///
/// Unlike [`crate::transformation::NMFFilter`] which applies a fixed dictionary
/// frame-by-frame, this struct performs the full offline factorization with control
/// over which matrices to update.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::decomposition::Nmf;
/// use flucoma_rs::data::Matrix;
///
/// let mut nmf = Nmf::new().unwrap();
/// let spectrogram = Matrix::from_vec(vec![0.1f64; 16 * 65], 16, 65).unwrap();
/// let result = nmf.process(&spectrogram, 4, 100, true, true, -1);
/// assert_eq!(result.bases.rows(), 4);
/// ```
///
/// See <https://learn.flucoma.org/reference/bufnmf>
pub struct Nmf {
    inner: *mut u8,
}

unsafe impl Send for Nmf {}

impl Nmf {
    /// Create a new NMF processor.
    ///
    /// # Errors
    /// Returns an error string if allocation fails.
    pub fn new() -> Result<Self, &'static str> {
        let inner = nmf_create();
        if inner.is_null() {
            return Err("failed to create NMF instance");
        }
        Ok(Self { inner })
    }

    /// Offline batch NMF decomposition of a magnitude spectrogram.
    ///
    /// Factorises `spectrogram` (shape `n_frames × n_bins`) into bases W and
    /// activations H using multiplicative update rules.
    ///
    /// # Arguments
    /// * `spectrogram`  - Row-major magnitude spectrogram, shape `n_frames × n_bins`.
    ///   Values should be non-negative (use magnitude spectra, not complex).
    /// * `rank`         - Number of NMF components. Must be > 0.
    /// * `n_iterations` - Number of multiplicative-update iterations.
    /// * `update_w`     - Whether to update the bases matrix W during iteration.
    /// * `update_h`     - Whether to update the activations matrix H during iteration.
    /// * `random_seed`  - Seed for random initialisation. Use -1 for a random seed.
    ///
    /// Returns an [`NmfResult`] with shapes:
    /// - `bases`:       `rank × n_bins`
    /// - `activations`: `n_frames × rank`
    /// - `estimate`:    `n_frames × n_bins`
    ///
    /// # Panics
    /// Panics if `rank == 0`.
    pub fn process(
        &mut self,
        spectrogram: &Matrix,
        rank: usize,
        n_iterations: usize,
        update_w: bool,
        update_h: bool,
        random_seed: i64,
    ) -> NmfResult {
        assert!(rank > 0, "rank must be > 0");
        let n_frames = spectrogram.rows();
        let n_bins = spectrogram.cols();
        let mut w = vec![0.0f64; rank * n_bins];
        let mut h = vec![0.0f64; n_frames * rank];
        let mut v = vec![0.0f64; n_frames * n_bins];
        nmf_process(
            self.inner,
            spectrogram.data().as_ptr(),
            n_frames as isize,
            n_bins as isize,
            w.as_mut_ptr(),
            h.as_mut_ptr(),
            v.as_mut_ptr(),
            rank as isize,
            n_iterations.max(1) as isize,
            update_w,
            update_h,
            random_seed as isize,
        );
        NmfResult {
            bases: Matrix::from_vec(w, rank, n_bins).unwrap(),
            activations: Matrix::from_vec(h, n_frames, rank).unwrap(),
            estimate: Matrix::from_vec(v, n_frames, n_bins).unwrap(),
        }
    }
}

impl Drop for Nmf {
    fn drop(&mut self) {
        nmf_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nmf_process_returns_correct_shapes() {
        let n_bins = 65usize;
        let n_frames = 16usize;
        let rank = 3usize;
        let mut nmf = Nmf::new().unwrap();
        let spectrogram =
            Matrix::from_vec(vec![0.1f64; n_frames * n_bins], n_frames, n_bins).unwrap();
        let result = nmf.process(&spectrogram, rank, 10, true, true, -1);
        assert_eq!(result.bases.rows(), rank);
        assert_eq!(result.bases.cols(), n_bins);
        assert_eq!(result.activations.rows(), n_frames);
        assert_eq!(result.activations.cols(), rank);
        assert_eq!(result.estimate.rows(), n_frames);
        assert_eq!(result.estimate.cols(), n_bins);
    }

    #[test]
    fn nmf_process_update_w_only() {
        let n_bins = 33usize;
        let n_frames = 8usize;
        let rank = 2usize;
        let mut nmf = Nmf::new().unwrap();
        let spectrogram =
            Matrix::from_vec(vec![0.5f64; n_frames * n_bins], n_frames, n_bins).unwrap();
        let result = nmf.process(&spectrogram, rank, 5, true, false, 42);
        assert_eq!(result.bases.rows(), rank);
        assert_eq!(result.activations.rows(), n_frames);
        for &v in result.bases.data() {
            assert!(v.is_finite(), "bases must be finite, got {v}");
        }
    }

    #[test]
    fn nmf_process_update_h_only() {
        let n_bins = 33usize;
        let n_frames = 8usize;
        let rank = 2usize;
        let mut nmf = Nmf::new().unwrap();
        let spectrogram =
            Matrix::from_vec(vec![0.5f64; n_frames * n_bins], n_frames, n_bins).unwrap();
        let result = nmf.process(&spectrogram, rank, 5, false, true, 42);
        assert_eq!(result.activations.cols(), rank);
        for &v in result.activations.data() {
            assert!(v.is_finite(), "activations must be finite, got {v}");
        }
    }
}
