use flucoma_sys::{nmf_create, nmf_destroy, nmf_process, nmf_process_frame};

use crate::matrix::Matrix;

// -------------------------------------------------------------------------------------------------

/// Result of an offline NMF batch decomposition via [`NMFFilter::process`].
pub struct NmfResult {
    /// Bases matrix W, shape `rank × n_bins`.
    pub bases: Matrix,
    /// Activations matrix H, shape `n_frames × rank`.
    pub activations: Matrix,
    /// Reconstruction matrix V ≈ H · W, shape `n_frames × n_bins`.
    pub estimate: Matrix,
}

// -------------------------------------------------------------------------------------------------

/// Real-time NMF filtering: compute per-frame activations of a fixed spectral dictionary.
///
/// Given a magnitude spectrum and a pre-trained bases matrix W (rank × n_bins),
/// estimates the activation vector H for that frame and the reconstructed magnitude
/// estimate V ≈ H · W.
///
/// # Setup
/// 1. Train NMF offline (e.g. via a full-spectrogram decomposition) to obtain bases W.
/// 2. Create `NMFFilter::new(n_bins, rank)`.
/// 3. Per frame: call `process_frame` with the current magnitude spectrum and W.
///
/// See <https://learn.flucoma.org/reference/nmffilter>
pub struct NMFFilter {
    inner: *mut u8,
    n_bins: usize,
    rank: usize,
    /// `[activations (rank) | estimate (n_bins)]`
    buf: Vec<f64>,
}

unsafe impl Send for NMFFilter {}

impl NMFFilter {
    /// Create a new NMFFilter.
    ///
    /// # Arguments
    /// * `n_bins` - Number of FFT bins (magnitude spectrum length). Must be > 0.
    /// * `rank`   - Number of NMF components / dictionary columns. Must be > 0.
    pub fn new(n_bins: usize, rank: usize) -> Result<Self, &'static str> {
        if n_bins == 0 {
            return Err("n_bins must be > 0");
        }
        if rank == 0 {
            return Err("rank must be > 0");
        }
        let inner = nmf_create();
        if inner.is_null() {
            return Err("failed to create NMF instance");
        }
        Ok(Self {
            inner,
            n_bins,
            rank,
            buf: vec![0.0f64; rank + n_bins],
        })
    }

    /// Process one magnitude-spectrum frame against a fixed bases matrix.
    ///
    /// # Arguments
    /// * `magnitudes`   - Magnitude spectrum of length `n_bins`.
    /// * `bases`        - Bases matrix of shape `rank × n_bins`.
    /// * `n_iterations` - Number of multiplicative-update NMF iterations (≥ 1).
    /// * `random_seed`  - Seed for random initialisation of activations. Use -1 for random.
    ///
    /// Returns `(activations, estimate)`:
    /// - `activations`: length `rank` — how strongly each basis is active this frame.
    /// - `estimate`:    length `n_bins` — reconstructed magnitude spectrum.
    ///
    /// Both slices are backed by an internal buffer valid until the next call.
    ///
    /// # Panics
    /// Panics if `magnitudes.len() != n_bins`, or `bases` shape is not `rank × n_bins`.
    pub fn process_frame<'a>(
        &'a mut self,
        magnitudes: &[f64],
        bases: &Matrix,
        n_iterations: usize,
        random_seed: i64,
    ) -> (&'a [f64], &'a [f64]) {
        assert_eq!(
            magnitudes.len(),
            self.n_bins,
            "magnitudes length ({}) must equal n_bins ({})",
            magnitudes.len(),
            self.n_bins
        );
        assert_eq!(
            bases.rows(),
            self.rank,
            "bases rows ({}) must equal rank ({})",
            bases.rows(),
            self.rank
        );
        assert_eq!(
            bases.cols(),
            self.n_bins,
            "bases cols ({}) must equal n_bins ({})",
            bases.cols(),
            self.n_bins
        );
        let n_iter = n_iterations.max(1) as isize;
        nmf_process_frame(
            self.inner,
            magnitudes.as_ptr(),
            self.n_bins as isize,
            bases.data().as_ptr(),
            self.rank as isize,
            self.n_bins as isize,
            self.buf.as_mut_ptr(),
            unsafe { self.buf.as_mut_ptr().add(self.rank) },
            n_iter,
            random_seed as isize,
        );
        let (activations, estimate) = self.buf.split_at(self.rank);
        (activations, estimate)
    }

    /// Offline batch NMF decomposition of a full magnitude spectrogram.
    ///
    /// Factorises `spectrogram` (shape `n_frames × n_bins`) into bases W and
    /// activations H using `rank` components.
    ///
    /// # Arguments
    /// * `spectrogram`  - Row-major magnitude spectrogram, shape `n_frames × n_bins`.
    /// * `rank`         - Number of NMF components (> 0).
    /// * `n_iterations` - Number of multiplicative-update iterations.
    /// * `random_seed`  - Seed for random initialisation. Use -1 for random.
    ///
    /// Returns an [`NmfResult`] whose matrices have shapes:
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
            true, // update_w
            true, // update_h
            random_seed as isize,
        );
        NmfResult {
            bases: Matrix::from_vec(w, rank, n_bins).unwrap(),
            activations: Matrix::from_vec(h, n_frames, rank).unwrap(),
            estimate: Matrix::from_vec(v, n_frames, n_bins).unwrap(),
        }
    }

    /// Number of FFT bins this filter was created for.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Number of NMF components.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl Drop for NMFFilter {
    fn drop(&mut self) {
        nmf_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nmf_filter_zero_magnitudes_produce_finite_output() {
        let n_bins = 513usize;
        let rank = 4usize;
        let mut f = NMFFilter::new(n_bins, rank).unwrap();

        let magnitudes = vec![0.0f64; n_bins];
        let bases = Matrix::from_vec(vec![1.0f64; rank * n_bins], rank, n_bins).unwrap();

        let (activations, estimate) = f.process_frame(&magnitudes, &bases, 10, -1);
        assert_eq!(activations.len(), rank);
        assert_eq!(estimate.len(), n_bins);
        for &v in activations {
            assert!(v.is_finite(), "activation must be finite, got {v}");
        }
        for &v in estimate {
            assert!(v.is_finite(), "estimate must be finite, got {v}");
        }
    }

    #[test]
    fn nmf_filter_output_has_correct_lengths() {
        let n_bins = 257usize;
        let rank = 8usize;
        let mut f = NMFFilter::new(n_bins, rank).unwrap();
        let magnitudes: Vec<f64> = (0..n_bins).map(|i| i as f64 / n_bins as f64).collect();
        let bases = Matrix::from_vec(vec![0.5f64; rank * n_bins], rank, n_bins).unwrap();
        let (activations, estimate) = f.process_frame(&magnitudes, &bases, 5, 42);
        assert_eq!(activations.len(), rank);
        assert_eq!(estimate.len(), n_bins);
    }

    #[test]
    fn nmf_filter_process_returns_correct_shapes() {
        let n_bins = 65usize;
        let n_frames = 16usize;
        let rank = 3usize;
        let mut f = NMFFilter::new(n_bins, rank).unwrap();
        let spectrogram =
            Matrix::from_vec(vec![0.1f64; n_frames * n_bins], n_frames, n_bins).unwrap();
        let result = f.process(&spectrogram, rank, 10, -1);
        assert_eq!(result.bases.rows(), rank);
        assert_eq!(result.bases.cols(), n_bins);
        assert_eq!(result.activations.rows(), n_frames);
        assert_eq!(result.activations.cols(), rank);
        assert_eq!(result.estimate.rows(), n_frames);
        assert_eq!(result.estimate.cols(), n_bins);
    }
}
