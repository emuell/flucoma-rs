use flucoma_sys::{
    running_stats_create, running_stats_destroy, running_stats_init, running_stats_process,
};

// -------------------------------------------------------------------------------------------------

/// Incremental running mean and sample standard deviation.
///
/// Maintains a sliding history of recent vectors and updates the running mean
/// and sample standard deviation for each element position every time a new
/// vector is processed.
///
/// This is the online counterpart to the offline statistics wrappers:
/// [`crate::data::MultiStats`] and [`crate::data::BufStats`].
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::RunningStats;
///
/// let mut stats = RunningStats::new(8, 2).unwrap();
/// let (mean, stddev) = stats.process(&[1.0, 2.0]);
/// assert_eq!(mean.len(), 2);
/// assert_eq!(stddev.len(), 2);
/// ```
///
/// See <https://learn.flucoma.org/reference/runningstats>
pub struct RunningStats {
    inner: *mut u8,
    history_size: usize,
    input_size: usize,
    mean_buf: Vec<f64>,
    stddev_buf: Vec<f64>,
}

unsafe impl Send for RunningStats {}

impl RunningStats {
    /// Create and initialize a running statistics processor.
    ///
    /// # Arguments
    /// * `history_size` - Number of past vectors kept in the running window.
    /// * `input_size` - Length of each processed input vector.
    ///
    /// # Errors
    /// Returns an error if `history_size < 2`, `input_size == 0`, or if the
    /// underlying FluCoMa instance cannot be allocated.
    pub fn new(history_size: usize, input_size: usize) -> Result<Self, &'static str> {
        if history_size < 2 {
            return Err("history_size must be >= 2");
        }
        if input_size == 0 {
            return Err("input_size must be > 0");
        }
        let inner = running_stats_create();
        if inner.is_null() {
            return Err("failed to create RunningStats instance");
        }
        running_stats_init(inner, history_size as isize, input_size as isize);
        Ok(Self {
            inner,
            history_size,
            input_size,
            mean_buf: vec![0.0; input_size],
            stddev_buf: vec![0.0; input_size],
        })
    }

    /// Process one input vector and return `(mean, sample_std_dev)`.
    ///
    /// Returned slices point to internal buffers and are valid until the next call.
    ///
    /// # Panics
    /// Panics if `input.len() != input_size()`.
    pub fn process(&mut self, input: &[f64]) -> (&[f64], &[f64]) {
        assert_eq!(
            input.len(),
            self.input_size,
            "input length ({}) must equal input_size ({})",
            input.len(),
            self.input_size
        );
        running_stats_process(
            self.inner,
            input.as_ptr(),
            self.input_size as isize,
            self.mean_buf.as_mut_ptr(),
            self.stddev_buf.as_mut_ptr(),
        );
        (&self.mean_buf, &self.stddev_buf)
    }

    /// Reset internal history.
    ///
    /// After `clear`, the next call to [`RunningStats::process`] behaves as if
    /// it were the first observation in a new running window.
    pub fn clear(&mut self) {
        running_stats_init(
            self.inner,
            self.history_size as isize,
            self.input_size as isize,
        );
    }

    /// Number of past vectors kept in the running window.
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Length of each processed input vector.
    pub fn input_size(&self) -> usize {
        self.input_size
    }
}

impl Drop for RunningStats {
    fn drop(&mut self) {
        running_stats_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_running_mean_and_std() {
        let mut rs = RunningStats::new(4, 1).unwrap();

        let (m1, s1) = rs.process(&[1.0]);
        assert!((m1[0] - 1.0).abs() < 1e-12);
        assert!(s1[0].abs() < 1e-12);

        let (m2, s2) = rs.process(&[2.0]);
        assert!((m2[0] - 1.5).abs() < 1e-12);
        assert!((s2[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn clear_resets_history() {
        let mut rs = RunningStats::new(8, 2).unwrap();
        let _ = rs.process(&[10.0, -10.0]);
        let _ = rs.process(&[20.0, -20.0]);
        rs.clear();
        let (mean, stddev) = rs.process(&[1.0, 2.0]);
        assert!((mean[0] - 1.0).abs() < 1e-12);
        assert!((mean[1] - 2.0).abs() < 1e-12);
        assert!(stddev[0].abs() < 1e-12);
        assert!(stddev[1].abs() < 1e-12);
    }

    #[test]
    fn nan_input_is_cleaned_to_zero() {
        let mut rs = RunningStats::new(4, 1).unwrap();
        let (mean, stddev) = rs.process(&[f64::NAN]);
        assert!(mean[0].abs() < 1e-12);
        assert!(stddev[0].abs() < 1e-12);
    }
}
