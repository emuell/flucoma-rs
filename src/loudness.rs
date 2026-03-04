use flucoma_sys::{loudness_create, loudness_destroy, loudness_init, loudness_process_frame};

// -------------------------------------------------------------------------------------------------

/// Loudness measurement result (EBU R128-style).
#[derive(Debug, Clone, Copy)]
pub struct LoudnessResult {
    /// Integrated loudness in dBFS (K-weighted if enabled).
    pub loudness_db: f64,
    /// Peak level in dBFS (true peak if enabled, otherwise absolute max).
    pub peak_db: f64,
}

// -------------------------------------------------------------------------------------------------

/// Measures loudness and peak level of audio frames.
///
/// See <https://learn.flucoma.org/reference/loudness>
pub struct Loudness {
    inner: *mut u8,
    frame_size: usize,
}

unsafe impl Send for Loudness {}

impl Loudness {
    /// Create and fully initialise a Loudness analyser.
    ///
    /// # Arguments
    /// * `frame_size`  - Number of samples per frame (also the max size).
    /// * `sample_rate` - Audio sample rate in Hz.
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid.
    pub fn new(frame_size: usize, sample_rate: f64) -> Result<Self, &'static str> {
        if frame_size == 0 {
            return Err("frame_size must be > 0");
        }
        if sample_rate <= 0.0 {
            return Err("sample_rate must be > 0");
        }
        let inner = loudness_create(frame_size as isize);
        if inner.is_null() {
            return Err("failed to create Loudness instance");
        }
        loudness_init(inner, frame_size as isize, sample_rate);
        Ok(Self { inner, frame_size })
    }

    /// Process a single audio frame.
    ///
    /// # Arguments
    /// * `input`      - Audio samples; must have exactly `frame_size` elements.
    /// * `k_weighting` - Apply K-weighting filter (as per EBU R128).
    /// * `true_peak`  - Use true peak detection (interpolated); otherwise
    ///   reports the absolute maximum sample.
    ///
    /// # Panics
    /// Panics if `input.len() != frame_size`.
    pub fn process_frame(
        &mut self,
        input: &[f64],
        k_weighting: bool,
        true_peak: bool,
    ) -> LoudnessResult {
        assert_eq!(
            input.len(),
            self.frame_size,
            "input length ({}) must equal frame_size ({})",
            input.len(),
            self.frame_size
        );
        let mut out = [0.0f64; 2];
        loudness_process_frame(
            self.inner,
            input.as_ptr(),
            input.len() as isize,
            out.as_mut_ptr(),
            k_weighting,
            true_peak,
        );
        LoudnessResult {
            loudness_db: out[0],
            peak_db: out[1],
        }
    }

    /// Analysis frame size in samples.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }
}

impl Drop for Loudness {
    fn drop(&mut self) {
        loudness_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loudness_silent_frame() {
        let mut l = Loudness::new(1024, 44100.0).unwrap();
        let silence = vec![0.0f64; 1024];
        let r = l.process_frame(&silence, true, true);
        // Silence produces a very low (negative) loudness value
        assert!(r.loudness_db < -60.0, "loudness_db = {}", r.loudness_db);
        assert!(r.peak_db < -60.0, "peak_db = {}", r.peak_db);
    }

    #[test]
    fn loudness_sine_frame() {
        use std::f64::consts::PI;
        let n = 1024usize;
        let sine: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let mut l = Loudness::new(n, 44100.0).unwrap();
        let r = l.process_frame(&sine, false, false);
        // 0 dBFS sine should land around -3 dB loudness and 0 dB peak
        assert!(
            r.loudness_db > -10.0 && r.loudness_db < 0.0,
            "loudness_db = {}",
            r.loudness_db
        );
        assert!(
            r.peak_db > -3.0 && r.peak_db < 1.0,
            "peak_db = {}",
            r.peak_db
        );
    }
}
