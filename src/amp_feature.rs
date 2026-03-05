use flucoma_sys::{
    amp_feature_create, amp_feature_destroy, amp_feature_init, amp_feature_process_sample,
};

// -------------------------------------------------------------------------------------------------

/// Amplitude envelope follower.
///
/// Two-phase setup:
/// 1. [`AmpFeature::new`] -- allocates and initialises the follower.
/// 2. Call [`AmpFeature::process_sample`] per audio sample.
///
/// Returns the difference between a fast and slow envelope tracker, giving a
/// measure of amplitude change rather than absolute level. A hi-pass filter
/// can be applied before envelope detection to focus on transient content.
///
/// See <https://learn.flucoma.org/reference/ampfeature>
pub struct AmpFeature {
    inner: *mut u8,
}

unsafe impl Send for AmpFeature {}

impl AmpFeature {
    /// Create and initialise an amplitude envelope follower.
    ///
    /// # Arguments
    /// * `floor`        - Noise floor in dB; amplitudes below this are clamped.
    /// * `hi_pass_freq` - Hi-pass filter cutoff in Hz (0.0 to disable).
    ///
    /// # Errors
    /// Returns an error string if allocation fails.
    pub fn new(floor: f64, hi_pass_freq: f64) -> Result<Self, &'static str> {
        if hi_pass_freq < 0.0 {
            return Err("hi_pass_freq must be >= 0.0 (use 0.0 to disable)");
        }
        let inner = amp_feature_create();
        if inner.is_null() {
            return Err("failed to create AmpFeature instance");
        }
        amp_feature_init(inner, floor, hi_pass_freq);
        Ok(Self { inner })
    }

    /// Process one audio sample and return the envelope value.
    ///
    /// # Arguments
    /// * `input`         - Audio sample.
    /// * `floor`         - Noise floor in dB.
    /// * `fast_ramp_up`  - Rise time for the fast envelope tracker (samples).
    /// * `slow_ramp_up`  - Rise time for the slow envelope tracker (samples).
    /// * `fast_ramp_down`- Fall time for the fast envelope tracker (samples).
    /// * `slow_ramp_down`- Fall time for the slow envelope tracker (samples).
    /// * `hi_pass_freq`  - Hi-pass filter cutoff in Hz (0.0 to disable).
    ///
    /// Returns `fast_envelope - slow_envelope` in dB.
    #[allow(clippy::too_many_arguments)]
    pub fn process_sample(
        &mut self,
        input: f64,
        floor: f64,
        fast_ramp_up: usize,
        slow_ramp_up: usize,
        fast_ramp_down: usize,
        slow_ramp_down: usize,
        hi_pass_freq: f64,
    ) -> f64 {
        assert!(
            hi_pass_freq >= 0.0,
            "hi_pass_freq must be >= 0.0, got {hi_pass_freq}"
        );
        amp_feature_process_sample(
            self.inner,
            input,
            floor,
            fast_ramp_up as isize,
            slow_ramp_up as isize,
            fast_ramp_down as isize,
            slow_ramp_down as isize,
            hi_pass_freq,
        )
    }
}

impl Drop for AmpFeature {
    fn drop(&mut self) {
        amp_feature_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amp_feature_silence_returns_finite_value() {
        let mut af = AmpFeature::new(-60.0, 0.0).unwrap();
        let val = af.process_sample(0.0, -60.0, 10, 100, 10, 100, 0.0);
        assert!(
            val.is_finite(),
            "expected finite value for silence, got {val}"
        );
    }

    #[test]
    fn amp_feature_impulse_produces_response() {
        let mut af = AmpFeature::new(-60.0, 0.0).unwrap();
        // Warm up with silence
        for _ in 0..100 {
            af.process_sample(0.0, -60.0, 10, 100, 10, 100, 0.0);
        }
        // Impulse
        let val = af.process_sample(1.0, -60.0, 10, 100, 10, 100, 0.0);
        assert!(
            val > 0.0,
            "expected positive envelope response to impulse, got {val}"
        );
    }
}
