use flucoma_sys::{amp_seg_create, amp_seg_destroy, amp_seg_init, amp_seg_process_sample};

// -------------------------------------------------------------------------------------------------

/// Amplitude-envelope-based audio segmenter, operating sample by sample.
///
/// Two-phase setup:
/// 1. [`AmpSegmentation::new`] -- allocates and initialises the follower.
/// 2. Call [`AmpSegmentation::process_sample`] per audio sample.
///
/// Uses a dual-ramp envelope follower with separate fast/slow attack and
/// release times, plus on/off thresholds for hysteresis.
///
/// See <https://learn.flucoma.org/reference/ampslice>
pub struct AmpSegmentation {
    inner: *mut u8,
}

unsafe impl Send for AmpSegmentation {}

impl AmpSegmentation {
    /// Create and initialise an envelope segmenter.
    ///
    /// # Arguments
    /// * `floor`        - Noise floor in dB. Signals below this level are ignored.
    /// * `hi_pass_freq` - Hi-pass filter frequency in Hz applied before the follower.
    ///
    /// # Errors
    /// Returns an error string if allocation fails.
    pub fn new(floor: f64, hi_pass_freq: f64) -> Result<Self, &'static str> {
        let inner = amp_seg_create();
        if inner.is_null() {
            return Err("failed to create AmpSegmentation instance");
        }
        amp_seg_init(inner, floor, hi_pass_freq);
        Ok(Self { inner })
    }

    /// Process a single audio sample.
    ///
    /// # Arguments
    /// * `sample`        - The audio sample value.
    /// * `on_threshold`  - dB level above which an onset is declared.
    /// * `off_threshold` - dB level below which the gate closes (must be <= `on_threshold`).
    /// * `floor`         - Noise floor in dB.
    /// * `fast_ramp_up`  - Fast attack time in samples.
    /// * `slow_ramp_up`  - Slow attack time in samples.
    /// * `fast_ramp_down`- Fast release time in samples.
    /// * `slow_ramp_down`- Slow release time in samples.
    /// * `hi_pass_freq`  - Hi-pass filter frequency in Hz.
    /// * `debounce`      - Minimum samples between successive onsets.
    ///
    /// Returns 1.0 on an onset event, 0.0 otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn process_sample(
        &mut self,
        sample: f64,
        on_threshold: f64,
        off_threshold: f64,
        floor: f64,
        fast_ramp_up: usize,
        slow_ramp_up: usize,
        fast_ramp_down: usize,
        slow_ramp_down: usize,
        hi_pass_freq: f64,
        debounce: usize,
    ) -> f64 {
        amp_seg_process_sample(
            self.inner,
            sample,
            on_threshold,
            off_threshold,
            floor,
            fast_ramp_up as isize,
            slow_ramp_up as isize,
            fast_ramp_down as isize,
            slow_ramp_down as isize,
            hi_pass_freq,
            debounce as isize,
        )
    }
}

impl Drop for AmpSegmentation {
    fn drop(&mut self) {
        amp_seg_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_seg_silence_returns_zero() {
        let mut seg = AmpSegmentation::new(-60.0, 20.0).unwrap();
        for _ in 0..100 {
            let val = seg.process_sample(0.0, -30.0, -40.0, -60.0, 10, 100, 10, 100, 20.0, 10);
            assert_eq!(val, 0.0, "silence should produce 0.0, got {val}");
        }
    }

    #[test]
    fn env_seg_loud_signal_can_trigger() {
        let mut seg = AmpSegmentation::new(-60.0, 20.0).unwrap();
        let silence = vec![0.0f64; 10];
        let peak: Vec<f64> = (0..10).map(|i| ((10 - i) as f64) * 0.1).collect();
        let mut triggered = false;
        for i in 0..100 {
            let buffer = if i % 20 < 10 { &silence } else { &peak };
            let frame = buffer[i % 20 % buffer.len()];
            let val = seg.process_sample(frame, -10.0, -40.0, -60.0, 1, 2, 2, 4, 20.0, 1);
            if val == 1.0 {
                triggered = true;
                break;
            }
        }
        assert!(triggered, "loud signal should eventually trigger an onset");
    }
}
