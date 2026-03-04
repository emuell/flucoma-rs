use flucoma_sys::{
    multistats_create, multistats_destroy, multistats_init, multistats_process, FlucomaIndex,
};

const STATS_PER_DERIVATIVE: usize = 7;

/// Configuration for [`MultiStats`].
#[derive(Debug, Clone)]
pub struct MultiStatsConfig {
    pub num_derivatives: u8,
    pub low_percentile: f64,
    pub middle_percentile: f64,
    pub high_percentile: f64,
    pub outliers_cutoff: Option<f64>,
}

impl Default for MultiStatsConfig {
    fn default() -> Self {
        Self {
            num_derivatives: 0,
            low_percentile: 0.0,
            middle_percentile: 50.0,
            high_percentile: 100.0,
            outliers_cutoff: None,
        }
    }
}

/// Full MultiStats output in channel-major layout.
///
/// For each channel, the output is:
/// `[mean, std, skew, kurtosis, low, mid, high]` for derivative 0,
/// followed by derivative 1 (if enabled), then derivative 2 (if enabled).
#[derive(Debug, Clone)]
pub struct MultiStatsOutput {
    values: Vec<f64>,
    num_channels: usize,
    values_per_channel: usize,
}

impl MultiStatsOutput {
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn values_per_channel(&self) -> usize {
        self.values_per_channel
    }

    pub fn channel(&self, channel: usize) -> Option<&[f64]> {
        if channel >= self.num_channels {
            return None;
        }
        let start = channel * self.values_per_channel;
        let end = start + self.values_per_channel;
        self.values.get(start..end)
    }

    fn get_stat(&self, channel: usize, derivative: usize, stat_index: usize) -> Option<f64> {
        let channel_data = self.channel(channel)?;
        let idx = derivative * STATS_PER_DERIVATIVE + stat_index;
        channel_data.get(idx).copied()
    }

    pub fn mean(&self, channel: usize, derivative: usize) -> Option<f64> {
        self.get_stat(channel, derivative, 0)
    }

    pub fn std(&self, channel: usize, derivative: usize) -> Option<f64> {
        self.get_stat(channel, derivative, 1)
    }

    pub fn skew(&self, channel: usize, derivative: usize) -> Option<f64> {
        self.get_stat(channel, derivative, 2)
    }

    pub fn kurt(&self, channel: usize, derivative: usize) -> Option<f64> {
        self.get_stat(channel, derivative, 3)
    }

    pub fn low(&self, channel: usize, derivative: usize) -> Option<f64> {
        self.get_stat(channel, derivative, 4)
    }

    pub fn mid(&self, channel: usize, derivative: usize) -> Option<f64> {
        self.get_stat(channel, derivative, 5)
    }

    pub fn high(&self, channel: usize, derivative: usize) -> Option<f64> {
        self.get_stat(channel, derivative, 6)
    }
}

/// Thin safe wrapper for `flucoma::algorithm::MultiStats`.
pub struct MultiStats {
    inner: *mut u8,
    config: MultiStatsConfig,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for MultiStats {}

impl MultiStats {
    pub fn new(config: MultiStatsConfig) -> Result<Self, &'static str> {
        validate_config(&config)?;
        let inner = multistats_create();
        if inner.is_null() {
            return Err("failed to create MultiStats instance");
        }
        Ok(Self { inner, config })
    }

    pub fn config(&self) -> &MultiStatsConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: MultiStatsConfig) -> Result<(), &'static str> {
        validate_config(&config)?;
        self.config = config;
        Ok(())
    }

    /// Process channel-major input (`[channel0_frames..., channel1_frames..., ...]`).
    pub fn process(
        &mut self,
        input: &[f64],
        num_frames: usize,
        num_channels: usize,
        weights: Option<&[f64]>,
    ) -> Result<MultiStatsOutput, &'static str> {
        if num_frames == 0 {
            return Err("num_frames must be > 0");
        }
        if num_channels == 0 {
            return Err("num_channels must be > 0");
        }
        if input.len() != num_frames * num_channels {
            return Err("input length does not match num_frames * num_channels");
        }
        if num_frames <= self.config.num_derivatives as usize {
            return Err("num_frames must be > num_derivatives");
        }
        if let Some(w) = weights {
            if w.len() != num_frames {
                return Err("weights length must equal num_frames");
            }
            if !w.iter().copied().any(|x| x > 0.0) {
                return Ok(MultiStatsOutput {
                    values: vec![0.0; num_channels * self.values_per_channel()],
                    num_channels,
                    values_per_channel: self.values_per_channel(),
                });
            }
        }

        multistats_init(
            self.inner,
            self.config.num_derivatives as FlucomaIndex,
            self.config.low_percentile,
            self.config.middle_percentile,
            self.config.high_percentile,
        );

        let values_per_channel = self.values_per_channel();
        let mut output = vec![0.0; num_channels * values_per_channel];
        let (weights_ptr, weights_len) = match weights {
            Some(w) => (w.as_ptr(), w.len() as FlucomaIndex),
            None => (std::ptr::null(), 0),
        };
        multistats_process(
            self.inner,
            input.as_ptr(),
            num_channels as FlucomaIndex,
            num_frames as FlucomaIndex,
            output.as_mut_ptr(),
            values_per_channel as FlucomaIndex,
            self.config.outliers_cutoff.unwrap_or(-1.0),
            weights_ptr,
            weights_len,
        );

        Ok(MultiStatsOutput {
            values: output,
            num_channels,
            values_per_channel,
        })
    }

    pub fn values_per_channel(&self) -> usize {
        STATS_PER_DERIVATIVE * (self.config.num_derivatives as usize + 1)
    }
}

impl Drop for MultiStats {
    fn drop(&mut self) {
        multistats_destroy(self.inner);
    }
}

fn validate_config(config: &MultiStatsConfig) -> Result<(), &'static str> {
    if config.num_derivatives > 2 {
        return Err("num_derivatives must be in [0, 2]");
    }
    if !(0.0..=100.0).contains(&config.low_percentile) {
        return Err("low_percentile must be in [0, 100]");
    }
    if !(0.0..=100.0).contains(&config.middle_percentile) {
        return Err("middle_percentile must be in [0, 100]");
    }
    if !(0.0..=100.0).contains(&config.high_percentile) {
        return Err("high_percentile must be in [0, 100]");
    }
    if config.low_percentile > config.middle_percentile {
        return Err("low_percentile must be <= middle_percentile");
    }
    if config.middle_percentile > config.high_percentile {
        return Err("middle_percentile must be <= high_percentile");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_mean_std_are_correct_slots() {
        let mut ms = MultiStats::new(MultiStatsConfig::default()).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let out = ms.process(&input, 4, 1, None).unwrap();
        let ch = out.channel(0).unwrap();
        assert!((ch[0] - 2.5).abs() < 1e-12, "mean slot mismatch");
        assert!(ch[1].is_finite() && ch[1] > 0.0, "std slot mismatch");
    }

    #[test]
    fn derivatives_expand_output_width() {
        let cfg = MultiStatsConfig {
            num_derivatives: 2,
            ..MultiStatsConfig::default()
        };
        let mut ms = MultiStats::new(cfg).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let out = ms.process(&input, 4, 1, None).unwrap();
        assert_eq!(out.values_per_channel(), 21);
        let ch = out.channel(0).unwrap();
        assert!((ch[0] - 2.5).abs() < 1e-12);
        assert!((ch[7] - 1.0).abs() < 1e-12);
        assert!(ch[14].abs() < 1e-12);
    }

    #[test]
    fn helper_methods_return_correct_values() {
        let mut ms = MultiStats::new(MultiStatsConfig::default()).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let out = ms.process(&input, 4, 1, None).unwrap();

        assert_eq!(out.mean(0, 0), Some(2.5));
        assert!(out.std(0, 0).unwrap() > 0.0);
        assert!(out.skew(0, 0).is_some());
        assert!(out.kurt(0, 0).is_some());
        assert!(out.low(0, 0).is_some());
        assert!(out.mid(0, 0).is_some());
        assert!(out.high(0, 0).is_some());

        assert_eq!(out.mean(1, 0), None);
        assert_eq!(out.mean(0, 1), None);
    }
}
