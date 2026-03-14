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

/// Seven summary statistics for one derivative order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiStatsValues {
    pub mean: f64,
    pub std: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub low: f64,
    pub mid: f64,
    pub high: f64,
}

impl MultiStatsValues {
    pub(crate) fn from_slice(slice: &[f64]) -> Self {
        Self {
            mean: slice[0],
            std: slice[1],
            skew: slice[2],
            kurtosis: slice[3],
            low: slice[4],
            mid: slice[5],
            high: slice[6],
        }
    }

    pub(crate) fn zero() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            skew: 0.0,
            kurtosis: 0.0,
            low: 0.0,
            mid: 0.0,
            high: 0.0,
        }
    }
}

/// Per-channel output of `MultiStats`/`BufStats`, with optional derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiStatsOutput {
    pub stats: MultiStatsValues,
    pub derivative_1: Option<MultiStatsValues>,
    pub derivative_2: Option<MultiStatsValues>,
}

pub(crate) fn zero_outputs(
    num_channels: usize,
    num_derivatives: u8,
) -> Vec<MultiStatsOutput> {
    let zero = MultiStatsValues::zero();
    let channel = MultiStatsOutput {
        stats: zero,
        derivative_1: (num_derivatives >= 1).then_some(zero),
        derivative_2: (num_derivatives >= 2).then_some(zero),
    };
    vec![channel; num_channels]
}

pub(crate) fn outputs_from_raw(
    raw: &[f64],
    num_channels: usize,
    num_derivatives: u8,
) -> Vec<MultiStatsOutput> {
    let values_per_channel = STATS_PER_DERIVATIVE * (num_derivatives as usize + 1);
    raw.chunks_exact(values_per_channel)
        .take(num_channels)
        .map(|channel| MultiStatsOutput {
            stats: MultiStatsValues::from_slice(&channel[0..7]),
            derivative_1: (num_derivatives >= 1)
                .then(|| MultiStatsValues::from_slice(&channel[7..14])),
            derivative_2: (num_derivatives >= 2)
                .then(|| MultiStatsValues::from_slice(&channel[14..21])),
        })
        .collect()
}

/// Computes summary statistics over channel-major multi-channel data.
///
/// Input layout is channel-major:
/// `[channel0_frames..., channel1_frames..., ...]`.
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

    /// Compute summary statistics over channel-major input data.
    ///
    /// `input` layout is `[channel0_frames..., channel1_frames..., ...]`.
    pub fn process(
        &mut self,
        input: &[f64],
        num_frames: usize,
        num_channels: usize,
        weights: Option<&[f64]>,
    ) -> Result<Vec<MultiStatsOutput>, &'static str> {
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
        if let Some(weight_slice) = weights {
            if weight_slice.len() != num_frames {
                return Err("weights length must equal num_frames");
            }
            if !weight_slice.iter().copied().any(|value| value > 0.0) {
                return Ok(zero_outputs(num_channels, self.config.num_derivatives));
            }
        }

        multistats_init(
            self.inner,
            self.config.num_derivatives as FlucomaIndex,
            self.config.low_percentile,
            self.config.middle_percentile,
            self.config.high_percentile,
        );

        let values_per_channel = STATS_PER_DERIVATIVE * (self.config.num_derivatives as usize + 1);
        let mut raw = vec![0.0; num_channels * values_per_channel];
        let (weights_ptr, weights_len) = match weights {
            Some(weight_slice) => (weight_slice.as_ptr(), weight_slice.len() as FlucomaIndex),
            None => (std::ptr::null(), 0),
        };
        multistats_process(
            self.inner,
            input.as_ptr(),
            num_channels as FlucomaIndex,
            num_frames as FlucomaIndex,
            raw.as_mut_ptr(),
            values_per_channel as FlucomaIndex,
            self.config.outliers_cutoff.unwrap_or(-1.0),
            weights_ptr,
            weights_len,
        );

        Ok(outputs_from_raw(
            &raw,
            num_channels,
            self.config.num_derivatives,
        ))
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
        let mut multi_stats = MultiStats::new(MultiStatsConfig::default()).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let channels = multi_stats.process(&input, 4, 1, None).unwrap();
        assert_eq!(channels.len(), 1);
        assert!((channels[0].stats.mean - 2.5).abs() < 1e-12);
        assert!(channels[0].stats.std.is_finite() && channels[0].stats.std > 0.0);
    }

    #[test]
    fn derivatives_are_exposed_as_optional_structs() {
        let config = MultiStatsConfig {
            num_derivatives: 2,
            ..MultiStatsConfig::default()
        };
        let mut multi_stats = MultiStats::new(config).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let channels = multi_stats.process(&input, 4, 1, None).unwrap();
        let output = &channels[0];
        assert!((output.stats.mean - 2.5).abs() < 1e-12);
        assert!((output.derivative_1.unwrap().mean - 1.0).abs() < 1e-12);
        assert!(output.derivative_2.unwrap().mean.abs() < 1e-12);
    }

    #[test]
    fn zero_weights_return_zeroed_outputs() {
        let mut multi_stats = MultiStats::new(MultiStatsConfig::default()).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [0.0, 0.0, 0.0, 0.0];
        let channels = multi_stats.process(&input, 4, 1, Some(&weights)).unwrap();
        assert_eq!(channels[0].stats, MultiStatsValues::zero());
        assert!(channels[0].derivative_1.is_none());
        assert!(channels[0].derivative_2.is_none());
    }
}
