use flucoma_sys::{
    multistats_create, multistats_destroy, multistats_init, multistats_process, FlucomaIndex,
};

// -------------------------------------------------------------------------------------------------

/// Offline summary statistics over multi-channel time-series data.
///
/// Computes the standard FluCoMa statistics (mean, standard deviation, skew,
/// kurtosis, and three percentile values) for each channel, optionally also
/// summarising the first and second derivatives.
///
/// The result contains one [`MultiStatsOutput`] per channel.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{MultiStats, MultiStatsConfig};
///
/// let mut stats = MultiStats::new(MultiStatsConfig::default()).unwrap();
/// let input = vec![1.0f64, 2.0, 3.0, 4.0];
/// let channels = stats.process(&input, 4, 1, None).unwrap();
/// assert_eq!(channels.len(), 1);
/// println!("mean = {}", channels[0].stats.mean);
/// ```
///
/// See <https://learn.flucoma.org/reference/bufstats>
///
/// Input layout is interleaved:
/// `[channel0_frames..., channel1_frames..., ...]`.
pub struct MultiStats {
    inner: *mut u8,
    config: MultiStatsConfig,
}

// -------------------------------------------------------------------------------------------------

/// Seven summary statistics for one derivative order.
///
/// Each value stores the standard FluCoMa summary descriptors for one signal
/// stream or derivative stream:
/// - `mean`
/// - `std`
/// - `skew`
/// - `kurtosis`
/// - `low`
/// - `mid`
/// - `high`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiStatsValues {
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub std: f64,
    /// Skewness (third standardised moment).
    pub skew: f64,
    /// Excess kurtosis (fourth standardised moment).
    pub kurtosis: f64,
    /// Low percentile value (controlled by [`MultiStatsConfig::low_percentile`]).
    pub low: f64,
    /// Middle percentile value (controlled by [`MultiStatsConfig::middle_percentile`]).
    pub mid: f64,
    /// High percentile value (controlled by [`MultiStatsConfig::high_percentile`]).
    pub high: f64,
}

impl MultiStatsValues {
    pub const NUM_VALUES: usize = 7;

    /// Create new stats from a f64 slice with at least 7 values.
    pub fn from_slice(slice: &[f64]) -> Self {
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

    /// Create new zeroed out, empty stats.
    pub fn zero() -> Self {
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

// -------------------------------------------------------------------------------------------------

/// Configuration for [`MultiStats`].
#[derive(Debug, Clone)]
pub struct MultiStatsConfig {
    /// Number of temporal derivatives to summarise in addition to the signal itself. In `[0, 2]`.
    pub num_derivatives: u8,
    /// Low percentile value. Must be in `[0, 100]` and ≤ `middle_percentile`.
    pub low_percentile: f64,
    /// Middle (median) percentile value. Must be in `[0, 100]`.
    pub middle_percentile: f64,
    /// High percentile value. Must be in `[0, 100]` and ≥ `middle_percentile`.
    pub high_percentile: f64,
    /// If `Some(z)`, values further than `z` standard deviations from the mean are excluded.
    /// `None` disables outlier removal.
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

impl MultiStatsConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_derivatives > 2 {
            return Err("num_derivatives must be in [0, 2]");
        }
        if !(0.0..=100.0).contains(&self.low_percentile) {
            return Err("low_percentile must be in [0, 100]");
        }
        if !(0.0..=100.0).contains(&self.middle_percentile) {
            return Err("middle_percentile must be in [0, 100]");
        }
        if !(0.0..=100.0).contains(&self.high_percentile) {
            return Err("high_percentile must be in [0, 100]");
        }
        if self.low_percentile > self.middle_percentile {
            return Err("low_percentile must be <= middle_percentile");
        }
        if self.middle_percentile > self.high_percentile {
            return Err("middle_percentile must be <= high_percentile");
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------------

/// Per-channel output of [`MultiStats`] and [`BufStats`](crate::data::BufStats).
///
/// `stats` contains the summary values for the original signal. When
/// derivatives are enabled in the configuration, `derivative_1` and
/// `derivative_2` contain the same seven-value summary for the first and
/// second temporal derivative respectively.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiStatsOutput {
    pub stats: MultiStatsValues,
    pub derivative_1: Option<MultiStatsValues>,
    pub derivative_2: Option<MultiStatsValues>,
}

pub(crate) fn outputs_from_raw(
    raw: &[f64],
    num_channels: usize,
    num_derivatives: u8,
) -> Vec<MultiStatsOutput> {
    let values_per_channel = MultiStatsValues::NUM_VALUES * (num_derivatives as usize + 1);
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

pub(crate) fn zero_outputs(num_channels: usize, num_derivatives: u8) -> Vec<MultiStatsOutput> {
    let zero = MultiStatsValues::zero();
    let channel = MultiStatsOutput {
        stats: zero,
        derivative_1: (num_derivatives >= 1).then_some(zero),
        derivative_2: (num_derivatives >= 2).then_some(zero),
    };
    vec![channel; num_channels]
}

unsafe impl Send for MultiStats {}

// -------------------------------------------------------------------------------------------------

impl MultiStats {
    /// Create a new `MultiStats` processor with the given configuration.
    ///
    /// # Errors
    /// Returns an error if the percentile or derivative settings are invalid,
    /// or if the underlying FluCoMa instance cannot be allocated.
    pub fn new(config: MultiStatsConfig) -> Result<Self, &'static str> {
        config.validate()?;
        let inner = multistats_create();
        if inner.is_null() {
            return Err("failed to create MultiStats instance");
        }
        Ok(Self { inner, config })
    }

    /// Access to the current configuration.
    pub fn config(&self) -> &MultiStatsConfig {
        &self.config
    }

    /// Replace the current configuration.
    ///
    /// # Errors
    /// Returns an error if the new configuration is invalid.
    pub fn set_config(&mut self, config: MultiStatsConfig) -> Result<(), &'static str> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Compute summary statistics over an interleaved multi-channel buffer.
    ///
    /// `input` layout is `[channel0_frames..., channel1_frames..., ...]`.
    ///
    /// # Arguments
    /// * `input` - Interleaved samples for all channels.
    /// * `num_frames` - Number of frames per channel.
    /// * `num_channels` - Number of channels in `input`.
    /// * `weights` - Optional per-frame weights, shared across channels.
    ///
    /// # Errors
    /// Returns an error if the input shape is inconsistent, if
    /// `num_frames <= num_derivatives`, or if the weight vector length does
    /// not match `num_frames`.
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

        let values_per_channel =
            MultiStatsValues::NUM_VALUES * (self.config.num_derivatives as usize + 1);
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

// -------------------------------------------------------------------------------------------------

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
