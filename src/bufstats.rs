use flucoma_sys::{
    multistats_create, multistats_destroy, multistats_init, multistats_process, FlucomaIndex,
};

use crate::{
    data::MultiStatsValues,
    multi_stats::{outputs_from_raw, zero_outputs, MultiStatsOutput},
};

// -------------------------------------------------------------------------------------------------

/// Configuration for [`BufStats`].
#[derive(Debug, Clone)]
pub struct BufStatsConfig {
    /// First frame index of the analysis window (0-based). Default: `0`.
    pub start_frame: usize,
    /// Number of frames to analyse. `None` selects all frames from `start_frame`. Default: `None`.
    pub num_frames: Option<usize>,
    /// First channel index of the analysis window (0-based). Default: `0`.
    pub start_channel: usize,
    /// Number of channels to analyse. `None` selects all channels from `start_channel`. Default: `None`.
    pub num_channels: Option<usize>,
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

impl Default for BufStatsConfig {
    fn default() -> Self {
        Self {
            start_frame: 0,
            num_frames: None,
            start_channel: 0,
            num_channels: None,
            num_derivatives: 0,
            low_percentile: 0.0,
            middle_percentile: 50.0,
            high_percentile: 100.0,
            outliers_cutoff: None,
        }
    }
}

impl BufStatsConfig {
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

/// Offline summary statistics over a selected region of a multichannel buffer.
///
/// This wrapper mirrors the buffer-slicing behaviour of FluCoMa `BufStats`:
/// it selects a frame range and channel range from the source buffer and then
/// returns one [`MultiStatsOutput`] per selected channel.
///
/// Unlike [`MultiStats`](crate::data::MultiStats), which always analyses the whole input
/// region passed to it, `BufStats` carries the slice selection in
/// [`BufStatsConfig`].
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{BufStats, BufStatsConfig};
///
/// let mut stats = BufStats::new(BufStatsConfig::default()).unwrap();
/// let source = vec![1.0f64, 2.0, 3.0, 4.0];
/// let channels = stats.process(&source, 4, 1, None).unwrap();
/// assert_eq!(channels.len(), 1);
/// println!("mean = {}", channels[0].stats.mean);
/// ```
///
/// See <https://learn.flucoma.org/reference/bufstats>
///
/// Input layout is interleaved:
/// `[channel0_frames..., channel1_frames..., ...]`.
pub struct BufStats {
    inner: *mut u8,
    config: BufStatsConfig,
}

unsafe impl Send for BufStats {}

impl BufStats {
    /// Create a new `BufStats` processor with the given slicing/statistics
    /// configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid or if the underlying
    /// FluCoMa instance cannot be allocated.
    pub fn new(config: BufStatsConfig) -> Result<Self, &'static str> {
        config.validate()?;
        let inner = multistats_create();
        if inner.is_null() {
            return Err("failed to create MultiStats instance");
        }
        Ok(Self { inner, config })
    }

    /// Access to the current configuration.
    pub fn config(&self) -> &BufStatsConfig {
        &self.config
    }

    /// Replace the current configuration.
    ///
    /// # Errors
    /// Returns an error if the new configuration is invalid.
    pub fn set_config(&mut self, config: BufStatsConfig) -> Result<(), &'static str> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Compute summary statistics over a selected region of an interleaved
    /// multichannel buffer.
    ///
    /// `source` layout is `[channel0_frames..., channel1_frames..., ...]`.
    ///
    /// # Arguments
    /// * `source` - Interleaved samples for all channels.
    /// * `source_num_frames` - Number of frames per channel in `source`.
    /// * `source_num_channels` - Number of channels in `source`.
    /// * `weights` - Optional per-frame weights for the selected frame span.
    ///
    /// # Errors
    /// Returns an error if the source shape is inconsistent, the configured
    /// frame/channel selection is out of range, or the weights length does not
    /// match the selected frame span.
    pub fn process(
        &mut self,
        source: &[f64],
        source_num_frames: usize,
        source_num_channels: usize,
        weights: Option<&[f64]>,
    ) -> Result<Vec<MultiStatsOutput>, &'static str> {
        if source_num_frames == 0 {
            return Err("source_num_frames must be > 0");
        }
        if source_num_channels == 0 {
            return Err("source_num_channels must be > 0");
        }
        if source.len() != source_num_frames * source_num_channels {
            return Err("source length does not match source_num_frames * source_num_channels");
        }

        let start_frame = self.config.start_frame;
        if start_frame >= source_num_frames {
            return Err("start_frame out of range");
        }
        let selected_num_frames = self
            .config
            .num_frames
            .unwrap_or(source_num_frames.saturating_sub(start_frame));
        if selected_num_frames == 0 {
            return Err("selected frame span must be > 0");
        }
        if start_frame + selected_num_frames > source_num_frames {
            return Err("start_frame + num_frames out of range");
        }
        if selected_num_frames <= self.config.num_derivatives as usize {
            return Err("selected frame span must be > num_derivatives");
        }

        let start_channel = self.config.start_channel;
        if start_channel >= source_num_channels {
            return Err("start_channel out of range");
        }
        let selected_num_channels = self
            .config
            .num_channels
            .unwrap_or(source_num_channels.saturating_sub(start_channel));
        if selected_num_channels == 0 {
            return Err("selected channel count must be > 0");
        }
        if start_channel + selected_num_channels > source_num_channels {
            return Err("start_channel + num_channels out of range");
        }

        let mut selected_source = vec![0.0; selected_num_channels * selected_num_frames];
        for channel in 0..selected_num_channels {
            let source_channel = start_channel + channel;
            let src_start = source_channel * source_num_frames + start_frame;
            let src_end = src_start + selected_num_frames;
            let dst_start = channel * selected_num_frames;
            let dst_end = dst_start + selected_num_frames;
            selected_source[dst_start..dst_end].copy_from_slice(&source[src_start..src_end]);
        }

        if let Some(weight_slice) = weights {
            if weight_slice.len() != selected_num_frames {
                return Err("weights length must match selected frame span");
            }
            if !weight_slice.iter().copied().any(|value| value > 0.0) {
                return Ok(zero_outputs(
                    selected_num_channels,
                    self.config.num_derivatives,
                ));
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
        let mut raw = vec![0.0; selected_num_channels * values_per_channel];
        let (weights_ptr, weights_len) = match weights {
            Some(weight_slice) => (weight_slice.as_ptr(), weight_slice.len() as FlucomaIndex),
            None => (std::ptr::null(), 0),
        };
        multistats_process(
            self.inner,
            selected_source.as_ptr(),
            selected_num_channels as FlucomaIndex,
            selected_num_frames as FlucomaIndex,
            raw.as_mut_ptr(),
            values_per_channel as FlucomaIndex,
            self.config.outliers_cutoff.unwrap_or(-1.0),
            weights_ptr,
            weights_len,
        );

        Ok(outputs_from_raw(
            &raw,
            selected_num_channels,
            self.config.num_derivatives,
        ))
    }
}

impl Drop for BufStats {
    fn drop(&mut self) {
        multistats_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_matches_expected() {
        let mut stats = BufStats::new(BufStatsConfig::default()).unwrap();
        let source = vec![1.0, 2.0, 3.0, 4.0];
        let channels = stats.process(&source, 4, 1, None).unwrap();
        assert_eq!(channels.len(), 1);
        assert!((channels[0].stats.mean - 2.5).abs() < 1e-12);
    }

    #[test]
    fn derivatives_are_exposed_in_output_structs() {
        let config = BufStatsConfig {
            num_derivatives: 2,
            ..BufStatsConfig::default()
        };
        let mut stats = BufStats::new(config).unwrap();
        let source = vec![1.0, 2.0, 3.0, 4.0];
        let channels = stats.process(&source, 4, 1, None).unwrap();
        let output = &channels[0];
        assert!((output.stats.mean - 2.5).abs() < 1e-12);
        assert!((output.derivative_1.unwrap().mean - 1.0).abs() < 1e-12);
        assert!(output.derivative_2.unwrap().mean.abs() < 1e-12);
    }

    #[test]
    fn weights_influence_mean() {
        let mut stats = BufStats::new(BufStatsConfig::default()).unwrap();
        let source = vec![0.0, 10.0];
        let weights = vec![0.9, 0.1];
        let channels = stats.process(&source, 2, 1, Some(&weights)).unwrap();
        assert!((channels[0].stats.mean - 1.0).abs() < 1e-9);
    }

    #[test]
    fn non_positive_weights_return_zeroed_outputs() {
        let config = BufStatsConfig {
            num_derivatives: 1,
            ..BufStatsConfig::default()
        };
        let mut stats = BufStats::new(config).unwrap();
        let source = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.0, -1.0, 0.0, -2.0];
        let channels = stats.process(&source, 4, 1, Some(&weights)).unwrap();
        assert_eq!(channels[0].stats.mean, 0.0);
        assert_eq!(channels[0].stats.std, 0.0);
        assert_eq!(channels[0].derivative_1.unwrap().mean, 0.0);
    }
}
