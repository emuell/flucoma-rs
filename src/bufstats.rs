use flucoma_sys::{
    multistats_create, multistats_destroy, multistats_init, multistats_process, FlucomaIndex,
};

const STATS_PER_DERIVATIVE: usize = 7;

/// Statistics emitted by BufStats/MultiStats in fixed order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufStat {
    Mean,
    Std,
    Skew,
    Kurtosis,
    Low,
    Mid,
    High,
}

impl BufStat {
    const fn index(self) -> usize {
        match self {
            Self::Mean => 0,
            Self::Std => 1,
            Self::Skew => 2,
            Self::Kurtosis => 3,
            Self::Low => 4,
            Self::Mid => 5,
            Self::High => 6,
        }
    }
}

/// Selection mask for BufStats output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufStatsSelect {
    mask: [bool; STATS_PER_DERIVATIVE],
}

impl Default for BufStatsSelect {
    fn default() -> Self {
        Self {
            mask: [true; STATS_PER_DERIVATIVE],
        }
    }
}

impl BufStatsSelect {
    /// Select all statistics.
    pub fn all() -> Self {
        Self::default()
    }

    /// Build a selection from an explicit list of statistics.
    pub fn from_stats(stats: &[BufStat]) -> Self {
        let mut mask = [false; STATS_PER_DERIVATIVE];
        for stat in stats.iter().copied() {
            mask[stat.index()] = true;
        }
        Self { mask }
    }

    fn selected_count(&self) -> usize {
        self.mask.iter().filter(|&&enabled| enabled).count()
    }
}

/// Configuration for [`BufStats`].
#[derive(Debug, Clone)]
pub struct BufStatsConfig {
    pub start_frame: usize,
    pub num_frames: Option<usize>,
    pub start_channel: usize,
    pub num_channels: Option<usize>,
    pub select: BufStatsSelect,
    pub num_derivatives: u8,
    pub low_percentile: f64,
    pub middle_percentile: f64,
    pub high_percentile: f64,
    pub outliers_cutoff: Option<f64>,
}

impl Default for BufStatsConfig {
    fn default() -> Self {
        Self {
            start_frame: 0,
            num_frames: None,
            start_channel: 0,
            num_channels: None,
            select: BufStatsSelect::default(),
            num_derivatives: 0,
            low_percentile: 0.0,
            middle_percentile: 50.0,
            high_percentile: 100.0,
            outliers_cutoff: None,
        }
    }
}

/// Channel-major (`[channel0_frames..., channel1_frames..., ...]`) BufStats output.
#[derive(Debug, Clone)]
pub struct BufStatsOutput {
    values: Vec<f64>,
    num_channels: usize,
    values_per_channel: usize,
}

impl BufStatsOutput {
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
}

/// BufStats-style offline statistics wrapper built on `MultiStats`.
pub struct BufStats {
    inner: *mut u8,
    config: BufStatsConfig,
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for BufStats {}

impl BufStats {
    pub fn new(config: BufStatsConfig) -> Result<Self, &'static str> {
        validate_config(&config)?;
        let inner = multistats_create();
        if inner.is_null() {
            return Err("failed to create MultiStats instance");
        }
        Ok(Self { inner, config })
    }

    pub fn config(&self) -> &BufStatsConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: BufStatsConfig) -> Result<(), &'static str> {
        validate_config(&config)?;
        self.config = config;
        Ok(())
    }

    /// Process a channel-major source buffer.
    ///
    /// `source` layout is `[channel0_frames..., channel1_frames..., ...]` where each
    /// channel has `source_num_frames` contiguous samples.
    /// `weights`, if provided, must match the selected frame span length.
    pub fn process(
        &mut self,
        source: &[f64],
        source_num_frames: usize,
        source_num_channels: usize,
        weights: Option<&[f64]>,
    ) -> Result<BufStatsOutput, &'static str> {
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

        let selected_per_derivative = self.config.select.selected_count();
        if selected_per_derivative == 0 {
            return Err("select must enable at least one statistic");
        }
        let values_per_channel =
            selected_per_derivative * (self.config.num_derivatives as usize + 1);

        let mut selected_source = vec![0.0; selected_num_channels * selected_num_frames];
        for ch in 0..selected_num_channels {
            let src_ch = start_channel + ch;
            let src_start = src_ch * source_num_frames + start_frame;
            let src_end = src_start + selected_num_frames;
            let dst_start = ch * selected_num_frames;
            let dst_end = dst_start + selected_num_frames;
            selected_source[dst_start..dst_end].copy_from_slice(&source[src_start..src_end]);
        }

        let weights_are_all_non_positive = weights
            .map(|w| {
                if w.len() != selected_num_frames {
                    return Err("weights length must match selected frame span");
                }
                Ok(!w.iter().copied().any(|v| v > 0.0))
            })
            .transpose()?
            .unwrap_or(false);

        if weights_are_all_non_positive {
            return Ok(BufStatsOutput {
                values: vec![0.0; selected_num_channels * values_per_channel],
                num_channels: selected_num_channels,
                values_per_channel,
            });
        }

        multistats_init(
            self.inner,
            self.config.num_derivatives as FlucomaIndex,
            self.config.low_percentile,
            self.config.middle_percentile,
            self.config.high_percentile,
        );

        let full_values_per_channel =
            STATS_PER_DERIVATIVE * (self.config.num_derivatives as usize + 1);
        let mut full_output = vec![0.0; selected_num_channels * full_values_per_channel];

        let (weights_ptr, weights_len) = match weights {
            Some(w) => (w.as_ptr(), w.len() as FlucomaIndex),
            None => (std::ptr::null(), 0),
        };
        let outliers_cutoff = self.config.outliers_cutoff.unwrap_or(-1.0);
        multistats_process(
            self.inner,
            selected_source.as_ptr(),
            selected_num_channels as FlucomaIndex,
            selected_num_frames as FlucomaIndex,
            full_output.as_mut_ptr(),
            full_values_per_channel as FlucomaIndex,
            outliers_cutoff,
            weights_ptr,
            weights_len,
        );

        let mut selected_output = vec![0.0; selected_num_channels * values_per_channel];
        for ch in 0..selected_num_channels {
            let full_start = ch * full_values_per_channel;
            let full_end = full_start + full_values_per_channel;
            let full_channel = &full_output[full_start..full_end];
            let mut write_idx = ch * values_per_channel;
            for (idx, value) in full_channel.iter().copied().enumerate() {
                if self.config.select.mask[idx % STATS_PER_DERIVATIVE] {
                    selected_output[write_idx] = value;
                    write_idx += 1;
                }
            }
        }

        Ok(BufStatsOutput {
            values: selected_output,
            num_channels: selected_num_channels,
            values_per_channel,
        })
    }
}

impl Drop for BufStats {
    fn drop(&mut self) {
        multistats_destroy(self.inner);
    }
}

fn validate_config(config: &BufStatsConfig) -> Result<(), &'static str> {
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
    fn mean_only_matches_expected() {
        let config = BufStatsConfig {
            select: BufStatsSelect::from_stats(&[BufStat::Mean]),
            ..BufStatsConfig::default()
        };
        let mut stats = BufStats::new(config).unwrap();
        let source = vec![1.0, 2.0, 3.0, 4.0];
        let output = stats.process(&source, 4, 1, None).unwrap();
        assert_eq!(output.values(), &[2.5]);
    }

    #[test]
    fn derivative_means_follow_expected_order() {
        let config = BufStatsConfig {
            select: BufStatsSelect::from_stats(&[BufStat::Mean]),
            num_derivatives: 2,
            ..BufStatsConfig::default()
        };
        let mut stats = BufStats::new(config).unwrap();
        let source = vec![1.0, 2.0, 3.0, 4.0];
        let output = stats.process(&source, 4, 1, None).unwrap();
        let values = output.values();
        assert!(
            (values[0] - 2.5).abs() < 1e-12,
            "unexpected d0 mean: {}",
            values[0]
        );
        assert!(
            (values[1] - 1.0).abs() < 1e-12,
            "unexpected d1 mean: {}",
            values[1]
        );
        assert!(values[2].abs() < 1e-12, "unexpected d2 mean: {}", values[2]);
    }

    #[test]
    fn weights_influence_mean() {
        let config = BufStatsConfig {
            select: BufStatsSelect::from_stats(&[BufStat::Mean]),
            ..BufStatsConfig::default()
        };
        let mut stats = BufStats::new(config).unwrap();
        let source = vec![0.0, 10.0];
        let weights = vec![0.9, 0.1];
        let output = stats.process(&source, 2, 1, Some(&weights)).unwrap();
        assert!((output.values()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn non_positive_weights_return_zeros() {
        let config = BufStatsConfig {
            select: BufStatsSelect::from_stats(&[BufStat::Mean, BufStat::Std]),
            num_derivatives: 1,
            ..BufStatsConfig::default()
        };
        let mut stats = BufStats::new(config).unwrap();
        let source = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.0, -1.0, 0.0, -2.0];
        let output = stats.process(&source, 4, 1, Some(&weights)).unwrap();
        assert_eq!(output.values(), &[0.0, 0.0, 0.0, 0.0]);
    }
}
