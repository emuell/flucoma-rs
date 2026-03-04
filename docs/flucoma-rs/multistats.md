# MultiStats

Module path: `flucoma_rs::data::{MultiStats, MultiStatsConfig, MultiStatsOutput}`

## Types

- `MultiStatsConfig`
- `MultiStatsOutput`
- `MultiStats`

## API

```rust
pub fn new(config: MultiStatsConfig) -> Result<MultiStats, &'static str>;
pub fn config(&self) -> &MultiStatsConfig;
pub fn set_config(&mut self, config: MultiStatsConfig) -> Result<(), &'static str>;

pub fn process(
    &mut self,
    input: &[f64],
    num_frames: usize,
    num_channels: usize,
    weights: Option<&[f64]>,
) -> Result<MultiStatsOutput, &'static str>;

pub fn values_per_channel(&self) -> usize;
```

## Output API

```rust
pub fn values(&self) -> &[f64];
pub fn num_channels(&self) -> usize;
pub fn values_per_channel(&self) -> usize;
pub fn channel(&self, channel: usize) -> Option<&[f64]>;

// Helpers to fetch specific statistics from raw data:
// (derivative 0 = data, 1 = first derivative, 2 = second derivative)
pub fn mean(&self, channel: usize, derivative: usize) -> Option<f64>;
pub fn std(&self, channel: usize, derivative: usize) -> Option<f64>;
pub fn skew(&self, channel: usize, derivative: usize) -> Option<f64>;
pub fn kurt(&self, channel: usize, derivative: usize) -> Option<f64>;
pub fn low(&self, channel: usize, derivative: usize) -> Option<f64>;
pub fn mid(&self, channel: usize, derivative: usize) -> Option<f64>;
pub fn high(&self, channel: usize, derivative: usize) -> Option<f64>;
```

## Notes

- Input layout is channel-major: `[ch0_frames..., ch1_frames..., ...]`.

## FluCoMa Reference Notes (Archival)

Source basis: `learn-website/src/routes/(content)/reference/bufstats/+page.svx`.

- MultiStats-style summaries provide compact descriptor aggregation over frames.
- Order statistics (low/middle/high percentiles) are robust alternatives to mean-centric summaries.
- Outlier control and derivative summaries are central to stable dataset-level features.
