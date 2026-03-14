//! Safe Rust bindings for [flucoma-core](https://github.com/flucoma/flucoma-core)
//! audio analysis algorithms.
//!
//! ## Examples
//!
//! ### Loudness
//!
//! ```rust,no_run
//! use flucoma_rs::analyzation::Loudness;
//!
//! let mut analyzer = Loudness::new(1024, 44100.0).unwrap();
//!
//! let frame = vec![0.0f64; 1024]; // fill with audio samples
//!
//! let result = analyzer.process_frame(
//!     &frame,
//!     true, // k_weighting
//!     true, // true_peak
//! );
//!
//! println!("Loudness: {:.1} dBFS", result.loudness_db);
//! println!("Peak:     {:.1} dBFS", result.peak_db);
//! ```
//!
//! ### Onset Detection
//!
//!```rust,no_run
//! use flucoma_rs::analyzation::{Onset, OnsetFunction};
//!
//! let window_size = 1024;
//! let filter_size = 5;
//! let mut odf = Onset::new(window_size, window_size, filter_size).unwrap();
//!
//! // Feed silent frame to seed history
//! let silence = vec![0.0f64; window_size];
//! let frame_delta = 0;
//! let _ = odf.process_frame(
//!   &silence, OnsetFunction::PowerSpectrum, filter_size, frame_delta);
//!
//! // Then feed a frame with audio
//! let mut audio_frame = vec![0.0f64; window_size];
//! audio_frame[512] = 1.0;
//! let value = odf.process_frame(
//!   &audio_frame, OnsetFunction::PowerSpectrum, filter_size, 0);
//!
//! println!("Onset value: {:.4}", value);
//!```

mod amp_feature;
mod amp_seg;
mod audio_transport;
mod bufstats;
mod loudness;
mod matrix;
mod mel_bands;
mod multi_stats;
mod nmf_filter;
mod nmf_morph;
mod normalize;
mod novelty_feature;
mod novelty_seg;
mod onset;
mod onset_seg;
mod running_stats;
mod sine;
mod stft;
mod transient_seg;

/// Raw data processing and helper types.
pub mod data {
    pub use super::bufstats::{BufStats, BufStatsConfig};
    pub use super::matrix::Matrix;
    pub use super::multi_stats::{
        MultiStats, MultiStatsConfig, MultiStatsOutput, MultiStatsValues,
    };
    pub use super::normalize::Normalize;
    pub use super::running_stats::RunningStats;
}

/// Fast Fourier transform types and functions.
pub mod fourier {
    pub use super::stft::{ComplexSpectrum, Istft, Stft, WindowType};
    pub use num_complex::Complex64 as Complex;
}

/// Audio feature extraction.
pub mod analyzation {
    pub use super::amp_feature::AmpFeature;
    pub use super::loudness::Loudness;
    pub use super::mel_bands::MelBands;
    pub use super::novelty_feature::Novelty;
    pub use super::onset::{Onset, OnsetFunction};
    pub use super::sine::{Sine, SortBy};
}

/// Spectral transformation.
pub mod transformation {
    pub use super::audio_transport::AudioTransport;
    pub use super::nmf_filter::{NMFFilter, NmfResult};
    pub use super::nmf_morph::NMFMorph;
}

/// Onset segmentation.
pub mod segmentation {
    pub use super::amp_seg::AmpSegmentation;
    pub use super::novelty_seg::NoveltySegmentation;
    pub use super::onset_seg::OnsetSegmentation;
    pub use super::transient_seg::TransientSegmentation;
}
