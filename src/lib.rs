#![cfg_attr(docsrs, feature(doc_cfg))]
//! Safe Rust bindings for [FluCoMa](https://www.flucoma.org/) (Fluid Corpus Manipulation),
//! a set of C++ audio analysis and transformation algorithms developed for creative
//! music applications. The underlying C++ library ([flucoma-core](https://github.com/flucoma/flucoma-core))
//! covers spectral analysis, source separation, feature extraction, and event segmentation.
//!
//! This crate exposes those algorithms through idiomatic Rust wrappers with owned types,
//! result-based error handling, and no unsafe code in user-facing APIs. Note that FluCoMa's
//! machine learning and statistical tools (such as PCA, KMeans, KDTree, etc.)
//! are **not wrapped on purpose**. There are excellent Rust-native libraries available for these
//! purposes and they can be use in combination with the tools provided here.
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
mod hpss;
mod loudness;
mod matrix;
mod mel_bands;
mod nmf;
mod nmf_filter;
mod nmf_morph;
mod novelty_feature;
mod novelty_seg;
mod onset;
mod onset_seg;
mod sine;
mod sine_extraction;
mod stft;
mod transient_extraction;
mod transient_seg;

/// Core matrix types and traits used throughout the FluCoMa algorithms.
///
/// # ndarray interop
/// Enable the `ndarray` feature to pass [`ndarray::Array2<f64>`] directly to
/// any algorithm that accepts [`AsMatrixView`](data::AsMatrixView) or
/// [`AsMatrixViewMut`](data::AsMatrixViewMut), and to convert [`Matrix`](data::Matrix)
/// results back to `ndarray::Array2` with `.into()`.
pub mod data {
    pub use super::matrix::{AsMatrixView, AsMatrixViewMut, Matrix, MatrixView, MatrixViewMut};
    #[cfg(feature = "ndarray")]
    pub use ::ndarray;
}

/// STFT / ISTFT — convert audio frames to/from complex spectra.
pub mod fourier {
    pub use super::stft::{ComplexSpectrum, Istft, Stft, WindowType};
    pub use num_complex::Complex64 as Complex;
}

/// Per-frame feature extraction: loudness, onsets, mel bands, partials, novelty.
pub mod analyzation {
    pub use super::amp_feature::AmpFeature;
    pub use super::loudness::Loudness;
    pub use super::mel_bands::MelBands;
    pub use super::novelty_feature::Novelty;
    pub use super::onset::{Onset, OnsetFunction};
    pub use super::sine::{Sine, SortBy};
}

/// Spectral transformation: NMF morphing and audio transport.
pub mod transformation {
    pub use super::audio_transport::AudioTransport;
    pub use super::nmf_filter::{NMFFilter, NmfResult};
    pub use super::nmf_morph::NMFMorph;
}

/// Source separation: HPSS, NMF, sinusoidal extraction, transient extraction.
pub mod decomposition {
    pub use super::hpss::{Hpss, HpssMode, HpssParams};
    pub use super::nmf::Nmf;
    pub use super::nmf_filter::NmfResult;
    pub use super::sine_extraction::{SineExtraction, SineExtractionParams};
    pub use super::transient_extraction::TransientExtraction;
}

/// Onset / event segmentation: amplitude, novelty, onset, transient slicers.
pub mod segmentation {
    pub use super::amp_seg::AmpSlice;
    pub use super::novelty_seg::NoveltySlice;
    pub use super::onset_seg::OnsetSlice;
    pub use super::transient_seg::TransientSlice;
}
