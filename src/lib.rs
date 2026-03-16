#![cfg_attr(docsrs, feature(doc_cfg))]
//! Safe Rust bindings for [FluCoMa](https://www.flucoma.org/) (Fluid Corpus Manipulation),
//! a set of C++ audio analysis and transformation algorithms developed for creative
//! music applications.
//!
//! The underlying C++ library ([flucoma-core](https://github.com/flucoma/flucoma-core))
//! covers spectral analysis, source separation, feature extraction, and event segmentation.
//! This crate exposes those algorithms through idiomatic Rust wrappers with owned types,
//! `Result`-based error handling, and no unsafe code in user-facing APIs.

mod amp_feature;
mod amp_seg;
mod audio_transport;
mod bufstats;
mod dataset_query;
mod grid;
mod hpss;
mod kdtree;
mod kmeans;
mod loudness;
mod matrix;
mod mds;
mod mel_bands;
mod multi_stats;
mod nmf;
mod nmf_filter;
mod nmf_morph;
mod normalize;
mod novelty_feature;
mod novelty_seg;
mod onset;
mod onset_seg;
mod pca;
mod robust_scale;
mod running_stats;
mod sine;
mod sine_extraction;
mod standardize;
mod stft;
mod transient_extraction;
mod transient_seg;

/// Low-level data processing, machine learning, and other shared data types.
///
/// # ndarray interop
/// Enable the `ndarray` feature to pass [`ndarray::Array2<f64>`] directly to
/// any algorithm that accepts [`AsMatrixView`](data::AsMatrixView) or [`AsMatrixViewMut`](data::AsMatrixViewMut),
/// and to convert [`Matrix`](data::Matrix) results back to `ndarray::Array2` with `.into()`.
pub mod data {
    pub use super::bufstats::{BufStats, BufStatsConfig};
    pub use super::dataset_query::{
        DataSetComparisonOp, DataSetQuery, DataSetQueryCondition, DataSetQueryResult,
    };
    pub use super::grid::Grid;
    pub use super::kdtree::KDTree;
    pub use super::kmeans::{KMeans, KMeansConfig, KMeansInit, KMeansResult, SKMeans};
    pub use super::matrix::{AsMatrixView, AsMatrixViewMut, Matrix, MatrixView, MatrixViewMut};
    pub use super::mds::{Mds, MdsDistance};
    pub use super::multi_stats::{
        MultiStats, MultiStatsConfig, MultiStatsOutput, MultiStatsValues,
    };
    pub use super::normalize::Normalize;
    pub use super::pca::{Pca, PcaConfig, PcaScaler};
    pub use super::robust_scale::RobustScale;
    pub use super::running_stats::RunningStats;
    pub use super::standardize::Standardize;

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
