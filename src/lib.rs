//! Safe Rust bindings for [flucoma-core](https://github.com/flucoma/flucoma-core)
//! audio analysis algorithms.

mod audio_transport;
mod envelope_seg;
mod loudness;
mod mel_bands;
mod novelty_seg;
mod onset;
mod onset_seg;
mod stft;
mod transient_seg;

pub mod analyzation {
    pub use super::loudness::Loudness;
    pub use super::mel_bands::MelBands;
    pub use super::onset::{OnsetDetectionFunctions, OnsetFunction};
    pub use super::stft::{ComplexSpectrum, Istft, Stft, WindowType};
}

pub mod transformation {
    pub use super::audio_transport::AudioTransport;
}

pub mod segmentation {
    pub use super::envelope_seg::EnvelopeSegmentation;
    pub use super::novelty_seg::NoveltySegmentation;
    pub use super::onset_seg::OnsetSegmentation;
    pub use super::transient_seg::TransientSegmentation;
}
