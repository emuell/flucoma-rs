//! Detects onsets in an audio file, computes a mean mel-band vector per slice, and writes up to
//! N timbrally unique slices as individual WAV files.
//!
//! Slices that are too similar to an already-kept slice are skipped.
//!
//! ```sh
//! cargo run --example unique-slices -- input.wav
//! ```
//!
//! Output: `<input_stem>_slices/slice1_<start>_<end>.wav`, etc.

use std::{error::Error, fs::File, path::Path};

use flucoma_rs::{
    analyzation::{MelBands, OnsetFunction},
    fourier::{Stft, WindowType},
    segmentation::OnsetSegmentation,
};

// -------------------------------------------------------------------------------------------------

// Shared STFT config
const HOP_SIZE: usize = 512;
const WINDOW_SIZE: usize = 1024;
const FFT_SIZE: usize = 4096;

// OnsetSegmentation config
const ONSET_FUNCTION: OnsetFunction = OnsetFunction::PowerSpectrum;
const ONSET_THRESHOLD: f64 = 0.0;
const ONSET_DEBOUNCE: usize = 0;
const MIN_SLICE_SAMPLES: usize = 512;

// MelBands config
const FILTER_SIZE: usize = 5;
const NUM_MEL_BANDS: usize = 40;
const MIN_FREQ_HZ: f64 = 20.0;

// Similarity config
const SIMILARITY_THRESHOLD: f64 = 0.15;

// -------------------------------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.is_empty() {
        eprintln!("Usage: unique-slices <input.wav>");
        std::process::exit(1);
    }
    let input_path = Path::new(args.get(1).unwrap().as_str());

    let (header, mono_sample_data) = read_sample_mono(input_path)?;

    let sample_rate = header.sample_rate;
    let channel_count = header.channels as usize;

    println!(
        "Read `{}`: {} samples, {} Hz, {} ch",
        input_path.display(),
        mono_sample_data.len(),
        sample_rate,
        channel_count
    );

    let boundaries = detect_onsets(&mono_sample_data);
    println!("Detected {} onset boundaries", boundaries.len());

    // Build slices from consecutive boundaries; discard slices shorter than MIN_SLICE_SAMPLES
    let slices = collect_slices(mono_sample_data, sample_rate, boundaries);

    println!(
        "{} slices after filtering (min {} samples)",
        slices.len(),
        MIN_SLICE_SAMPLES
    );

    if slices.is_empty() {
        println!("No slices found; nothing to write.");
        return Ok(());
    }

    let selected = deduplicate_slices(&slices);

    // Create output folder
    let input_parent_path = input_path.parent().unwrap_or(Path::new("."));
    let input_file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    let output_path = input_parent_path.join(format!("{}_slices", input_file_stem));
    std::fs::create_dir_all(&output_path)?;

    // Remove previously written slices matching slice*.wav
    if let Ok(entries) = std::fs::read_dir(&output_path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("slice") && name.ends_with(".wav") {
                println!(
                    "Removing existing slice {} from previous run.",
                    entry.path().display()
                );
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    // Read original file as interleaved f32 for slice writing
    let (_, raw_samples) = wav_io::read_from_file(File::open(input_path)?)?;

    println!(
        "\nWriting {} slices to `{}/`:",
        selected.len(),
        output_path.display()
    );

    for (rank, &index) in selected.iter().enumerate() {
        let slice = &slices[index];
        let file_name = format!(
            "slice{:02}_{:08}_{:08}.wav",
            rank + 1,
            slice.start,
            slice.end
        );
        let file_path = output_path.join(&file_name);

        let sample_start = slice.start * channel_count;
        let mut sample_end = (slice.end * channel_count).min(raw_samples.len());
        if (sample_end - sample_start) & 1 == 1 {
            // `wav_io` expects sample counts to be even for padding
            sample_end -= 1;
        }
        let slice_data = raw_samples[sample_start..sample_end].to_vec();

        wav_io::write_to_file(
            &mut File::create(&file_path)?,
            &wav_io::new_header(sample_rate, 16, false, channel_count == 1),
            &slice_data,
        )?;

        println!(
            "  [{}] {} -- samples {}..{} ({:.3}s)",
            rank + 1,
            file_name,
            slice.start,
            slice.end,
            (slice.end - slice.start) as f64 / sample_rate as f64,
        );
    }

    Ok(())
}

// -------------------------------------------------------------------------------------------------

/// Read a WAV file and mix down all channels to mono f64 samples.
fn read_sample_mono(path: &Path) -> Result<(wav_io::header::WavHeader, Vec<f64>), Box<dyn Error>> {
    let (header, data) = wav_io::read_from_file(File::open(path)?)?;

    if header.channels > 1 {
        let mono = data
            .chunks_exact(header.channels as usize)
            .map(|frame| frame.iter().map(|&s| s as f64).sum::<f64>() / header.channels as f64)
            .collect();
        Ok((header, mono))
    } else {
        let mono = data.iter().map(|frame| *frame as f64).collect();
        Ok((header, mono))
    }
}

// -------------------------------------------------------------------------------------------------

/// Run `OnsetSegmentation` hop-by-hop and return a sorted list of sample boundaries.
/// Always includes 0 and `mono.len()` as the outer sentinels.
fn detect_onsets(mono_sample_data: &[f64]) -> Vec<usize> {
    /// Find first zero crossing in the given frame buffer
    fn find_zero_crossing(frame: &[f64]) -> Option<usize> {
        let mut last_s = frame[0];
        for (index, &sample) in frame.iter().enumerate().skip(1) {
            if sample.signum() != last_s.signum() {
                return Some(index);
            }
            last_s = sample;
        }
        None
    }

    let mut segmentation =
        OnsetSegmentation::new(WINDOW_SIZE, FFT_SIZE, FILTER_SIZE).expect("OnsetSegmentation::new");

    let mut boundaries = vec![0usize];
    let mut frame = vec![0.0f64; WINDOW_SIZE];

    let hop_count = mono_sample_data.len().div_ceil(HOP_SIZE);
    for hop in 0..hop_count {
        let start = hop * HOP_SIZE;
        let end = start + WINDOW_SIZE;
        if end > mono_sample_data.len() {
            for i in 0..WINDOW_SIZE {
                frame[i] = if start + i < mono_sample_data.len() {
                    mono_sample_data[start + i]
                } else {
                    0.0
                };
            }
        } else {
            frame.copy_from_slice(&mono_sample_data[start..end]);
        }

        let onset = segmentation.process_frame(
            &frame,
            ONSET_FUNCTION,
            FILTER_SIZE,
            ONSET_THRESHOLD,
            ONSET_DEBOUNCE,
            0,
        );

        if onset == 1.0 {
            if let Some(zero_crossing_offset) = find_zero_crossing(&frame) {
                boundaries.push(start + zero_crossing_offset);
            } else {
                boundaries.push(start);
            }
        }
    }

    boundaries.push(mono_sample_data.len());

    // hop and crossfade lookup may have created duplicated boundaries
    boundaries.sort();
    boundaries.dedup();

    boundaries
}

// -------------------------------------------------------------------------------------------------

struct Slice {
    start: usize,
    end: usize,
    mel: Vec<f64>,
}

/// Compute the mean mel-band vector for a slice `mono[start..end]`.
fn mean_mel(mono_sample_data: &[f64], start: usize, end: usize, sample_rate: u32) -> Vec<f64> {
    let hi_hz = sample_rate as f64 / 2.0;
    let bin_count = FFT_SIZE / 2 + 1;

    let mut stft = Stft::new(WINDOW_SIZE, FFT_SIZE, HOP_SIZE, WindowType::Hann).expect("Stft::new");
    let mut mel = MelBands::new(
        NUM_MEL_BANDS,
        bin_count,
        MIN_FREQ_HZ,
        hi_hz,
        sample_rate as f64,
        WINDOW_SIZE,
    )
    .expect("MelBands::new");

    let mut accumulator = vec![0.0f64; NUM_MEL_BANDS];
    let mut count = 0usize;
    let mut frame = vec![0.0f64; WINDOW_SIZE];

    let slice_len = end.saturating_sub(start);
    let total_hops = slice_len.saturating_sub(WINDOW_SIZE) / HOP_SIZE + 1;

    for hop in 0..total_hops {
        let pos = start + hop * HOP_SIZE;
        for i in 0..WINDOW_SIZE {
            frame[i] = if pos + i < end && pos + i < mono_sample_data.len() {
                mono_sample_data[pos + i]
            } else {
                0.0
            };
        }

        let spec = stft.process_frame(&frame);
        let mags = spec.magnitudes();
        let bands = mel.process_frame(&mags, false, true, false);

        for (a, b) in accumulator.iter_mut().zip(bands.iter()) {
            *a += b;
        }
        count += 1;
    }

    if count > 0 {
        for v in &mut accumulator {
            *v /= count as f64;
        }
    }

    accumulator
}

// Build slices from consecutive boundaries; discard slices shorter than MIN_SLICE_SAMPLES
fn collect_slices(
    mono_sample_data: Vec<f64>,
    sample_rate: u32,
    boundaries: Vec<usize>,
) -> Vec<Slice> {
    boundaries
        .windows(2)
        .filter_map(|w| {
            let (start, end) = (w[0], w[1]);
            if end - start < MIN_SLICE_SAMPLES {
                return None;
            }
            let mel = mean_mel(&mono_sample_data, start, end, sample_rate);
            Some(Slice { start, end, mel })
        })
        .collect()
}

/// Walk in temporal order and skip any slice whose Pearson distance to an already-kept
/// slice is below the threshold.
fn deduplicate_slices(slices: &[Slice]) -> Vec<usize> {
    fn pearson_distance(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        let num: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum();
        let den_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>().sqrt();
        let den_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum::<f64>().sqrt();
        let denom = den_a * den_b;
        if denom < 1e-12 {
            return 1.0; // treat constant vectors as maximally distant
        }
        1.0 - (num / denom).clamp(-1.0, 1.0)
    }

    let mut kept: Vec<usize> = Vec::with_capacity(slices.len());
    for i in 0..slices.len() {
        let too_similar = kept
            .iter()
            .any(|&j| pearson_distance(&slices[i].mel, &slices[j].mel) < SIMILARITY_THRESHOLD);
        if !too_similar {
            kept.push(i);
        }
    }
    kept
}
