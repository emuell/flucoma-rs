//! Decomposes an audio file into two layers using a spectral separation algorithm.
//!
//! Each output layer is written as a separate WAV file next to the input.
//!
//! **Modes:**
//! * `hpss` (default) -- harmonic/percussive separation via median filtering
//! * `sine` -- sinusoidal/residual separation via partial tracking
//!
//! ```sh
//! cargo run --example decompose -- input.wav
//! cargo run --example decompose -- --mode sine input.wav
//! ```
//!
//! Output (hpss): `<input_stem>_harmonic.wav` and `<input_stem>_percussive.wav`.
//! Output (sine): `<input_stem>_sines.wav` and `<input_stem>_residual.wav`.

use std::{error::Error, fs::File, path::Path};

use arg::{parse_args, Args};
use flucoma_rs::{
    decomposition::{Hpss, HpssParams, SineExtraction, SineExtractionParams},
    fourier::{ComplexSpectrum, Istft, Stft, WindowType},
};

// -------------------------------------------------------------------------------------------------

// STFT config
const WINDOW_SIZE: usize = 1024;
const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = 512;

// HPSS median filter sizes (must be odd).
// Larger H_SIZE -> stronger harmonic smoothing across time.
// Larger V_SIZE -> stronger percussive smoothing across frequency.
const H_SIZE: usize = 17;
const V_SIZE: usize = 17;

// -------------------------------------------------------------------------------------------------

#[derive(Args, Debug)]
struct Arguments {
    /// Decomposition mode: hpss (default) or sine
    #[arg(long)]
    mode: Option<String>,
    /// Input audio file
    input: String,
}

// -------------------------------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args::<Arguments>();

    if args.input.is_empty() {
        return Err("Please specify an input file".into());
    }
    let mode = args.mode.as_deref().unwrap_or("hpss");
    if mode != "hpss" && mode != "sine" {
        return Err(format!("Unknown mode {mode:?}. Use \"hpss\" or \"sine\".").into());
    }

    let input_path = Path::new(args.input.as_str());
    let (header, samples) = read_mono_f64(input_path)?;
    let sample_rate = header.sample_rate;

    println!(
        "[{mode}] Read `{}`: {} samples, {} Hz",
        input_path.display(),
        samples.len(),
        sample_rate,
    );

    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    let parent = input_path.parent().unwrap_or(Path::new("."));

    let wav_header = wav_io::new_header(sample_rate, 16, false, true);

    let (out_a, out_b) = decompose(&samples, mode, sample_rate as f64)?;

    let (name_a, name_b) = if mode == "sine" {
        ("sines", "residual")
    } else {
        ("harmonic", "percussive")
    };

    write_wav(
        &parent.join(format!("{stem}_{name_a}.wav")),
        &wav_header,
        &out_a,
    )?;
    write_wav(
        &parent.join(format!("{stem}_{name_b}.wav")),
        &wav_header,
        &out_b,
    )?;

    println!(
        "Wrote `{stem}_{name_a}.wav` and `{stem}_{name_b}.wav` ({} samples each)",
        out_a.len()
    );

    Ok(())
}

// -------------------------------------------------------------------------------------------------

/// Run the selected decomposition frame by frame and return two output sample vectors,
/// both the same length as `samples`.
fn decompose(
    samples: &[f64],
    mode: &str,
    sample_rate: f64,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let mut stft = Stft::new(WINDOW_SIZE, FFT_SIZE, HOP_SIZE, WindowType::Hann)?;
    let mut istft_a = Istft::new(FFT_SIZE, FFT_SIZE, HOP_SIZE, WindowType::Hann)?;
    let mut istft_b = Istft::new(FFT_SIZE, FFT_SIZE, HOP_SIZE, WindowType::Hann)?;

    let n_bins = stft.num_bins();

    let hop_count = samples.len().div_ceil(HOP_SIZE);
    let output_len = hop_count * HOP_SIZE + FFT_SIZE;

    let mut out_a = vec![0.0f64; output_len];
    let mut out_b = vec![0.0f64; output_len];

    let mut frame = vec![0.0f64; WINDOW_SIZE];
    let mut hop_buf = vec![0.0f64; FFT_SIZE];
    let mut spec_a = ComplexSpectrum::zeros(n_bins);
    let mut spec_b = ComplexSpectrum::zeros(n_bins);

    if mode == "sine" {
        let mut sine = SineExtraction::new(WINDOW_SIZE, FFT_SIZE, FFT_SIZE)?;
        let params = SineExtractionParams {
            sample_rate,
            ..SineExtractionParams::default()
        };
        for hop in 0..hop_count {
            fill_frame(&mut frame, samples, hop * HOP_SIZE);
            let spectrum = stft.process_frame(&frame);
            let (sines, residual) = sine.process_frame(&spectrum.bins, &params);
            spec_a.bins.copy_from_slice(sines);
            spec_b.bins.copy_from_slice(residual);
            overlap_add(
                &mut out_a,
                &mut istft_a,
                &spec_a,
                &mut hop_buf,
                hop * HOP_SIZE,
            );
            overlap_add(
                &mut out_b,
                &mut istft_b,
                &spec_b,
                &mut hop_buf,
                hop * HOP_SIZE,
            );
        }
    } else {
        let mut hpss = Hpss::new(FFT_SIZE, H_SIZE, V_SIZE)?;
        let params = HpssParams::default();
        for hop in 0..hop_count {
            fill_frame(&mut frame, samples, hop * HOP_SIZE);
            let spectrum = stft.process_frame(&frame);
            let (harmonic, percussive, _residual) = hpss.process_frame(&spectrum.bins, &params);
            spec_a.bins.copy_from_slice(harmonic);
            spec_b.bins.copy_from_slice(percussive);
            overlap_add(
                &mut out_a,
                &mut istft_a,
                &spec_a,
                &mut hop_buf,
                hop * HOP_SIZE,
            );
            overlap_add(
                &mut out_b,
                &mut istft_b,
                &spec_b,
                &mut hop_buf,
                hop * HOP_SIZE,
            );
        }
    }

    // Trim to original length and convert to f32 for wav_io.
    let to_f32 =
        |v: Vec<f64>| -> Vec<f32> { v[..samples.len()].iter().map(|&s| s as f32).collect() };

    Ok((to_f32(out_a), to_f32(out_b)))
}

// -------------------------------------------------------------------------------------------------

fn read_mono_f64(path: &Path) -> Result<(wav_io::header::WavHeader, Vec<f64>), Box<dyn Error>> {
    let (header, data) = wav_io::read_from_file(File::open(path)?)?;
    let ch = header.channels as usize;
    let mono: Vec<f64> = if ch > 1 {
        data.chunks_exact(ch)
            .map(|frame| frame.iter().map(|&s| s as f64).sum::<f64>() / ch as f64)
            .collect()
    } else {
        data.iter().map(|&s| s as f64).collect()
    };
    Ok((header, mono))
}

fn write_wav(
    path: &Path,
    header: &wav_io::header::WavHeader,
    samples: &Vec<f32>,
) -> Result<(), Box<dyn Error>> {
    wav_io::write_to_file(&mut File::create(path)?, header, samples)?;
    Ok(())
}

// -------------------------------------------------------------------------------------------------

fn fill_frame(frame: &mut [f64], samples: &[f64], start: usize) {
    let end = start + frame.len();
    if end <= samples.len() {
        frame.copy_from_slice(&samples[start..end]);
    } else {
        for i in 0..frame.len() {
            frame[i] = samples.get(start + i).copied().unwrap_or(0.0);
        }
    }
}

fn overlap_add(
    output: &mut [f64],
    istft: &mut Istft,
    spectrum: &ComplexSpectrum,
    hop_buf: &mut [f64],
    start: usize,
) {
    istft.process_frame(spectrum, hop_buf);
    output[start..start + hop_buf.len()]
        .iter_mut()
        .zip(hop_buf.iter())
        .for_each(|(o, &s)| *o += s);
}
