//! Morphs two audio files together using one of two spectral interpolation methods:
//!
//! * `transport` (default) -- optimal-transport spectral interpolation via
//!   [`AudioTransport`](https://learn.flucoma.org/reference/audiotransport/).
//! * `morph` -- NMF decomposition + optimal-transport component morphing via
//!   [`NMFMorph`](https://learn.flucoma.org/reference/nmfmorph/).
//!
//! Both modes sweep linearly from file 1 to file 2.
//!
//! ```sh
//! cargo run --example transform -- input1.wav input2.wav output.wav
//! cargo run --example transform -- --mode morph input1.wav input2.wav output.wav
//! ```

use std::{error::Error, fs::File};

use arg::{parse_args, Args};
use flucoma_rs::{
    data::Matrix,
    fourier::{ComplexSpectrum, Istft, Stft, WindowType},
    transformation::{AudioTransport, NMFFilter, NMFMorph},
};

// -------------------------------------------------------------------------------------------------

// Shared STFT config
const WINDOW_SIZE: usize = 4096;
const FFT_SIZE: usize = 4096;
const HOP_SIZE: usize = WINDOW_SIZE / 2;

// NMF config (morph mode only)
const NMF_RANK: usize = 8;
const NMF_ITERATIONS: usize = 100;

// -------------------------------------------------------------------------------------------------

#[derive(Args, Debug)]
struct Arguments {
    /// Morphing mode: transport (default) or morph
    #[arg(long)]
    mode: Option<String>,
    /// First input file (crossfade starts here)
    input1: String,
    /// Second input file (crossfade ends here)
    input2: String,
    /// Output file path
    output: String,
}

// -------------------------------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args::<Arguments>();

    if args.input1.is_empty() {
        return Err("Please specify a first input file".into());
    }
    if args.input2.is_empty() {
        return Err("Please specify a second input file".into());
    }
    if args.output.is_empty() {
        return Err("Please specify an output file".into());
    }
    let mode = args.mode.as_deref().unwrap_or("transport");
    if mode != "transport" && mode != "morph" {
        return Err(format!("Unknown mode {mode:?}. Use \"transport\" or \"morph\".").into());
    }

    println!(
        "[{mode}] Crossfading `{}` -> `{}` into `{}`",
        args.input1, args.input2, args.output
    );

    let (wav1, mut wav1_data) = wav_io::read_from_file(File::open(&args.input1)?)?;
    let (wav2, mut wav2_data) = wav_io::read_from_file(File::open(&args.input2)?)?;

    let sample_rate = wav1.sample_rate;
    let channel_count = wav1.channels as usize;

    if wav2.sample_rate != sample_rate {
        println!(
            "WARNING: Sample rates don't match: {} vs {}",
            sample_rate, wav2.sample_rate
        );
    }
    if wav2.channels as usize != channel_count {
        return Err(format!(
            "Channel counts don't match: {} vs {}",
            channel_count, wav2.channels
        )
        .into());
    }

    let mut samples1 = deinterleave(&mut wav1_data, channel_count);
    let mut samples2 = deinterleave(&mut wav2_data, channel_count);

    let len1 = samples1[0].len();
    let len2 = samples2[0].len();

    // Prepend WINDOW_SIZE silence so OLA has full overlap before real audio starts
    for v in &mut samples1 {
        v.splice(0..0, vec![0.0; WINDOW_SIZE]);
    }
    for v in &mut samples2 {
        v.splice(0..0, vec![0.0; WINDOW_SIZE]);
    }

    println!(
        "  File 1: {} frames ({:.2}s)",
        len1,
        len1 as f64 / sample_rate as f64
    );
    println!(
        "  File 2: {} frames ({:.2}s)",
        len2,
        len2 as f64 / sample_rate as f64
    );

    let output_frames = samples1[0].len().min(samples2[0].len());
    let total_hops = (output_frames + WINDOW_SIZE).div_ceil(HOP_SIZE);

    // Use FFT_SIZE for tail room: morph mode ISTFT outputs FFT_SIZE samples per frame.
    let acc_len = total_hops * HOP_SIZE + FFT_SIZE;
    let mut audio_acc = vec![vec![0.0f64; acc_len]; channel_count];
    let mut norm_acc = vec![vec![0.0f64; acc_len]; channel_count];

    if mode == "transport" {
        run_transport(
            &samples1,
            &samples2,
            channel_count,
            total_hops,
            &mut audio_acc,
            &mut norm_acc,
        )?;
    } else {
        run_nmf_morph(
            &samples1,
            &samples2,
            channel_count,
            total_hops,
            &mut audio_acc,
            &mut norm_acc,
        )?;
    }

    // Normalize and interleave into interleaved f32 output, skipping the WINDOW_SIZE silence prefix
    let mut interleaved_out: Vec<f32> = Vec::with_capacity(len1.min(len2) * channel_count);
    for frame in WINDOW_SIZE..output_frames {
        for channel in 0..channel_count {
            let norm = norm_acc[channel][frame];
            let sample = if norm > 1e-9 {
                (audio_acc[channel][frame] / norm).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            interleaved_out.push(sample as f32);
        }
    }

    wav_io::write_to_file(
        &mut File::create(&args.output)?,
        &wav_io::new_header(sample_rate, 16, false, channel_count == 1),
        &interleaved_out,
    )?;

    println!("Done. Wrote {} frames to `{}`.", output_frames, args.output);

    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Transport mode

fn run_transport(
    samples1: &[Vec<f64>],
    samples2: &[Vec<f64>],
    channel_count: usize,
    total_hops: usize,
    audio_acc: &mut [Vec<f64>],
    norm_acc: &mut [Vec<f64>],
) -> Result<(), Box<dyn Error>> {
    let mut morphers: Vec<AudioTransport> = (0..channel_count)
        .map(|_| AudioTransport::new(WINDOW_SIZE, FFT_SIZE, HOP_SIZE))
        .collect::<Result<_, _>>()?;

    let mut frame_buffer1 = vec![0.0f64; WINDOW_SIZE];
    let mut frame_buffer2 = vec![0.0f64; WINDOW_SIZE];

    for hop in 0..total_hops {
        let start = hop * HOP_SIZE;
        let weight = hop as f64 / (total_hops - 1).max(1) as f64;

        for channel in 0..channel_count {
            extract_window(&samples1[channel], start, &mut frame_buffer1);
            extract_window(&samples2[channel], start, &mut frame_buffer2);

            let (audio, norm_window) =
                morphers[channel].process_frame(&frame_buffer1, &frame_buffer2, weight);

            for i in 0..WINDOW_SIZE {
                audio_acc[channel][start + i] += audio[i];
                norm_acc[channel][start + i] += norm_window[i];
            }
        }
    }

    Ok(())
}

// -------------------------------------------------------------------------------------------------
// NMF morph mode

fn run_nmf_morph(
    samples1: &[Vec<f64>],
    samples2: &[Vec<f64>],
    channel_count: usize,
    total_hops: usize,
    audio_acc: &mut [Vec<f64>],
    norm_acc: &mut [Vec<f64>],
) -> Result<(), Box<dyn Error>> {
    let bin_count = FFT_SIZE / 2 + 1;

    // Separate STFT instances per source per channel — STFT is stateful
    let mut stfts1: Vec<Stft> = (0..channel_count)
        .map(|_| Stft::new(WINDOW_SIZE, FFT_SIZE, HOP_SIZE, WindowType::Hann))
        .collect::<Result<_, _>>()?;
    let mut stfts2: Vec<Stft> = (0..channel_count)
        .map(|_| Stft::new(WINDOW_SIZE, FFT_SIZE, HOP_SIZE, WindowType::Hann))
        .collect::<Result<_, _>>()?;

    // Build magnitude spectrograms for both sources (n_frames × n_bins, row-major)
    let mut frame_buffer = vec![0.0f64; WINDOW_SIZE];
    let mut spec1_data = vec![vec![0.0f64; total_hops * bin_count]; channel_count];
    let mut spec2_data = vec![vec![0.0f64; total_hops * bin_count]; channel_count];

    for hop in 0..total_hops {
        let start = hop * HOP_SIZE;
        for channel in 0..channel_count {
            extract_window(&samples1[channel], start, &mut frame_buffer);
            let s = stfts1[channel].process_frame(&frame_buffer);
            for bin in 0..bin_count {
                spec1_data[channel][hop * bin_count + bin] = s.bins[bin].norm();
            }

            extract_window(&samples2[channel], start, &mut frame_buffer);
            let s = stfts2[channel].process_frame(&frame_buffer);
            for bin in 0..bin_count {
                spec2_data[channel][hop * bin_count + bin] = s.bins[bin].norm();
            }
        }
    }

    println!("  Running NMF decomposition (rank={NMF_RANK}, iters={NMF_ITERATIONS})...");

    // Decompose each channel's spectrograms
    let mut nmf = NMFFilter::new(bin_count, NMF_RANK)?;
    let mut morphers: Vec<NMFMorph> = (0..channel_count)
        .map(|_| NMFMorph::new(FFT_SIZE))
        .collect::<Result<_, _>>()?;

    // ISTFT must be constructed with window_size = FFT_SIZE here in case FFT_SIZE > WINDOW_SIZE.
    let mut istfts: Vec<Istft> = (0..channel_count)
        .map(|_| Istft::new(FFT_SIZE, FFT_SIZE, HOP_SIZE, WindowType::Hann))
        .collect::<Result<_, _>>()?;

    for channel in 0..channel_count {
        let spec1 = Matrix::from_vec(spec1_data[channel].clone(), total_hops, bin_count).unwrap();
        let spec2 = Matrix::from_vec(spec2_data[channel].clone(), total_hops, bin_count).unwrap();

        let res1 = nmf.process(&spec1, NMF_RANK, NMF_ITERATIONS, -1);
        let res2 = nmf.process(&spec2, NMF_RANK, NMF_ITERATIONS, -1);

        // NMFFilter::process returns activations as n_frames × rank.
        // NMFMorph::init expects H as rank × n_frames — transpose it.
        let h1_t = res1.activations.transposed();

        morphers[channel].init(
            &res1.bases,
            &res2.bases,
            &h1_t,
            WINDOW_SIZE,
            FFT_SIZE,
            HOP_SIZE,
            false,
        )?;
    }

    println!("  Synthesising morphed output...");

    // ISTFT window_size = FFT_SIZE, so output frame is FFT_SIZE samples.
    let mut audio_frame = vec![0.0f64; FFT_SIZE];
    // Periodic Hann window — matches FluCoMa's synthesis window
    let norm_window: Vec<f64> = (0..FFT_SIZE)
        .map(|i| {
            use std::f64::consts::PI;
            let hann = 0.5 * (1.0 - (2.0 * PI * i as f64 / FFT_SIZE as f64).cos());
            hann * hann
        })
        .collect();

    for hop in 0..total_hops {
        let start = hop * HOP_SIZE;
        let weight = hop as f64 / (total_hops - 1).max(1) as f64;

        for channel in 0..channel_count {
            let raw = morphers[channel].process_frame(weight, -1);
            let spec = ComplexSpectrum { bins: raw.to_vec() };
            istfts[channel].process_frame(&spec, &mut audio_frame);

            for i in 0..FFT_SIZE {
                audio_acc[channel][start + i] += audio_frame[i];
                norm_acc[channel][start + i] += norm_window[i];
            }
        }
    }

    Ok(())
}

// -------------------------------------------------------------------------------------------------

/// Convert interleaved f32 sample data to planar f64 data
fn deinterleave(samples: &mut [f32], n_channels: usize) -> Vec<Vec<f64>> {
    let mut channels = vec![Vec::new(); n_channels];
    for frame in samples.chunks_exact(n_channels) {
        for ch in 0..n_channels {
            channels[ch].push(frame[ch] as f64);
        }
    }
    channels
}

/// Read a window frame and write it into dest, padding with zeros if necessary
fn extract_window(src: &[f64], start: usize, dst: &mut [f64]) {
    for i in 0..dst.len() {
        dst[i] = if start + i < src.len() {
            src[start + i]
        } else {
            0.0
        };
    }
}
