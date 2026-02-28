//! Morphs two audio files together using optimal-transport spectral interpolation,
//! sweeping linearly from file 1 to file 2.
//!
//! See <https://learn.flucoma.org/reference/audiotransport/> for more info.
//!
//! ```sh
//! cargo run --example audio-transport -- input1.wav input2.wav output.wav
//! ```

use std::error::Error;

use arg::{parse_args, Args};
use flucoma_rs::transformation::AudioTransport;
use wavers::Wav;

// -------------------------------------------------------------------------------------------------
// AudioTransport config

const WINDOW_SIZE: usize = 1024;
const FFT_SIZE: usize = 4096;
const HOP_SIZE: usize = WINDOW_SIZE / 2;

// -------------------------------------------------------------------------------------------------
// Arguments

#[derive(Args, Debug)]
struct Arguments {
    /// First input file (crossfade start -- weight 0.0)
    input1: String,
    /// Second input file (crossfade end -- weight 1.0)
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

    println!(
        "Crossfading `{}` -> `{}` into `{}`",
        args.input1, args.input2, args.output
    );

    // Open both input files
    let mut wav1 = Wav::<f32>::from_path(&args.input1)?;
    let mut wav2 = Wav::<f32>::from_path(&args.input2)?;

    let sample_rate = wav1.sample_rate();
    let n_channels = wav1.n_channels() as usize;

    if wav2.sample_rate() != sample_rate {
        println!(
            "WARNING: Sample rates don't match: {} vs {}",
            sample_rate,
            wav2.sample_rate()
        );
    }
    if wav2.n_channels() as usize != n_channels {
        return Err(format!(
            "Channel counts don't match: {} vs {}",
            n_channels,
            wav2.n_channels()
        )
        .into());
    }

    // Read all samples into per-channel f64 buffers
    let mut ch1 = read_channels(&mut wav1, n_channels);
    let mut ch2 = read_channels(&mut wav2, n_channels);

    let len1 = ch1[0].len();
    let len2 = ch2[0].len();

    // Prepend WINDOW silence so OLA has full overlap before real audio starts
    for v in &mut ch1 {
        v.splice(0..0, vec![0.0; WINDOW_SIZE]);
    }
    for v in &mut ch2 {
        v.splice(0..0, vec![0.0; WINDOW_SIZE]);
    }

    let output_frames = ch1[0].len().min(ch2[0].len());
    let total_hops = (output_frames + WINDOW_SIZE).div_ceil(HOP_SIZE);

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

    // OLA accumulation buffers per channel
    let acc_len = total_hops * HOP_SIZE + WINDOW_SIZE;
    let mut audio_acc = vec![vec![0.0f64; acc_len]; n_channels];
    let mut norm_acc = vec![vec![0.0f64; acc_len]; n_channels];

    // One AudioTransport instance per channel (each holds independent state)
    let mut morphers: Vec<AudioTransport> = (0..n_channels)
        .map(|_| AudioTransport::new(WINDOW_SIZE, FFT_SIZE, HOP_SIZE))
        .collect::<Result<_, _>>()?;

    // Process frame by frame, linearly sweeping weight 0.0 -> 1.0
    let mut frame_buf1 = vec![0.0f64; WINDOW_SIZE];
    let mut frame_buf2 = vec![0.0f64; WINDOW_SIZE];

    for hop in 0..total_hops {
        let start = hop * HOP_SIZE;
        let weight = hop as f64 / (total_hops - 1).max(1) as f64;

        for ch in 0..n_channels {
            extract_window(&ch1[ch], start, &mut frame_buf1);
            extract_window(&ch2[ch], start, &mut frame_buf2);

            let (audio, window_sq) = morphers[ch].process_frame(&frame_buf1, &frame_buf2, weight);

            for i in 0..WINDOW_SIZE {
                audio_acc[ch][start + i] += audio[i];
                norm_acc[ch][start + i] += window_sq[i];
            }
        }
    }

    // Normalize and interleave into i16 output, skipping the WINDOW silence prefix.
    let mut output: Vec<i16> = Vec::with_capacity(len1.min(len2) * n_channels);
    for frame in WINDOW_SIZE..output_frames {
        for ch in 0..n_channels {
            let norm = norm_acc[ch][frame];
            let sample = if norm > 1e-9 {
                audio_acc[ch][frame] / norm
            } else {
                0.0
            };
            output.push((sample.clamp(-1.0, 1.0) * 32767.0) as i16);
        }
    }

    wavers::write(&args.output, &output, sample_rate as i32, n_channels as u16)?;

    println!("Done. Wrote {} frames to `{}`.", output_frames, args.output);

    Ok(())
}

// -------------------------------------------------------------------------------------------------

/// Read all frames from a wav file into per-channel f64 buffers.
fn read_channels(wav: &mut Wav<f32>, n_channels: usize) -> Vec<Vec<f64>> {
    let mut channels = vec![Vec::new(); n_channels];
    for frame in wav.frames() {
        for ch in 0..n_channels {
            channels[ch].push(frame[ch] as f64);
        }
    }
    channels
}

/// Copy a windowed slice of `src` into `dst`, zero-padding past the end.
fn extract_window(src: &[f64], start: usize, dst: &mut [f64]) {
    let len = dst.len();
    for i in 0..len {
        dst[i] = if start + i < src.len() {
            src[start + i]
        } else {
            0.0
        };
    }
}
