use flucoma_sys::{
    audio_transport_create, audio_transport_destroy, audio_transport_init,
    audio_transport_process_frame,
};

// -------------------------------------------------------------------------------------------------

/// Optimal-transport spectral morphing between two audio frames.
///
/// Two-phase setup:
/// 1. [`AudioTransport::new`] -- allocates buffers and initialises the internal STFT.
/// 2. Call [`AudioTransport::process_frame`] per frame pair.
///
/// Each call takes two time-domain input frames and an interpolation `weight`
/// (0.0 = all `in1`, 1.0 = all `in2`) and returns `(audio, window_sq)` -- two
/// slices into an internal preallocated buffer valid until the next call.
///
/// See <https://learn.flucoma.org/reference/audiotransport>
pub struct AudioTransport {
    inner: *mut u8,
    window_size: usize,
    fft_size: usize,
    hop_size: usize,
    buf: Vec<f64>,
}

unsafe impl Send for AudioTransport {}

impl AudioTransport {
    /// Create and initialise an AudioTransport morpher.
    ///
    /// # Arguments
    /// * `window_size` - Analysis/synthesis window size in samples.
    /// * `fft_size`    - FFT size (must be >= `window_size`, typically a power of 2).
    /// * `hop_size`    - Hop size between frames in samples (must be > 0).
    ///
    /// # Errors
    /// Returns an error string if parameters are invalid or allocation fails.
    pub fn new(window_size: usize, fft_size: usize, hop_size: usize) -> Result<Self, &'static str> {
        if window_size == 0 {
            return Err("window_size must be > 0");
        }
        if fft_size < window_size {
            return Err("fft_size must be >= window_size");
        }
        if hop_size == 0 {
            return Err("hop_size must be > 0");
        }
        let inner = audio_transport_create(fft_size as isize);
        if inner.is_null() {
            return Err("failed to create AudioTransport instance");
        }
        audio_transport_init(
            inner,
            window_size as isize,
            fft_size as isize,
            hop_size as isize,
        );
        Ok(Self {
            inner,
            window_size,
            fft_size,
            hop_size,
            buf: vec![0.0f64; 2 * window_size],
        })
    }

    /// Interpolate between two audio frames using optimal transport.
    ///
    /// # Arguments
    /// * `in1`    - First audio frame. Length must equal `window_size`.
    /// * `in2`    - Second audio frame. Length must equal `window_size`.
    /// * `weight` - Interpolation weight in `[0.0, 1.0]`. 0.0 = all `in1`, 1.0 = all `in2`.
    ///
    /// Returns `(audio, window_sq)`: slices into an internal buffer of length
    /// `window_size` each. The slices are valid until the next call to this method.
    ///
    /// # Panics
    /// Panics if input lengths differ from `window_size`.
    pub fn process_frame<'a>(
        &'a mut self,
        in1: &[f64],
        in2: &[f64],
        weight: f64,
    ) -> (&'a [f64], &'a [f64]) {
        assert_eq!(
            in1.len(),
            self.window_size,
            "in1 length ({}) must equal window_size ({})",
            in1.len(),
            self.window_size
        );
        assert_eq!(
            in2.len(),
            self.window_size,
            "in2 length ({}) must equal window_size ({})",
            in2.len(),
            self.window_size
        );
        let weight = weight.clamp(0.0, 1.0);
        audio_transport_process_frame(
            self.inner,
            in1.as_ptr(),
            in2.as_ptr(),
            self.window_size as isize,
            weight,
            self.buf.as_mut_ptr(),
        );
        self.buf.split_at(self.window_size)
    }

    /// Analysis/synthesis window size in samples.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// FFT size in samples.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size between frames in samples.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }
}

impl Drop for AudioTransport {
    fn drop(&mut self) {
        audio_transport_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_transport_silence_gives_silence() {
        let win = 1024usize;
        let mut at = AudioTransport::new(win, win, 512).unwrap();
        let silence = vec![0.0f64; win];
        let (audio, window_sq) = at.process_frame(&silence, &silence, 0.5);
        assert_eq!(audio.len(), win);
        assert_eq!(window_sq.len(), win);
        for (i, &v) in audio.iter().enumerate() {
            assert!(
                v.abs() < 1e-9,
                "silence should give near-zero audio, sample {i} = {v}"
            );
        }
    }

    #[test]
    fn audio_transport_output_has_correct_shape() {
        use std::f64::consts::PI;
        let win = 1024usize;
        let mut at = AudioTransport::new(win, win, 512).unwrap();
        let in1: Vec<f64> = (0..win)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let in2: Vec<f64> = (0..win)
            .map(|i| (2.0 * PI * 880.0 * i as f64 / 44100.0).sin())
            .collect();

        let (audio0, wsq0) = at.process_frame(&in1, &in2, 0.0);
        assert_eq!(audio0.len(), win);
        assert_eq!(wsq0.len(), win);
        let energy0: f64 = audio0.iter().map(|x| x * x).sum();

        let (audio1, wsq1) = at.process_frame(&in1, &in2, 1.0);
        assert_eq!(audio1.len(), win);
        assert_eq!(wsq1.len(), win);
        let energy1: f64 = audio1.iter().map(|x| x * x).sum();

        assert!(
            energy0 > 0.0 || energy1 > 0.0,
            "at least one output should have energy"
        );
    }
}
