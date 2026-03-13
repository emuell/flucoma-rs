#![allow(non_upper_case_globals)]
#![recursion_limit = "512"]
/// Raw bindings for flucoma-core algorithms via inline C++ (cpp! macros).
///
/// All handles are opaque `*mut u8` pointers. Do not use these functions
/// directly -- use the safe wrappers in the `flucoma-rs` crate instead.
use cpp::cpp;

/// Signed index type matching `ptrdiff_t` used by flucoma-core.
pub type FlucomaIndex = isize;

// -------------------------------------------------------------------------------------------------
// Cpp includes

cpp! {{
    #define FMT_HEADER_ONLY 1
    #include <complex>
    #include <flucoma/data/FluidMemory.hpp>
    #include <flucoma/algorithms/public/Envelope.hpp>
    #include <flucoma/algorithms/public/Loudness.hpp>
    #include <flucoma/algorithms/public/STFT.hpp>
    #include <flucoma/algorithms/public/MelBands.hpp>
    #include <flucoma/algorithms/public/OnsetDetectionFunctions.hpp>
    #include <flucoma/algorithms/public/OnsetSegmentation.hpp>
    #include <flucoma/algorithms/public/AudioTransport.hpp>
    #include <flucoma/algorithms/public/DataSetQuery.hpp>
    #include <flucoma/algorithms/public/Grid.hpp>
    #include <flucoma/algorithms/public/KDTree.hpp>
    #include <flucoma/algorithms/public/KMeans.hpp>
    #include <flucoma/algorithms/public/MDS.hpp>
    #include <flucoma/algorithms/public/MultiStats.hpp>
    #include <flucoma/algorithms/public/Normalization.hpp>
    #include <flucoma/algorithms/public/NMF.hpp>
    #include <flucoma/algorithms/public/NMFMorph.hpp>
    #include <flucoma/algorithms/public/PCA.hpp>
    #include <flucoma/algorithms/public/RobustScaling.hpp>
    #include <flucoma/algorithms/public/RunningStats.hpp>
    #include <flucoma/algorithms/public/SKMeans.hpp>
    #include <flucoma/algorithms/public/Standardization.hpp>
    #include <flucoma/algorithms/public/EnvelopeSegmentation.hpp>
    #include <flucoma/algorithms/public/NoveltyFeature.hpp>
    #include <flucoma/algorithms/public/NoveltySegmentation.hpp>
    #include <flucoma/algorithms/public/SineFeature.hpp>
    #include <flucoma/algorithms/public/TransientSegmentation.hpp>
    #include <flucoma/algorithms/public/HPSS.hpp>
    #include <flucoma/algorithms/public/SineExtraction.hpp>
    #include <flucoma/algorithms/public/TransientExtraction.hpp>
    using namespace fluid;
    using namespace fluid::algorithm;
}}

// -------------------------------------------------------------------------------------------------
// Envelope (AmpFeature)

pub fn amp_feature_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new Envelope());
        })
    }
}

pub fn amp_feature_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "Envelope*"] {
            delete ptr;
        })
    }
}

pub fn amp_feature_init(ptr: *mut u8, floor: f64, hi_pass_freq: f64) {
    unsafe {
        cpp!([ptr as "Envelope*", floor as "double", hi_pass_freq as "double"] {
            ptr->init(floor, hi_pass_freq);
        })
    }
}

pub fn amp_feature_process_sample(
    ptr: *mut u8,
    input: f64,
    floor: f64,
    fast_ramp_up: FlucomaIndex,
    slow_ramp_up: FlucomaIndex,
    fast_ramp_down: FlucomaIndex,
    slow_ramp_down: FlucomaIndex,
    hi_pass_freq: f64,
) -> f64 {
    unsafe {
        cpp!([
            ptr as "Envelope*",
            input as "double", floor as "double",
            fast_ramp_up as "ptrdiff_t", slow_ramp_up as "ptrdiff_t",
            fast_ramp_down as "ptrdiff_t", slow_ramp_down as "ptrdiff_t",
            hi_pass_freq as "double"
        ] -> f64 as "double" {
            return ptr->processSample(input, floor, fast_ramp_up, slow_ramp_up,
                                      fast_ramp_down, slow_ramp_down, hi_pass_freq);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// Loudness

pub fn loudness_create(max_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(new Loudness(max_size));
        })
    }
}

pub fn loudness_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "Loudness*"] {
            delete ptr;
        })
    }
}

pub fn loudness_init(ptr: *mut u8, size: FlucomaIndex, sample_rate: f64) {
    unsafe {
        cpp!([ptr as "Loudness*", size as "ptrdiff_t", sample_rate as "double"] {
            ptr->init(size, sample_rate);
        })
    }
}

pub fn loudness_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    output: *mut f64,
    weighting: bool,
    true_peak: bool,
) {
    unsafe {
        cpp!([
            ptr as "Loudness*",
            input as "const double*", input_len as "ptrdiff_t",
            output as "double*",
            weighting as "bool", true_peak as "bool"
        ] {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            FluidTensorView<double, 1> out_v(output, 0, 2);
            ptr->processFrame(in_v, out_v, weighting, true_peak);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// STFT

pub fn stft_create(
    window_size: FlucomaIndex,
    fft_size: FlucomaIndex,
    hop_size: FlucomaIndex,
    window_type: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            window_size as "ptrdiff_t", fft_size as "ptrdiff_t",
            hop_size as "ptrdiff_t", window_type as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            return static_cast<void*>(new STFT(window_size, fft_size, hop_size, window_type));
        })
    }
}

pub fn stft_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "STFT*"] {
            delete ptr;
        })
    }
}

pub fn stft_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    out_complex: *mut f64,
    num_bins: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "STFT*",
            input as "const double*", input_len as "ptrdiff_t",
            out_complex as "double*", num_bins as "ptrdiff_t"
        ] {
            auto* cptr = reinterpret_cast<std::complex<double>*>(out_complex);
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            FluidTensorView<std::complex<double>, 1> out_v(cptr, 0, num_bins);
            ptr->processFrame(in_v, out_v);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// ISTFT

pub fn istft_create(
    window_size: FlucomaIndex,
    fft_size: FlucomaIndex,
    hop_size: FlucomaIndex,
    window_type: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            window_size as "ptrdiff_t", fft_size as "ptrdiff_t",
            hop_size as "ptrdiff_t", window_type as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            return static_cast<void*>(new ISTFT(window_size, fft_size, hop_size, window_type));
        })
    }
}

pub fn istft_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "ISTFT*"] {
            delete ptr;
        })
    }
}

pub fn istft_process_frame(
    ptr: *mut u8,
    in_complex: *const f64,
    num_bins: FlucomaIndex,
    output: *mut f64,
    output_len: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "ISTFT*",
            in_complex as "const double*", num_bins as "ptrdiff_t",
            output as "double*", output_len as "ptrdiff_t"
        ] {
            using namespace Eigen;
            auto* cptr = reinterpret_cast<std::complex<double>*>(
                const_cast<double*>(in_complex));
            Map<ArrayXcd> in_m(cptr, num_bins);
            Map<ArrayXd> out_m(output, output_len);
            ptr->processFrame(in_m, out_m);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// MelBands

pub fn melbands_create(max_bands: FlucomaIndex, max_fft: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_bands as "ptrdiff_t", max_fft as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(new MelBands(max_bands, max_fft));
        })
    }
}

pub fn melbands_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "MelBands*"] {
            delete ptr;
        })
    }
}

pub fn melbands_init(
    ptr: *mut u8,
    lo_hz: f64,
    hi_hz: f64,
    n_bands: FlucomaIndex,
    n_bins: FlucomaIndex,
    sample_rate: f64,
    window_size: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "MelBands*",
            lo_hz as "double", hi_hz as "double",
            n_bands as "ptrdiff_t", n_bins as "ptrdiff_t",
            sample_rate as "double", window_size as "ptrdiff_t"
        ] {
            ptr->init(lo_hz, hi_hz, n_bands, n_bins, sample_rate, window_size);
        })
    }
}

pub fn melbands_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    output: *mut f64,
    output_len: FlucomaIndex,
    mag_norm: bool,
    use_power: bool,
    log_output: bool,
) {
    unsafe {
        cpp!([
            ptr as "MelBands*",
            input as "const double*", input_len as "ptrdiff_t",
            output as "double*", output_len as "ptrdiff_t",
            mag_norm as "bool", use_power as "bool", log_output as "bool"
        ] {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            FluidTensorView<double, 1> out_v(output, 0, output_len);
            ptr->processFrame(in_v, out_v, mag_norm, use_power, log_output, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// AudioTransport

pub fn audio_transport_create(max_fft_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_fft_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(new AudioTransport(max_fft_size, FluidDefaultAllocator()));
        })
    }
}

pub fn audio_transport_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "AudioTransport*"] {
            delete ptr;
        })
    }
}

pub fn audio_transport_init(
    ptr: *mut u8,
    window_size: FlucomaIndex,
    fft_size: FlucomaIndex,
    hop_size: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "AudioTransport*",
            window_size as "ptrdiff_t", fft_size as "ptrdiff_t", hop_size as "ptrdiff_t"
        ] {
            ptr->init(window_size, fft_size, hop_size);
        })
    }
}

pub fn audio_transport_process_frame(
    ptr: *mut u8,
    in1: *const f64,
    in2: *const f64,
    frame_len: FlucomaIndex,
    weight: f64,
    output: *mut f64,
) {
    unsafe {
        cpp!([
            ptr as "AudioTransport*",
            in1 as "const double*", in2 as "const double*",
            frame_len as "ptrdiff_t",
            weight as "double",
            output as "double*"
        ] {
            FluidTensorView<double, 1> in1_v(const_cast<double*>(in1), 0, frame_len);
            FluidTensorView<double, 1> in2_v(const_cast<double*>(in2), 0, frame_len);
            FluidTensorView<double, 2> out_v(output, 0, 2, frame_len);
            ptr->processFrame(in1_v, in2_v, weight, out_v, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// NMF (used by NMFFilter)

pub fn nmf_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new NMF());
        })
    }
}

pub fn nmf_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "NMF*"] {
            delete ptr;
        })
    }
}

pub fn nmf_process(
    ptr: *mut u8,
    x: *const f64,
    n_frames: FlucomaIndex,
    n_bins: FlucomaIndex,
    w1: *mut f64,
    h1: *mut f64,
    v1: *mut f64,
    rank: FlucomaIndex,
    n_iterations: FlucomaIndex,
    update_w: bool,
    update_h: bool,
    random_seed: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "NMF*",
            x  as "const double*", n_frames as "ptrdiff_t", n_bins as "ptrdiff_t",
            w1 as "double*", h1 as "double*", v1 as "double*",
            rank as "ptrdiff_t", n_iterations as "ptrdiff_t",
            update_w as "bool", update_h as "bool", random_seed as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> x_v (const_cast<double*>(x),  0, n_frames, n_bins);
            FluidTensorView<double, 2> w1_v(w1,                      0, rank,     n_bins);
            FluidTensorView<double, 2> h1_v(h1,                      0, n_frames, rank);
            FluidTensorView<double, 2> v1_v(v1,                      0, n_frames, n_bins);
            ptr->process(x_v, w1_v, h1_v, v1_v, rank, n_iterations, update_w,
                         update_h, random_seed);
        })
    }
}

pub fn nmf_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    bases: *const f64,
    bases_rows: FlucomaIndex,
    bases_cols: FlucomaIndex,
    output: *mut f64,
    estimate: *mut f64,
    n_iterations: FlucomaIndex,
    random_seed: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "NMF*",
            input as "const double*", input_len as "ptrdiff_t",
            bases as "const double*", bases_rows as "ptrdiff_t", bases_cols as "ptrdiff_t",
            output as "double*",
            estimate as "double*",
            n_iterations as "ptrdiff_t",
            random_seed as "ptrdiff_t"
        ] {
            FluidTensorView<double, 1> x_v(const_cast<double*>(input), 0, input_len);
            FluidTensorView<double, 2> w_v(const_cast<double*>(bases), 0, bases_rows, bases_cols);
            FluidTensorView<double, 1> out_v(output, 0, bases_rows);
            FluidTensorView<double, 1> est_v(estimate, 0, input_len);
            ptr->processFrame(x_v, w_v, out_v, n_iterations, est_v, random_seed, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// NMFMorph

pub fn nmf_morph_create(max_fft_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_fft_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(new NMFMorph(max_fft_size, FluidDefaultAllocator()));
        })
    }
}

pub fn nmf_morph_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "NMFMorph*"] {
            delete ptr;
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn nmf_morph_init(
    ptr: *mut u8,
    w1: *const f64,
    w1_rows: FlucomaIndex,
    w1_cols: FlucomaIndex,
    w2: *const f64,
    w2_rows: FlucomaIndex,
    w2_cols: FlucomaIndex,
    h: *const f64,
    h_rows: FlucomaIndex,
    h_cols: FlucomaIndex,
    win_size: FlucomaIndex,
    fft_size: FlucomaIndex,
    hop_size: FlucomaIndex,
    assign: bool,
) {
    unsafe {
        cpp!([
            ptr as "NMFMorph*",
            w1 as "const double*", w1_rows as "ptrdiff_t", w1_cols as "ptrdiff_t",
            w2 as "const double*", w2_rows as "ptrdiff_t", w2_cols as "ptrdiff_t",
            h  as "const double*", h_rows  as "ptrdiff_t", h_cols  as "ptrdiff_t",
            win_size as "ptrdiff_t", fft_size as "ptrdiff_t", hop_size as "ptrdiff_t",
            assign as "bool"
        ] {
            FluidTensorView<double, 2> w1_v(const_cast<double*>(w1), 0, w1_rows, w1_cols);
            FluidTensorView<double, 2> w2_v(const_cast<double*>(w2), 0, w2_rows, w2_cols);
            FluidTensorView<double, 2> h_v (const_cast<double*>(h),  0, h_rows,  h_cols);
            ptr->init(w1_v, w2_v, h_v, win_size, fft_size, hop_size, assign, FluidDefaultAllocator());
        })
    }
}

pub fn nmf_morph_process_frame(
    ptr: *mut u8,
    out_complex: *mut f64,
    num_bins: FlucomaIndex,
    interpolation: f64,
    seed: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "NMFMorph*",
            out_complex as "double*", num_bins as "ptrdiff_t",
            interpolation as "double",
            seed as "ptrdiff_t"
        ] {
            auto* cptr = reinterpret_cast<std::complex<double>*>(out_complex);
            FluidTensorView<std::complex<double>, 1> v(cptr, 0, num_bins);
            ptr->processFrame(v, interpolation, seed, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// OnsetDetectionFunctions

pub fn onset_create(max_size: FlucomaIndex, max_filter_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_size as "ptrdiff_t", max_filter_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(
                new OnsetDetectionFunctions(max_size, max_filter_size, FluidDefaultAllocator()));
        })
    }
}

pub fn onset_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "OnsetDetectionFunctions*"] {
            delete ptr;
        })
    }
}

pub fn onset_init(
    ptr: *mut u8,
    window_size: FlucomaIndex,
    fft_size: FlucomaIndex,
    filter_size: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "OnsetDetectionFunctions*",
            window_size as "ptrdiff_t", fft_size as "ptrdiff_t", filter_size as "ptrdiff_t"
        ] {
            ptr->init(window_size, fft_size, filter_size);
        })
    }
}

pub fn onset_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    function: FlucomaIndex,
    filter_size: FlucomaIndex,
    frame_delta: FlucomaIndex,
) -> f64 {
    unsafe {
        cpp!([
            ptr as "OnsetDetectionFunctions*",
            input as "const double*", input_len as "ptrdiff_t",
            function as "ptrdiff_t", filter_size as "ptrdiff_t", frame_delta as "ptrdiff_t"
        ] -> f64 as "double" {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            return ptr->processFrame(in_v, function, filter_size, frame_delta, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// OnsetSlice

pub fn onset_seg_create(max_size: FlucomaIndex, max_filter_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_size as "ptrdiff_t", max_filter_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(
                new OnsetSegmentation(max_size, max_filter_size, FluidDefaultAllocator()));
        })
    }
}

pub fn onset_seg_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "OnsetSegmentation*"] {
            delete ptr;
        })
    }
}

pub fn onset_seg_init(
    ptr: *mut u8,
    window_size: FlucomaIndex,
    fft_size: FlucomaIndex,
    filter_size: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "OnsetSegmentation*",
            window_size as "ptrdiff_t", fft_size as "ptrdiff_t", filter_size as "ptrdiff_t"
        ] {
            ptr->init(window_size, fft_size, filter_size);
        })
    }
}

pub fn onset_seg_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    function: FlucomaIndex,
    filter_size: FlucomaIndex,
    threshold: f64,
    debounce: FlucomaIndex,
    frame_delta: FlucomaIndex,
) -> f64 {
    unsafe {
        cpp!([
            ptr as "OnsetSegmentation*",
            input as "const double*", input_len as "ptrdiff_t",
            function as "ptrdiff_t", filter_size as "ptrdiff_t",
            threshold as "double", debounce as "ptrdiff_t", frame_delta as "ptrdiff_t"
        ] -> f64 as "double" {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            return ptr->processFrame(in_v, function, filter_size, threshold, debounce, frame_delta, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// AmpSlice

pub fn amp_seg_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new EnvelopeSegmentation());
        })
    }
}

pub fn amp_seg_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "EnvelopeSegmentation*"] {
            delete ptr;
        })
    }
}

pub fn amp_seg_init(ptr: *mut u8, floor: f64, hi_pass_freq: f64) {
    unsafe {
        cpp!([
            ptr as "EnvelopeSegmentation*",
            floor as "double", hi_pass_freq as "double"
        ] {
            ptr->init(floor, hi_pass_freq);
        })
    }
}

pub fn amp_seg_process_sample(
    ptr: *mut u8,
    sample: f64,
    on_threshold: f64,
    off_threshold: f64,
    floor: f64,
    fast_ramp_up: FlucomaIndex,
    slow_ramp_up: FlucomaIndex,
    fast_ramp_down: FlucomaIndex,
    slow_ramp_down: FlucomaIndex,
    hi_pass_freq: f64,
    debounce: FlucomaIndex,
) -> f64 {
    unsafe {
        cpp!([
            ptr as "EnvelopeSegmentation*",
            sample as "double",
            on_threshold as "double", off_threshold as "double",
            floor as "double",
            fast_ramp_up as "ptrdiff_t", slow_ramp_up as "ptrdiff_t",
            fast_ramp_down as "ptrdiff_t", slow_ramp_down as "ptrdiff_t",
            hi_pass_freq as "double", debounce as "ptrdiff_t"
        ] -> f64 as "double" {
            return ptr->processSample(sample, on_threshold, off_threshold, floor,
                fast_ramp_up, slow_ramp_up, fast_ramp_down, slow_ramp_down,
                hi_pass_freq, debounce);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// NoveltySlice

pub fn novelty_seg_create(
    max_kernel_size: FlucomaIndex,
    max_dims: FlucomaIndex,
    max_filter_size: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            max_kernel_size as "ptrdiff_t", max_dims as "ptrdiff_t", max_filter_size as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            return static_cast<void*>(
                new NoveltySegmentation(max_kernel_size, max_dims, max_filter_size, FluidDefaultAllocator()));
        })
    }
}

pub fn novelty_seg_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "NoveltySegmentation*"] {
            delete ptr;
        })
    }
}

pub fn novelty_seg_init(
    ptr: *mut u8,
    kernel_size: FlucomaIndex,
    filter_size: FlucomaIndex,
    n_dims: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "NoveltySegmentation*",
            kernel_size as "ptrdiff_t", filter_size as "ptrdiff_t", n_dims as "ptrdiff_t"
        ] {
            ptr->init(kernel_size, filter_size, n_dims, FluidDefaultAllocator());
        })
    }
}

pub fn novelty_seg_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    threshold: f64,
    min_slice_length: FlucomaIndex,
) -> f64 {
    unsafe {
        cpp!([
            ptr as "NoveltySegmentation*",
            input as "const double*", input_len as "ptrdiff_t",
            threshold as "double", min_slice_length as "ptrdiff_t"
        ] -> f64 as "double" {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            return ptr->processFrame(in_v, threshold, min_slice_length, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// NoveltyFeature

pub fn novelty_feature_create(
    max_kernel_size: FlucomaIndex,
    max_dims: FlucomaIndex,
    max_filter_size: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            max_kernel_size as "ptrdiff_t", max_dims as "ptrdiff_t", max_filter_size as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            return static_cast<void*>(
                new NoveltyFeature(max_kernel_size, max_dims, max_filter_size, FluidDefaultAllocator()));
        })
    }
}

pub fn novelty_feature_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "NoveltyFeature*"] {
            delete ptr;
        })
    }
}

pub fn novelty_feature_init(
    ptr: *mut u8,
    kernel_size: FlucomaIndex,
    filter_size: FlucomaIndex,
    n_dims: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "NoveltyFeature*",
            kernel_size as "ptrdiff_t", filter_size as "ptrdiff_t", n_dims as "ptrdiff_t"
        ] {
            ptr->init(kernel_size, filter_size, n_dims, FluidDefaultAllocator());
        })
    }
}

pub fn novelty_feature_process_frame(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
) -> f64 {
    unsafe {
        cpp!([
            ptr as "NoveltyFeature*",
            input as "const double*", input_len as "ptrdiff_t"
        ] -> f64 as "double" {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            return ptr->processFrame(in_v, FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// SineFeature

pub fn sine_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new SineFeature(FluidDefaultAllocator()));
        })
    }
}

pub fn sine_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "SineFeature*"] {
            delete ptr;
        })
    }
}

pub fn sine_init(ptr: *mut u8, window_size: FlucomaIndex, fft_size: FlucomaIndex) {
    unsafe {
        cpp!([ptr as "SineFeature*", window_size as "ptrdiff_t", fft_size as "ptrdiff_t"] {
            ptr->init(window_size, fft_size);
        })
    }
}

pub fn sine_process_frame(
    ptr: *mut u8,
    in_complex: *const f64,
    in_len: FlucomaIndex,
    freq_out: *mut f64,
    mag_out: *mut f64,
    out_len: FlucomaIndex,
    sample_rate: f64,
    detection_threshold: f64,
    sort_by: FlucomaIndex,
) -> FlucomaIndex {
    unsafe {
        cpp!([
            ptr as "SineFeature*",
            in_complex as "const double*", in_len as "ptrdiff_t",
            freq_out as "double*", mag_out as "double*", out_len as "ptrdiff_t",
            sample_rate as "double", detection_threshold as "double",
            sort_by as "ptrdiff_t"
        ] -> FlucomaIndex as "ptrdiff_t" {
            auto* cptr = reinterpret_cast<std::complex<double>*>(
                const_cast<double*>(in_complex));
            FluidTensorView<std::complex<double>, 1> in_v(cptr, 0, in_len);
            FluidTensorView<double, 1> freq_v(freq_out, 0, out_len);
            FluidTensorView<double, 1> mag_v(mag_out, 0, out_len);
            return ptr->processFrame(in_v, freq_v, mag_v, sample_rate,
                                     detection_threshold, sort_by,
                                     FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// TransientSlice

pub fn transient_seg_create(
    max_order: FlucomaIndex,
    max_block_size: FlucomaIndex,
    max_pad_size: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            max_order as "ptrdiff_t", max_block_size as "ptrdiff_t", max_pad_size as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            return static_cast<void*>(
                new TransientSegmentation(max_order, max_block_size, max_pad_size, FluidDefaultAllocator()));
        })
    }
}

pub fn transient_seg_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "TransientSegmentation*"] {
            delete ptr;
        })
    }
}

pub fn transient_seg_init(
    ptr: *mut u8,
    order: FlucomaIndex,
    block_size: FlucomaIndex,
    pad_size: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "TransientSegmentation*",
            order as "ptrdiff_t", block_size as "ptrdiff_t", pad_size as "ptrdiff_t"
        ] {
            ptr->init(order, block_size, pad_size);
        })
    }
}

pub fn transient_seg_set_detection_params(
    ptr: *mut u8,
    power: f64,
    thresh_hi: f64,
    thresh_lo: f64,
    half_window: FlucomaIndex,
    hold: FlucomaIndex,
    min_segment: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "TransientSegmentation*",
            power as "double", thresh_hi as "double", thresh_lo as "double",
            half_window as "ptrdiff_t", hold as "ptrdiff_t", min_segment as "ptrdiff_t"
        ] {
            ptr->setDetectionParameters(power, thresh_hi, thresh_lo, half_window, hold, min_segment);
        })
    }
}

pub fn transient_seg_process(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    output: *mut f64,
    output_len: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "TransientSegmentation*",
            input as "const double*", input_len as "ptrdiff_t",
            output as "double*", output_len as "ptrdiff_t"
        ] {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            FluidTensorView<double, 1> out_v(output, 0, output_len);
            ptr->process(in_v, out_v, FluidDefaultAllocator());
        })
    }
}

pub fn transient_seg_hop_size(ptr: *mut u8) -> FlucomaIndex {
    unsafe {
        cpp!([ptr as "TransientSegmentation*"] -> FlucomaIndex as "ptrdiff_t" {
            return ptr->hopSize();
        })
    }
}

pub fn transient_seg_input_size(ptr: *mut u8) -> FlucomaIndex {
    unsafe {
        cpp!([ptr as "TransientSegmentation*"] -> FlucomaIndex as "ptrdiff_t" {
            return ptr->inputSize();
        })
    }
}

// -------------------------------------------------------------------------------------------------
// HPSS

pub fn hpss_create(max_fft_size: FlucomaIndex, max_h_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_fft_size as "ptrdiff_t", max_h_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(new HPSS(max_fft_size, max_h_size, FluidDefaultAllocator()));
        })
    }
}

pub fn hpss_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "HPSS*"] {
            delete ptr;
        })
    }
}

pub fn hpss_init(ptr: *mut u8, n_bins: FlucomaIndex, h_size: FlucomaIndex) {
    unsafe {
        cpp!([ptr as "HPSS*", n_bins as "ptrdiff_t", h_size as "ptrdiff_t"] {
            ptr->init(n_bins, h_size);
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn hpss_process_frame(
    ptr: *mut u8,
    in_complex: *const f64,
    n_bins: FlucomaIndex,
    out_complex: *mut f64,
    v_size: FlucomaIndex,
    h_size: FlucomaIndex,
    mode: FlucomaIndex,
    h_thresh_x1: f64,
    h_thresh_y1: f64,
    h_thresh_x2: f64,
    h_thresh_y2: f64,
    p_thresh_x1: f64,
    p_thresh_y1: f64,
    p_thresh_x2: f64,
    p_thresh_y2: f64,
) {
    unsafe {
        cpp!([
            ptr as "HPSS*",
            in_complex as "const double*", n_bins as "ptrdiff_t",
            out_complex as "double*",
            v_size as "ptrdiff_t", h_size as "ptrdiff_t", mode as "ptrdiff_t",
            h_thresh_x1 as "double", h_thresh_y1 as "double",
            h_thresh_x2 as "double", h_thresh_y2 as "double",
            p_thresh_x1 as "double", p_thresh_y1 as "double",
            p_thresh_x2 as "double", p_thresh_y2 as "double"
        ] {
            auto* in_cptr = reinterpret_cast<std::complex<double>*>(
                const_cast<double*>(in_complex));
            FluidTensorView<std::complex<double>, 1> in_v(in_cptr, 0, n_bins);
            auto* out_cptr = reinterpret_cast<std::complex<double>*>(out_complex);
            FluidTensorView<std::complex<double>, 2> out_v(out_cptr, 0, n_bins, 3);
            ptr->processFrame(in_v, out_v, v_size, h_size,
                              static_cast<HPSS::HPSSMode>(mode),
                              h_thresh_x1, h_thresh_y1, h_thresh_x2, h_thresh_y2,
                              p_thresh_x1, p_thresh_y1, p_thresh_x2, p_thresh_y2);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// SineExtraction

pub fn sine_ext_create(max_fft_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_fft_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            return static_cast<void*>(new SineExtraction(max_fft_size, FluidDefaultAllocator()));
        })
    }
}

pub fn sine_ext_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "SineExtraction*"] {
            delete ptr;
        })
    }
}

pub fn sine_ext_init(
    ptr: *mut u8,
    window_size: FlucomaIndex,
    fft_size: FlucomaIndex,
    transform_size: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "SineExtraction*",
            window_size as "ptrdiff_t", fft_size as "ptrdiff_t",
            transform_size as "ptrdiff_t"
        ] {
            ptr->init(window_size, fft_size, transform_size, FluidDefaultAllocator());
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sine_ext_process_frame(
    ptr: *mut u8,
    in_complex: *const f64,
    n_bins: FlucomaIndex,
    out_complex: *mut f64,
    sample_rate: f64,
    detection_threshold: f64,
    min_track_length: FlucomaIndex,
    birth_low_threshold: f64,
    birth_high_threshold: f64,
    track_method: FlucomaIndex,
    zeta_a: f64,
    zeta_f: f64,
    delta: f64,
    bandwidth: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "SineExtraction*",
            in_complex as "const double*", n_bins as "ptrdiff_t",
            out_complex as "double*",
            sample_rate as "double", detection_threshold as "double",
            min_track_length as "ptrdiff_t",
            birth_low_threshold as "double", birth_high_threshold as "double",
            track_method as "ptrdiff_t",
            zeta_a as "double", zeta_f as "double", delta as "double",
            bandwidth as "ptrdiff_t"
        ] {
            auto* in_cptr = reinterpret_cast<std::complex<double>*>(
                const_cast<double*>(in_complex));
            FluidTensorView<std::complex<double>, 1> in_v(in_cptr, 0, n_bins);
            auto* out_cptr = reinterpret_cast<std::complex<double>*>(out_complex);
            FluidTensorView<std::complex<double>, 2> out_v(out_cptr, 0, n_bins, 2);
            ptr->processFrame(in_v, out_v, sample_rate, detection_threshold,
                              min_track_length, birth_low_threshold, birth_high_threshold,
                              track_method, zeta_a, zeta_f, delta, bandwidth,
                              FluidDefaultAllocator());
        })
    }
}

// -------------------------------------------------------------------------------------------------
// TransientExtraction

pub fn transient_ext_create(
    max_order: FlucomaIndex,
    max_block_size: FlucomaIndex,
    max_pad_size: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            max_order as "ptrdiff_t",
            max_block_size as "ptrdiff_t",
            max_pad_size as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            return static_cast<void*>(
                new TransientExtraction(max_order, max_block_size, max_pad_size,
                                        FluidDefaultAllocator()));
        })
    }
}

pub fn transient_ext_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "TransientExtraction*"] {
            delete ptr;
        })
    }
}

pub fn transient_ext_init(
    ptr: *mut u8,
    order: FlucomaIndex,
    block_size: FlucomaIndex,
    pad_size: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "TransientExtraction*",
            order as "ptrdiff_t", block_size as "ptrdiff_t", pad_size as "ptrdiff_t"
        ] {
            ptr->init(order, block_size, pad_size);
        })
    }
}

pub fn transient_ext_set_detection_params(
    ptr: *mut u8,
    power: f64,
    thresh_hi: f64,
    thresh_lo: f64,
    half_window: FlucomaIndex,
    hold: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "TransientExtraction*",
            power as "double", thresh_hi as "double", thresh_lo as "double",
            half_window as "ptrdiff_t", hold as "ptrdiff_t"
        ] {
            ptr->setDetectionParameters(power, thresh_hi, thresh_lo, half_window, hold);
        })
    }
}

pub fn transient_ext_process(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    transients_out: *mut f64,
    residual_out: *mut f64,
    output_len: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "TransientExtraction*",
            input as "const double*", input_len as "ptrdiff_t",
            transients_out as "double*", residual_out as "double*",
            output_len as "ptrdiff_t"
        ] {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            FluidTensorView<double, 1> trans_v(transients_out, 0, output_len);
            FluidTensorView<double, 1> resid_v(residual_out, 0, output_len);
            ptr->process(in_v, trans_v, resid_v, FluidDefaultAllocator());
        })
    }
}

pub fn transient_ext_hop_size(ptr: *mut u8) -> FlucomaIndex {
    unsafe {
        cpp!([ptr as "TransientExtraction*"] -> FlucomaIndex as "ptrdiff_t" {
            return ptr->hopSize();
        })
    }
}

pub fn transient_ext_input_size(ptr: *mut u8) -> FlucomaIndex {
    unsafe {
        cpp!([ptr as "TransientExtraction*"] -> FlucomaIndex as "ptrdiff_t" {
            return ptr->inputSize();
        })
    }
}

// -------------------------------------------------------------------------------------------------
// MultiStats

pub fn multistats_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new MultiStats());
        })
    }
}

pub fn multistats_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "MultiStats*"] {
            delete ptr;
        })
    }
}

pub fn multistats_init(
    ptr: *mut u8,
    num_derivatives: FlucomaIndex,
    low_percentile: f64,
    middle_percentile: f64,
    high_percentile: f64,
) {
    unsafe {
        cpp!([
            ptr as "MultiStats*",
            num_derivatives as "ptrdiff_t",
            low_percentile as "double",
            middle_percentile as "double",
            high_percentile as "double"
        ] {
            ptr->init(num_derivatives, low_percentile, middle_percentile, high_percentile);
        })
    }
}

pub fn multistats_process(
    ptr: *mut u8,
    input: *const f64,
    num_channels: FlucomaIndex,
    num_frames: FlucomaIndex,
    output: *mut f64,
    output_cols: FlucomaIndex,
    outliers_cutoff: f64,
    weights: *const f64,
    weights_len: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "MultiStats*",
            input as "const double*",
            num_channels as "ptrdiff_t",
            num_frames as "ptrdiff_t",
            output as "double*",
            output_cols as "ptrdiff_t",
            outliers_cutoff as "double",
            weights as "const double*",
            weights_len as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> in_v(
                const_cast<double*>(input),
                0,
                num_channels,
                num_frames
            );
            FluidTensorView<double, 2> out_v(output, 0, num_channels, output_cols);
            if (weights_len > 0 && weights != nullptr) {
                RealVectorView weight_v(const_cast<double*>(weights), 0, weights_len);
                ptr->process(in_v, out_v, outliers_cutoff, weight_v);
            } else {
                RealVectorView no_weights(nullptr, 0, 0);
                ptr->process(in_v, out_v, outliers_cutoff, no_weights);
            }
        })
    }
}

// -------------------------------------------------------------------------------------------------
// RunningStats

pub fn running_stats_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new RunningStats());
        })
    }
}

pub fn running_stats_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "RunningStats*"] {
            delete ptr;
        })
    }
}

pub fn running_stats_init(ptr: *mut u8, history_size: FlucomaIndex, input_size: FlucomaIndex) {
    unsafe {
        cpp!([
            ptr as "RunningStats*",
            history_size as "ptrdiff_t",
            input_size as "ptrdiff_t"
        ] {
            ptr->init(history_size, input_size);
        })
    }
}

pub fn running_stats_process(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    mean_out: *mut f64,
    stddev_out: *mut f64,
) {
    unsafe {
        cpp!([
            ptr as "RunningStats*",
            input as "const double*",
            input_len as "ptrdiff_t",
            mean_out as "double*",
            stddev_out as "double*"
        ] {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            FluidTensorView<double, 1> mean_v(mean_out, 0, input_len);
            FluidTensorView<double, 1> std_v(stddev_out, 0, input_len);
            ptr->process(in_v, mean_v, std_v);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// Normalization

pub fn normalization_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new Normalization());
        })
    }
}

pub fn normalization_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "Normalization*"] {
            delete ptr;
        })
    }
}

pub fn normalization_fit(
    ptr: *mut u8,
    min: f64,
    max: f64,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "Normalization*",
            min as "double",
            max as "double",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            ptr->init(min, max, in_v);
        })
    }
}

pub fn normalization_process(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    output: *mut f64,
    inverse: bool,
) {
    unsafe {
        cpp!([
            ptr as "Normalization*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            output as "double*",
            inverse as "bool"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            FluidTensorView<double, 2> out_v(output, 0, rows, cols);
            ptr->process(in_v, out_v, inverse);
        })
    }
}

pub fn normalization_initialized(ptr: *mut u8) -> bool {
    unsafe {
        cpp!([ptr as "Normalization*"] -> bool as "bool" {
            return ptr->initialized();
        })
    }
}

// -------------------------------------------------------------------------------------------------
// Standardization

pub fn standardization_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new Standardization());
        })
    }
}

pub fn standardization_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "Standardization*"] {
            delete ptr;
        })
    }
}

pub fn standardization_fit(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "Standardization*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            ptr->init(in_v);
        })
    }
}

pub fn standardization_process(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    output: *mut f64,
    inverse: bool,
) {
    unsafe {
        cpp!([
            ptr as "Standardization*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            output as "double*",
            inverse as "bool"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            FluidTensorView<double, 2> out_v(output, 0, rows, cols);
            ptr->process(in_v, out_v, inverse);
        })
    }
}

pub fn standardization_initialized(ptr: *mut u8) -> bool {
    unsafe {
        cpp!([ptr as "Standardization*"] -> bool as "bool" {
            return ptr->initialized();
        })
    }
}

// -------------------------------------------------------------------------------------------------
// RobustScaling

pub fn robust_scaling_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new RobustScaling());
        })
    }
}

pub fn robust_scaling_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "RobustScaling*"] {
            delete ptr;
        })
    }
}

pub fn robust_scaling_fit(
    ptr: *mut u8,
    low: f64,
    high: f64,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "RobustScaling*",
            low as "double",
            high as "double",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            ptr->init(low, high, in_v);
        })
    }
}

pub fn robust_scaling_process(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    output: *mut f64,
    inverse: bool,
) {
    unsafe {
        cpp!([
            ptr as "RobustScaling*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            output as "double*",
            inverse as "bool"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            FluidTensorView<double, 2> out_v(output, 0, rows, cols);
            ptr->process(in_v, out_v, inverse);
        })
    }
}

pub fn robust_scaling_initialized(ptr: *mut u8) -> bool {
    unsafe {
        cpp!([ptr as "RobustScaling*"] -> bool as "bool" {
            return ptr->initialized();
        })
    }
}

// -------------------------------------------------------------------------------------------------
// PCA

pub fn pca_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new PCA());
        })
    }
}

pub fn pca_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "PCA*"] {
            delete ptr;
        })
    }
}

pub fn pca_fit(ptr: *mut u8, input: *const f64, rows: FlucomaIndex, cols: FlucomaIndex) {
    unsafe {
        cpp!([
            ptr as "PCA*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            ptr->init(in_v);
        })
    }
}

pub fn pca_transform(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    output: *mut f64,
    k: FlucomaIndex,
    whiten: bool,
) -> f64 {
    unsafe {
        cpp!([
            ptr as "PCA*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            output as "double*",
            k as "ptrdiff_t",
            whiten as "bool"
        ] -> f64 as "double" {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            FluidTensorView<double, 2> out_v(output, 0, rows, k);
            return ptr->process(in_v, out_v, k, whiten);
        })
    }
}

pub fn pca_inverse_transform(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    output: *mut f64,
    out_cols: FlucomaIndex,
    whiten: bool,
) {
    unsafe {
        cpp!([
            ptr as "PCA*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            output as "double*",
            out_cols as "ptrdiff_t",
            whiten as "bool"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            FluidTensorView<double, 2> out_v(output, 0, rows, out_cols);
            ptr->inverseProcess(in_v, out_v, whiten);
        })
    }
}

pub fn pca_initialized(ptr: *mut u8) -> bool {
    unsafe {
        cpp!([ptr as "PCA*"] -> bool as "bool" {
            return ptr->initialized();
        })
    }
}

pub fn pca_dims(ptr: *mut u8) -> FlucomaIndex {
    unsafe {
        cpp!([ptr as "PCA*"] -> FlucomaIndex as "ptrdiff_t" {
            return ptr->dims();
        })
    }
}

// -------------------------------------------------------------------------------------------------
// KDTree

pub fn kdtree_create(dims: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([dims as "ptrdiff_t"] -> *mut u8 as "void*" {
            KDTree::DataSet data_set(dims);
            return static_cast<void*>(new KDTree(data_set));
        })
    }
}

pub fn kdtree_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "KDTree*"] {
            delete ptr;
        })
    }
}

pub fn kdtree_add_node(ptr: *mut u8, id: *const u8, data: *const f64, len: FlucomaIndex) {
    unsafe {
        cpp!([ptr as "KDTree*", id as "const char*", data as "const double*", len as "ptrdiff_t"] {
            FluidTensorView<double, 1> data_v(const_cast<double*>(data), 0, len);
            auto flat = ptr->toFlat();
            KDTree::DataSet data_set(flat.ids, flat.data);
            data_set.add(std::string(id), data_v);
            *ptr = KDTree(data_set);
        })
    }
}

pub fn kdtree_k_nearest(
    ptr: *mut u8,
    input: *const f64,
    input_len: FlucomaIndex,
    k: FlucomaIndex,
    radius: f64,
    out_distances: *mut f64,
    out_ids: *mut *const u8,
) {
    unsafe {
        cpp!([
            ptr as "KDTree*",
            input as "const double*",
            input_len as "ptrdiff_t",
            k as "ptrdiff_t",
            radius as "double",
            out_distances as "double*",
            out_ids as "const char**"
        ] {
            FluidTensorView<double, 1> in_v(const_cast<double*>(input), 0, input_len);
            Allocator alloc{};
            auto result = ptr->kNearest(in_v, k, radius, alloc);
            for (fluid::index i = 0; i < static_cast<fluid::index>(result.first.size()); ++i) {
                out_distances[i] = result.first[i];
                out_ids[i] = result.second[i]->c_str();
            }
        })
    }
}

// -------------------------------------------------------------------------------------------------
// MDS

pub fn mds_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new MDS());
        })
    }
}

pub fn mds_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "MDS*"] {
            delete ptr;
        })
    }
}

pub fn mds_process(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    output: *mut f64,
    target_dims: FlucomaIndex,
    distance: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "MDS*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            output as "double*",
            target_dims as "ptrdiff_t",
            distance as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            FluidTensorView<double, 2> out_v(output, 0, rows, target_dims);
            ptr->process(in_v, out_v, distance, target_dims);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// KMeans

pub fn kmeans_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new KMeans());
        })
    }
}

pub fn kmeans_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "KMeans*"] {
            delete ptr;
        })
    }
}

pub fn kmeans_fit(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    k: FlucomaIndex,
    max_iter: FlucomaIndex,
    init_method: FlucomaIndex,
    seed: FlucomaIndex,
    means_out: *mut f64,
    assignments_out: *mut FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "KMeans*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            k as "ptrdiff_t",
            max_iter as "ptrdiff_t",
            init_method as "ptrdiff_t",
            seed as "ptrdiff_t",
            means_out as "double*",
            assignments_out as "ptrdiff_t*"
        ] {
            FluidDataSet<std::string, double, 1> ds(cols);
            for (ptrdiff_t r = 0; r < rows; ++r) {
                RealVector point(cols);
                for (ptrdiff_t c = 0; c < cols; ++c) point(c) = input[r * cols + c];
                ds.add(std::to_string(r), point);
            }

            auto init = static_cast<KMeans::InitMethod>(init_method);
            ptr->train(ds, k, max_iter, init, seed);

            FluidTensor<double, 2> means(k, cols);
            ptr->getMeans(means);
            for (ptrdiff_t i = 0; i < k * cols; ++i) means_out[i] = means.data()[i];

            FluidTensor<fluid::index, 1> assignments(rows);
            ptr->getAssignments(assignments);
            for (ptrdiff_t i = 0; i < rows; ++i) assignments_out[i] = assignments(i);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// SKMeans

pub fn skmeans_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new SKMeans());
        })
    }
}

pub fn skmeans_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "SKMeans*"] {
            delete ptr;
        })
    }
}

pub fn skmeans_fit(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    k: FlucomaIndex,
    max_iter: FlucomaIndex,
    init_method: FlucomaIndex,
    seed: FlucomaIndex,
    means_out: *mut f64,
    assignments_out: *mut FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "SKMeans*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            k as "ptrdiff_t",
            max_iter as "ptrdiff_t",
            init_method as "ptrdiff_t",
            seed as "ptrdiff_t",
            means_out as "double*",
            assignments_out as "ptrdiff_t*"
        ] {
            FluidDataSet<std::string, double, 1> ds(cols);
            for (ptrdiff_t r = 0; r < rows; ++r) {
                RealVector point(cols);
                for (ptrdiff_t c = 0; c < cols; ++c) point(c) = input[r * cols + c];
                ds.add(std::to_string(r), point);
            }

            auto init = static_cast<SKMeans::InitMethod>(init_method);
            ptr->train(ds, k, max_iter, init, seed);

            FluidTensor<double, 2> means(k, cols);
            ptr->getMeans(means);
            for (ptrdiff_t i = 0; i < k * cols; ++i) means_out[i] = means.data()[i];

            FluidTensor<fluid::index, 1> assignments(rows);
            ptr->getAssignments(assignments);
            for (ptrdiff_t i = 0; i < rows; ++i) assignments_out[i] = assignments(i);
        })
    }
}

pub fn skmeans_encode(
    ptr: *mut u8,
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    alpha: f64,
    out: *mut f64,
    out_cols: FlucomaIndex,
) {
    unsafe {
        cpp!([
            ptr as "SKMeans*",
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            alpha as "double",
            out as "double*",
            out_cols as "ptrdiff_t"
        ] {
            FluidTensorView<double, 2> in_v(const_cast<double*>(input), 0, rows, cols);
            FluidTensorView<double, 2> out_v(out, 0, rows, out_cols);
            ptr->encode(in_v, out_v, alpha);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// Grid

pub fn grid_process(
    input: *const f64,
    rows: FlucomaIndex,
    over_sample: FlucomaIndex,
    extent: FlucomaIndex,
    axis: FlucomaIndex,
    output: *mut f64,
) -> bool {
    unsafe {
        cpp!([
            input as "const double*",
            rows as "ptrdiff_t",
            over_sample as "ptrdiff_t",
            extent as "ptrdiff_t",
            axis as "ptrdiff_t",
            output as "double*"
        ] -> bool as "bool" {
            if (rows <= 0) return false;
            Grid::DataSet ds(2);
            for (ptrdiff_t r = 0; r < rows; ++r) {
                RealVector point(2);
                point(0) = input[r * 2];
                point(1) = input[r * 2 + 1];
                ds.add(std::to_string(r), point);
            }
            Grid g;
            auto result = g.process(ds, over_sample, extent, axis);
            if (result.size() != rows) return false;
            for (ptrdiff_t r = 0; r < rows; ++r) {
                RealVector point(2);
                if (!result.get(std::to_string(r), point)) return false;
                output[r * 2] = point(0);
                output[r * 2 + 1] = point(1);
            }
            return true;
        })
    }
}

// -------------------------------------------------------------------------------------------------
// DataSetQuery

pub fn dataset_query_process(
    input: *const f64,
    rows: FlucomaIndex,
    cols: FlucomaIndex,
    selected_cols: *const FlucomaIndex,
    selected_count: FlucomaIndex,
    cond_cols: *const FlucomaIndex,
    cond_ops: *const FlucomaIndex,
    cond_vals: *const f64,
    cond_and_flags: *const FlucomaIndex,
    cond_count: FlucomaIndex,
    limit: FlucomaIndex,
    out_data: *mut f64,
    out_ids: *mut FlucomaIndex,
    out_count: *mut FlucomaIndex,
) -> bool {
    unsafe {
        cpp!([
            input as "const double*",
            rows as "ptrdiff_t",
            cols as "ptrdiff_t",
            selected_cols as "const ptrdiff_t*",
            selected_count as "ptrdiff_t",
            cond_cols as "const ptrdiff_t*",
            cond_ops as "const ptrdiff_t*",
            cond_vals as "const double*",
            cond_and_flags as "const ptrdiff_t*",
            cond_count as "ptrdiff_t",
            limit as "ptrdiff_t",
            out_data as "double*",
            out_ids as "ptrdiff_t*",
            out_count as "ptrdiff_t*"
        ] -> bool as "bool" {
            if (rows <= 0 || cols <= 0 || selected_count <= 0) return false;

            DataSetQuery::DataSet in_ds(cols);
            for (ptrdiff_t r = 0; r < rows; ++r) {
                RealVector point(cols);
                for (ptrdiff_t c = 0; c < cols; ++c) point(c) = input[r * cols + c];
                in_ds.add(std::to_string(r), point);
            }

            DataSetQuery query;
            for (ptrdiff_t i = 0; i < selected_count; ++i) query.addColumn(selected_cols[i]);

            auto op_str = [](ptrdiff_t op) -> const char* {
                switch (op) {
                    case 0: return "==";
                    case 1: return "!=";
                    case 2: return "<";
                    case 3: return "<=";
                    case 4: return ">";
                    case 5: return ">=";
                    default: return "==";
                }
            };

            for (ptrdiff_t i = 0; i < cond_count; ++i) {
                bool conjunction = cond_and_flags[i] != 0;
                if (!query.addCondition(cond_cols[i], op_str(cond_ops[i]), cond_vals[i], conjunction)) {
                    return false;
                }
            }

            if (limit > 0) query.limit(limit);

            DataSetQuery::DataSet current(0);
            DataSetQuery::DataSet out_ds(selected_count);
            query.process(in_ds, current, out_ds);

            ptrdiff_t n = out_ds.size();
            *out_count = n;
            auto ids = out_ds.getIds();
            auto data = out_ds.getData();
            for (ptrdiff_t r = 0; r < n; ++r) {
                out_ids[r] = std::stoll(ids(r));
                for (ptrdiff_t c = 0; c < selected_count; ++c) {
                    out_data[r * selected_count + c] = data(r, c);
                }
            }
            return true;
        })
    }
}
