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
    #include <flucoma/algorithms/public/Loudness.hpp>
    #include <flucoma/algorithms/public/STFT.hpp>
    #include <flucoma/algorithms/public/MelBands.hpp>
    #include <flucoma/algorithms/public/OnsetDetectionFunctions.hpp>
    #include <flucoma/algorithms/public/OnsetSegmentation.hpp>
    #include <flucoma/algorithms/public/AudioTransport.hpp>
    #include <flucoma/algorithms/public/NMF.hpp>
    #include <flucoma/algorithms/public/NMFMorph.hpp>
    #include <flucoma/algorithms/public/EnvelopeSegmentation.hpp>
    #include <flucoma/algorithms/public/NoveltySegmentation.hpp>
    #include <flucoma/algorithms/public/TransientSegmentation.hpp>
    using namespace fluid;
    using namespace fluid::algorithm;
}}

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
            auto* cptr = reinterpret_cast<std::complex<double>*>(
                const_cast<double*>(in_complex));
            FluidTensorView<std::complex<double>, 1> in_v(cptr, 0, num_bins);
            FluidTensorView<double, 1> out_v(output, 0, output_len);
            ptr->processFrame(in_v, out_v);
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
            Allocator alloc{};
            ptr->processFrame(in_v, out_v, mag_norm, use_power, log_output, alloc);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// AudioTransport

pub fn audio_transport_create(max_fft_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_fft_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            Allocator alloc{};
            return static_cast<void*>(new AudioTransport(max_fft_size, alloc));
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
            Allocator alloc{};
            ptr->processFrame(in1_v, in2_v, weight, out_v, alloc);
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
            Allocator alloc{};
            ptr->processFrame(x_v, w_v, out_v, n_iterations, est_v, random_seed, alloc);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// NMFMorph

pub fn nmf_morph_create(max_fft_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_fft_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            Allocator alloc{};
            return static_cast<void*>(new NMFMorph(max_fft_size, alloc));
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
            Allocator alloc{};
            ptr->init(w1_v, w2_v, h_v, win_size, fft_size, hop_size, assign, alloc);
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
            Allocator alloc{};
            ptr->processFrame(v, interpolation, seed, alloc);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// OnsetDetectionFunctions

pub fn onset_create(max_size: FlucomaIndex, max_filter_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_size as "ptrdiff_t", max_filter_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            Allocator alloc{};
            return static_cast<void*>(
                new OnsetDetectionFunctions(max_size, max_filter_size, alloc));
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
            Allocator alloc{};
            return ptr->processFrame(in_v, function, filter_size, frame_delta, alloc);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// OnsetSegmentation

pub fn onset_seg_create(max_size: FlucomaIndex, max_filter_size: FlucomaIndex) -> *mut u8 {
    unsafe {
        cpp!([max_size as "ptrdiff_t", max_filter_size as "ptrdiff_t"] -> *mut u8 as "void*" {
            Allocator alloc{};
            return static_cast<void*>(
                new OnsetSegmentation(max_size, max_filter_size, alloc));
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
            Allocator alloc{};
            return ptr->processFrame(in_v, function, filter_size, threshold, debounce, frame_delta, alloc);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// EnvelopeSegmentation

pub fn env_seg_create() -> *mut u8 {
    unsafe {
        cpp!([] -> *mut u8 as "void*" {
            return static_cast<void*>(new EnvelopeSegmentation());
        })
    }
}

pub fn env_seg_destroy(ptr: *mut u8) {
    unsafe {
        cpp!([ptr as "EnvelopeSegmentation*"] {
            delete ptr;
        })
    }
}

pub fn env_seg_init(ptr: *mut u8, floor: f64, hi_pass_freq: f64) {
    unsafe {
        cpp!([
            ptr as "EnvelopeSegmentation*",
            floor as "double", hi_pass_freq as "double"
        ] {
            ptr->init(floor, hi_pass_freq);
        })
    }
}

pub fn env_seg_process_sample(
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
// NoveltySegmentation

pub fn novelty_seg_create(
    max_kernel_size: FlucomaIndex,
    max_dims: FlucomaIndex,
    max_filter_size: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            max_kernel_size as "ptrdiff_t", max_dims as "ptrdiff_t", max_filter_size as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            Allocator alloc{};
            return static_cast<void*>(
                new NoveltySegmentation(max_kernel_size, max_dims, max_filter_size, alloc));
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
            Allocator alloc{};
            ptr->init(kernel_size, filter_size, n_dims, alloc);
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
            Allocator alloc{};
            return ptr->processFrame(in_v, threshold, min_slice_length, alloc);
        })
    }
}

// -------------------------------------------------------------------------------------------------
// TransientSegmentation

pub fn transient_seg_create(
    max_order: FlucomaIndex,
    max_block_size: FlucomaIndex,
    max_pad_size: FlucomaIndex,
) -> *mut u8 {
    unsafe {
        cpp!([
            max_order as "ptrdiff_t", max_block_size as "ptrdiff_t", max_pad_size as "ptrdiff_t"
        ] -> *mut u8 as "void*" {
            Allocator alloc{};
            return static_cast<void*>(
                new TransientSegmentation(max_order, max_block_size, max_pad_size, alloc));
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
            Allocator alloc{};
            ptr->process(in_v, out_v, alloc);
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
