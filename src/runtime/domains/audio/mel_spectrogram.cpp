#include "runtime/domains/audio/mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace trtmc {

namespace {

// Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
// Matches the periodic Hann window convention used by HF feature extractors.
std::vector<float> make_hann_window(int32_t length) {
    std::vector<float> window(length);
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < length; ++i) {
        window[i] = static_cast<float>(
            0.5 * (1.0 - std::cos(pi2 * static_cast<double>(i) / static_cast<double>(length))));
    }
    return window;
}

std::vector<float> make_symmetric_hann_window(int32_t length) {
    std::vector<float> window(length);
    if (length <= 1)
        return window;
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < length; ++i) {
        window[i] = static_cast<float>(
            0.5 * (1.0 - std::cos(pi2 * static_cast<double>(i) / static_cast<double>(length - 1))));
    }
    return window;
}

std::vector<float> make_centered_stft_window(int32_t n_fft, int32_t win_length) {
    std::vector<float> window(static_cast<std::size_t>(n_fft), 0.0F);
    const auto inner = make_symmetric_hann_window(win_length);
    const int32_t offset = std::max(0, (n_fft - win_length) / 2);
    for (int32_t i = 0; i < win_length && offset + i < n_fft; ++i)
        window[static_cast<std::size_t>(offset + i)] = inner[static_cast<std::size_t>(i)];
    return window;
}

// Direct real-to-complex DFT for the first n_out bins.
// Computes X[k] = sum_{n=0}^{N-1} x[n] * e^{-j*2*pi*k*n/N} for k = 0..n_out-1.
// Uses double precision for twiddle factors to match numpy's FFT accuracy.
// Writes squared magnitude |X[k]|^2 directly into power_out.
void rfft_power_direct(const float* x, int32_t n, int32_t n_out, float* power_out) {
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t k = 0; k < n_out; ++k) {
        double re = 0.0, im = 0.0;
        const double w = pi2 * static_cast<double>(k) / static_cast<double>(n);
        for (int32_t t = 0; t < n; ++t) {
            const double angle = w * static_cast<double>(t);
            re += static_cast<double>(x[t]) * std::cos(angle);
            im -= static_cast<double>(x[t]) * std::sin(angle);
        }
        power_out[k] = static_cast<float>(re * re + im * im);
    }
}

std::vector<float> build_center_padded_audio(const float* samples, int32_t n_samples,
                                             int32_t chunk_length_s, int32_t sample_rate,
                                             int32_t n_fft, float preemphasis = 0.0F) {
    const int32_t audio_length = chunk_length_s * sample_rate;
    std::vector<float> audio_padded(audio_length, 0.0F);
    const int32_t copy_len = std::min(n_samples, audio_length);
    if (copy_len > 0) {
        if (preemphasis != 0.0F) {
            audio_padded[0] = samples[0];
            for (int32_t i = 1; i < copy_len; ++i) {
                audio_padded[static_cast<std::size_t>(i)] =
                    samples[i] - preemphasis * samples[i - 1];
            }
        } else {
            std::memcpy(audio_padded.data(), samples, copy_len * sizeof(float));
        }
    }

    const int32_t pad_size = n_fft / 2;
    const int32_t padded_length = pad_size + audio_length + pad_size;
    std::vector<float> padded(padded_length, 0.0F);
    std::memcpy(padded.data() + pad_size, audio_padded.data(), audio_length * sizeof(float));
    return padded;
}

int32_t resolve_num_freq_bins(int32_t n_freq_bins, int32_t n_fft) {
    const int32_t expected_freq_bins = n_fft / 2 + 1;
    return (n_freq_bins == expected_freq_bins) ? n_freq_bins : expected_freq_bins;
}

std::vector<float> compute_power_spectrogram_with_window(const std::vector<float>& padded,
                                                         const std::vector<float>& window,
                                                         int32_t n_fft, int32_t hop_length,
                                                         int32_t n_freq_bins,
                                                         int32_t& n_frames_raw) {
    n_frames_raw = 1 + (static_cast<int32_t>(padded.size()) - n_fft) / hop_length;
    std::vector<float> power(static_cast<std::size_t>(n_freq_bins) * n_frames_raw, 0.0F);
    std::vector<float> windowed(n_fft);
    std::vector<float> frame_power(n_freq_bins);

    for (int32_t t = 0; t < n_frames_raw; ++t) {
        const int32_t start = t * hop_length;
        for (int32_t i = 0; i < n_fft; ++i)
            windowed[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(start + i)] * window[static_cast<std::size_t>(i)];

        rfft_power_direct(windowed.data(), n_fft, n_freq_bins, frame_power.data());

        for (int32_t f = 0; f < n_freq_bins; ++f)
            power[static_cast<std::size_t>(f) * n_frames_raw + t] = frame_power[f];
    }

    return power;
}

std::vector<float> project_power_to_mel(const std::vector<float>& power, const float* mel_filters,
                                        int32_t n_freq_bins, int32_t n_mel_bins,
                                        int32_t n_frames_raw) {
    std::vector<float> mel_spec(static_cast<std::size_t>(n_mel_bins) * n_frames_raw, 0.0F);

    for (int32_t t = 0; t < n_frames_raw; ++t) {
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = power[static_cast<std::size_t>(f) * n_frames_raw + t];
            if (p == 0.0F) {
                continue;
            }
            for (int32_t m = 0; m < n_mel_bins; ++m) {
                mel_spec[static_cast<std::size_t>(m) * n_frames_raw + t] +=
                    p * mel_filters[static_cast<std::size_t>(f) * n_mel_bins + m];
            }
        }
    }

    return mel_spec;
}

void normalize_log_mel_inplace(std::vector<float>& mel_spec) {
    float global_max = -1e10F;
    for (float& value : mel_spec) {
        value = std::log10(std::max(value, 1e-10F));
        if (value > global_max) {
            global_max = value;
        }
    }

    const float floor = global_max - 8.0F;
    for (float& value : mel_spec) {
        value = std::max(value, floor);
        value = (value + 4.0F) / 4.0F;
    }
}

void natural_log_mel_inplace(std::vector<float>& mel_spec) {
    constexpr float kLogGuard = 5.960464477539063e-08F; // 2**-24
    for (float& value : mel_spec)
        value = std::log(value + kLogGuard);
}

void normalize_per_feature_inplace(std::vector<float>& mel_spec, int32_t n_mel_bins,
                                   int32_t n_frames_raw, int32_t valid_frames) {
    if (n_mel_bins <= 0 || n_frames_raw <= 0) {
        return;
    }

    valid_frames = std::clamp(valid_frames, 0, n_frames_raw);
    if (valid_frames <= 0) {
        std::fill(mel_spec.begin(), mel_spec.end(), 0.0F);
        return;
    }

    constexpr float kStdGuard = 1e-5F;
    for (int32_t m = 0; m < n_mel_bins; ++m) {
        const std::size_t base = static_cast<std::size_t>(m) * n_frames_raw;

        double mean = 0.0;
        for (int32_t t = 0; t < valid_frames; ++t)
            mean += static_cast<double>(mel_spec[base + static_cast<std::size_t>(t)]);
        mean /= static_cast<double>(valid_frames);

        if (valid_frames == 1) {
            mel_spec[base] = 0.0F;
        } else {
            double var = 0.0;
            for (int32_t t = 0; t < valid_frames; ++t) {
                const double diff =
                    static_cast<double>(mel_spec[base + static_cast<std::size_t>(t)]) - mean;
                var += diff * diff;
            }
            const double stddev =
                std::sqrt(var / static_cast<double>(valid_frames - 1)) + kStdGuard;
            for (int32_t t = 0; t < valid_frames; ++t) {
                const std::size_t idx = base + static_cast<std::size_t>(t);
                mel_spec[idx] =
                    static_cast<float>((static_cast<double>(mel_spec[idx]) - mean) / stddev);
            }
        }

        for (int32_t t = valid_frames; t < n_frames_raw; ++t)
            mel_spec[base + static_cast<std::size_t>(t)] = 0.0F;
    }
}

std::vector<float> trim_last_frame_if_needed(std::vector<float> mel_spec, int32_t n_mel_bins,
                                             int32_t n_frames_raw, int32_t& n_frames_out) {
    n_frames_out = n_frames_raw;
    if (n_frames_raw <= 1) {
        return mel_spec;
    }

    n_frames_out = n_frames_raw - 1;
    std::vector<float> trimmed(static_cast<std::size_t>(n_mel_bins) * n_frames_out);
    for (int32_t m = 0; m < n_mel_bins; ++m) {
        std::memcpy(trimmed.data() + static_cast<std::size_t>(m) * n_frames_out,
                    mel_spec.data() + static_cast<std::size_t>(m) * n_frames_raw,
                    n_frames_out * sizeof(float));
    }
    return trimmed;
}

std::vector<float> make_stft_window(const MelSpectrogramOptions& options) {
    const int32_t win_length = options.win_length > 0 ? options.win_length : options.n_fft;
    if (options.center_window_in_fft) {
        return make_centered_stft_window(options.n_fft, win_length);
    }
    if (options.symmetric_window) {
        return make_symmetric_hann_window(options.n_fft);
    }
    return make_hann_window(options.n_fft);
}

void apply_log_scale_inplace(std::vector<float>& mel_spec, MelLogScale log_scale) {
    switch (log_scale) {
    case MelLogScale::kNaturalLog:
        natural_log_mel_inplace(mel_spec);
        break;
    case MelLogScale::kLog10Normalized:
        normalize_log_mel_inplace(mel_spec);
        break;
    }
}

} // anonymous namespace

MelResult extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                                  int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft,
                                  int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate) {
    MelSpectrogramOptions options;
    options.n_fft = n_fft;
    options.hop_length = hop_length;
    options.chunk_length_s = chunk_length_s;
    options.sample_rate = sample_rate;
    options.log_scale = MelLogScale::kLog10Normalized;
    return extract_configured_mel_spectrogram(samples, n_samples, mel_filters, n_freq_bins,
                                              n_mel_bins, options);
}

MelResult extract_configured_mel_spectrogram(const float* samples, int32_t n_samples,
                                             const float* mel_filters, int32_t n_freq_bins,
                                             int32_t n_mel_bins,
                                             const MelSpectrogramOptions& options) {
    const std::vector<float> padded =
        build_center_padded_audio(samples, n_samples, options.chunk_length_s, options.sample_rate,
                                  options.n_fft, options.preemphasis);
    const int32_t freq_bins = resolve_num_freq_bins(n_freq_bins, options.n_fft);

    int32_t n_frames_raw = 0;
    const auto window = make_stft_window(options);
    const std::vector<float> power = compute_power_spectrogram_with_window(
        padded, window, options.n_fft, options.hop_length, freq_bins, n_frames_raw);
    std::vector<float> mel_spec =
        project_power_to_mel(power, mel_filters, freq_bins, n_mel_bins, n_frames_raw);
    apply_log_scale_inplace(mel_spec, options.log_scale);
    if (options.normalize_per_feature) {
        const int32_t valid_frames = (options.hop_length > 0) ? (n_samples / options.hop_length) : 0;
        normalize_per_feature_inplace(mel_spec, n_mel_bins, n_frames_raw, valid_frames);
    }

    int32_t n_frames_out = 0;
    mel_spec =
        trim_last_frame_if_needed(std::move(mel_spec), n_mel_bins, n_frames_raw, n_frames_out);

    MelResult result;
    result.data = std::move(mel_spec);
    result.n_mels = n_mel_bins;
    result.n_frames = n_frames_out;
    return result;
}

} // namespace trtmc
