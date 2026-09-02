/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_helpers.h"

#include "utils/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace tdt {

namespace {

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
    if (length <= 1) {
        return window;
    }
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
    for (int32_t i = 0; i < win_length && offset + i < n_fft; ++i) {
        window[static_cast<std::size_t>(offset + i)] = inner[static_cast<std::size_t>(i)];
    }
    return window;
}

void rfft_power_direct(const float* x, int32_t n, int32_t n_out, float* power_out) {
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t k = 0; k < n_out; ++k) {
        double re = 0.0;
        double im = 0.0;
        const double w = pi2 * static_cast<double>(k) / static_cast<double>(n);
        for (int32_t t = 0; t < n; ++t) {
            const double angle = w * static_cast<double>(t);
            re += static_cast<double>(x[t]) * std::cos(angle);
            im -= static_cast<double>(x[t]) * std::sin(angle);
        }
        power_out[k] = static_cast<float>(re * re + im * im);
    }
}

bool is_power_of_two(int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void rfft_power_radix2(const float* x, int32_t n, int32_t n_out, float* power_out,
                       std::vector<std::complex<double>>& workspace,
                       const std::vector<std::complex<double>>& twiddles) {
    workspace.resize(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        workspace[static_cast<std::size_t>(i)] =
            std::complex<double>(static_cast<double>(x[i]), 0.0);
    }

    for (int32_t i = 1, j = 0; i < n; ++i) {
        int32_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(workspace[static_cast<std::size_t>(i)],
                      workspace[static_cast<std::size_t>(j)]);
        }
    }

    for (int32_t length = 2; length <= n; length <<= 1) {
        const int32_t half = length / 2;
        for (int32_t offset = 0; offset < n; offset += length) {
            for (int32_t i = 0; i < half; ++i) {
                const auto even = workspace[static_cast<std::size_t>(offset + i)];
                const auto twiddle = twiddles[static_cast<std::size_t>(i * n / length)];
                const auto odd = workspace[static_cast<std::size_t>(offset + i + half)] * twiddle;
                workspace[static_cast<std::size_t>(offset + i)] = even + odd;
                workspace[static_cast<std::size_t>(offset + i + half)] = even - odd;
            }
        }
    }

    const int32_t bins = std::min(n_out, n / 2 + 1);
    for (int32_t k = 0; k < bins; ++k) {
        power_out[k] = static_cast<float>(std::norm(workspace[static_cast<std::size_t>(k)]));
    }
    std::fill(power_out + bins, power_out + n_out, 0.0F);
}

class RfftPowerPlan {
  public:
    explicit RfftPowerPlan(int32_t n) : n_(n) {
        if (is_power_of_two(n_)) {
            const double pi2 = 2.0 * 3.14159265358979323846;
            twiddles_.reserve(static_cast<std::size_t>(n_ / 2));
            for (int32_t k = 0; k < n_ / 2; ++k) {
                const double angle = -pi2 * static_cast<double>(k) / static_cast<double>(n_);
                twiddles_.emplace_back(std::cos(angle), std::sin(angle));
            }
        }
    }

    void execute(const float* input, int32_t n_out, float* power_out) {
        if (is_power_of_two(n_)) {
            rfft_power_radix2(input, n_, n_out, power_out, workspace_, twiddles_);
            return;
        }
        rfft_power_direct(input, n_, n_out, power_out);
    }

  private:
    int32_t n_{0};
    std::vector<std::complex<double>> workspace_;
    std::vector<std::complex<double>> twiddles_;
};

std::vector<float> build_center_padded_audio(const float* samples, int32_t n_samples,
                                             int32_t chunk_length_s, int32_t sample_rate,
                                             int32_t n_fft, float preemphasis) {
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

std::vector<float> compute_mel_spectrogram(const std::vector<float>& padded,
                                           const std::vector<float>& window,
                                           const float* mel_filters, int32_t n_fft,
                                           int32_t hop_length, int32_t n_freq_bins,
                                           int32_t n_mel_bins, int32_t frames_to_compute,
                                           int32_t& n_frames_raw) {
    n_frames_raw = 1 + (static_cast<int32_t>(padded.size()) - n_fft) / hop_length;
    std::vector<float> mel_spec(static_cast<std::size_t>(n_mel_bins) * n_frames_raw, 0.0F);
    std::vector<float> windowed(n_fft);
    std::vector<float> frame_power(n_freq_bins);
    std::vector<float> frame_mel(n_mel_bins);
    RfftPowerPlan fft_plan(n_fft);

    const int32_t computed_frames = std::clamp(frames_to_compute, 0, n_frames_raw);
    for (int32_t t = 0; t < computed_frames; ++t) {
        const int32_t start = t * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(start + i)] * window[static_cast<std::size_t>(i)];
        }

        fft_plan.execute(windowed.data(), n_freq_bins, frame_power.data());

        std::fill(frame_mel.begin(), frame_mel.end(), 0.0F);
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = frame_power[static_cast<std::size_t>(f)];
            if (p == 0.0F) {
                continue;
            }
            const float* filter_row = mel_filters + static_cast<std::size_t>(f) * n_mel_bins;
            for (int32_t m = 0; m < n_mel_bins; ++m) {
                frame_mel[static_cast<std::size_t>(m)] +=
                    p * filter_row[static_cast<std::size_t>(m)];
            }
        }
        for (int32_t m = 0; m < n_mel_bins; ++m) {
            mel_spec[static_cast<std::size_t>(m) * n_frames_raw + t] =
                frame_mel[static_cast<std::size_t>(m)];
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
    constexpr float kLogGuard = 5.960464477539063e-08F;
    for (float& value : mel_spec) {
        value = std::log(value + kLogGuard);
    }
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

std::vector<float> trim_last_frame(std::vector<float> mel_spec, int32_t n_mel_bins,
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

} // namespace

namespace detail {

std::vector<float> rfft_power(const std::vector<float>& input) {
    const int32_t n = static_cast<int32_t>(input.size());
    const int32_t n_out = n / 2 + 1;
    std::vector<float> power(static_cast<std::size_t>(n_out), 0.0F);
    if (n > 0) {
        RfftPowerPlan plan(n);
        plan.execute(input.data(), n_out, power.data());
    }
    return power;
}

} // namespace detail

class IncrementalMelSpectrogram::Impl {
  public:
    Impl(const float* mel_filters, int32_t n_freq_bins, int32_t n_mel_bins,
         MelSpectrogramOptions options, int32_t input_sample_rate)
        : options_(std::move(options)), input_sample_rate_(input_sample_rate),
          n_freq_bins_(n_freq_bins), n_mel_bins_(n_mel_bins), window_(make_stft_window(options_)),
          fft_plan_(options_.n_fft), windowed_(static_cast<std::size_t>(options_.n_fft)),
          frame_power_(static_cast<std::size_t>(n_freq_bins)),
          frame_mel_(static_cast<std::size_t>(n_mel_bins)) {
        if (mel_filters == nullptr || n_freq_bins != options_.n_fft / 2 + 1 || n_mel_bins <= 0) {
            throw std::invalid_argument("incremental TDT mel requires a complete filterbank");
        }
        if (input_sample_rate_ <= 0 || options_.sample_rate <= 0 || options_.n_fft <= 0 ||
            options_.hop_length <= 0) {
            throw std::invalid_argument("incremental TDT mel requires positive dimensions");
        }
        if (options_.log_scale != MelLogScale::kNaturalLog || options_.normalize_per_feature) {
            throw std::invalid_argument(
                "incremental TDT mel only supports natural-log features without normalization");
        }
        mel_filters_.assign(mel_filters,
                            mel_filters + static_cast<std::size_t>(n_freq_bins_) * n_mel_bins_);
    }

    void accept_audio(const float* samples, int32_t n_samples) {
        if (n_samples < 0 || (n_samples > 0 && samples == nullptr)) {
            throw std::invalid_argument("incremental TDT mel received invalid audio");
        }
        if (n_samples == 0) {
            return;
        }
        raw_audio_.insert(raw_audio_.end(), samples, samples + n_samples);
    }

    int32_t target_sample_count() const {
        const int64_t converted =
            static_cast<int64_t>(raw_audio_.size()) * options_.sample_rate / input_sample_rate_;
        const int64_t chunk_limit =
            static_cast<int64_t>(options_.chunk_length_s) * options_.sample_rate;
        return static_cast<int32_t>(std::min(converted, chunk_limit));
    }

    int32_t stable_target_sample_count(bool final) const {
        if (final || input_sample_rate_ == options_.sample_rate) {
            return target_sample_count();
        }
        constexpr int32_t kResampleHalfTaps = 16;
        const int64_t stable_source =
            std::max<int64_t>(0, static_cast<int64_t>(raw_audio_.size()) - kResampleHalfTaps);
        const int64_t stable_target = stable_source * options_.sample_rate / input_sample_rate_;
        return static_cast<int32_t>(std::min<int64_t>(stable_target, target_sample_count()));
    }

    int32_t available_frames() const { return target_sample_count() / options_.hop_length; }

    void ensure_resampled_samples(int32_t required_samples, bool final) {
        if (!final && required_samples > stable_target_sample_count(false)) {
            throw std::runtime_error(
                "incremental TDT mel requested samples before resampler lookahead was ready");
        }
        const int32_t bounded_required = std::min(required_samples, target_sample_count());
        const int32_t start = static_cast<int32_t>(resampled_audio_.size());
        if (bounded_required <= start) {
            return;
        }
        auto next = resample_linear_range(
            raw_audio_.data(), static_cast<int32_t>(raw_audio_.size()), input_sample_rate_,
            options_.sample_rate, start, bounded_required - start);
        resampled_audio_.insert(resampled_audio_.end(), next.begin(), next.end());
    }

    float preemphasized_sample(int32_t audio_index) const {
        if (audio_index < 0 || audio_index >= static_cast<int32_t>(resampled_audio_.size())) {
            return 0.0F;
        }
        const float current = resampled_audio_[static_cast<std::size_t>(audio_index)];
        if (options_.preemphasis == 0.0F || audio_index == 0) {
            return current;
        }
        return current -
               options_.preemphasis * resampled_audio_[static_cast<std::size_t>(audio_index - 1)];
    }

    void compute_frame(int32_t frame) {
        const int32_t pad_size = options_.n_fft / 2;
        const int32_t frame_start = frame * options_.hop_length - pad_size;
        for (int32_t i = 0; i < options_.n_fft; ++i) {
            windowed_[static_cast<std::size_t>(i)] =
                preemphasized_sample(frame_start + i) * window_[static_cast<std::size_t>(i)];
        }
        fft_plan_.execute(windowed_.data(), n_freq_bins_, frame_power_.data());

        std::fill(frame_mel_.begin(), frame_mel_.end(), 0.0F);
        for (int32_t f = 0; f < n_freq_bins_; ++f) {
            const float power = frame_power_[static_cast<std::size_t>(f)];
            const float* filter_row =
                mel_filters_.data() + static_cast<std::size_t>(f) * n_mel_bins_;
            for (int32_t m = 0; m < n_mel_bins_; ++m) {
                frame_mel_[static_cast<std::size_t>(m)] +=
                    power * filter_row[static_cast<std::size_t>(m)];
            }
        }
        constexpr float kLogGuard = 5.960464477539063e-08F;
        for (float value : frame_mel_) {
            frames_.push_back(std::log(value + kLogGuard));
        }
    }

    void ensure_frames(int32_t end_frame, bool final) {
        if (end_frame < 0 || end_frame > available_frames()) {
            throw std::out_of_range("incremental TDT mel frame request exceeds available audio");
        }
        const int32_t current_frames = frame_count();
        if (end_frame <= current_frames) {
            return;
        }
        const int32_t last_frame = end_frame - 1;
        const int32_t required_samples =
            last_frame * options_.hop_length + options_.n_fft - options_.n_fft / 2;
        ensure_resampled_samples(required_samples, final);
        for (int32_t frame = current_frames; frame < end_frame; ++frame) {
            compute_frame(frame);
        }
    }

    void reset() {
        raw_audio_.clear();
        resampled_audio_.clear();
        frames_.clear();
    }

    int32_t frame_count() const {
        return static_cast<int32_t>(frames_.size() / static_cast<std::size_t>(n_mel_bins_));
    }

    int32_t n_mels() const { return n_mel_bins_; }

    float value(int32_t mel_bin, int32_t frame) const {
        if (mel_bin < 0 || mel_bin >= n_mel_bins_ || frame < 0 || frame >= frame_count()) {
            throw std::out_of_range("incremental TDT mel feature index is out of range");
        }
        return frames_[static_cast<std::size_t>(frame) * n_mel_bins_ + mel_bin];
    }

    IncrementalMelStats stats() const {
        return {static_cast<int64_t>(raw_audio_.size()),
                static_cast<int64_t>(resampled_audio_.size()), frame_count()};
    }

    MelSpectrogramOptions options_;
    int32_t input_sample_rate_{0};
    int32_t n_freq_bins_{0};
    int32_t n_mel_bins_{0};
    std::vector<float> mel_filters_;
    std::vector<float> window_;
    RfftPowerPlan fft_plan_;
    std::vector<float> raw_audio_;
    std::vector<float> resampled_audio_;
    std::vector<float> frames_;
    std::vector<float> windowed_;
    std::vector<float> frame_power_;
    std::vector<float> frame_mel_;
};

IncrementalMelSpectrogram::IncrementalMelSpectrogram(const float* mel_filters, int32_t n_freq_bins,
                                                     int32_t n_mel_bins,
                                                     MelSpectrogramOptions options,
                                                     int32_t input_sample_rate)
    : impl_(std::make_unique<Impl>(mel_filters, n_freq_bins, n_mel_bins, std::move(options),
                                   input_sample_rate)) {}

IncrementalMelSpectrogram::~IncrementalMelSpectrogram() = default;
IncrementalMelSpectrogram::IncrementalMelSpectrogram(IncrementalMelSpectrogram&&) noexcept =
    default;
IncrementalMelSpectrogram&
IncrementalMelSpectrogram::operator=(IncrementalMelSpectrogram&&) noexcept = default;

void IncrementalMelSpectrogram::accept_audio(const float* samples, int32_t n_samples) {
    impl_->accept_audio(samples, n_samples);
}

void IncrementalMelSpectrogram::ensure_frames(int32_t end_frame, bool final) {
    impl_->ensure_frames(end_frame, final);
}

void IncrementalMelSpectrogram::reset() {
    impl_->reset();
}

int32_t IncrementalMelSpectrogram::available_frames() const {
    return impl_->available_frames();
}

int32_t IncrementalMelSpectrogram::stable_target_sample_count(bool final) const {
    return impl_->stable_target_sample_count(final);
}

int32_t IncrementalMelSpectrogram::frame_count() const {
    return impl_->frame_count();
}

int32_t IncrementalMelSpectrogram::n_mels() const {
    return impl_->n_mels();
}

float IncrementalMelSpectrogram::value(int32_t mel_bin, int32_t frame) const {
    return impl_->value(mel_bin, frame);
}

IncrementalMelStats IncrementalMelSpectrogram::stats() const {
    return impl_->stats();
}

MelResult extract_configured_mel_spectrogram(const float* samples, int32_t n_samples,
                                             const float* mel_filters, int32_t n_freq_bins,
                                             int32_t n_mel_bins,
                                             const MelSpectrogramOptions& options) {
    const int32_t expected_freq_bins = options.n_fft / 2 + 1;
    const int32_t freq_bins = n_freq_bins == expected_freq_bins ? n_freq_bins : expected_freq_bins;
    const int32_t chunk_samples = options.chunk_length_s * options.sample_rate;
    const int32_t valid_audio = std::min(std::max(n_samples, 0), chunk_samples);
    const int32_t pad_size = options.n_fft / 2;
    const int32_t frames_to_compute = valid_audio > 0 && options.hop_length > 0
                                          ? 1 + (pad_size + valid_audio - 1) / options.hop_length
                                          : 0;
    const std::vector<float> padded =
        build_center_padded_audio(samples, n_samples, options.chunk_length_s, options.sample_rate,
                                  options.n_fft, options.preemphasis);

    int32_t n_frames_raw = 0;
    std::vector<float> mel_spec = compute_mel_spectrogram(
        padded, make_stft_window(options), mel_filters, options.n_fft, options.hop_length,
        freq_bins, n_mel_bins, frames_to_compute, n_frames_raw);
    apply_log_scale_inplace(mel_spec, options.log_scale);
    if (options.normalize_per_feature) {
        const int32_t valid_frames =
            (options.hop_length > 0) ? (n_samples / options.hop_length) : 0;
        normalize_per_feature_inplace(mel_spec, n_mel_bins, n_frames_raw, valid_frames);
    }

    int32_t n_frames_out = 0;
    mel_spec = trim_last_frame(std::move(mel_spec), n_mel_bins, n_frames_raw, n_frames_out);

    MelResult result;
    result.data = std::move(mel_spec);
    result.n_mels = n_mel_bins;
    result.n_frames = n_frames_out;
    return result;
}

MelResult extract_tdt_mel_spectrogram(const float* samples, int32_t n_samples,
                                      const float* mel_filters, int32_t n_freq_bins,
                                      int32_t n_mel_bins, int32_t n_fft, int32_t win_length,
                                      int32_t hop_length, int32_t chunk_length_s,
                                      int32_t sample_rate, float preemphasis) {
    MelSpectrogramOptions options;
    options.n_fft = n_fft;
    options.win_length = win_length;
    options.hop_length = hop_length;
    options.chunk_length_s = chunk_length_s;
    options.sample_rate = sample_rate;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = preemphasis;
    options.log_scale = MelLogScale::kNaturalLog;
    options.normalize_per_feature = true;
    return extract_configured_mel_spectrogram(samples, n_samples, mel_filters, n_freq_bins,
                                              n_mel_bins, options);
}

} // namespace tdt
} // namespace trtmc
