/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_helpers.h"

#include "families/nemotron_voicechat/runtime/resampler.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace voicechat_audio {

namespace {

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
                       std::vector<std::complex<double>>& workspace) {
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

    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t length = 2; length <= n; length <<= 1) {
        const double angle = -pi2 / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        const int32_t half = length / 2;
        for (int32_t offset = 0; offset < n; offset += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for (int32_t i = 0; i < half; ++i) {
                const auto even = workspace[static_cast<std::size_t>(offset + i)];
                const auto odd = workspace[static_cast<std::size_t>(offset + i + half)] * twiddle;
                workspace[static_cast<std::size_t>(offset + i)] = even + odd;
                workspace[static_cast<std::size_t>(offset + i + half)] = even - odd;
                twiddle *= step;
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
    explicit RfftPowerPlan(int32_t n) : n_(n) {}

    void execute(const float* input, int32_t n_out, float* power_out) {
        if (is_power_of_two(n_)) {
            rfft_power_radix2(input, n_, n_out, power_out, workspace_);
            return;
        }
        rfft_power_direct(input, n_, n_out, power_out);
    }

  private:
    int32_t n_{0};
    std::vector<std::complex<double>> workspace_;
};

} // namespace

namespace detail {

int32_t reflect_index(int32_t index, int32_t length) {
    if (length <= 1)
        return 0;
    const int32_t period = 2 * (length - 1);
    int32_t folded = index % period;
    if (folded < 0)
        folded += period;
    return folded < length ? folded : period - folded;
}

} // namespace detail

namespace {

void validate_incremental_mel_configuration(const float* mel_filters, int32_t n_freq_bins,
                                            int32_t n_mel_bins,
                                            const MelSpectrogramOptions& options,
                                            int32_t input_sample_rate) {
    if (mel_filters == nullptr || n_freq_bins != options.n_fft / 2 + 1 || n_mel_bins <= 0) {
        throw std::invalid_argument("incremental VoiceChat mel requires a complete filterbank");
    }
    if (input_sample_rate <= 0 || options.sample_rate <= 0 || options.n_fft <= 0 ||
        options.hop_length <= 0) {
        throw std::invalid_argument("incremental VoiceChat mel requires positive dimensions");
    }
    if (options.log_scale != MelLogScale::kNaturalLog || options.normalize_per_feature) {
        throw std::invalid_argument("incremental VoiceChat mel only supports natural-log "
                                    "features without normalization");
    }
}

std::vector<float> make_incremental_mel_window(const MelSpectrogramOptions& options,
                                               const float* exact_window,
                                               int32_t exact_window_length) {
    const int32_t expected = options.win_length > 0 ? options.win_length : options.n_fft;
    if (exact_window == nullptr) {
        throw std::invalid_argument("incremental VoiceChat mel requires the checkpoint window");
    }
    if (exact_window_length != expected) {
        throw std::invalid_argument("incremental VoiceChat mel checkpoint window length mismatch");
    }
    if (expected == options.n_fft) {
        return {exact_window, exact_window + exact_window_length};
    }
    if (!options.center_window_in_fft) {
        throw std::invalid_argument(
            "incremental VoiceChat mel requires centering for a short checkpoint window");
    }

    std::vector<float> window(static_cast<std::size_t>(options.n_fft), 0.0F);
    const int32_t offset = (options.n_fft - expected) / 2;
    std::copy_n(exact_window, expected, window.begin() + static_cast<std::ptrdiff_t>(offset));
    return window;
}

} // namespace

class IncrementalMelSpectrogram::Impl {
  public:
    Impl(const float* mel_filters, int32_t n_freq_bins, int32_t n_mel_bins,
         MelSpectrogramOptions options, int32_t input_sample_rate, const float* exact_window,
         int32_t exact_window_length)
        : options_(std::move(options)), input_sample_rate_(input_sample_rate),
          n_freq_bins_(n_freq_bins), n_mel_bins_(n_mel_bins), fft_plan_(options_.n_fft),
          windowed_(static_cast<std::size_t>(options_.n_fft)),
          frame_power_(static_cast<std::size_t>(n_freq_bins)),
          frame_mel_(static_cast<std::size_t>(n_mel_bins)) {
        validate_incremental_mel_configuration(mel_filters, n_freq_bins_, n_mel_bins_, options_,
                                               input_sample_rate_);
        window_ = make_incremental_mel_window(options_, exact_window, exact_window_length);
        mel_filters_.assign(mel_filters,
                            mel_filters + static_cast<std::size_t>(n_freq_bins_) * n_mel_bins_);
    }

    void accept_audio(const float* samples, int32_t n_samples) {
        if (n_samples < 0 || (n_samples > 0 && samples == nullptr)) {
            throw std::invalid_argument("incremental VoiceChat mel received invalid audio");
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

    int32_t available_frames() const {
        const int32_t samples = target_sample_count();
        return samples > 0 ? samples / options_.hop_length + 1 : 0;
    }

    void ensure_resampled_samples(int32_t required_samples, bool final) {
        if (!final && required_samples > stable_target_sample_count(false)) {
            throw std::runtime_error(
                "incremental VoiceChat mel requested samples before resampler lookahead was ready");
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
        const float current = resampled_audio_[static_cast<std::size_t>(audio_index)];
        if (options_.preemphasis == 0.0F || audio_index == 0) {
            return current;
        }
        return current -
               options_.preemphasis * resampled_audio_[static_cast<std::size_t>(audio_index - 1)];
    }

    float reflected_preemphasized_sample(int32_t audio_index, bool final) const {
        const int32_t length = static_cast<int32_t>(resampled_audio_.size());
        if (length <= 0)
            return 0.0F;
        if (audio_index >= length && !final) {
            throw std::runtime_error(
                "incremental VoiceChat mel requested unstable right reflection");
        }
        const int32_t reflected = detail::reflect_index(audio_index, length);
        return preemphasized_sample(reflected);
    }

    void compute_frame(int32_t frame, bool final) {
        const int32_t pad_size = options_.n_fft / 2;
        const int32_t frame_start = frame * options_.hop_length - pad_size;
        for (int32_t i = 0; i < options_.n_fft; ++i) {
            windowed_[static_cast<std::size_t>(i)] =
                reflected_preemphasized_sample(frame_start + i, final) *
                window_[static_cast<std::size_t>(i)];
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
            throw std::out_of_range(
                "incremental VoiceChat mel frame request exceeds available audio");
        }
        const int32_t current_frames = frame_count();
        if (end_frame <= current_frames) {
            return;
        }
        const int32_t last_frame = end_frame - 1;
        const int32_t required_samples =
            last_frame * options_.hop_length + options_.n_fft - options_.n_fft / 2 + 1;
        ensure_resampled_samples(required_samples, final);
        for (int32_t frame = current_frames; frame < end_frame; ++frame) {
            compute_frame(frame, final);
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
            throw std::out_of_range("incremental VoiceChat mel feature index is out of range");
        }
        return frames_[static_cast<std::size_t>(frame) * n_mel_bins_ + mel_bin];
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
                                                     int32_t input_sample_rate,
                                                     const float* exact_window,
                                                     int32_t exact_window_length)
    : impl_(std::make_unique<Impl>(mel_filters, n_freq_bins, n_mel_bins, std::move(options),
                                   input_sample_rate, exact_window, exact_window_length)) {}

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

int32_t IncrementalMelSpectrogram::frame_count() const {
    return impl_->frame_count();
}

int32_t IncrementalMelSpectrogram::n_mels() const {
    return impl_->n_mels();
}

float IncrementalMelSpectrogram::value(int32_t mel_bin, int32_t frame) const {
    return impl_->value(mel_bin, frame);
}

} // namespace voicechat_audio
} // namespace trtmc
