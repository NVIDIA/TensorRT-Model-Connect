/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for RNNT-owned audio feature extraction helpers.

#include "families/nemotron_speech_streaming/runtime/audio_helpers.h"
#include "families/nemotron_speech_streaming/runtime/resampler.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

std::vector<float> make_identity_filterbank(int32_t n_freq_bins) {
    std::vector<float> filters(static_cast<std::size_t>(n_freq_bins) * n_freq_bins, 0.0F);
    for (int32_t i = 0; i < n_freq_bins; ++i) {
        filters[static_cast<std::size_t>(i) * n_freq_bins + i] = 1.0F;
    }
    return filters;
}

std::vector<float> reference_rfft_power(const std::vector<float>& input) {
    const int32_t n = static_cast<int32_t>(input.size());
    const int32_t n_out = n / 2 + 1;
    const double pi2 = 2.0 * 3.14159265358979323846;
    std::vector<float> power(static_cast<std::size_t>(n_out), 0.0F);
    for (int32_t k = 0; k < n_out; ++k) {
        double real = 0.0;
        double imag = 0.0;
        for (int32_t t = 0; t < n; ++t) {
            const double angle =
                pi2 * static_cast<double>(k) * static_cast<double>(t) / static_cast<double>(n);
            real += static_cast<double>(input[static_cast<std::size_t>(t)]) * std::cos(angle);
            imag -= static_cast<double>(input[static_cast<std::size_t>(t)]) * std::sin(angle);
        }
        power[static_cast<std::size_t>(k)] = static_cast<float>(real * real + imag * imag);
    }
    return power;
}

void check_vectors_close(const std::vector<float>& actual, const std::vector<float>& expected,
                         float tolerance, const char* name) {
    check(actual.size() == expected.size(), name);
    for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        const float scaled_tolerance = tolerance * std::max(1.0F, std::abs(expected[i]));
        check_close(actual[i], expected[i], scaled_tolerance, name);
    }
}

void test_rnnt_fft_matches_direct_dft() {
    std::vector<float> input(512);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.17) +
                                      0.25 * std::cos(static_cast<double>(i) * 0.07));
    }
    check_vectors_close(trtmc::rnnt::detail::rfft_power(input), reference_rfft_power(input), 1e-5F,
                        "rnnt 512-point FFT matches direct DFT");
}

trtmc::rnnt::MelSpectrogramOptions make_incremental_options() {
    trtmc::rnnt::MelSpectrogramOptions options;
    options.n_fft = 8;
    options.win_length = 6;
    options.hop_length = 2;
    options.chunk_length_s = 1;
    options.sample_rate = 32;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = 0.97F;
    options.log_scale = trtmc::rnnt::MelLogScale::kNaturalLog;
    return options;
}

std::vector<float> make_test_audio(int32_t count) {
    std::vector<float> audio(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        audio[static_cast<std::size_t>(i)] =
            static_cast<float>(0.35 * std::sin(static_cast<double>(i) * 0.47) +
                               0.2 * std::cos(static_cast<double>(i) * 0.19));
    }
    return audio;
}

std::vector<float> materialize_incremental(const trtmc::rnnt::IncrementalMelSpectrogram& state) {
    std::vector<float> output;
    output.reserve(static_cast<std::size_t>(state.n_mels()) * state.frame_count());
    for (int32_t mel = 0; mel < state.n_mels(); ++mel) {
        for (int32_t frame = 0; frame < state.frame_count(); ++frame) {
            output.push_back(state.value(mel, frame));
        }
    }
    return output;
}

std::vector<float> run_resampled_chunks(const std::vector<float>& source_audio,
                                        const std::vector<int32_t>& chunk_sizes,
                                        trtmc::rnnt::IncrementalMelStats& stats) {
    const auto options = make_incremental_options();
    const auto filters = make_identity_filterbank(5);
    trtmc::rnnt::IncrementalMelSpectrogram state(filters.data(), 5, 5, options, 96);
    int32_t accepted = 0;
    for (const int32_t chunk_size : chunk_sizes) {
        const int32_t take =
            std::min(chunk_size, static_cast<int32_t>(source_audio.size()) - accepted);
        state.accept_audio(source_audio.data() + accepted, take);
        accepted += take;
        const bool final = accepted == static_cast<int32_t>(source_audio.size());
        if (final) {
            state.ensure_frames(16, true);
            break;
        }

        constexpr int32_t kResampleHalfTaps = 16;
        const int32_t stable_target = std::max(0, accepted - kResampleHalfTaps) / 3;
        const int32_t stable_frames =
            stable_target >= options.n_fft / 2
                ? 1 + (stable_target - options.n_fft / 2) / options.hop_length
                : 0;
        state.ensure_frames(stable_frames, false);
    }
    stats = state.stats();
    return materialize_incremental(state);
}

void test_incremental_mel_processes_each_frame_once() {
    const auto options = make_incremental_options();
    const auto filters = make_identity_filterbank(5);
    const auto audio = make_test_audio(32);
    const auto offline = trtmc::rnnt::extract_configured_mel_spectrogram(
        audio.data(), static_cast<int32_t>(audio.size()), filters.data(), 5, 5, options);

    trtmc::rnnt::IncrementalMelSpectrogram state(filters.data(), 5, 5, options, 32);
    state.accept_audio(audio.data(), 8);
    state.ensure_frames(2, false);
    state.accept_audio(audio.data() + 8, 8);
    state.ensure_frames(6, false);
    state.accept_audio(audio.data() + 16, 16);
    state.ensure_frames(16, true);

    check_vectors_close(materialize_incremental(state), offline.data, 1e-5F,
                        "incremental mel matches one-shot features");
    const auto stats = state.stats();
    check(stats.accepted_source_samples == 32, "incremental mel accepts each source sample once");
    check(stats.generated_resampled_samples == 32,
          "incremental mel materializes each identity-rate sample once");
    check(stats.computed_mel_frames == 16, "incremental mel computes each frame once");
}

void test_incremental_resample_matches_one_shot() {
    const auto options = make_incremental_options();
    const auto filters = make_identity_filterbank(5);
    const auto source_audio = make_test_audio(96);
    const auto resampled = trtmc::resample_linear(source_audio.data(), 96, 96, 32);
    const auto offline = trtmc::rnnt::extract_configured_mel_spectrogram(
        resampled.data(), static_cast<int32_t>(resampled.size()), filters.data(), 5, 5, options);

    trtmc::rnnt::IncrementalMelSpectrogram state(filters.data(), 5, 5, options, 96);
    state.accept_audio(source_audio.data(), 72);
    state.ensure_frames(6, false);
    state.accept_audio(source_audio.data() + 72, 24);
    state.ensure_frames(16, true);

    check_vectors_close(materialize_incremental(state), offline.data, 1e-5F,
                        "incremental resampling matches one-shot features");
    const auto stats = state.stats();
    check(stats.accepted_source_samples == 96,
          "incremental resampling accepts each source sample once");
    check(stats.generated_resampled_samples == 32,
          "incremental resampling generates each target sample once");
    check(stats.computed_mel_frames == 16, "incremental resampling computes each mel frame once");
}

void test_incremental_resample_is_chunk_size_independent() {
    const auto source_audio = make_test_audio(96);
    trtmc::rnnt::IncrementalMelStats coarse_stats;
    trtmc::rnnt::IncrementalMelStats irregular_stats;
    const auto coarse = run_resampled_chunks(source_audio, {24, 24, 24, 24}, coarse_stats);
    const auto irregular = run_resampled_chunks(source_audio, {17, 31, 11, 37}, irregular_stats);

    check_vectors_close(irregular, coarse, 0.0F,
                        "incremental features are independent of source chunk sizes");
    check(coarse_stats.accepted_source_samples == 96 &&
              irregular_stats.accepted_source_samples == 96,
          "chunk schedules accept each source sample once");
    check(coarse_stats.generated_resampled_samples == 32 &&
              irregular_stats.generated_resampled_samples == 32,
          "chunk schedules generate each resampled sample once");
    check(coarse_stats.computed_mel_frames == 16 && irregular_stats.computed_mel_frames == 16,
          "chunk schedules compute each mel frame once");
}

void test_rnnt_mel_matches_configured_owned_options() {
    const std::vector<float> samples{0.0F, 0.1F, -0.2F, 0.4F, -0.1F, 0.0F, 0.2F, -0.3F};
    const int32_t n_fft = 4;
    const int32_t n_freq_bins = 3;
    const int32_t n_mel_bins = 2;
    const std::vector<float> mel_filters{
        0.5F, 0.1F, 0.2F, 0.7F, 0.3F, 0.4F,
    };

    const auto rnnt = trtmc::rnnt::extract_rnnt_mel_spectrogram(
        samples.data(), static_cast<int32_t>(samples.size()), mel_filters.data(), n_freq_bins,
        n_mel_bins, n_fft, /*win_length=*/3, /*hop_length=*/2, /*chunk_length_s=*/1,
        /*sample_rate=*/8, /*preemphasis=*/0.97F);

    trtmc::rnnt::MelSpectrogramOptions options;
    options.n_fft = n_fft;
    options.win_length = 3;
    options.hop_length = 2;
    options.chunk_length_s = 1;
    options.sample_rate = 8;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = 0.97F;
    options.log_scale = trtmc::rnnt::MelLogScale::kNaturalLog;
    const auto configured = trtmc::rnnt::extract_configured_mel_spectrogram(
        samples.data(), static_cast<int32_t>(samples.size()), mel_filters.data(), n_freq_bins,
        n_mel_bins, options);

    check(rnnt.n_mels == configured.n_mels, "rnnt mel keeps configured mel count");
    check(rnnt.n_frames == configured.n_frames, "rnnt mel keeps configured frame count");
    check(rnnt.data.size() == configured.data.size(), "rnnt mel keeps configured data size");
    for (std::size_t i = 0; i < rnnt.data.size(); ++i) {
        check_close(rnnt.data[i], configured.data[i], 1e-6F, "rnnt mel value matches configured");
    }
}

// The front end's straightforward definition, which the runtime is held to: a
// resampler that weights every tap of every output sample from its own
// distance and adds up the weights it used, and a mel product over every
// filterbank coefficient. The power spectrum between them is this file's own
// copy of the runtime's arithmetic, a radix-2 transform that updates its
// twiddle factor after every butterfly (the direct sum for a length that is
// not a power of two).
namespace definition {

constexpr double kPi = 3.14159265358979323846;
constexpr float kLogGuard = 5.960464477539063e-08F;

std::vector<float> rfft_power(const std::vector<float>& input) {
    const int32_t n = static_cast<int32_t>(input.size());
    const int32_t bins = n / 2 + 1;
    std::vector<float> power(static_cast<std::size_t>(bins), 0.0F);
    if (n <= 0) {
        return power;
    }
    const double pi2 = 2.0 * kPi;
    if ((n & (n - 1)) != 0) {
        for (int32_t k = 0; k < bins; ++k) {
            double re = 0.0;
            double im = 0.0;
            const double w = pi2 * static_cast<double>(k) / static_cast<double>(n);
            for (int32_t t = 0; t < n; ++t) {
                const double angle = w * static_cast<double>(t);
                re += static_cast<double>(input[static_cast<std::size_t>(t)]) * std::cos(angle);
                im -= static_cast<double>(input[static_cast<std::size_t>(t)]) * std::sin(angle);
            }
            power[static_cast<std::size_t>(k)] = static_cast<float>(re * re + im * im);
        }
        return power;
    }

    std::vector<std::complex<double>> values(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        values[static_cast<std::size_t>(i)] =
            std::complex<double>(static_cast<double>(input[static_cast<std::size_t>(i)]), 0.0);
    }
    for (int32_t i = 1, j = 0; i < n; ++i) {
        int32_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[static_cast<std::size_t>(i)], values[static_cast<std::size_t>(j)]);
        }
    }
    for (int32_t length = 2; length <= n; length <<= 1) {
        const double angle = -pi2 / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        const int32_t half = length / 2;
        for (int32_t offset = 0; offset < n; offset += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for (int32_t i = 0; i < half; ++i) {
                const auto even = values[static_cast<std::size_t>(offset + i)];
                const auto odd = values[static_cast<std::size_t>(offset + i + half)] * twiddle;
                values[static_cast<std::size_t>(offset + i)] = even + odd;
                values[static_cast<std::size_t>(offset + i + half)] = even - odd;
                twiddle *= step;
            }
        }
    }
    for (int32_t k = 0; k < bins; ++k) {
        power[static_cast<std::size_t>(k)] =
            static_cast<float>(std::norm(values[static_cast<std::size_t>(k)]));
    }
    return power;
}

double scaled_sinc(double distance, double cutoff) {
    if (std::abs(distance) < 1.0e-12) {
        return cutoff;
    }
    return cutoff * std::sin(kPi * distance * cutoff) / (kPi * distance * cutoff);
}

double hann_window(double distance, int32_t half_taps) {
    const double position =
        (distance + static_cast<double>(half_taps)) / (2.0 * static_cast<double>(half_taps));
    return 0.5 * (1.0 - std::cos(2.0 * kPi * position));
}

// Output samples [output_start, output_start + output_count) of `samples`
// resampled from source_rate to target_rate, clamped to the output's length.
std::vector<float> resample_range(const std::vector<float>& samples, int32_t source_rate,
                                  int32_t target_rate, int32_t output_start, int32_t output_count) {
    const int32_t num_samples = static_cast<int32_t>(samples.size());
    const int32_t full_length =
        static_cast<int32_t>(static_cast<int64_t>(num_samples) * target_rate / source_rate);
    const int32_t start = std::clamp(output_start, 0, full_length);
    const int32_t count = std::clamp(output_count, 0, full_length - start);
    if (source_rate == target_rate) {
        return {samples.begin() + start, samples.begin() + start + count};
    }
    constexpr int32_t half_taps = 16;
    const double cutoff =
        std::min(1.0, static_cast<double>(target_rate) / static_cast<double>(source_rate));
    const int32_t rate_gcd = std::gcd(source_rate, target_rate);
    const int32_t phase_count = target_rate / rate_gcd;
    std::vector<float> output(static_cast<std::size_t>(count));
    for (int32_t local = 0; local < count; ++local) {
        const int32_t index = start + local;
        double value = 0.0;
        double weight_sum = 0.0;
        if (phase_count <= 2048) {
            // The source position is center + fraction, the fraction one of
            // phase_count values.
            const int64_t numerator = static_cast<int64_t>(index) * source_rate;
            const int32_t center = static_cast<int32_t>(numerator / target_rate);
            const int32_t phase = static_cast<int32_t>(numerator % target_rate) / rate_gcd;
            const double fraction =
                static_cast<double>(phase * rate_gcd) / static_cast<double>(target_rate);
            const int32_t first = std::max(-half_taps + 1, -center);
            const int32_t last = std::min(half_taps, num_samples - 1 - center);
            for (int32_t offset = first; offset <= last; ++offset) {
                const double distance = static_cast<double>(offset) - fraction;
                const double weight =
                    scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
                value += static_cast<double>(samples[static_cast<std::size_t>(center + offset)]) *
                         weight;
                weight_sum += weight;
            }
        } else {
            const double position = static_cast<double>(index) * source_rate / target_rate;
            const int32_t center = static_cast<int32_t>(std::floor(position));
            const int32_t low = std::max(0, center - half_taps + 1);
            const int32_t high = std::min(num_samples - 1, center + half_taps);
            for (int32_t tap = low; tap <= high; ++tap) {
                const double distance = static_cast<double>(tap) - position;
                const double weight =
                    scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
                value += static_cast<double>(samples[static_cast<std::size_t>(tap)]) * weight;
                weight_sum += weight;
            }
        }
        output[static_cast<std::size_t>(local)] =
            weight_sum > 1.0e-12 ? static_cast<float>(value / weight_sum) : 0.0F;
    }
    return output;
}

std::vector<float> resample(const std::vector<float>& samples, int32_t source_rate,
                            int32_t target_rate) {
    return resample_range(samples, source_rate, target_rate, 0,
                          std::numeric_limits<int32_t>::max());
}

// A symmetric Hann window of win_length in the middle of an n_fft frame.
std::vector<float> centered_window(int32_t n_fft, int32_t win_length) {
    std::vector<float> window(static_cast<std::size_t>(n_fft), 0.0F);
    const double pi2 = 2.0 * kPi;
    const int32_t offset = std::max(0, (n_fft - win_length) / 2);
    for (int32_t i = 0; i < win_length && offset + i < n_fft && win_length > 1; ++i) {
        window[static_cast<std::size_t>(offset + i)] = static_cast<float>(
            0.5 *
            (1.0 - std::cos(pi2 * static_cast<double>(i) / static_cast<double>(win_length - 1))));
    }
    return window;
}

// The mel values, before the log, of frames [0, frames) of `audio` (samples at
// the model's rate): frame t is centered on sample t * hop_length, samples
// outside the audio are zero. Row t holds frame t's n_mel_bins values. The
// whole-clip path skips a power of exactly zero; the incremental one does not.
std::vector<float> mel_frames(const std::vector<float>& audio, const std::vector<float>& filters,
                              int32_t n_mel_bins, const trtmc::rnnt::MelSpectrogramOptions& o,
                              int32_t frames, bool skip_zero_power) {
    const int32_t n_freq_bins = o.n_fft / 2 + 1;
    const int32_t count = static_cast<int32_t>(audio.size());
    const auto window = centered_window(o.n_fft, o.win_length);
    std::vector<float> mel(static_cast<std::size_t>(frames) * n_mel_bins, 0.0F);
    std::vector<float> frame(static_cast<std::size_t>(o.n_fft));
    for (int32_t t = 0; t < frames; ++t) {
        const int32_t frame_start = t * o.hop_length - o.n_fft / 2;
        for (int32_t i = 0; i < o.n_fft; ++i) {
            const int32_t index = frame_start + i;
            float sample = 0.0F;
            if (index >= 0 && index < count) {
                sample = audio[static_cast<std::size_t>(index)];
                if (o.preemphasis != 0.0F && index != 0) {
                    sample = audio[static_cast<std::size_t>(index)] -
                             o.preemphasis * audio[static_cast<std::size_t>(index - 1)];
                }
            }
            frame[static_cast<std::size_t>(i)] = sample * window[static_cast<std::size_t>(i)];
        }
        const auto power = rfft_power(frame);
        float* frame_mel = mel.data() + static_cast<std::size_t>(t) * n_mel_bins;
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = power[static_cast<std::size_t>(f)];
            if (skip_zero_power && p == 0.0F) {
                continue;
            }
            const float* row = filters.data() + static_cast<std::size_t>(f) * n_mel_bins;
            for (int32_t m = 0; m < n_mel_bins; ++m) {
                frame_mel[m] += p * row[m];
            }
        }
    }
    return mel;
}

// What an IncrementalMelSpectrogram holds after its first `frames` frames of a
// stream whose whole input is `source`: [mel][frame], as the tests read it.
std::vector<float> incremental_log_mel(const std::vector<float>& source, int32_t source_rate,
                                       const std::vector<float>& filters, int32_t n_mel_bins,
                                       const trtmc::rnnt::MelSpectrogramOptions& o,
                                       int32_t frames) {
    const int64_t converted = static_cast<int64_t>(source.size()) * o.sample_rate / source_rate;
    const int64_t chunk_limit = static_cast<int64_t>(o.chunk_length_s) * o.sample_rate;
    const auto audio = resample_range(source, source_rate, o.sample_rate, 0,
                                      static_cast<int32_t>(std::min(converted, chunk_limit)));
    const auto mel = mel_frames(audio, filters, n_mel_bins, o, frames, false);
    std::vector<float> features(mel.size());
    for (int32_t m = 0; m < n_mel_bins; ++m) {
        for (int32_t t = 0; t < frames; ++t) {
            features[static_cast<std::size_t>(m) * frames + t] =
                std::log(mel[static_cast<std::size_t>(t) * n_mel_bins + m] + kLogGuard);
        }
    }
    return features;
}

// extract_configured_mel_spectrogram for natural-log features of `audio` (at
// the model's rate): the frames that hold audio, then silence up to the chunk,
// the last frame dropped.
std::vector<float> whole_clip_log_mel(const std::vector<float>& audio,
                                      const std::vector<float>& filters, int32_t n_mel_bins,
                                      const trtmc::rnnt::MelSpectrogramOptions& o,
                                      int32_t& n_frames) {
    const int32_t chunk_samples = o.chunk_length_s * o.sample_rate;
    const int32_t valid = std::min(static_cast<int32_t>(audio.size()), chunk_samples);
    const int32_t pad = o.n_fft / 2;
    const int32_t n_frames_raw = 1 + (chunk_samples + 2 * pad - o.n_fft) / o.hop_length;
    const int32_t computed =
        std::clamp(valid > 0 ? 1 + (pad + valid - 1) / o.hop_length : 0, 0, n_frames_raw);
    const auto mel =
        mel_frames({audio.begin(), audio.begin() + valid}, filters, n_mel_bins, o, computed, true);
    n_frames = n_frames_raw > 1 ? n_frames_raw - 1 : n_frames_raw;
    std::vector<float> features(static_cast<std::size_t>(n_mel_bins) * n_frames);
    for (int32_t m = 0; m < n_mel_bins; ++m) {
        for (int32_t t = 0; t < n_frames; ++t) {
            const float value =
                t < computed ? mel[static_cast<std::size_t>(t) * n_mel_bins + m] : 0.0F;
            features[static_cast<std::size_t>(m) * n_frames + t] = std::log(value + kLogGuard);
        }
    }
    return features;
}

} // namespace definition

// Equal bit for bit, except that a NaN matches any NaN: which operand's NaN an
// addition returns follows the operand order the compiler picks, so a NaN's
// sign and payload bits are not part of the definition. Its place is.
bool same_values(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0 &&
            !(std::isnan(a[i]) && std::isnan(b[i]))) {
            return false;
        }
    }
    return true;
}

void check_same(const std::vector<float>& actual, const std::vector<float>& expected,
                const std::string& name) {
    if (!same_values(actual, expected)) {
        std::cerr << "FAIL: " << name << " (not bit-identical to the definition)\n";
        ++g_failures;
    }
}

// Features whose expected values come through definition::rfft_power, this
// file's copy of the transform. Where the compiler fuses multiply-adds
// (__FP_FAST_FMA), which products it fuses depends on how it compiles each
// copy: the test's copy is not the library's object code, and one of its
// powers can differ from the library's by an ulp. A correct library must not
// fail for that, so the features are compared there with a tolerance: a NaN
// matches any NaN, an infinity only itself, and finite values may differ by
// 1e-5 or by four float ulps of the larger one, whichever is more. Without
// fused multiply-adds every operation rounds the same in both copies, and the
// features must be equal bit for bit. The resampler does not use the
// transform: its tests are bit for bit everywhere.
#if defined(__FP_FAST_FMA)
bool same_features(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            if (!(std::isnan(a[i]) && std::isnan(b[i]))) {
                return false;
            }
        } else if (std::isinf(a[i]) || std::isinf(b[i])) {
            if (a[i] != b[i]) {
                return false;
            }
        } else {
            const float larger = std::max(std::abs(a[i]), std::abs(b[i]));
            const float ulp =
                std::nextafter(larger, std::numeric_limits<float>::infinity()) - larger;
            if (std::abs(a[i] - b[i]) > std::max(1e-5F, 4.0F * ulp)) {
                return false;
            }
        }
    }
    return true;
}
#else
bool same_features(const std::vector<float>& a, const std::vector<float>& b) {
    return same_values(a, b);
}
#endif

void check_features(const std::vector<float>& actual, const std::vector<float>& expected,
                    const std::string& name) {
    if (!same_features(actual, expected)) {
        std::cerr << "FAIL: " << name << " (features differ from the definition)\n";
        ++g_failures;
    }
}

std::vector<float> noise(std::mt19937& rng, int samples, float scale = 1.0F) {
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    std::vector<float> audio(static_cast<std::size_t>(samples));
    for (float& value : audio) {
        value = scale * uniform(rng);
    }
    return audio;
}

void test_resampler_equals_definition() {
    std::mt19937 rng(4816);
    struct Rates {
        int32_t source;
        int32_t target;
    };
    // Output phases: 1, 160, 320, 2 (upsampling), 7, 3; then two pairs above
    // the 2048 phases that the table path stops at; then equal rates.
    for (const Rates rates :
         {Rates{48000, 16000}, Rates{44100, 16000}, Rates{22050, 16000}, Rates{8000, 16000},
          Rates{3, 7}, Rates{7, 3}, Rates{44101, 16000}, Rates{4099, 2049}, Rates{16000, 16000}}) {
        const std::string pair =
            std::to_string(rates.source) + " -> " + std::to_string(rates.target) + " Hz";
        // With fewer than 32 samples no output sample has all its 32 taps
        // inside the input; the long inputs are mostly such samples, between
        // two edges.
        for (const int samples : {1, 2, 15, 16, 17, 31, 32, 33, 34, 48, 64, 65, 100, 300}) {
            for (int kind = 0; kind < 4; ++kind) {
                auto input = noise(rng, samples);
                if (kind == 1) {
                    std::fill(input.begin(), input.end(), 1.0F);
                } else if (kind == 2) {
                    input[static_cast<std::size_t>(samples / 2)] =
                        std::numeric_limits<float>::quiet_NaN();
                } else if (kind == 3) {
                    input[static_cast<std::size_t>(samples - 1)] =
                        std::numeric_limits<float>::infinity();
                }
                const std::string name = "resample " + pair + ", " + std::to_string(samples) +
                                         " samples, input kind " + std::to_string(kind);
                const auto expected = definition::resample(input, rates.source, rates.target);
                check_same(
                    trtmc::resample_linear(input.data(), samples, rates.source, rates.target),
                    expected, name);
                if (kind != 0) {
                    continue;
                }
                // Every split of the output into two calls, on the free
                // function and on one resampler object that keeps its table
                // from call to call: each call returns its part of the
                // one-call output.
                const int32_t full = static_cast<int32_t>(expected.size());
                trtmc::LinearResampler resampler(rates.source, rates.target);
                bool ranges_match = true;
                for (int32_t split = 0; split <= full; ++split) {
                    for (const bool reuse : {false, true}) {
                        for (const auto& [start, count] :
                             {std::pair<int32_t, int32_t>{0, split}, {split, full - split}}) {
                            const auto part =
                                reuse ? resampler.range(input.data(), samples, start, count)
                                      : trtmc::resample_linear_range(input.data(), samples,
                                                                     rates.source, rates.target,
                                                                     start, count);
                            ranges_match = ranges_match &&
                                           same_values(part, {expected.begin() + start,
                                                              expected.begin() + start + count});
                        }
                    }
                }
                check(ranges_match, (name + ": every split into two calls").c_str());
                // Ranges past either end are clamped.
                check_same(trtmc::resample_linear_range(input.data(), samples, rates.source,
                                                        rates.target, -3, full + 9),
                           expected, name + ": a range past both ends");
                check(resampler.range(input.data(), samples, full, 5).empty(),
                      (name + ": a range that starts at the end").c_str());
            }
        }
        // One resampler object over an input that grows, as a stream's does:
        // each call sees more samples and takes the next range.
        const auto input = noise(rng, 2000);
        trtmc::LinearResampler resampler(rates.source, rates.target);
        std::vector<float> made;
        bool growing_match = true;
        for (int32_t have = 40; have <= 2000; have += 280) {
            const std::vector<float> seen(input.begin(), input.begin() + have);
            const int32_t stable =
                static_cast<int32_t>(static_cast<int64_t>(have - 16) * rates.target / rates.source);
            const int32_t start = static_cast<int32_t>(made.size());
            const auto part = resampler.range(input.data(), have, start, stable - start);
            growing_match =
                growing_match &&
                same_values(part, definition::resample_range(seen, rates.source, rates.target,
                                                             start, stable - start));
            made.insert(made.end(), part.begin(), part.end());
        }
        // Held back by 16 source samples, no call's output depends on the
        // samples that came later.
        auto expected = definition::resample(input, rates.source, rates.target);
        expected.resize(made.size());
        check(growing_match && same_values(made, expected),
              ("resample " + pair + ": one resampler over a growing input").c_str());
    }
}

// Overlapping triangular filters over the frequency bins: each row holds at
// most two adjacent nonzero coefficients, as the Slaney filterbank's rows do.
std::vector<float> triangular_filterbank(int n_freq_bins, int mels) {
    std::vector<float> filters(static_cast<std::size_t>(n_freq_bins) * mels, 0.0F);
    for (int f = 0; f < n_freq_bins; ++f) {
        const double position = static_cast<double>(f) * (mels + 1) / n_freq_bins;
        const int below = static_cast<int>(position);
        const float weight = static_cast<float>(position - below);
        if (below >= 1 && below <= mels) {
            filters[static_cast<std::size_t>(f) * mels + (below - 1)] = 1.0F - weight;
        }
        if (below < mels && weight != 0.0F) {
            filters[static_cast<std::size_t>(f) * mels + below] = weight;
        }
    }
    return filters;
}

trtmc::rnnt::MelSpectrogramOptions make_model_options() {
    trtmc::rnnt::MelSpectrogramOptions options;
    options.n_fft = 512;
    options.win_length = 400;
    options.hop_length = 160;
    options.chunk_length_s = 1;
    options.sample_rate = 16000;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = 0.97F;
    options.log_scale = trtmc::rnnt::MelLogScale::kNaturalLog;
    return options;
}

// One stream: the source handed over in pieces of chunk_samples, the frames
// asked for as far as the resampler's lookahead allows after every piece, and
// to the end after the last one when `final` is set.
void feed_stream(trtmc::rnnt::IncrementalMelSpectrogram& state, const std::vector<float>& source,
                 int32_t source_rate, const trtmc::rnnt::MelSpectrogramOptions& options,
                 int32_t chunk_samples, bool final) {
    const int32_t total = static_cast<int32_t>(source.size());
    for (int32_t accepted = 0; accepted < total;) {
        const int32_t take = std::min(chunk_samples, total - accepted);
        state.accept_audio(source.data() + accepted, take);
        accepted += take;
        if (final && accepted == total) {
            state.ensure_frames(state.available_frames(), true);
            break;
        }
        constexpr int32_t kResampleHalfTaps = 16;
        const int64_t held_back = source_rate == options.sample_rate ? 0 : kResampleHalfTaps;
        const int64_t stable_target =
            std::max<int64_t>(0, accepted - held_back) * options.sample_rate / source_rate;
        const int64_t first_frame_samples = options.n_fft - options.n_fft / 2;
        const int64_t stable_frames =
            stable_target >= first_frame_samples
                ? 1 + (stable_target - first_frame_samples) / options.hop_length
                : 0;
        state.ensure_frames(
            static_cast<int32_t>(std::min<int64_t>(stable_frames, state.available_frames())),
            false);
    }
}

void test_mel_product_equals_definition() {
    std::mt19937 rng(75918);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    const auto options = make_model_options();
    constexpr int kFreqBins = 257;
    constexpr int kMels = 80;
    const float infinity = std::numeric_limits<float>::infinity();

    struct Filterbank {
        const char* name;
        std::vector<float> filters;
    };
    std::vector<Filterbank> filterbanks;
    const auto sparse = triangular_filterbank(kFreqBins, kMels);
    filterbanks.push_back({"sparse", sparse});
    std::vector<float> dense(static_cast<std::size_t>(kFreqBins) * kMels);
    for (float& value : dense) {
        value = 0.5F + 0.5F * uniform(rng);
    }
    filterbanks.push_back({"dense", dense});
    auto zero_rows = sparse;
    for (int f = 0; f < kFreqBins; f += 3) {
        std::fill_n(zero_rows.begin() + static_cast<std::ptrdiff_t>(f) * kMels, kMels, 0.0F);
    }
    filterbanks.push_back({"zero rows", zero_rows});
    filterbanks.push_back(
        {"all zero", std::vector<float>(static_cast<std::size_t>(kFreqBins) * kMels, 0.0F)});
    // Every row is nonzero, and none reaches the first or the last column.
    auto inner_columns = dense;
    for (int f = 0; f < kFreqBins; ++f) {
        inner_columns[static_cast<std::size_t>(f) * kMels] = 0.0F;
        inner_columns[static_cast<std::size_t>(f) * kMels + kMels - 1] = 0.0F;
    }
    filterbanks.push_back({"empty outer columns", inner_columns});
    auto interior_zero = dense;
    for (int f = 0; f < kFreqBins; ++f) {
        interior_zero[static_cast<std::size_t>(f) * kMels + 40] = 0.0F;
        interior_zero[static_cast<std::size_t>(f) * kMels + (f % kMels)] = 0.0F;
    }
    filterbanks.push_back({"zeros inside the rows", interior_zero});
    // The ends of a row's span are whatever is not zero: a negative, a small
    // or a NaN coefficient there still counts, and a negative zero does not.
    // Each row is four columns wide; NaN ends are in every tenth row only, so
    // most columns keep finite values.
    std::vector<float> negative_ends(static_cast<std::size_t>(kFreqBins) * kMels, 0.0F);
    auto nan_ends = negative_ends;
    auto small_ends = negative_ends;
    auto negative_zero_ends = negative_ends;
    for (int f = 0; f < kFreqBins; ++f) {
        const std::size_t first = static_cast<std::size_t>(f) * kMels + (f * 5) % (kMels - 3);
        for (auto* filters : {&negative_ends, &nan_ends, &small_ends, &negative_zero_ends}) {
            (*filters)[first] = 0.125F;
            (*filters)[first + 1] = 0.5F;
            (*filters)[first + 2] = 0.25F;
            (*filters)[first + 3] = 0.125F;
        }
        negative_ends[first] = -0.25F;
        negative_ends[first + 3] = -0.125F;
        small_ends[first] = 5e-7F;
        small_ends[first + 3] = 5e-7F;
        negative_zero_ends[first] = -0.0F;
        negative_zero_ends[first + 3] = -0.0F;
        if ((f % 10) == 3) {
            nan_ends[first + ((f % 20) == 3 ? 0 : 3)] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    filterbanks.push_back({"negative coefficients at span ends", negative_ends});
    filterbanks.push_back({"small coefficients at span ends", small_ends});
    filterbanks.push_back({"negative zeros at span ends", negative_zero_ends});
    filterbanks.push_back({"NaN coefficients at span ends", nan_ends});

    struct Audio {
        const char* name;
        std::vector<float> samples;
    };
    constexpr int kSamples = 4000; // 25 frames
    std::vector<Audio> clips;
    const auto base = noise(rng, kSamples);
    clips.push_back({"noise", base});
    clips.push_back({"silence", std::vector<float>(kSamples, 0.0F)});
    auto clip = base;
    std::fill(clip.begin() + 1000, clip.begin() + 2500, 0.0F);
    clips.push_back({"noise around silent frames", clip});
    for (int i = 0; i < kSamples; ++i) {
        clip[static_cast<std::size_t>(i)] =
            static_cast<float>(0.25 * std::cos(2.0 * definition::kPi * 3.0 * i / 512.0) +
                               0.15 * std::sin(2.0 * definition::kPi * 7.0 * i / 512.0));
    }
    clips.push_back({"tones on bins", clip});
    // A power that is not finite keeps the whole filterbank row: a sample
    // that is not finite, and finite samples too loud for a float power.
    clip = base;
    clip[kSamples / 2] = infinity;
    clips.push_back({"one infinite sample", clip});
    clip = base;
    clip[0] = -infinity;
    clips.push_back({"an infinite first sample", clip});
    clip = base;
    clip[kSamples / 3] = std::numeric_limits<float>::quiet_NaN();
    clips.push_back({"one NaN sample", clip});
    clip = base;
    for (float& value : clip) {
        value *= 3.0e38F;
    }
    clips.push_back({"samples near the largest float", clip});
    // Loud in a few bins of the first frames only: powers that are not finite
    // need a row first, finite ones need it afterwards.
    clip = base;
    for (int i = 0; i < 800; ++i) {
        clip[static_cast<std::size_t>(i)] =
            3.0e18F * static_cast<float>(std::cos(2.0 * definition::kPi * 30.0 * i / 512.0));
    }
    clips.push_back({"a start too loud for a float power", clip});

    for (const Filterbank& fb : filterbanks) {
        for (const Audio& audio : clips) {
            const std::string name = std::string("mel of ") + audio.name + ", " + fb.name;
            // The incremental object, three pieces.
            trtmc::rnnt::IncrementalMelSpectrogram state(fb.filters.data(), kFreqBins, kMels,
                                                         options, options.sample_rate);
            feed_stream(state, audio.samples, options.sample_rate, options, 1500, true);
            check(state.frame_count() == kSamples / options.hop_length,
                  (name + ": incremental frame count").c_str());
            check_features(materialize_incremental(state),
                           definition::incremental_log_mel(audio.samples, options.sample_rate,
                                                           fb.filters, kMels, options,
                                                           state.frame_count()),
                           name + ", incremental");
            // The whole-clip path.
            const auto whole = trtmc::rnnt::extract_configured_mel_spectrogram(
                audio.samples.data(), kSamples, fb.filters.data(), kFreqBins, kMels, options);
            int32_t frames = 0;
            const auto expected =
                definition::whole_clip_log_mel(audio.samples, fb.filters, kMels, options, frames);
            check(whole.n_mels == kMels && whole.n_frames == frames,
                  (name + ": whole-clip shape").c_str());
            check_features(whole.data, expected, name + ", whole clip");
        }
    }

    // The whole-clip path reads no filterbank row that no nonzero power needs.
    const std::vector<float> silence(kSamples, 0.0F);
    const auto silent = trtmc::rnnt::extract_configured_mel_spectrogram(
        silence.data(), kSamples, nullptr, kFreqBins, kMels, options);
    check(std::all_of(silent.data.begin(), silent.data.end(),
                      [](float value) { return value == std::log(definition::kLogGuard); }),
          "silence gives the log guard without reading the filterbank");
}

void test_incremental_equals_one_shot_definition() {
    std::mt19937 rng(1120);
    struct Geometry {
        int32_t n_fft;
        int32_t win_length;
        int32_t hop_length;
        int32_t sample_rate;
        int32_t n_mel_bins;
    };
    // The model's frame, then a short power-of-two frame, a frame that takes
    // the direct sum, and a frame as long as its window.
    for (const Geometry& geometry :
         {Geometry{512, 400, 160, 16000, 128}, Geometry{64, 50, 16, 1600, 20},
          Geometry{100, 80, 40, 4000, 16}, Geometry{256, 256, 80, 8000, 40}}) {
        auto options = make_model_options();
        options.n_fft = geometry.n_fft;
        options.win_length = geometry.win_length;
        options.hop_length = geometry.hop_length;
        options.sample_rate = geometry.sample_rate;
        const int32_t n_freq_bins = geometry.n_fft / 2 + 1;
        const auto filters = triangular_filterbank(n_freq_bins, geometry.n_mel_bins);
        // No resampling, one output phase, many phases, upsampling.
        for (const int32_t source_rate :
             {geometry.sample_rate, 3 * geometry.sample_rate, geometry.sample_rate * 441 / 160,
              geometry.sample_rate / 2}) {
            const int32_t samples = source_rate * 3 / 10 + 37;
            // One object serves every request of this rate, reset() between
            // them: its tables outlive a request.
            trtmc::rnnt::IncrementalMelSpectrogram state(filters.data(), n_freq_bins,
                                                         geometry.n_mel_bins, options, source_rate);
            int request = 0;
            // The whole clip at once, then pieces of 100 ms, of 997 samples,
            // and of one sample.
            for (const int32_t chunk_samples : {samples, source_rate / 10, 997, 1}) {
                if (chunk_samples == 1 && geometry.n_fft != 64) {
                    continue;
                }
                for (const bool final : {true, false}) {
                    const auto source = noise(rng, samples);
                    const std::string name = "stream of " + std::to_string(samples) +
                                             " samples at " + std::to_string(source_rate) +
                                             " Hz for " + std::to_string(geometry.sample_rate) +
                                             " Hz, frame " + std::to_string(geometry.n_fft) +
                                             ", pieces of " + std::to_string(chunk_samples) +
                                             (final ? ", final" : ", not final") + ", request " +
                                             std::to_string(request);
                    feed_stream(state, source, source_rate, options, chunk_samples, final);
                    if (final) {
                        check(state.frame_count() == state.available_frames(),
                              (name + ": every frame computed").c_str());
                    }
                    check(state.frame_count() > 0, (name + ": some frames computed").c_str());
                    check_features(materialize_incremental(state),
                                   definition::incremental_log_mel(source, source_rate, filters,
                                                                   geometry.n_mel_bins, options,
                                                                   state.frame_count()),
                                   name);
                    state.reset();
                    check(state.frame_count() == 0 && state.available_frames() == 0,
                          (name + ": reset empties the object").c_str());
                    ++request;
                }
            }
        }
    }
}

template <class Call>
bool throws_invalid_argument(Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void test_invalid_arguments_throw() {
    const std::vector<float> input(64, 0.25F);
    struct BadCall {
        const char* what;
        const float* samples;
        int32_t count;
        int32_t source_rate;
        int32_t target_rate;
    };
    // The resampler checks its arguments before it uses any of them.
    for (const BadCall& bad :
         {BadCall{"a null input", nullptr, 64, 48000, 16000},
          BadCall{"a null input at equal rates", nullptr, 64, 16000, 16000},
          BadCall{"no samples", input.data(), 0, 48000, 16000},
          BadCall{"a negative sample count", input.data(), -1, 48000, 16000},
          BadCall{"a source rate of zero", input.data(), 64, 0, 16000},
          BadCall{"a negative source rate", input.data(), 64, -48000, 16000},
          BadCall{"a target rate of zero", input.data(), 64, 48000, 0},
          BadCall{"a negative target rate", input.data(), 64, 48000, -16000}}) {
        const std::string what = std::string(" throws for ") + bad.what;
        check(throws_invalid_argument([&] {
                  trtmc::LinearResampler resampler(bad.source_rate, bad.target_rate);
                  (void)resampler.range(bad.samples, bad.count, 0, 8);
              }),
              ("LinearResampler::range" + what).c_str());
        check(throws_invalid_argument([&] {
                  (void)trtmc::resample_linear_range(bad.samples, bad.count, bad.source_rate,
                                                     bad.target_rate, 0, 8);
              }),
              ("resample_linear_range" + what).c_str());
        // resample_linear returns no samples for a count that is not positive
        // and works out its output length before the range is checked, which
        // divides by the source rate: a source rate of zero is not for it.
        if (bad.count > 0 && bad.source_rate != 0) {
            check(throws_invalid_argument([&] {
                      (void)trtmc::resample_linear(bad.samples, bad.count, bad.source_rate,
                                                   bad.target_rate);
                  }),
                  ("resample_linear" + what).c_str());
        }
    }
    check(trtmc::resample_linear(input.data(), 0, 48000, 16000).empty() &&
              trtmc::resample_linear(nullptr, 0, 48000, 16000).empty() &&
              trtmc::resample_linear(input.data(), -5, 48000, 16000).empty(),
          "resample_linear returns no samples for a count that is not positive");

    // The incremental object checks its arguments before it reads the
    // filterbank: a null one, or one too small for the dimensions given, is
    // refused, not read.
    const auto options = make_model_options();
    const std::vector<float> filters(257 * 128, 0.5F);
    const std::vector<float> one(1, 0.5F);
    using trtmc::rnnt::IncrementalMelSpectrogram;
    check(throws_invalid_argument(
              [&] { IncrementalMelSpectrogram state(nullptr, 257, 128, options, 16000); }),
          "incremental mel refuses a null filterbank");
    check(throws_invalid_argument(
              [&] { IncrementalMelSpectrogram state(one.data(), 256, 128, options, 16000); }),
          "incremental mel refuses a row count that is not n_fft / 2 + 1");
    check(throws_invalid_argument(
              [&] { IncrementalMelSpectrogram state(one.data(), 257, 0, options, 16000); }),
          "incremental mel refuses no mel bins");
    check(throws_invalid_argument(
              [&] { IncrementalMelSpectrogram state(filters.data(), 257, 128, options, 0); }),
          "incremental mel refuses a source rate of zero");
    check(throws_invalid_argument(
              [&] { IncrementalMelSpectrogram state(filters.data(), 257, 128, options, -1); }),
          "incremental mel refuses a negative source rate");
    auto bad_options = options;
    bad_options.sample_rate = 0;
    check(throws_invalid_argument([&] {
              IncrementalMelSpectrogram state(filters.data(), 257, 128, bad_options, 16000);
          }),
          "incremental mel refuses a model rate of zero");
    bad_options = options;
    bad_options.hop_length = 0;
    check(throws_invalid_argument([&] {
              IncrementalMelSpectrogram state(filters.data(), 257, 128, bad_options, 16000);
          }),
          "incremental mel refuses a hop of zero");
    bad_options = options;
    bad_options.log_scale = trtmc::rnnt::MelLogScale::kLog10Normalized;
    check(throws_invalid_argument([&] {
              IncrementalMelSpectrogram state(filters.data(), 257, 128, bad_options, 16000);
          }),
          "incremental mel refuses the log10 scale");
    bad_options = options;
    bad_options.normalize_per_feature = true;
    check(throws_invalid_argument([&] {
              IncrementalMelSpectrogram state(filters.data(), 257, 128, bad_options, 16000);
          }),
          "incremental mel refuses per-feature normalization");
}

} // namespace

int main() {
    test_rnnt_fft_matches_direct_dft();
    test_rnnt_mel_matches_configured_owned_options();
    test_incremental_mel_processes_each_frame_once();
    test_incremental_resample_matches_one_shot();
    test_incremental_resample_is_chunk_size_independent();
    test_resampler_equals_definition();
    test_mel_product_equals_definition();
    test_incremental_equals_one_shot_definition();
    test_invalid_arguments_throw();

    if (g_failures != 0) {
        std::cerr << g_failures << " rnnt audio helper test(s) failed\n";
        return 1;
    }
    return 0;
}
