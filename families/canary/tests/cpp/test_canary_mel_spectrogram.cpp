/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/canary/runtime/canary_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

std::vector<float> make_identity_filterbank(int32_t n_freq_bins, int32_t n_mel_bins) {
    std::vector<float> fb(static_cast<std::size_t>(n_freq_bins) * n_mel_bins, 0.0F);
    const int32_t mapped = std::min(n_freq_bins, n_mel_bins);
    for (int32_t i = 0; i < mapped; ++i) {
        fb[static_cast<std::size_t>(i) * n_mel_bins + i] = 1.0F;
    }
    return fb;
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

void check_fft_matches_direct(const std::vector<float>& input, const char* test_name) {
    const auto actual = trtmc::canary::detail::rfft_power(input);
    const auto reference = reference_rfft_power(input);
    bool values_match = actual.size() == reference.size();
    for (std::size_t i = 0; values_match && i < actual.size(); ++i) {
        const float tolerance = 1e-5F * std::max(1.0F, std::abs(reference[i]));
        values_match = std::abs(actual[i] - reference[i]) <= tolerance;
    }
    check(values_match, test_name);
}

void test_canary_fft_matches_direct_dft() {
    check_fft_matches_direct({0.25F, -0.5F, 0.75F, 1.0F, -0.25F, 0.125F, 0.5F, -0.75F},
                             "canary radix-2 FFT power matches direct DFT");
    check_fft_matches_direct({0.1F, 0.2F, -0.3F, 0.4F, 0.5F, -0.6F, 0.7F, 0.8F, -0.9F, 1.0F},
                             "canary non-radix-2 fallback matches direct DFT");

    std::vector<float> canary_input(512);
    for (std::size_t i = 0; i < canary_input.size(); ++i) {
        canary_input[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.17) +
                                             0.25 * std::cos(static_cast<double>(i) * 0.07));
    }
    check_fft_matches_direct(canary_input, "canary 512-point FFT power matches direct DFT");
}

// The frontend's straightforward definition, which the production code must
// equal bit for bit: per frame, the window, a radix-2 transform that updates
// its twiddle factor inline in every butterfly group (the direct DFT for other
// lengths), the dense filterbank product, the log of every frame, then
// per-feature normalization (or zeroed padding) and the last frame dropped.
constexpr double kPi2 = 2.0 * 3.14159265358979323846;

void definition_rfft_power(const float* x, int32_t n, int32_t n_out, float* power_out) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        for (int32_t k = 0; k < n_out; ++k) {
            double re = 0.0;
            double im = 0.0;
            const double w = kPi2 * static_cast<double>(k) / static_cast<double>(n);
            for (int32_t t = 0; t < n; ++t) {
                const double angle = w * static_cast<double>(t);
                re += static_cast<double>(x[t]) * std::cos(angle);
                im -= static_cast<double>(x[t]) * std::sin(angle);
            }
            power_out[k] = static_cast<float>(re * re + im * im);
        }
        return;
    }
    std::vector<std::complex<double>> work(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i)
        work[static_cast<std::size_t>(i)] = std::complex<double>(static_cast<double>(x[i]), 0.0);
    for (int32_t i = 1, j = 0; i < n; ++i) {
        int32_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(work[static_cast<std::size_t>(i)], work[static_cast<std::size_t>(j)]);
    }
    for (int32_t length = 2; length <= n; length <<= 1) {
        const double angle = -kPi2 / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        const int32_t half = length / 2;
        for (int32_t offset = 0; offset < n; offset += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for (int32_t i = 0; i < half; ++i) {
                const auto even = work[static_cast<std::size_t>(offset + i)];
                const auto odd = work[static_cast<std::size_t>(offset + i + half)] * twiddle;
                work[static_cast<std::size_t>(offset + i)] = even + odd;
                work[static_cast<std::size_t>(offset + i + half)] = even - odd;
                twiddle *= step;
            }
        }
    }
    const int32_t bins = std::min(n_out, n / 2 + 1);
    for (int32_t k = 0; k < bins; ++k)
        power_out[k] = static_cast<float>(std::norm(work[static_cast<std::size_t>(k)]));
    std::fill(power_out + bins, power_out + n_out, 0.0F);
}

struct MelCase {
    int32_t n_fft;
    int32_t win_length;
    int32_t hop_length;
    int32_t chunk_length_s;
    int32_t sample_rate;
    int32_t n_mel_bins;
    float preemph;
    bool normalize;
};

std::vector<float> definition_mel(const std::vector<float>& audio, const std::vector<float>& fb,
                                  const MelCase& c) {
    const int32_t n_freq_bins = c.n_fft / 2 + 1;
    const int32_t n_samples = static_cast<int32_t>(audio.size());
    const int32_t chunk_samples = c.chunk_length_s * c.sample_rate;
    const int32_t valid_audio = std::min(n_samples, chunk_samples);
    const int32_t valid_frames = std::min(valid_audio / c.hop_length, chunk_samples / c.hop_length);

    std::vector<float> emphasized(static_cast<std::size_t>(chunk_samples), 0.0F);
    if (valid_audio > 0) {
        emphasized[0] = audio[0];
        for (int32_t i = 1; i < valid_audio; ++i)
            emphasized[static_cast<std::size_t>(i)] =
                audio[static_cast<std::size_t>(i)] -
                c.preemph * audio[static_cast<std::size_t>(i - 1)];
    }
    const int32_t pad = c.n_fft / 2;
    std::vector<float> padded(static_cast<std::size_t>(chunk_samples + 2 * pad), 0.0F);
    std::copy(emphasized.begin(), emphasized.end(), padded.begin() + pad);

    std::vector<float> window(static_cast<std::size_t>(c.n_fft), 0.0F);
    const int32_t offset = std::max(0, (c.n_fft - c.win_length) / 2);
    for (int32_t i = 0; i < c.win_length && offset + i < c.n_fft; ++i) {
        window[static_cast<std::size_t>(offset + i)] =
            c.win_length <= 1
                ? 0.0F
                : static_cast<float>(0.5 * (1.0 - std::cos(kPi2 * static_cast<double>(i) /
                                                           static_cast<double>(c.win_length - 1))));
    }

    const int32_t n_frames = 1 + (static_cast<int32_t>(padded.size()) - c.n_fft) / c.hop_length;
    std::vector<float> mel(static_cast<std::size_t>(c.n_mel_bins) * n_frames, 0.0F);
    std::vector<float> frame(static_cast<std::size_t>(c.n_fft));
    std::vector<float> power(static_cast<std::size_t>(n_freq_bins));
    std::vector<float> frame_mel(static_cast<std::size_t>(c.n_mel_bins));
    for (int32_t t = 0; t < std::min(valid_frames, n_frames); ++t) {
        for (int32_t i = 0; i < c.n_fft; ++i)
            frame[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(t * c.hop_length + i)] *
                window[static_cast<std::size_t>(i)];
        definition_rfft_power(frame.data(), c.n_fft, n_freq_bins, power.data());
        std::fill(frame_mel.begin(), frame_mel.end(), 0.0F);
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = power[static_cast<std::size_t>(f)];
            if (p == 0.0F)
                continue;
            const float* row = fb.data() + static_cast<std::size_t>(f) * c.n_mel_bins;
            for (int32_t m = 0; m < c.n_mel_bins; ++m)
                frame_mel[static_cast<std::size_t>(m)] += p * row[m];
        }
        for (int32_t m = 0; m < c.n_mel_bins; ++m)
            mel[static_cast<std::size_t>(m) * n_frames + t] =
                frame_mel[static_cast<std::size_t>(m)];
    }
    for (float& value : mel)
        value = std::log(value + 5.960464477539063e-08F);

    for (int32_t m = 0; m < c.n_mel_bins; ++m) {
        float* row = mel.data() + static_cast<std::size_t>(m) * n_frames;
        if (c.normalize && valid_frames > 0) {
            double mean = 0.0;
            for (int32_t t = 0; t < valid_frames; ++t)
                mean += static_cast<double>(row[t]);
            mean /= static_cast<double>(valid_frames);
            double variance = 0.0;
            for (int32_t t = 0; t < valid_frames; ++t) {
                const double delta = static_cast<double>(row[t]) - mean;
                variance += delta * delta;
            }
            const double stddev =
                valid_frames > 1
                    ? std::sqrt(variance / static_cast<double>(valid_frames - 1)) + 1e-5
                    : 1e-5;
            for (int32_t t = 0; t < valid_frames; ++t)
                row[t] = static_cast<float>((static_cast<double>(row[t]) - mean) / stddev);
        }
        const int32_t keep = c.normalize && valid_frames <= 0 ? 0 : valid_frames;
        std::fill(row + keep, row + n_frames, 0.0F);
    }

    std::vector<float> trimmed(static_cast<std::size_t>(c.n_mel_bins) * (n_frames - 1));
    for (int32_t m = 0; m < c.n_mel_bins; ++m)
        std::copy_n(mel.begin() + static_cast<std::ptrdiff_t>(m) * n_frames, n_frames - 1,
                    trimmed.begin() + static_cast<std::ptrdiff_t>(m) * (n_frames - 1));
    return trimmed;
}

bool same_bits(const std::vector<float>& actual, const std::vector<float>& expected) {
    return actual.size() == expected.size() &&
           (actual.empty() ||
            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
}

// Triangular filters on the mel scale with area normalization: each frequency
// row has at most two nonzero columns, like the bundled slaney filterbank.
std::vector<float> triangular_filterbank(int32_t n_freq_bins, int32_t n_mel_bins,
                                         int32_t sample_rate) {
    const auto to_mel = [](double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };
    const auto to_hz = [](double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); };
    const double top = to_mel(sample_rate / 2.0);
    std::vector<double> edges(static_cast<std::size_t>(n_mel_bins) + 2);
    for (std::size_t i = 0; i < edges.size(); ++i)
        edges[i] = to_hz(top * static_cast<double>(i) / static_cast<double>(n_mel_bins + 1));
    std::vector<float> fb(static_cast<std::size_t>(n_freq_bins) * n_mel_bins, 0.0F);
    for (int32_t f = 0; f < n_freq_bins; ++f) {
        const double hz = (sample_rate / 2.0) * f / static_cast<double>(n_freq_bins - 1);
        for (int32_t m = 0; m < n_mel_bins; ++m) {
            const double lo = edges[static_cast<std::size_t>(m)];
            const double mid = edges[static_cast<std::size_t>(m) + 1];
            const double hi = edges[static_cast<std::size_t>(m) + 2];
            const double rise = (hz - lo) / (mid - lo);
            const double fall = (hi - hz) / (hi - mid);
            const double weight = std::max(0.0, std::min(rise, fall)) * 2.0 / (hi - lo);
            fb[static_cast<std::size_t>(f) * n_mel_bins + m] = static_cast<float>(weight);
        }
    }
    return fb;
}

std::vector<float> seeded_speechlike(int32_t length, int32_t sample_rate, uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> noise(0.0F, 0.05F);
    std::vector<float> audio(static_cast<std::size_t>(length));
    for (int32_t i = 0; i < length; ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        const double envelope = 0.5 + 0.5 * std::sin(kPi2 * 3.0 * t);
        audio[static_cast<std::size_t>(i)] =
            static_cast<float>(envelope * (0.3 * std::sin(kPi2 * 220.0 * t) +
                                           0.15 * std::sin(kPi2 * 1730.0 * t) +
                                           0.05 * std::sin(kPi2 * 5100.0 * t))) +
            noise(generator);
    }
    return audio;
}

void check_mel_matches_definition(const std::vector<float>& audio, const std::vector<float>& fb,
                                  const MelCase& c, const std::string& test_name) {
    const auto actual = trtmc::canary::extract_mel_spectrogram(
        audio.data(), static_cast<int32_t>(audio.size()), fb.data(), c.n_fft / 2 + 1, c.n_mel_bins,
        c.n_fft, c.win_length, c.hop_length, c.chunk_length_s, c.sample_rate, c.preemph,
        c.normalize);
    check(same_bits(actual.data, definition_mel(audio, fb, c)), test_name);
}

void test_canary_fft_power_matches_definition_bits() {
    std::mt19937 generator(11);
    std::normal_distribution<float> noise(0.0F, 0.5F);
    for (const int32_t n : {1, 2, 4, 8, 16, 64, 256, 400, 512, 1024, 10}) {
        std::vector<float> input(static_cast<std::size_t>(n));
        for (float& value : input)
            value = noise(generator);
        std::vector<float> expected(static_cast<std::size_t>(n / 2 + 1));
        definition_rfft_power(input.data(), n, n / 2 + 1, expected.data());
        check(same_bits(trtmc::canary::detail::rfft_power(input), expected),
              "canary FFT power equals the inline-twiddle definition, n = " + std::to_string(n));
    }
}

void test_canary_mel_matches_definition_bits() {
    const MelCase canary{512, 400, 160, 30, 16000, 128, 0.97F, true};
    const auto slaney_like = triangular_filterbank(257, 128, 16000);
    const auto speech = seeded_speechlike(66560, 16000, 5);
    check_mel_matches_definition(speech, slaney_like, canary,
                                 "canary 4.16 s request mel equals the definition");
    MelCase unnormalized = canary;
    unnormalized.normalize = false;
    check_mel_matches_definition(speech, slaney_like, unnormalized,
                                 "canary unnormalized mel equals the definition");

    // Small geometry: every filterbank shape, lengths around the chunk.
    const MelCase small{64, 50, 13, 1, 1000, 20, 0.97F, true};
    const int32_t small_bins = 33;
    std::mt19937 generator(23);
    std::uniform_real_distribution<float> positive(0.01F, 1.0F);
    std::vector<float> dense(static_cast<std::size_t>(small_bins) * small.n_mel_bins);
    for (float& value : dense)
        value = positive(generator);
    std::vector<float> holes = dense;
    for (std::size_t i = 0; i < holes.size(); ++i) {
        const auto row = i / static_cast<std::size_t>(small.n_mel_bins);
        const auto column = i % static_cast<std::size_t>(small.n_mel_bins);
        if (row % 5 == 0 || column % 3 == 1 || (row % 4 == 1 && column > 7))
            holes[i] = (row % 2 == 0) ? 0.0F : -0.0F;
    }
    const auto sparse = triangular_filterbank(small_bins, small.n_mel_bins, small.sample_rate);
    const struct {
        const std::vector<float>* fb;
        const char* name;
    } banks[] = {{&dense, "dense"}, {&holes, "zero rows and holes"}, {&sparse, "triangular"}};
    for (const auto& bank : banks) {
        for (const int32_t length : {0, 1, 12, 13, 14, 500, 999, 1000, 1001, 1700}) {
            for (const bool normalize : {true, false}) {
                MelCase c = small;
                c.normalize = normalize;
                c.preemph = normalize ? 0.97F : 0.0F;
                check_mel_matches_definition(
                    seeded_speechlike(length, small.sample_rate, static_cast<uint32_t>(length)),
                    *bank.fb, c,
                    std::string("canary mel equals the definition, ") + bank.name + ", " +
                        std::to_string(length) + " samples" + (normalize ? "" : ", unnormalized"));
            }
        }
    }

    // Powers past float range: inf * 0 makes NaN wherever a zero coefficient
    // meets it, which the full-row fallback keeps.
    auto loud = seeded_speechlike(1000, small.sample_rate, 31);
    for (float& sample : loud)
        sample *= 1.0e30F;
    loud[400] = std::numeric_limits<float>::infinity();
    for (const bool normalize : {true, false}) {
        MelCase c = small;
        c.normalize = normalize;
        check_mel_matches_definition(loud, holes, c,
                                     std::string("canary mel with non-finite power equals the "
                                                 "definition") +
                                         (normalize ? "" : ", unnormalized"));
    }
}

// Rounding-sensitive spectra: Hann-windowed tone frames and on-bin cosines put
// most bins near zero, where a one-ulp twiddle difference flips the float
// power; noise frames almost never do. Where no multiply-add is contracted
// (x86-64 by default) every operation rounds the same in any compile context,
// so the transform must equal the definition bit for bit. Where FMAs are
// contracted (__FP_FAST_FMA: aarch64, -mfma), whether a product fuses depends
// on how each loop is vectorized, so this file's own copy of the definition
// can round differently from the library in some builds (aarch64 GCC 13.3 at
// -O2); there each bin may differ by 1e-12 of the frame's peak power, far
// above a double rounding (~1e-16 of the peak) and far below any wrong twiddle.
#ifdef __FP_FAST_FMA
constexpr double kToneFramePeakTolerance = 1e-12;
#else
constexpr double kToneFramePeakTolerance = 0.0;
#endif

bool matches_tone_frame(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (kToneFramePeakTolerance == 0.0)
        return same_bits(actual, expected);
    if (actual.size() != expected.size())
        return false;
    float peak = 0.0F;
    for (const float value : expected)
        peak = std::max(peak, value);
    const double tolerance = kToneFramePeakTolerance * static_cast<double>(peak);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!(std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i])) <=
              tolerance))
            return false;
    }
    return true;
}

void test_canary_fft_tone_frames_match_definition_bits() {
    std::vector<float> window(512, 0.0F);
    for (int32_t i = 0; i < 400; ++i)
        window[static_cast<std::size_t>(56 + i)] =
            static_cast<float>(0.5 * (1.0 - std::cos(kPi2 * i / 399.0)));
    int32_t differing = 0;
    for (int32_t frame_index = 0; frame_index < 200; ++frame_index) {
        std::vector<float> frame(512);
        for (int32_t i = 0; i < 512; ++i) {
            frame[static_cast<std::size_t>(i)] =
                static_cast<float>(0.4 *
                                   std::sin(kPi2 * 220.0 * (frame_index * 160 + i) / 16000.0)) *
                window[static_cast<std::size_t>(i)];
        }
        std::vector<float> expected(257);
        definition_rfft_power(frame.data(), 512, 257, expected.data());
        differing += matches_tone_frame(trtmc::canary::detail::rfft_power(frame), expected) ? 0 : 1;
    }
    for (int32_t bin = 1; bin < 256; bin += 7) {
        std::vector<float> cosine(512);
        for (int32_t i = 0; i < 512; ++i)
            cosine[static_cast<std::size_t>(i)] =
                static_cast<float>(std::cos(kPi2 * bin * i / 512.0));
        std::vector<float> expected(257);
        definition_rfft_power(cosine.data(), 512, 257, expected.data());
        differing +=
            matches_tone_frame(trtmc::canary::detail::rfft_power(cosine), expected) ? 0 : 1;
    }
    check(differing == 0, "canary FFT power equals the definition on tone frames (" +
                              std::to_string(differing) + " differ)");

    // A loud tone without pre-emphasis keeps the low-power bins above the log
    // guard, so the extraction itself sees them.
    const MelCase loud{512, 400, 160, 1, 16000, 128, 0.0F, false};
    std::vector<float> tone(16000);
    for (std::size_t i = 0; i < tone.size(); ++i)
        tone[i] =
            static_cast<float>(1000.0 * std::sin(kPi2 * 220.0 * static_cast<double>(i) / 16000.0));
    check_mel_matches_definition(tone, triangular_filterbank(257, 128, 16000), loud,
                                 "canary loud-tone mel equals the definition");
}

void test_canary_shape_and_empty_audio() {
    const int32_t sample_rate = 16000;
    const int32_t n_fft = 512;
    const int32_t win_length = 400;
    const int32_t hop_length = 160;
    const int32_t chunk_length_s = 30;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = 80;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

    const auto mel = trtmc::canary::extract_mel_spectrogram(
        nullptr, 0, fb.data(), n_freq_bins, n_mel_bins, n_fft, win_length, hop_length,
        chunk_length_s, sample_rate, 0.97F, true);

    check(mel.n_mels == n_mel_bins, "canary mel keeps mel bin count");
    check(mel.n_frames == 3000, "canary mel frame count matches 30s HF window");
    check(static_cast<int32_t>(mel.data.size()) == n_mel_bins * mel.n_frames,
          "canary mel data size matches shape");
    check(mel.valid_frames == 0, "canary empty audio has no valid feature frames");
    check(std::all_of(mel.data.begin(), mel.data.end(), [](float value) { return value == 0.0F; }),
          "canary empty audio uses NeMo's zero pad value");
}

trtmc::canary::MelResult extract_test_mel(const std::vector<float>& audio) {
    const int32_t sample_rate = 32;
    const int32_t n_fft = 8;
    const int32_t win_length = 6;
    const int32_t hop_length = 2;
    const int32_t chunk_length_s = 1;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = n_freq_bins;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);
    return trtmc::canary::extract_mel_spectrogram(
        audio.data(), static_cast<int32_t>(audio.size()), fb.data(), n_freq_bins, n_mel_bins, n_fft,
        win_length, hop_length, chunk_length_s, sample_rate, 0.97F, true);
}

void test_canary_low_volume_uses_per_feature_normalization() {
    std::vector<float> audio(24);
    for (std::size_t i = 0; i < audio.size(); ++i) {
        audio[i] = static_cast<float>(0.35 * std::sin(static_cast<double>(i) * 0.47) +
                                      0.2 * std::cos(static_cast<double>(i) * 0.19));
    }
    for (float& sample : audio)
        sample *= 0.12F;

    const auto quiet = extract_test_mel(audio);
    bool centered = true;
    bool has_normalized_variation = false;
    for (int32_t m = 0; m < quiet.n_mels; ++m) {
        const std::size_t base = static_cast<std::size_t>(m) * quiet.n_frames;
        double mean = 0.0;
        double square_sum = 0.0;
        for (int32_t t = 0; t < quiet.valid_frames; ++t) {
            mean += quiet.data[base + t];
            square_sum += static_cast<double>(quiet.data[base + t]) * quiet.data[base + t];
        }
        mean /= quiet.valid_frames;
        centered = centered && std::abs(mean) < 1e-5;
        has_normalized_variation = has_normalized_variation || square_sum > 1.0;
    }
    check(centered, "canary low-volume features use NeMo per-feature centering");
    check(has_normalized_variation,
          "canary low-volume features retain normalized speech variation");
}

void test_canary_silence_padding_uses_valid_frame_statistics() {
    std::vector<float> audio(24, 0.0F);
    for (std::size_t i = 6; i < 18; ++i) {
        audio[i] = static_cast<float>(0.4 * std::sin(static_cast<double>(i - 6) * 0.61));
    }
    const auto mel = extract_test_mel(audio);
    check(mel.valid_frames == static_cast<int32_t>(audio.size()) / 2,
          "canary silence-padded audio preserves its valid duration");

    bool normalized = true;
    bool padded_tail_zero = true;
    for (int32_t m = 0; m < mel.n_mels; ++m) {
        const std::size_t base = static_cast<std::size_t>(m) * mel.n_frames;
        double mean = 0.0;
        for (int32_t t = 0; t < mel.valid_frames; ++t)
            mean += mel.data[base + t];
        mean /= mel.valid_frames;
        normalized = normalized && std::abs(mean) < 1e-5;
        for (int32_t t = mel.valid_frames; t < mel.n_frames; ++t)
            padded_tail_zero = padded_tail_zero && mel.data[base + t] == 0.0F;
    }
    check(normalized, "canary silence-padded audio is normalized over valid frames only");
    check(padded_tail_zero, "canary chunk padding uses NeMo's zero feature value");
}

} // namespace

int main() {
    test_canary_fft_matches_direct_dft();
    test_canary_fft_power_matches_definition_bits();
    test_canary_mel_matches_definition_bits();
    test_canary_fft_tone_frames_match_definition_bits();
    test_canary_shape_and_empty_audio();
    test_canary_low_volume_uses_per_feature_normalization();
    test_canary_silence_padding_uses_valid_frame_statistics();
    return failures;
}
