/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ast/runtime/fbank.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(double actual, double expected, double tolerance, const char* name) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

constexpr std::int32_t kFrame = 400;
constexpr std::int32_t kShift = 160;
constexpr std::int32_t kFft = 512;
constexpr std::int32_t kMels = 4;
constexpr std::int32_t kMaxLength = 8;
constexpr float kMean = -4.2677393F;
constexpr float kScale = 1.0F / (2.0F * 4.5689974F);

trtmc::ast::FbankConfig base_config() {
    trtmc::ast::FbankConfig config;
    config.num_mel_bins = kMels;
    config.max_length = kMaxLength;
    config.feature_mean = kMean;
    config.feature_scale = kScale;
    // A flat window keeps the expected values below readable; the shape of the
    // window is pinned by the Python unit test against torchaudio instead.
    config.window.assign(static_cast<std::size_t>(kFrame), 1.0F);
    // One filter per quarter of the spectrum, each a single unit-weight bin, so
    // a tone lands in exactly one mel row.
    config.mel_filters.assign(
        static_cast<std::size_t>(kMels) * static_cast<std::size_t>(kFft / 2 + 1), 0.0F);
    const auto bins = static_cast<std::size_t>(kFft / 2 + 1);
    for (std::int32_t mel = 0; mel < kMels; ++mel)
        config.mel_filters[static_cast<std::size_t>(mel) * bins + (mel + 1) * 16] = 1.0F;
    return config;
}

std::vector<float> tone(std::int64_t count, double cycles_per_sample, float amplitude,
                        float offset) {
    std::vector<float> samples(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        samples[static_cast<std::size_t>(i)] =
            offset + amplitude * static_cast<float>(std::cos(2.0 * M_PI * cycles_per_sample *
                                                             static_cast<double>(i)));
    }
    return samples;
}

void test_frame_count_snips_the_edges() {
    // Whole frames only, no centring and no padding at either end.
    check(trtmc::ast::compute_ast_frame_count(400, kFrame, kShift) == 1, "exactly one frame");
    check(trtmc::ast::compute_ast_frame_count(399, kFrame, kShift) == 0, "short clip has none");
    check(trtmc::ast::compute_ast_frame_count(559, kFrame, kShift) == 1, "partial frame dropped");
    check(trtmc::ast::compute_ast_frame_count(560, kFrame, kShift) == 2, "second frame fits");
    // The 4.16 s asset the E2E uses, at 16 kHz.
    check(trtmc::ast::compute_ast_frame_count(66560, kFrame, kShift) == 414, "414 real frames");
}

void test_frame_removes_dc_then_preemphasises() {
    const auto config = base_config();
    // A constant signal is pure DC. Removing the offset leaves zero, and the
    // pre-emphasis of zero is zero: the frame must vanish entirely.
    const std::vector<float> flat(static_cast<std::size_t>(kFrame), 0.75F);
    std::vector<float> frame(static_cast<std::size_t>(kFrame));
    trtmc::ast::compute_ast_frame(flat.data(), config, frame.data());
    double worst = 0.0;
    for (const auto value : frame)
        worst = std::max(worst, static_cast<double>(std::fabs(value)));
    check_close(worst, 0.0, 1e-6, "constant input leaves nothing after DC removal");

    // A unit ramp, whose mean is (kFrame - 1) / 2. After removing it the
    // centred value at i is i - centre, and the pre-emphasis of that is
    // 0.03 * (i - centre) + 0.97 for every sample after the first.
    const double centre = 0.5 * (kFrame - 1);
    std::vector<float> ramp(static_cast<std::size_t>(kFrame));
    for (std::int32_t i = 0; i < kFrame; ++i)
        ramp[static_cast<std::size_t>(i)] = static_cast<float>(i);
    trtmc::ast::compute_ast_frame(ramp.data(), config, frame.data());
    // Kaldi replicates the frame's own first centred sample rather than
    // reaching back into the waveform, so the first output is 0.03 of it.
    // Reaching back to a zero instead would leave -centre here, which is
    // larger by a factor of about 33.
    check_close(frame[0], 0.03 * (0.0 - centre), 1e-3, "the first sample is replicated");
    check(std::fabs(static_cast<double>(frame[0]) + centre) > 1.0,
          "and not differenced against a zero outside the frame");
    for (const std::int32_t i : {1, 2, kFrame / 2, kFrame - 1}) {
        check_close(frame[static_cast<std::size_t>(i)], 0.03 * (i - centre) + 0.97, 1e-3,
                    "pre-emphasis follows the centred ramp");
    }
}

void test_power_spectrum_finds_a_pure_tone() {
    const auto config = base_config();
    // Exactly 32 cycles across the 512-point transform, so the energy lands in
    // bin 32 alone and nothing leaks.
    const auto samples = tone(kFrame, 32.0 / static_cast<double>(kFft), 1.0F, 0.0F);
    std::vector<float> frame(static_cast<std::size_t>(kFrame));
    trtmc::ast::compute_ast_frame(samples.data(), config, frame.data());
    const auto power = trtmc::ast::compute_ast_power_spectrum(frame.data(), config);
    check(power.size() == static_cast<std::size_t>(kFft / 2 + 1), "one-sided spectrum length");

    std::size_t peak = 0;
    for (std::size_t bin = 1; bin < power.size(); ++bin)
        if (power[bin] > power[peak])
            peak = bin;
    check(peak == 32, "the tone peaks in its own bin");
    // The transform is not normalised, so Parseval holds against the windowed
    // frame rather than against unity.
    double spectrum_total = power[0] + power[power.size() - 1];
    for (std::size_t bin = 1; bin + 1 < power.size(); ++bin)
        spectrum_total += 2.0 * power[bin];
    double frame_total = 0.0;
    for (const auto value : frame)
        frame_total += static_cast<double>(value) * value;
    check_close(spectrum_total / static_cast<double>(kFft), frame_total, frame_total * 1e-5,
                "Parseval holds across the transform");
}

void test_padding_is_normalised_not_zeroed() {
    const auto config = base_config();
    // One frame of audio, seven frames of padding.
    const auto samples = tone(kFrame, 32.0 / static_cast<double>(kFft), 0.5F, 0.0F);
    const auto features = trtmc::ast::compute_ast_features(
        samples.data(), static_cast<std::int64_t>(samples.size()), config);
    check(features.size() == static_cast<std::size_t>(kMaxLength) * kMels, "feature extent");

    // The raw filter bank is padded with zero and the normalisation runs
    // afterwards, so the padded rows carry (0 - mean) * scale, not zero.
    const double expected = (0.0 - static_cast<double>(kMean)) * static_cast<double>(kScale);
    for (std::int32_t row = 1; row < kMaxLength; ++row)
        for (std::int32_t mel = 0; mel < kMels; ++mel)
            check_close(features[static_cast<std::size_t>(row) * kMels + mel], expected, 1e-5,
                        "padded rows carry the normalised zero");
    check(std::fabs(features[0] - expected) > 1e-3, "the real row is not the padded value");
}

void test_long_input_is_truncated_to_the_trained_length() {
    const auto config = base_config();
    // Twenty frames of audio into an eight-frame spectrogram.
    const auto samples = tone(kFrame + 19 * kShift, 32.0 / static_cast<double>(kFft), 0.5F, 0.0F);
    const auto features = trtmc::ast::compute_ast_features(
        samples.data(), static_cast<std::int64_t>(samples.size()), config);
    check(features.size() == static_cast<std::size_t>(kMaxLength) * kMels, "truncated extent");
    const double padded = (0.0 - static_cast<double>(kMean)) * static_cast<double>(kScale);
    // Every row is real audio now, so none of them may be the padded constant.
    for (std::int32_t row = 0; row < kMaxLength; ++row)
        check(std::fabs(features[static_cast<std::size_t>(row) * kMels] - padded) > 1e-3,
              "a truncated clip leaves no padding behind");
}

void test_mono_mix_averages_the_channels() {
    const std::vector<float> stereo{1.0F, 3.0F, -2.0F, 2.0F};
    const auto mono = trtmc::ast::mix_ast_mono(stereo.data(), 4, 2);
    check(mono.size() == 2, "two interleaved frames become two samples");
    check_close(mono[0], 2.0, 1e-6, "first frame averages");
    check_close(mono[1], 0.0, 1e-6, "second frame averages");
    const auto passthrough = trtmc::ast::mix_ast_mono(stereo.data(), 4, 1);
    check(passthrough.size() == 4, "mono input is copied through");
}

void test_unusable_input_is_refused() {
    auto config = base_config();
    const std::vector<float> samples(static_cast<std::size_t>(kFrame), 0.0F);
    bool threw = false;
    try {
        // Not a power of two, so Kaldi's padded window size is wrong.
        config.n_fft = 500;
        trtmc::ast::compute_ast_features(samples.data(), kFrame, config);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "a non-power-of-two transform is refused");

    threw = false;
    try {
        const auto good = base_config();
        trtmc::ast::compute_ast_features(samples.data(), kFrame - 1, good);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "a clip shorter than one frame is refused");
}

} // namespace

int main() {
    test_frame_count_snips_the_edges();
    test_frame_removes_dc_then_preemphasises();
    test_power_spectrum_finds_a_pure_tone();
    test_padding_is_normalised_not_zeroed();
    test_long_input_is_truncated_to_the_trained_length();
    test_mono_mix_averages_the_channels();
    test_unusable_input_is_refused();
    if (g_failures == 0)
        std::cout << "ast filter bank: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
