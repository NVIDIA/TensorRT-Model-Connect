/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wav2vec2/runtime/ctc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void check_close(double actual, double expected, double tolerance, const char* what) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::printf("FAIL: %s (got %.8f, expected %.8f)\n", what, actual, expected);
        ++failures;
    }
}

trtmc::wav2vec2::CtcVocabulary vocabulary() {
    trtmc::wav2vec2::CtcVocabulary v;
    v.tokens = {"<pad>", "|", "A", "B"};
    v.blank_token_id = 0;
    v.word_delimiter_token = "|";
    return v;
}

// One logit row per frame, one-hot on the wanted id.
std::vector<float> rows(const std::vector<int>& ids, int vocab) {
    std::vector<float> block(ids.size() * static_cast<std::size_t>(vocab), -10.0F);
    for (std::size_t frame = 0; frame < ids.size(); ++frame)
        block[frame * static_cast<std::size_t>(vocab) + static_cast<std::size_t>(ids[frame])] =
            10.0F;
    return block;
}

std::string decode(const std::vector<int>& ids) {
    const auto block = rows(ids, 4);
    return trtmc::wav2vec2::greedy_decode(block.data(), static_cast<std::int64_t>(ids.size()), 4,
                                          vocabulary())
        .text;
}

void test_normalize() {
    // A ramp has a known mean and a known population variance, so the
    // normalised clip can be checked in closed form rather than by eye.
    std::vector<float> ramp(100);
    for (std::size_t i = 0; i < ramp.size(); ++i)
        ramp[i] = static_cast<float>(i);
    trtmc::wav2vec2::normalize_waveform(ramp);

    double mean = 0.0;
    for (const auto value : ramp)
        mean += value;
    mean /= static_cast<double>(ramp.size());
    double variance = 0.0;
    for (const auto value : ramp)
        variance += (value - mean) * (value - mean);
    variance /= static_cast<double>(ramp.size());
    check_close(mean, 0.0, 1e-5, "normalised clip has zero mean");
    check_close(variance, 1.0, 1e-5, "normalised clip has unit population variance");

    // Population, not sample, variance: with Bessel's correction the result
    // would be sqrt(99/100) = 0.99499 rather than 1.0, which this margin sees.
    check(std::fabs(variance - 99.0 / 100.0) > 1e-3, "the sample-variance spelling is excluded");

    // A constant clip has zero variance; the epsilon is what keeps it finite.
    std::vector<float> flat(16, 3.5F);
    trtmc::wav2vec2::normalize_waveform(flat);
    for (const auto value : flat)
        check(std::isfinite(value), "a constant clip normalises to a finite value");
}

void test_mix_mono() {
    const std::vector<float> stereo{1.0F, 3.0F, -2.0F, 0.0F};
    const auto mono = trtmc::wav2vec2::mix_mono(stereo.data(), 4, 2);
    check(mono.size() == 2, "stereo mixes to half as many frames");
    check_close(mono[0], 2.0, 1e-6, "first stereo frame averages");
    check_close(mono[1], -1.0, 1e-6, "second stereo frame averages");

    const auto passthrough = trtmc::wav2vec2::mix_mono(stereo.data(), 4, 1);
    check(passthrough.size() == 4, "mono passes through unchanged");
}

void test_feature_frames() {
    const std::vector<std::int32_t> kernels{10, 3, 3, 3, 3, 2, 2};
    const std::vector<std::int32_t> strides{5, 2, 2, 2, 2, 2, 2};
    // Measured against the reference: 66560 samples produce 207 frames and
    // 16000 produce 49, which is the 320-sample hop this geometry implies.
    check(trtmc::wav2vec2::feature_frames(66560, kernels, strides) == 207,
          "66560 samples give 207 frames");
    check(trtmc::wav2vec2::feature_frames(16000, kernels, strides) == 49,
          "16000 samples give 49 frames");
    check(trtmc::wav2vec2::feature_frames(100, kernels, strides) == 0,
          "a clip shorter than the receptive field gives no frames");
}

void test_greedy_decode() {
    check(decode({0, 0, 0}) == "", "a run of blanks decodes to nothing");
    check(decode({2, 2, 2}) == "A", "a run of one token collapses to one letter");
    check(decode({2, 0, 2}) == "AA", "a blank between repeats keeps both letters");
    check(decode({2, 2, 0, 3}) == "AB", "collapse happens before the blank is dropped");
    check(decode({2, 1, 3}) == "A B", "the word delimiter becomes a space");

    // The control: if the blank were dropped before the collapse, 'A<pad>A'
    // would become a single 'A'. The assertion above is what excludes that.
    check(decode({2, 0, 2}) != "A", "dropping the blank first is excluded");

    const auto decoded =
        trtmc::wav2vec2::greedy_decode(rows({2, 0, 3}, 4).data(), 3, 4, vocabulary());
    check(decoded.token_ids.size() == 2, "token ids skip the blank");
    check(decoded.token_ids[0] == 2 && decoded.token_ids[1] == 3, "token ids keep their order");
}

} // namespace

int test_pipeline_language();

int main() {
    test_normalize();
    test_mix_mono();
    test_feature_frames();
    test_greedy_decode();
    failures += test_pipeline_language();
    if (failures != 0) {
        std::printf("wav2vec2 ctc: %d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("wav2vec2 ctc: all checks passed\n");
    return EXIT_SUCCESS;
}
