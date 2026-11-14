/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hubert/runtime/ctc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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

// The official checkpoint's first five ids: blank, three special tokens, delimiter.
trtmc::hubert::CtcVocabulary vocabulary() {
    trtmc::hubert::CtcVocabulary v;
    v.tokens = {"<pad>", "<s>", "</s>", "<unk>", "|", "A", "B"};
    v.blank_token_id = 0;
    v.word_delimiter_token = "|";
    return v;
}

constexpr int kVocab = 7;

// One logit row per frame, one-hot on the wanted id.
std::vector<float> rows(const std::vector<int>& ids) {
    std::vector<float> block(ids.size() * static_cast<std::size_t>(kVocab), -10.0F);
    for (std::size_t frame = 0; frame < ids.size(); ++frame)
        block[frame * static_cast<std::size_t>(kVocab) + static_cast<std::size_t>(ids[frame])] =
            10.0F;
    return block;
}

trtmc::hubert::CtcDecoded decode(const std::vector<int>& ids) {
    const auto block = rows(ids);
    return trtmc::hubert::greedy_decode(block.data(), static_cast<std::int64_t>(ids.size()), kVocab,
                                        vocabulary());
}

void test_normalize() {
    // A ramp has a known mean and population variance, so the normalised clip
    // is checked in closed form.
    std::vector<float> ramp(100);
    for (std::size_t i = 0; i < ramp.size(); ++i)
        ramp[i] = static_cast<float>(i);
    trtmc::hubert::normalize_waveform(ramp);

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
    // Population, not sample, variance: Bessel's correction would give 99/100.
    check(std::fabs(variance - 99.0 / 100.0) > 1e-3, "the sample-variance spelling is excluded");

    // A constant clip has zero variance; the epsilon keeps it finite.
    std::vector<float> flat(16, 3.5F);
    trtmc::hubert::normalize_waveform(flat);
    for (const auto value : flat)
        check(std::isfinite(value), "a constant clip normalises to a finite value");
}

void test_mix_mono() {
    const std::vector<float> stereo{1.0F, 3.0F, -2.0F, 0.0F};
    const auto mono = trtmc::hubert::mix_mono(stereo.data(), 4, 2);
    check(mono.size() == 2, "stereo mixes to half as many frames");
    check_close(mono[0], 2.0, 1e-6, "first stereo frame averages");
    check_close(mono[1], -1.0, 1e-6, "second stereo frame averages");
    check(trtmc::hubert::mix_mono(stereo.data(), 4, 1).size() == 4, "mono passes through");

    bool refused = false;
    try {
        trtmc::hubert::mix_mono(stereo.data(), 3, 2);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "a partial stereo frame is refused");
}

void test_feature_frames() {
    const std::vector<std::int32_t> kernels{10, 3, 3, 3, 3, 2, 2};
    const std::vector<std::int32_t> strides{5, 2, 2, 2, 2, 2, 2};
    // The official geometry has a 320-sample hop and a 400-sample receptive
    // field: 66560 samples (the 4.16 s fixture) give 207 frames, 16000 give 49
    // and 373040 (the 23.3 s fixture) give 1165.
    check(trtmc::hubert::feature_frames(66560, kernels, strides) == 207, "66560 -> 207 frames");
    check(trtmc::hubert::feature_frames(16000, kernels, strides) == 49, "16000 -> 49 frames");
    check(trtmc::hubert::feature_frames(373040, kernels, strides) == 1165, "373040 -> 1165");
    check(trtmc::hubert::feature_frames(400, kernels, strides) == 1, "the receptive field is 400");
    check(trtmc::hubert::feature_frames(399, kernels, strides) == 0,
          "a clip shorter than the receptive field gives no frames");
}

void test_greedy_decode() {
    check(decode({0, 0, 0}).text.empty(), "a run of blanks decodes to nothing");
    check(decode({5, 5, 5}).text == "A", "a run of one token collapses to one letter");
    check(decode({5, 0, 5}).text == "AA", "a blank between repeats keeps both letters");
    check(decode({5, 5, 0, 6}).text == "AB", "collapse happens before the blank is dropped");
    check(decode({5, 4, 6}).text == "A B", "the word delimiter becomes a space");
    // Control: dropping the blank before collapsing would turn 'A<pad>A' into 'A'.
    check(decode({5, 0, 5}).text != "A", "dropping the blank first is excluded");

    // The reference tokenizer strips the joined text; the ids keep the delimiters.
    const auto edged = decode({4, 4, 0, 5, 0, 4, 6, 4});
    check(edged.text == "A B", "delimiters at either end leave no space");
    check(edged.token_ids == std::vector<std::int32_t>({4, 5, 4, 6, 4}),
          "ids keep every non-blank token, delimiters included");
    // Two delimiters separated by a blank stay two spaces, as in the reference.
    check(decode({5, 4, 0, 4, 6}).text == "A  B", "a doubled delimiter is not merged");
    check(decode({4, 0, 4}).text.empty(), "a transcript of delimiters only is empty");
    // Special tokens other than the blank are emitted verbatim, as the
    // reference's decode(skip_special_tokens=False) does.
    check(decode({3, 0, 1}).text == "<unk><s>", "special tokens keep their spelling");

    const auto decoded = decode({5, 0, 6});
    check(decoded.token_ids == std::vector<std::int32_t>({5, 6}), "token ids skip the blank");
}

void test_greedy_decode_refuses_bad_logits() {
    auto block = rows({5, 6});
    block[5] = INFINITY; // frame 0's best logit
    bool refused = false;
    try {
        trtmc::hubert::greedy_decode(block.data(), 2, kVocab, vocabulary());
    } catch (const std::runtime_error&) {
        refused = true;
    }
    check(refused, "a non-finite best logit is refused");
}

} // namespace

int test_pipeline_language();
int test_pipeline_refusals();

int main() {
    test_normalize();
    test_mix_mono();
    test_feature_frames();
    test_greedy_decode();
    test_greedy_decode_refuses_bad_logits();
    failures += test_pipeline_language();
    failures += test_pipeline_refusals();
    if (failures != 0) {
        std::printf("hubert ctc: %d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("hubert ctc: all checks passed\n");
    return EXIT_SUCCESS;
}
