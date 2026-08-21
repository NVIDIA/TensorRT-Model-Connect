/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/codec_reconstruction.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using trtmc::nemotron_voicechat::CodecCausalCache;
using trtmc::nemotron_voicechat::CodecReconstruction;
using trtmc::nemotron_voicechat::kCodecConvCacheChannels;
using trtmc::nemotron_voicechat::kCodecConvCacheWidth;
using trtmc::nemotron_voicechat::kCodecFrameSamples;
using trtmc::nemotron_voicechat::kCodecSpectralBins;
using trtmc::nemotron_voicechat::kCodecSpectralChannels;
using trtmc::nemotron_voicechat::kCodecSpectralFramesPerFrame;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

std::vector<float> make_oracle_input(int32_t codec_frames) {
    const int32_t spectral_frames = codec_frames * kCodecSpectralFramesPerFrame;
    std::vector<float> input(static_cast<std::size_t>(kCodecSpectralChannels) * spectral_frames);
    for (int32_t bin = 0; bin < kCodecSpectralBins; ++bin) {
        for (int32_t frame = 0; frame < spectral_frames; ++frame) {
            input[static_cast<std::size_t>(bin) * spectral_frames + frame] =
                -4.0F + 0.1F * static_cast<float>(bin) + 0.0005F * static_cast<float>(frame);
            input[static_cast<std::size_t>(kCodecSpectralBins + bin) * spectral_frames + frame] =
                0.013F * static_cast<float>(frame) + 0.17F * static_cast<float>(bin);
        }
    }
    return input;
}

bool near(float actual, float expected, float tolerance = 2e-7F) {
    return std::abs(actual - expected) <= tolerance;
}

void test_cpp_istft_matches_public_nemo_oracle() {
    // Expected values were produced by the public NeMo
    // ear_tts_vae_codec.spec_to_wav implementation at Speech commit
    // 097dfe9e2f55baf653b83035868bdc89849f1b47, with its four zero cache frames,
    // constrain_value_range=True, and exact eight-sample streaming trim.
    const std::vector<float> expected_prefix = {
        0.0F,
        0.0F,
        0.0F,
        -0.0002385340485488996F,
        0.00045015805517323315F,
        -0.0006814252119511366F,
        0.0006793595384806395F,
        -0.000921293452847749F,
        0.0008780672797001898F,
        -0.0009189462289214134F,
        0.000514875864610076F,
        -0.0005254276329651475F,
    };
    const std::vector<float> expected_suffix = {
        -0.0004921572399325669F,  -0.0003682767564896494F, -0.0003809256013482809F,
        -0.0001725396141409874F,  -0.0005017001531086862F, -0.00035479225334711373F,
        -0.00038785728975199163F, -0.0001604830176802352F,
    };
    const std::vector<float> expected_tail = {
        -0.0005111655918881297F, -0.0003412267833482474F, -0.0003947364166378975F,
        0.00003802729770541191F, -0.0011103010037913918F, 0.00015932628593873233F,
        -0.001281136879697442F,  0.0005493240314535797F,
    };

    CodecReconstruction reconstruction;
    const auto waveform = reconstruction.push(make_oracle_input(1), 1);
    check(waveform.size() == static_cast<std::size_t>(kCodecFrameSamples),
          "one 80 ms frame produces exactly 1764 samples");
    bool prefix_matches = waveform.size() >= expected_prefix.size();
    for (std::size_t index = 0; prefix_matches && index < expected_prefix.size(); ++index)
        prefix_matches = near(waveform[index], expected_prefix[index]);
    check(prefix_matches, "C++ ISTFT prefix matches pinned NeMo oracle");

    bool suffix_matches = waveform.size() >= expected_suffix.size();
    const std::size_t suffix_start = waveform.size() - expected_suffix.size();
    for (std::size_t index = 0; suffix_matches && index < expected_suffix.size(); ++index)
        suffix_matches = near(waveform[suffix_start + index], expected_suffix[index]);
    check(suffix_matches, "C++ ISTFT suffix matches pinned NeMo oracle");

    const auto tail = reconstruction.flush();
    check(tail.size() == expected_tail.size(), "flush emits NeMo's eight-sample right tail");
    bool tail_matches = tail.size() == expected_tail.size();
    for (std::size_t index = 0; tail_matches && index < expected_tail.size(); ++index)
        tail_matches = near(tail[index], expected_tail[index]);
    check(tail_matches, "C++ flush matches pinned NeMo oracle");
    check(!reconstruction.active(), "flush resets reconstruction state");
    check(reconstruction.flush().empty(), "second flush is empty");
}

void test_chunked_reconstruction_preserves_causal_overlap() {
    const auto first_input = make_oracle_input(1);
    auto second_input = make_oracle_input(1);
    for (int32_t bin = 0; bin < kCodecSpectralBins; ++bin) {
        for (int32_t frame = 0; frame < kCodecSpectralFramesPerFrame; ++frame) {
            second_input[static_cast<std::size_t>(bin) * kCodecSpectralFramesPerFrame + frame] +=
                0.03F;
            second_input[static_cast<std::size_t>(kCodecSpectralBins + bin) *
                             kCodecSpectralFramesPerFrame +
                         frame] += 0.11F;
        }
    }

    CodecReconstruction chunked;
    const auto first = chunked.push(first_input, 1);
    const auto second = chunked.push(second_input, 1);

    // Build the same two frames as one channel-major TRT output.
    std::vector<float> joined(static_cast<std::size_t>(kCodecSpectralChannels) * 2 *
                              kCodecSpectralFramesPerFrame);
    for (int32_t channel = 0; channel < kCodecSpectralChannels; ++channel) {
        const std::size_t dst =
            static_cast<std::size_t>(channel) * 2 * kCodecSpectralFramesPerFrame;
        const std::size_t src = static_cast<std::size_t>(channel) * kCodecSpectralFramesPerFrame;
        std::copy_n(first_input.begin() + static_cast<std::ptrdiff_t>(src),
                    kCodecSpectralFramesPerFrame,
                    joined.begin() + static_cast<std::ptrdiff_t>(dst));
        std::copy_n(
            second_input.begin() + static_cast<std::ptrdiff_t>(src), kCodecSpectralFramesPerFrame,
            joined.begin() + static_cast<std::ptrdiff_t>(dst + kCodecSpectralFramesPerFrame));
    }
    CodecReconstruction batched;
    const auto both = batched.push(joined, 2);
    check(both.size() == 2 * static_cast<std::size_t>(kCodecFrameSamples),
          "two-frame call produces exactly 3528 samples");

    bool equal = first.size() + second.size() == both.size();
    for (std::size_t index = 0; equal && index < first.size(); ++index)
        equal = near(first[index], both[index], 4e-7F);
    for (std::size_t index = 0; equal && index < second.size(); ++index)
        equal = near(second[index], both[first.size() + index], 4e-7F);
    check(equal, "frame-at-a-time overlap equals a batched decode");
}

void test_reset_and_explicit_convnext_cache_contract() {
    CodecReconstruction reconstruction;
    const auto input = make_oracle_input(1);
    const auto initial = reconstruction.push(input, 1);
    reconstruction.reset();
    const auto after_reset = reconstruction.push(input, 1);
    check(initial == after_reset, "reset restores zero spectral history");

    CodecCausalCache cache;
    bool sizes_match = true;
    bool initially_zero = true;
    for (int32_t block = 0; block < 9; ++block) {
        const std::size_t expected =
            static_cast<std::size_t>(kCodecConvCacheChannels[block]) * kCodecConvCacheWidth;
        sizes_match = sizes_match && cache.element_count(block) == expected;
        initially_zero = initially_zero && std::all_of(cache.current_data(block),
                                                       cache.current_data(block) + expected,
                                                       [](float value) { return value == 0.0F; });
        std::fill_n(cache.next_data(block), expected, static_cast<float>(block + 1));
    }
    check(sizes_match, "nine TensorRT causal caches have exact channel sizes");
    check(initially_zero, "new TensorRT causal caches are zero-initialized");
    cache.commit();
    check(cache.current_data(8)[0] == 9.0F, "cache commit advances decoder state");
    cache.reset();
    check(cache.current_data(8)[0] == 0.0F, "cache reset clears decoder state");
}

void test_invalid_spectral_input_is_rejected() {
    CodecReconstruction reconstruction;
    bool rejected_size = false;
    try {
        reconstruction.push(std::vector<float>(3, 0.0F), 1);
    } catch (const std::invalid_argument&) {
        rejected_size = true;
    }
    check(rejected_size, "wrong spectral tensor size is rejected");

    auto input = make_oracle_input(1);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected_nan = false;
    try {
        reconstruction.push(input, 1);
    } catch (const std::invalid_argument&) {
        rejected_nan = true;
    }
    check(rejected_nan, "non-finite spectral tensor is rejected");
}

} // namespace

int main() {
    test_cpp_istft_matches_public_nemo_oracle();
    test_chunked_reconstruction_preserves_causal_overlap();
    test_reset_and_explicit_convnext_cache_contract();
    test_invalid_spectral_input_is_rejected();
    return failures;
}
