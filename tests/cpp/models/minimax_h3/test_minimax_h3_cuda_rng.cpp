/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/torch_cuda_normal.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

uint32_t bits(float value) {
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

uint64_t update_fnv1a(uint64_t hash, const std::vector<float>& values) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
    for (std::size_t index = 0; index < values.size() * sizeof(float); ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

int main() {
    constexpr std::size_t video_count = 24ULL * 37ULL * 48ULL * 84ULL;
    constexpr std::size_t audio_count = 414ULL * 32ULL;
    const auto video = trtmc::minimax_h3::torch_cuda_normal(video_count, 0);
    const auto offset = trtmc::minimax_h3::torch_cuda_normal_consumed_offset(video_count);
    const auto audio = trtmc::minimax_h3::torch_cuda_normal(audio_count, 0, offset);
    constexpr uint32_t expected_video[] = {0xbf901b85U, 0xbf93808aU, 0xbe804bd6U, 0xbede255eU,
                                           0x3f594515U, 0x3f312784U, 0xbea1cc6dU, 0xc0075fc2U};
    constexpr uint32_t expected_audio[] = {0x3f6c3b90U, 0xbf07dfc2U, 0xbfb125e1U, 0xbe1fae64U,
                                           0x3e1c49afU, 0xbf9724beU, 0xbf320601U, 0xc02f1fdaU};
    int failures = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        if (bits(video[index]) != expected_video[index]) {
            std::cerr << "FAIL: H3 video torch.randn mismatch at " << index << " actual=0x"
                      << std::hex << bits(video[index]) << " expected=0x" << expected_video[index]
                      << std::dec << '\n';
            ++failures;
        }
        if (bits(audio[index]) != expected_audio[index]) {
            std::cerr << "FAIL: H3 sequential audio torch.randn mismatch at " << index
                      << " actual=0x" << std::hex << bits(audio[index]) << " expected=0x"
                      << expected_audio[index] << std::dec << '\n';
            ++failures;
        }
    }
    uint64_t hash = update_fnv1a(14695981039346656037ULL, video);
    hash = update_fnv1a(hash, audio);
    if (hash != 0xb68438da31d3c096ULL) {
        std::cerr << "FAIL: H3 full video+audio torch.randn hash mismatch actual=0x" << std::hex
                  << hash << std::dec << '\n';
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
