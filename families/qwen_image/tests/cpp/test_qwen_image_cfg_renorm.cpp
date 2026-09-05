/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image/runtime/pipeline.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void fill_seeded(std::vector<float>& buffer, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-3.0F, 3.0F);
    for (auto& value : buffer)
        value = distribution(generator);
}

bool buffers_equal(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

void test_per_sample_independence_at_b2() {
    constexpr int kImageTokens = 16;
    constexpr std::size_t kChannels = 64;
    constexpr float kCfgScale = 4.0F;
    std::vector<float> pos_s0(static_cast<std::size_t>(kImageTokens) * kChannels);
    std::vector<float> neg_s0(pos_s0.size());
    std::vector<float> pos_s1(pos_s0.size());
    std::vector<float> neg_s1(pos_s0.size());
    fill_seeded(pos_s0, 1001U);
    fill_seeded(neg_s0, 2002U);
    fill_seeded(pos_s1, 3003U);
    fill_seeded(neg_s1, 4004U);

    std::vector<float> out_s0(pos_s0.size());
    std::vector<float> out_s1(pos_s1.size());
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(pos_s0, neg_s0, kCfgScale, kImageTokens,
                                                      kChannels, out_s0);
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(pos_s1, neg_s1, kCfgScale, kImageTokens,
                                                      kChannels, out_s1);

    std::vector<float> pos_batch(2 * pos_s0.size());
    std::vector<float> neg_batch(2 * neg_s0.size());
    std::memcpy(pos_batch.data(), pos_s0.data(), pos_s0.size() * sizeof(float));
    std::memcpy(pos_batch.data() + pos_s0.size(), pos_s1.data(), pos_s1.size() * sizeof(float));
    std::memcpy(neg_batch.data(), neg_s0.data(), neg_s0.size() * sizeof(float));
    std::memcpy(neg_batch.data() + neg_s0.size(), neg_s1.data(), neg_s1.size() * sizeof(float));
    std::vector<float> out_batch(pos_batch.size());
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(pos_batch, neg_batch, kCfgScale,
                                                      2 * kImageTokens, kChannels, out_batch);

    std::vector<float> out_s0_batched(out_s0.size());
    std::vector<float> out_s1_batched(out_s1.size());
    std::memcpy(out_s0_batched.data(), out_batch.data(), out_s0_batched.size() * sizeof(float));
    std::memcpy(out_s1_batched.data(), out_batch.data() + out_s0_batched.size(),
                out_s1_batched.size() * sizeof(float));
    check(buffers_equal(out_s0, out_s0_batched), "sample 0 batched output matches independent run");
    check(buffers_equal(out_s1, out_s1_batched), "sample 1 batched output matches independent run");
}

void test_renorm_basic_invariants() {
    constexpr int kImageTokens = 1;
    constexpr std::size_t kChannels = 8;
    std::vector<float> positive(kChannels);
    for (std::size_t index = 0; index < kChannels; ++index)
        positive[index] = static_cast<float>(index + 1);
    std::vector<float> negative(kChannels, 0.0F);
    std::vector<float> output;
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(positive, negative, 1.0F, kImageTokens,
                                                      kChannels, output);
    bool matches = true;
    for (std::size_t index = 0; index < kChannels; ++index)
        matches = matches && std::fabs(output[index] - positive[index]) <= 1e-6F;
    check(matches, "cfg=1.0 makes the renorm an identity");
    check(output.size() == kChannels, "renorm resizes output to tokens times channels");
}

} // namespace

int main() {
    test_per_sample_independence_at_b2();
    test_renorm_basic_invariants();
    return failures == 0 ? 0 : 1;
}
