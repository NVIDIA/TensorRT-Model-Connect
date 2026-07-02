/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// test_qwen_image_cfg_renorm.cpp
//
// Unit test for ``QwenImagePipeline::combine_cfg_with_renorm`` exercising the
// per-token L2 renorm under batch. The audit comment in pipeline.cpp claims
// the reduction is strictly per-token (no cross-sample / cross-token pooling).
// This test verifies that claim numerically by:
//
//   (a) running the combiner on a B=2 stacked buffer (two independent
//       (cond, uncond) pairs) and
//   (b) running it twice separately on each sample,
//
// then comparing every output element bit-for-bit identical between (a) and
// (b). If the renorm ever accidentally pooled across samples, sample 0's
// output in (a) would diverge from (b)'s sample-0 output.
//
// No engine bytes required — runs in regular CI.
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
// =============================================================================

#include "runtime/models/qwen_image/pipeline.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Fill ``buf`` with deterministic per-element values using ``seed``. We avoid
// any normal-distribution helpers so the bytes are reproducible across runs
// and platforms.
void fill_seeded(std::vector<float>& buf, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-3.0F, 3.0F);
    for (auto& v : buf) {
        v = dist(gen);
    }
}

// Compare two flat fp32 buffers element-wise. Uses exact equality because the
// renorm path is deterministic: same inputs -> same outputs to the last ULP.
bool buffers_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void test_per_sample_independence_at_b2() {
    constexpr int kNImg = 16;             // tokens per sample
    constexpr std::size_t kChannels = 64; // Qwen-Image out_channels * patch^2
    constexpr float kCfgScale = 4.0F;

    // Two independent (pos, neg) sample pairs with disjoint RNG seeds so the
    // numerical contents differ substantially between them.
    std::vector<float> pos_s0(static_cast<std::size_t>(kNImg) * kChannels);
    std::vector<float> neg_s0(pos_s0.size());
    std::vector<float> pos_s1(pos_s0.size());
    std::vector<float> neg_s1(pos_s0.size());
    fill_seeded(pos_s0, /*seed=*/1001U);
    fill_seeded(neg_s0, /*seed=*/2002U);
    fill_seeded(pos_s1, /*seed=*/3003U);
    fill_seeded(neg_s1, /*seed=*/4004U);

    // Reference: run the combiner per sample independently.
    std::vector<float> out_s0_ref(pos_s0.size());
    std::vector<float> out_s1_ref(pos_s0.size());
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(pos_s0, neg_s0, kCfgScale, kNImg, kChannels,
                                                      out_s0_ref);
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(pos_s1, neg_s1, kCfgScale, kNImg, kChannels,
                                                      out_s1_ref);

    // Stack into a B=2 batched buffer (sample 0 tokens, then sample 1 tokens)
    // and call the combiner once for the whole batch.
    std::vector<float> pos_batch(2 * pos_s0.size());
    std::vector<float> neg_batch(2 * neg_s0.size());
    std::memcpy(pos_batch.data(), pos_s0.data(), pos_s0.size() * sizeof(float));
    std::memcpy(pos_batch.data() + pos_s0.size(), pos_s1.data(), pos_s1.size() * sizeof(float));
    std::memcpy(neg_batch.data(), neg_s0.data(), neg_s0.size() * sizeof(float));
    std::memcpy(neg_batch.data() + neg_s0.size(), neg_s1.data(), neg_s1.size() * sizeof(float));

    std::vector<float> out_batch(pos_batch.size());
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(pos_batch, neg_batch, kCfgScale,
                                                      /*n_tokens=*/2 * kNImg, kChannels, out_batch);

    // Slice out each sample from the batched output.
    std::vector<float> out_s0_from_batch(out_s0_ref.size());
    std::vector<float> out_s1_from_batch(out_s1_ref.size());
    std::memcpy(out_s0_from_batch.data(), out_batch.data(),
                out_s0_from_batch.size() * sizeof(float));
    std::memcpy(out_s1_from_batch.data(), out_batch.data() + out_s0_from_batch.size(),
                out_s1_from_batch.size() * sizeof(float));

    check(buffers_equal(out_s0_ref, out_s0_from_batch),
          "sample 0 batched output matches independent run");
    check(buffers_equal(out_s1_ref, out_s1_from_batch),
          "sample 1 batched output matches independent run");
}

void test_renorm_basic_invariants() {
    // For a single token with cfg_scale = 1.0, comb == pos so ||comb|| == ||pos||
    // and the renorm scale is exactly 1.0 -> out == pos.
    constexpr int kNImg = 1;
    constexpr std::size_t kChannels = 8;
    std::vector<float> pos(kChannels);
    for (std::size_t i = 0; i < kChannels; ++i) {
        pos[i] = static_cast<float>(i + 1);
    }
    std::vector<float> neg(kChannels, 0.0F); // irrelevant when cfg=1
    std::vector<float> out;
    trtmc::QwenImagePipeline::combine_cfg_with_renorm(pos, neg, /*cfg_scale=*/1.0F, kNImg,
                                                      kChannels, out);
    bool matches = true;
    for (std::size_t i = 0; i < kChannels; ++i) {
        if (std::fabs(out[i] - pos[i]) > 1e-6F) {
            matches = false;
            break;
        }
    }
    check(matches, "cfg=1.0 makes the renorm an identity");
    check(out.size() == kChannels, "combine_cfg_with_renorm resizes output to n_tokens * channels");
}

} // namespace

int main() {
    test_per_sample_independence_at_b2();
    test_renorm_basic_invariants();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All qwen_image cfg_renorm tests passed.\n";
    return 0;
}
