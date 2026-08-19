/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kv_cache.h"

#include <cstdio>
#include <vector>

namespace {

constexpr float kExpectedMaskedScore = -1.0e9F;
int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

void check_masked(float value, const char* name) {
    check(value == kExpectedMaskedScore, name);
}

void test_decode_mask_uses_numerically_safe_sentinel() {
    // Zero layers keeps this mask-only unit test CPU-only: no KV device buffers
    // are allocated, while the production mask construction remains unchanged.
    trtmc::GptNeoxKvCache cache(0, 4, 2, nullptr);
    cache.set_position(2);

    std::vector<float> mask;
    cache.build_attention_mask(mask);

    check(mask.size() == 5, "decode mask has cache rows plus current token");
    check(mask[0] == 0.0F && mask[1] == 0.0F, "valid decode cache rows are visible");
    check_masked(mask[2], "first stale decode cache row uses -1e9");
    check_masked(mask[3], "last stale decode cache row uses -1e9");
    check(mask[4] == 0.0F, "current decode token is visible");
}

void test_batched_prefill_mask_uses_numerically_safe_sentinel() {
    trtmc::GptNeoxKvCache cache(0, 4, 2, nullptr);
    trtmc::TensorMap inputs;
    cache.prepare_step(inputs, 3);

    const auto it = inputs.find("attention_mask");
    check(it != inputs.end(), "batched prefill emits an attention mask");
    if (it == inputs.end())
        return;

    const auto& tensor = it->second;
    check(tensor.shape == std::vector<int64_t>({3, 7}),
          "batched prefill mask shape covers cache and prompt rows");
    const auto* mask = static_cast<const float*>(tensor.data);

    // With an empty cache, every fixed cache column must stay masked.
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column)
            check_masked(mask[row * 7 + column], "prefill cache slot uses -1e9");
    }

    // Prompt columns are causal.
    check(mask[0 * 7 + 4] == 0.0F, "prefill row 0 sees prompt token 0");
    check_masked(mask[0 * 7 + 5], "prefill row 0 masks future token 1");
    check_masked(mask[0 * 7 + 6], "prefill row 0 masks future token 2");
    check(mask[1 * 7 + 4] == 0.0F && mask[1 * 7 + 5] == 0.0F,
          "prefill row 1 sees prompt tokens 0 and 1");
    check_masked(mask[1 * 7 + 6], "prefill row 1 masks future token 2");
    check(mask[2 * 7 + 4] == 0.0F && mask[2 * 7 + 5] == 0.0F && mask[2 * 7 + 6] == 0.0F,
          "prefill row 2 sees the full causal prefix");
}

} // namespace

int main() {
    test_decode_mask_uses_numerically_safe_sentinel();
    test_batched_prefill_mask_uses_numerically_safe_sentinel();

    if (failures == 0)
        std::fprintf(stderr, "All GPT-NeoX attention mask tests passed.\n");
    return failures;
}
