/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace trtmc::smolvlm {

struct SamplingParams {
    double temperature{1.0}; // 0 selects greedy decoding.
    int64_t top_k{1};        // 1 selects greedy decoding; 0 keeps every token.
    double top_p{1.0};       // Nucleus mass kept when sampling, in (0, 1].
    int64_t seed{-1};        // -1 uses a fixed default seed.
};

// Host token selection: argmax (first maximum) when greedy, otherwise
// temperature softmax over the top-k tokens, nucleus-filtered, xorshift64 draws.
class Sampler {
  public:
    explicit Sampler(const SamplingParams& params);
    bool greedy() const;
    int32_t next(const float* logits, int32_t vocab_size);

  private:
    SamplingParams params_;
    uint64_t state_;
};

} // namespace trtmc::smolvlm
