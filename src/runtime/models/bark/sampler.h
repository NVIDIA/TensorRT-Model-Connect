/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

// Bark-owned sampling state shared by the semantic, coarse, and fine stages.
//
// The CUDA implementation follows PyTorch's CUDA multinomial RNG mapping so
// that a request seed has the same meaning as the Hugging Face Bark reference.
// Builds without the optional CUDA sampling bridge retain a host fallback.
class BarkSampler {
  public:
    explicit BarkSampler(void* stream);
    ~BarkSampler();

    BarkSampler(const BarkSampler&) = delete;
    BarkSampler& operator=(const BarkSampler&) = delete;

    void reset(int64_t seed);
    int32_t sample(const float* logits, int32_t vocab_size, float temperature, int32_t top_k);
    std::vector<int32_t> sample_rows(const float* logits, int32_t rows, int32_t row_stride,
                                     int32_t vocab_size, float temperature, int32_t top_k);

    const char* sampler_type() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc
