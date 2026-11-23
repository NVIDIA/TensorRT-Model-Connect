/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::minicpm_ocr {

// Geometry of the one-token hybrid decoder engine (runtime.json).
struct DecoderGeometry {
    int32_t hidden_size{0};
    int32_t vocab_size{0};
    int32_t max_cache_length{0};
    int32_t linear_layers{0};    // Gated DeltaNet layers
    int32_t attention_layers{0}; // gated full-attention layers
    int32_t conv_dim{0};
    int32_t conv_kernel{0};
    int32_t linear_heads{0};
    int32_t linear_key_dim{0};
    int32_t linear_value_dim{0};
    int32_t kv_dim{0};
    DType dtype{DType::kFloat16}; // visual rows and KV cache storage
};

struct SamplingOptions {
    double temperature{1.0};
    int64_t top_k{1};
    double top_p{1.0};
    int64_t seed{-1};
};

// A prompt whose image-pad positions take rows of `features` instead of the
// token embedding. features is row-major [rows, hidden] in geometry.dtype.
struct DecoderPrompt {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> feature_rows; // -1 for text tokens
    std::vector<uint8_t> features;
};

struct DecoderOutput {
    std::vector<int32_t> token_ids; // generated tokens, including a final EOS
    bool stopped{false};            // true when an EOS token ended generation
    double prefill_ms{0.0};
    double decode_ms{0.0};
};

class Decoder {
  public:
    Decoder(std::unique_ptr<ITrtModule> engine, DecoderGeometry geometry,
            std::vector<int32_t> eos_token_ids);

    DecoderOutput generate(const DecoderPrompt& prompt, const SamplingOptions& sampling,
                           int32_t max_new_tokens);
    const DecoderGeometry& geometry() const { return geometry_; }

  private:
    void reset();
    const std::vector<float>& step(int32_t token, const uint8_t* visual_row);
    int32_t sample(const std::vector<float>& logits, const SamplingOptions& sampling,
                   uint64_t& rng) const;
    bool is_eos(int32_t token) const;

    std::unique_ptr<ITrtModule> engine_;
    DecoderGeometry geometry_;
    std::vector<int32_t> eos_token_ids_;
    cudaStream_t stream_{nullptr};
    std::vector<DeviceTensor> conv_state_, conv_present_, ssm_state_, ssm_present_;
    std::vector<DeviceTensor> cache_k_, cache_v_, present_k_, present_v_;
    std::vector<float> mask_;
    std::vector<float> logits_;
    std::vector<uint8_t> zero_row_;
    void* logits_device_{nullptr};
    int32_t position_{0};
};

} // namespace trtmc::minicpm_ocr
