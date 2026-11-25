/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minicpmv4_6/runtime/sampler.h"
#include "families/minicpmv4_6/runtime/state.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::minicpmv4_6 {

// Prompt rows: a token id per row and, for vision rows, the index of the
// feature row (in `features`, [rows, hidden] at the decoder precision) that
// replaces the token embedding; -1 for text rows.
struct PromptRows {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> feature_rows;
    std::vector<std::uint8_t> features;
};

struct Generation {
    std::vector<std::int32_t> token_ids; // generated tokens only
    double prefill_ms{0.0}, decode_ms{0.0};
};

// The hybrid decoder: chunked prefill (prefill.plan) then one-token decoding
// (engine.plan) on one CUDA stream.
class Decoder {
  public:
    Decoder(std::unique_ptr<ITrtModule> decoder, std::unique_ptr<ITrtModule> prefill,
            const DecoderGeometry& geometry, std::int32_t vocab_size, std::int32_t hidden_size);

    // Generated ids stay below `token_limit`, the tokenizer's vocabulary (the logits may have
    // more rows: a padded embedding table).
    Generation generate(const PromptRows& prompt, std::int32_t max_new_tokens,
                        const std::vector<std::int32_t>& eos_ids, std::int32_t token_limit,
                        Sampler& sampler);
    std::int32_t capacity() const { return state_.capacity(); }
    DType precision() const { return dtype_; }

  private:
    void prefill(const PromptRows& prompt, std::vector<float>& logits);
    void step(std::int32_t token, std::vector<float>& logits);
    void copy_logits(const ITrtModule& module, std::vector<float>& logits);

    std::unique_ptr<ITrtModule> decoder_, prefill_;
    DecoderState state_;
    std::int32_t vocab_size_{0}, hidden_size_{0}, prefill_rows_{0};
    DType dtype_{DType::kFloat16};
    std::vector<std::uint8_t> chunk_features_;
    std::vector<std::int32_t> chunk_selector_;
};

} // namespace trtmc::minicpmv4_6
