/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/lfm2_moe/conv_state.h"
#include "runtime/models/lfm2_moe/inference_state.h"
#include "runtime/models/lfm2_moe/kv_cache.h"

#include <memory>

namespace trtmc {

class Lfm2MoeHybridState final : public Lfm2MoeInferenceState {
  public:
    Lfm2MoeHybridState(std::unique_ptr<Lfm2MoeKvCache> kv, std::unique_ptr<Lfm2MoeConvState> conv);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;

    int32_t position() const override;
    int32_t max_length() const override;
    int32_t num_layers() const override;
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "lfm2_moe_hybrid_conv_attention"; }
    bool ok() const override;

    Lfm2MoeKvCache& kv_cache() { return *kv_; }
    Lfm2MoeConvState& conv_state() { return *conv_; }

  private:
    void validate_positions() const;

    std::unique_ptr<Lfm2MoeKvCache> kv_;
    std::unique_ptr<Lfm2MoeConvState> conv_;
};

} // namespace trtmc
