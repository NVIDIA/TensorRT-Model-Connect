/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "inference_state.h"
#include "kv_cache.h"
#include "recurrent_state.h"

#include <memory>

namespace trtmc {

class Qwen35HybridState final : public Qwen35InferenceState {
  public:
    Qwen35HybridState(std::unique_ptr<Qwen35KvCache> kv, std::unique_ptr<Qwen35RecurrentState> ssm);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override;
    int32_t max_length() const override;
    int32_t num_layers() const override;
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "qwen3_5_hybrid_kv_recurrent"; }
    bool ok() const override;

    Qwen35KvCache* kv_cache() { return kv_.get(); }
    const Qwen35KvCache* kv_cache() const { return kv_.get(); }
    Qwen35RecurrentState* recurrent_state() { return ssm_.get(); }
    const Qwen35RecurrentState* recurrent_state() const { return ssm_.get(); }

  private:
    std::unique_ptr<Qwen35KvCache> kv_;
    std::unique_ptr<Qwen35RecurrentState> ssm_;
};

} // namespace trtmc
