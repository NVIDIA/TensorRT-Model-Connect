/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lfm2_audio/runtime/conv_state.h"
#include "families/lfm2_audio/runtime/inference_state.h"
#include "families/lfm2_audio/runtime/kv_cache.h"

#include <memory>

namespace trtmc::lfm2_audio {

class Lfm2AudioHybridState final : public Lfm2AudioInferenceState {
  public:
    Lfm2AudioHybridState(std::unique_ptr<Lfm2AudioKvCache> kv,
                         std::unique_ptr<Lfm2AudioConvState> conv);

    void reset() override;
    void bind_to(ITrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;

    int32_t position() const override;
    int32_t max_length() const override;
    int32_t num_layers() const override;
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "lfm2_audio_hybrid_conv_attention"; }
    bool ok() const override;

    Lfm2AudioKvCache& kv_cache() { return *kv_; }
    Lfm2AudioConvState& conv_state() { return *conv_; }

  private:
    void validate_positions() const;

    std::unique_ptr<Lfm2AudioKvCache> kv_;
    std::unique_ptr<Lfm2AudioConvState> conv_;
};

} // namespace trtmc::lfm2_audio
