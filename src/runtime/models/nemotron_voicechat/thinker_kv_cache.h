/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Static FP32 KV state for VoiceChat's single-token thinker engine.

#include "runtime/models/nemotron_voicechat/thinker_inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct VoiceChatThinkerKvCacheNames {
    explicit VoiceChatThinkerKvCacheNames(int32_t num_layers);

    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string attention_mask{"attention_mask"};
};

class VoiceChatThinkerKvCache : public VoiceChatThinkerInferenceState {
  public:
    VoiceChatThinkerKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                            cudaStream_t stream);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs) override;
    void advance() override;
    bool ok() const override;

  private:
    VoiceChatThinkerKvCacheNames names_;
    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    std::vector<DeviceTensor> present_k_;
    std::vector<DeviceTensor> present_v_;
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t kv_dim_{0};
    int32_t position_{0};
    cudaStream_t stream_{nullptr};
    std::vector<float> mask_buf_;
};

} // namespace trtmc
