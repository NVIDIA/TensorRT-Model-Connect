/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/nemotron_voicechat/thinker_inference_state.h"
#include "runtime/models/nemotron_voicechat/thinker_kv_cache.h"
#include "runtime/models/nemotron_voicechat/thinker_mamba_state.h"

#include <memory>

namespace trtmc {

class VoiceChatThinkerHybridState final : public VoiceChatThinkerInferenceState {
  public:
    VoiceChatThinkerHybridState(std::unique_ptr<VoiceChatThinkerKvCache> kv,
                                std::unique_ptr<VoiceChatThinkerMambaState> mamba);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs) override;
    void advance() override;
    bool ok() const override;

  private:
    std::unique_ptr<VoiceChatThinkerKvCache> kv_;
    std::unique_ptr<VoiceChatThinkerMambaState> mamba_;
};

} // namespace trtmc
