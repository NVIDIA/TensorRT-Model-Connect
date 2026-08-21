/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/thinker_hybrid_state.h"

#include <utility>

namespace trtmc {

VoiceChatThinkerHybridState::VoiceChatThinkerHybridState(
    std::unique_ptr<VoiceChatThinkerKvCache> kv, std::unique_ptr<VoiceChatThinkerMambaState> mamba)
    : kv_(std::move(kv)), mamba_(std::move(mamba)) {}

void VoiceChatThinkerHybridState::reset() {
    kv_->reset();
    mamba_->reset();
}

void VoiceChatThinkerHybridState::bind_to(TrtModule& module) {
    kv_->bind_to(module);
    mamba_->bind_to(module);
}

void VoiceChatThinkerHybridState::prepare_step(TensorMap& inputs) {
    kv_->prepare_step(inputs);
}

void VoiceChatThinkerHybridState::advance() {
    kv_->advance();
    mamba_->advance();
}

bool VoiceChatThinkerHybridState::ok() const {
    return kv_ && kv_->ok() && mamba_ && mamba_->ok();
}

} // namespace trtmc
