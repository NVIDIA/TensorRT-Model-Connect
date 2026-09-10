/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_voicechat/runtime/thinker_hybrid_state.h"

#include <stdexcept>
#include <utility>

namespace trtmc {

VoiceChatThinkerHybridState::VoiceChatThinkerHybridState(
    std::unique_ptr<VoiceChatThinkerKvCache> kv, std::unique_ptr<VoiceChatThinkerMambaState> mamba)
    : kv_(std::move(kv)), mamba_(std::move(mamba)) {}

void VoiceChatThinkerHybridState::reset() {
    kv_->reset();
    mamba_->reset();
}

void VoiceChatThinkerHybridState::bind_to(ITrtModule& module) {
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

void VoiceChatThinkerHybridState::pin_kv_prefix() {
    if (!kv_)
        throw std::logic_error("VoiceChat thinker KV state is unavailable");
    kv_->pin_current_prefix();
}

void VoiceChatThinkerHybridState::capture_prompt_snapshot() {
    if (!kv_ || !mamba_)
        throw std::logic_error("VoiceChat thinker hybrid state is unavailable");
    kv_->capture_prompt_snapshot();
    mamba_->capture_prompt_snapshot();
}

void VoiceChatThinkerHybridState::restore_prompt_snapshot() {
    if (!prompt_snapshot_ready())
        throw std::logic_error("VoiceChat thinker prompt snapshot is unavailable");
    kv_->restore_prompt_snapshot();
    mamba_->restore_prompt_snapshot();
}

bool VoiceChatThinkerHybridState::prompt_snapshot_ready() const noexcept {
    return kv_ && mamba_ && kv_->prompt_snapshot_ready() && mamba_->prompt_snapshot_ready();
}

} // namespace trtmc
