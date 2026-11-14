/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_5_moe/runtime/hybrid_state.h"

#include <utility>

namespace trtmc {

Qwen35MoeHybridState::Qwen35MoeHybridState(std::unique_ptr<Qwen35MoeKvCache> kv,
                                           std::unique_ptr<Qwen35MoeRecurrentState> ssm)
    : kv_(std::move(kv)), ssm_(std::move(ssm)) {}

void Qwen35MoeHybridState::reset() {
    kv_->reset();
    ssm_->reset();
}

void Qwen35MoeHybridState::bind_to(ITrtModule& module) {
    kv_->bind_to(module);
    ssm_->bind_to(module);
}

void Qwen35MoeHybridState::prepare_step(TensorMap& inputs, int32_t seq_len) {
    kv_->prepare_step(inputs, seq_len);
}

void Qwen35MoeHybridState::advance(int32_t n_tokens) {
    kv_->advance(n_tokens);
    ssm_->advance(n_tokens);
}

int32_t Qwen35MoeHybridState::position() const {
    return kv_->position();
}

int32_t Qwen35MoeHybridState::max_length() const {
    return kv_->max_length();
}

int32_t Qwen35MoeHybridState::num_layers() const {
    return kv_->num_layers() + ssm_->num_layers();
}

std::size_t Qwen35MoeHybridState::device_memory_bytes() const {
    return kv_->device_memory_bytes() + ssm_->device_memory_bytes();
}

bool Qwen35MoeHybridState::ok() const {
    return kv_ && kv_->ok() && ssm_ && ssm_->ok();
}

} // namespace trtmc
