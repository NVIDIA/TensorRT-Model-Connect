/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_8/hybrid_state.h"

#include <utility>

namespace trtmc {

Qwen38HybridState::Qwen38HybridState(std::unique_ptr<Qwen38KvCache> kv,
                                     std::unique_ptr<Qwen38RecurrentState> ssm)
    : kv_(std::move(kv)), ssm_(std::move(ssm)) {}

void Qwen38HybridState::reset() {
    kv_->reset();
    ssm_->reset();
}

void Qwen38HybridState::bind_to(TrtModule& module) {
    kv_->bind_to(module);
    ssm_->bind_to(module);
}

void Qwen38HybridState::prepare_step(TensorMap& inputs, int32_t seq_len) {
    kv_->prepare_step(inputs, seq_len);
}

void Qwen38HybridState::advance(int32_t n_tokens) {
    kv_->advance(n_tokens);
    ssm_->advance(n_tokens);
}

int32_t Qwen38HybridState::position() const {
    return kv_->position();
}

int32_t Qwen38HybridState::max_length() const {
    return kv_->max_length();
}

int32_t Qwen38HybridState::num_layers() const {
    return kv_->num_layers() + ssm_->num_layers();
}

std::size_t Qwen38HybridState::device_memory_bytes() const {
    return kv_->device_memory_bytes() + ssm_->device_memory_bytes();
}

bool Qwen38HybridState::ok() const {
    return kv_ && kv_->ok() && ssm_ && ssm_->ok();
}

} // namespace trtmc
