/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_state.h"

#include <utility>

namespace trtmc {

Qwen35HybridState::Qwen35HybridState(std::unique_ptr<Qwen35KvCache> kv,
                                     std::unique_ptr<Qwen35RecurrentState> ssm)
    : kv_(std::move(kv)), ssm_(std::move(ssm)) {}

void Qwen35HybridState::reset() {
    kv_->reset();
    ssm_->reset();
}

void Qwen35HybridState::bind_to(TrtModule& module) {
    kv_->bind_to(module);
    ssm_->bind_to(module);
}

void Qwen35HybridState::prepare_step(TensorMap& inputs, int32_t seq_len) {
    kv_->prepare_step(inputs, seq_len);
}

void Qwen35HybridState::advance(int32_t n_tokens) {
    kv_->advance(n_tokens);
    ssm_->advance(n_tokens);
}

int32_t Qwen35HybridState::position() const {
    return kv_->position();
}

int32_t Qwen35HybridState::max_length() const {
    return kv_->max_length();
}

int32_t Qwen35HybridState::num_layers() const {
    return kv_->num_layers() + ssm_->num_layers();
}

std::size_t Qwen35HybridState::device_memory_bytes() const {
    return kv_->device_memory_bytes() + ssm_->device_memory_bytes();
}

bool Qwen35HybridState::ok() const {
    return kv_ && kv_->ok() && ssm_ && ssm_->ok();
}

} // namespace trtmc
