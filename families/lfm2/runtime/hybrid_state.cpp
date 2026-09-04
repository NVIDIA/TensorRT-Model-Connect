/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/hybrid_state.h"

#include <stdexcept>
#include <utility>

namespace trtmc {

Lfm2HybridState::Lfm2HybridState(std::unique_ptr<Lfm2KvCache> kv,
                                 std::unique_ptr<Lfm2ConvState> conv)
    : kv_(std::move(kv)), conv_(std::move(conv)) {
    if (!kv_ || !conv_)
        throw std::invalid_argument("Lfm2HybridState requires KV and convolution state");
}

void Lfm2HybridState::validate_positions() const {
    if (kv_->position() != conv_->position())
        throw std::runtime_error("LFM2 KV and convolution state positions diverged");
}

void Lfm2HybridState::reset() {
    kv_->reset();
    conv_->reset();
    validate_positions();
}

void Lfm2HybridState::bind_to(ITrtModule& module) {
    kv_->bind_to(module);
    conv_->bind_to(module);
}

void Lfm2HybridState::prepare_step(TensorMap& inputs, int32_t seq_len) {
    validate_positions();
    kv_->prepare_step(inputs, seq_len);
}

void Lfm2HybridState::advance(int32_t n_tokens) {
    // Both operations are committed only after a successful TensorRT enqueue.
    // Native KV has already updated in place; the convolution output still
    // needs its same-stream D2D copy into the stable input buffer.
    kv_->advance(n_tokens);
    conv_->advance(n_tokens);
    validate_positions();
}

int32_t Lfm2HybridState::position() const {
    validate_positions();
    return kv_->position();
}

int32_t Lfm2HybridState::max_length() const {
    return kv_->max_length();
}

int32_t Lfm2HybridState::num_layers() const {
    return kv_->num_layers() + conv_->num_layers();
}

std::size_t Lfm2HybridState::device_memory_bytes() const {
    return kv_->device_memory_bytes() + conv_->device_memory_bytes();
}

bool Lfm2HybridState::ok() const {
    return kv_ && conv_ && kv_->ok() && conv_->ok();
}

} // namespace trtmc
