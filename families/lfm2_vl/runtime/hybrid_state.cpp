/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2_vl/runtime/hybrid_state.h"

#include <stdexcept>
#include <utility>

namespace trtmc {

Lfm2VlHybridState::Lfm2VlHybridState(std::unique_ptr<Lfm2VlKvCache> kv,
                                     std::unique_ptr<Lfm2VlConvState> conv)
    : kv_(std::move(kv)), conv_(std::move(conv)) {
    if (!kv_ || !conv_)
        throw std::invalid_argument("Lfm2VlHybridState requires KV and convolution state");
}

void Lfm2VlHybridState::validate_positions() const {
    if (kv_->position() != conv_->position())
        throw std::runtime_error("LFM2-VL KV and convolution state positions diverged");
}

void Lfm2VlHybridState::reset() {
    kv_->reset();
    conv_->reset();
    validate_positions();
}

void Lfm2VlHybridState::bind_to(ITrtModule& module) {
    kv_->bind_to(module);
    conv_->bind_to(module);
}

void Lfm2VlHybridState::prepare_step(TensorMap& inputs, int32_t seq_len) {
    validate_positions();
    kv_->prepare_step(inputs, seq_len);
}

void Lfm2VlHybridState::advance(int32_t n_tokens) {
    // Both operations are committed only after a successful TensorRT enqueue.
    // Native KV has already updated in place; the convolution output still
    // needs its same-stream D2D copy into the stable input buffer.
    kv_->advance(n_tokens);
    conv_->advance(n_tokens);
    validate_positions();
}

int32_t Lfm2VlHybridState::position() const {
    validate_positions();
    return kv_->position();
}

int32_t Lfm2VlHybridState::max_length() const {
    return kv_->max_length();
}

int32_t Lfm2VlHybridState::num_layers() const {
    return kv_->num_layers() + conv_->num_layers();
}

std::size_t Lfm2VlHybridState::device_memory_bytes() const {
    return kv_->device_memory_bytes() + conv_->device_memory_bytes();
}

bool Lfm2VlHybridState::ok() const {
    return kv_ && conv_ && kv_->ok() && conv_->ok();
}

} // namespace trtmc
