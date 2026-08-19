/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_state.h"

#include <utility>

namespace trtmc {

NemotronHHybridState::NemotronHHybridState(std::unique_ptr<NemotronHKvCache> kv,
                                           std::unique_ptr<NemotronHRecurrentState> ssm)
    : kv_(std::move(kv)), ssm_(std::move(ssm)) {}

void NemotronHHybridState::reset() {
    kv_->reset();
    ssm_->reset();
}

void NemotronHHybridState::bind_to(TrtModule& module) {
    kv_->bind_to(module);
    ssm_->bind_to(module);
}

void NemotronHHybridState::prepare_step(TensorMap& inputs, int32_t seq_len) {
    kv_->prepare_step(inputs, seq_len);
}

void NemotronHHybridState::advance(int32_t n_tokens) {
    kv_->advance(n_tokens);
    ssm_->advance(n_tokens);
}

int32_t NemotronHHybridState::position() const {
    return kv_->position();
}

int32_t NemotronHHybridState::max_length() const {
    return kv_->max_length();
}

int32_t NemotronHHybridState::num_layers() const {
    return kv_->num_layers() + ssm_->num_layers();
}

std::size_t NemotronHHybridState::device_memory_bytes() const {
    return kv_->device_memory_bytes() + ssm_->device_memory_bytes();
}

bool NemotronHHybridState::ok() const {
    return kv_ && kv_->ok() && ssm_ && ssm_->ok();
}

} // namespace trtmc
