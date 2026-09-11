/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_voicechat/runtime/thinker_mamba_state.h"

#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

void require_cuda_success(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("VoiceChat thinker ") + operation +
                                 " failed: " + cudaGetErrorString(status));
}

} // namespace

VoiceChatThinkerMambaState::VoiceChatThinkerMambaState(int32_t num_layers,
                                                       std::vector<TensorSpec> specs,
                                                       cudaStream_t stream)
    : specs_(std::move(specs)), num_layers_(num_layers), stream_(stream) {
    state_.resize(specs_.size());
    present_.resize(specs_.size());

    for (std::size_t si = 0; si < specs_.size(); ++si) {
        state_[si].reserve(static_cast<std::size_t>(num_layers));
        present_[si].reserve(static_cast<std::size_t>(num_layers));
        for (int32_t li = 0; li < num_layers; ++li) {
            state_[si].emplace_back(specs_[si].shape, DType::kFloat32, stream);
            present_[si].emplace_back(specs_[si].shape, DType::kFloat32, stream);
        }
    }

    reset();
}

void VoiceChatThinkerMambaState::bind_to(ITrtModule& module) {
    for (std::size_t si = 0; si < specs_.size(); ++si) {
        const auto& name = specs_[si].name;
        const auto& out_prefix =
            specs_[si].output_prefix.empty() ? ("present_" + name) : specs_[si].output_prefix;
        for (int32_t li = 0; li < num_layers_; ++li) {
            auto suffix = "_" + std::to_string(li);
            auto uli = static_cast<std::size_t>(li);

            // TensorRT does not guarantee that this model's recurrent input
            // remains intact while its matching output is produced. Keep
            // distinct stable addresses so CUDA Graph replay is safe without
            // introducing an input/output write-after-read hazard.
            module.bind_external(name + suffix, state_[si][uli].data());
            module.bind_external(out_prefix + suffix, present_[si][uli].data());
        }
    }
}

void VoiceChatThinkerMambaState::prepare_step(TensorMap& /*inputs*/) {}

void VoiceChatThinkerMambaState::advance() {
    for (std::size_t si = 0; si < specs_.size(); ++si) {
        for (int32_t li = 0; li < num_layers_; ++li) {
            auto uli = static_cast<std::size_t>(li);
            state_[si][uli].copy_from(present_[si][uli]);
        }
    }
}

void VoiceChatThinkerMambaState::reset() {
    for (std::size_t si = 0; si < specs_.size(); ++si) {
        for (int32_t li = 0; li < num_layers_; ++li) {
            auto uli = static_cast<std::size_t>(li);
            cudaMemsetAsync(state_[si][uli].data(), 0, state_[si][uli].nbytes(), stream_);
            cudaMemsetAsync(present_[si][uli].data(), 0, present_[si][uli].nbytes(), stream_);
        }
    }
    cudaStreamSynchronize(stream_);
}

void VoiceChatThinkerMambaState::capture_prompt_snapshot() {
    if (prompt_snapshot_ready_)
        throw std::logic_error("VoiceChat thinker Mamba prompt snapshot is already captured");
    std::vector<std::vector<DeviceTensor>> snapshot(specs_.size());
    for (std::size_t spec = 0; spec < specs_.size(); ++spec) {
        snapshot[spec].reserve(static_cast<std::size_t>(num_layers_));
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            snapshot[spec].emplace_back(specs_[spec].shape, DType::kFloat32, stream_);
            if (!snapshot[spec].back().ok())
                throw std::runtime_error(
                    "VoiceChat failed to allocate thinker Mamba prompt snapshot");
        }
    }

    for (std::size_t spec = 0; spec < specs_.size(); ++spec) {
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            const auto index = static_cast<std::size_t>(layer);
            require_cuda_success(
                cudaMemcpyAsync(snapshot[spec][index].data(), state_[spec][index].data(),
                                state_[spec][index].nbytes(), cudaMemcpyDeviceToDevice, stream_),
                "Mamba prompt-state capture");
        }
    }
    require_cuda_success(cudaStreamSynchronize(stream_), "Mamba prompt snapshot sync");
    prompt_snapshot_ = std::move(snapshot);
    prompt_snapshot_ready_ = true;
}

void VoiceChatThinkerMambaState::restore_prompt_snapshot() {
    if (!prompt_snapshot_ready_)
        throw std::logic_error("VoiceChat thinker Mamba prompt snapshot is unavailable");
    for (std::size_t spec = 0; spec < specs_.size(); ++spec) {
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            const auto index = static_cast<std::size_t>(layer);
            require_cuda_success(
                cudaMemcpyAsync(state_[spec][index].data(), prompt_snapshot_[spec][index].data(),
                                state_[spec][index].nbytes(), cudaMemcpyDeviceToDevice, stream_),
                "Mamba prompt-state restore");
        }
    }
    require_cuda_success(cudaStreamSynchronize(stream_), "Mamba prompt restore sync");
}

bool VoiceChatThinkerMambaState::ok() const {
    for (std::size_t si = 0; si < specs_.size(); ++si) {
        if (state_[si].size() != static_cast<std::size_t>(num_layers_))
            return false;
        for (const auto& t : state_[si]) {
            if (!t.ok())
                return false;
        }
        if (present_[si].size() != static_cast<std::size_t>(num_layers_))
            return false;
        for (const auto& t : present_[si]) {
            if (!t.ok())
                return false;
        }
    }
    return true;
}

} // namespace trtmc
