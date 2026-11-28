/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/vibevoice/runtime/runtime_config.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace trtmc::vibevoice {

// One engine tensor as a loaded module reports it.
struct EngineTensor {
    bool input{false};
    // An input with a dynamic dimension; its extents are the profile bounds below.
    bool dynamic{false};
    DType dtype{DType::kFloat32};
    // A static input's shape, or an output's shape as the module reports it: the module
    // resolves an output at the largest input shapes of its optimization profile.
    std::vector<std::int64_t> shape;
    // A dynamic input's minimum and maximum in the module's optimization profile.
    std::vector<std::int64_t> profile_min;
    std::vector<std::int64_t> profile_max;
};
using EngineTensors = std::map<std::string, EngineTensor>;

// The tensor names each check reads; the language model's list also names the
// first layer past runtime.json's, which must not exist.
std::vector<std::string> lm_tensor_names(const RuntimeConfig& config);
std::vector<std::string> embed_tensor_names();
std::vector<std::string> head_tensor_names();
std::vector<std::string> feedback_tensor_names();
std::vector<std::string> voice_tensor_names();

// The checks run when the bundle loads, so an engine that does not match
// runtime.json fails there rather than when the runtime binds buffers sized from
// runtime.json to it. Outputs are checked at the profile's largest inputs, where each
// must have exactly the extent the runtime copies or reserves (present_key_<l> holds
// only the new rows, never the past). They throw std::runtime_error naming the engine
// and tensor.

// One language-model profile: inputs_embeds [1, rows, hidden] and rope_cos / rope_sin
// [rows, head_dim] for rows 1..max_rows, past_key_<l> / past_value_<l> [1, past,
// kv_heads, head_dim] for past 0..capacity - 1; outputs present_key_<l> /
// present_value_<l>, hidden [1, hidden], log_normalizer [1] and control_logits [1, 4].
void check_lm_io(const EngineTensors& tensors, const RuntimeConfig& config, std::int64_t max_rows);
// token_ids [n] for n 1..prefill_chunk -> embeds [n, hidden].
void check_embed_io(const EngineTensors& tensors, const RuntimeConfig& config);
// noisy_latent [2, latent], timestep [2], condition [2, hidden] -> velocity [2, latent].
void check_head_io(const EngineTensors& tensors, const RuntimeConfig& config);
// latent [1, latent], state_in [state] -> audio [1, hop], next_embed [1, hidden], state_out.
void check_feedback_io(const EngineTensors& tensors, const RuntimeConfig& config);
// audio [1, 1, 1, samples] for hop..max_voice_samples, noise [1, frames, latent],
// noise_scale [1] -> embeds [1, frames, hidden].
void check_voice_io(const EngineTensors& tensors, const RuntimeConfig& config);

} // namespace trtmc::vibevoice
