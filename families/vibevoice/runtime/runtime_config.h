/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/vibevoice/runtime/script.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::vibevoice {

struct RuntimeConfig {
    bool bfloat16{true};
    std::int32_t hidden_size{0};
    std::int32_t num_layers{0};
    std::int32_t num_kv_heads{0};
    std::int32_t head_dim{0};
    std::int32_t vocab_size{0};
    std::int32_t capacity{0};
    std::int32_t prefill_chunk{0};
    std::vector<float> rope_inv_freq;
    std::vector<std::int32_t> control_token_ids; // ascending: ties resolve to the lowest id
    std::int32_t eos_token_id{0};
    std::int32_t audio_bos_token_id{0};
    std::int32_t audio_eos_token_id{0};
    std::int32_t audio_token_id{0};
    PromptFrame frame;
    std::int32_t sample_rate{0};
    std::int64_t hop_length{0};
    std::int32_t latent_size{0};
    float target_dbfs{0.0F};
    float normalize_eps{0.0F};
    std::int64_t max_voice_samples{0};
    std::int64_t stream_state_size{0};
    std::vector<float> train_sigmas;
    double default_guidance_scale{1.0};
    std::int64_t default_diffusion_steps{10};
};

// Parses and checks the bundle's runtime.json before anything is allocated from it:
// every integer is a JSON integer inside its type's range (no truncation of 384.5,
// no wraparound), every float is finite and a float32, every token id indexes the
// vocabulary, a duplicated key is refused, and the sizes the runtime allocates from
// it fit. Throws std::runtime_error naming the field.
RuntimeConfig parse_runtime_config(const std::string& json_text);

// prepare_voice divides float(10^(target / 20)) by (rms + float(eps)) in float with
// rms from 0 (silence) to FLT_MAX (the loudest finite input); the gain must stay a
// finite, nonzero float over that whole range.
bool usable_gain(double target_dbfs, double eps);

} // namespace trtmc::vibevoice
