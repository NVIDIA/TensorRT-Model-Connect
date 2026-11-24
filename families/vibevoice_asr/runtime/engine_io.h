/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/vibevoice_asr/runtime/runtime_config.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace trtmc::vibevoice_asr {

// One engine tensor as a loaded module reports it.
struct EngineTensor {
    bool input{false};
    // An input with a dynamic dimension; its extents are the profile bounds below.
    bool dynamic{false};
    DType dtype{DType::kFloat32};
    // A static input's shape, or an output's shape as the module reports it (its
    // dynamic dimensions hold whatever extent the module last resolved).
    std::vector<std::int64_t> shape;
    // A dynamic input's minimum and maximum in the module's optimization profile.
    std::vector<std::int64_t> profile_min;
    std::vector<std::int64_t> profile_max;
};
using EngineTensors = std::map<std::string, EngineTensor>;

// The checks run when the bundle loads, so an engine that does not match
// runtime.json fails there rather than at the first request that reaches the
// mismatch. They throw std::runtime_error naming the engine and the tensor.

// The audio encoder takes FP32 audio [1, 1, 1, samples] for every chunk length
// from 1 sample to (encoder_chunk_frames + encoder_context_frames) * hop_length,
// and returns BF16 audio_features [frames, hidden_size].
void check_encoder_io(const EngineTensors& tensors, const RuntimeConfig& config);

// One decoder profile takes input_ids and audio_mask (INT32 [rows]) and
// audio_embeds (BF16 [rows, hidden_size]) for every row count from 1 to
// max_rows, cache_write_index (INT32 [1]), and returns FP32 logits [1, vocab_size].
void check_decoder_io(const EngineTensors& tensors, const RuntimeConfig& config,
                      std::int64_t max_rows);

} // namespace trtmc::vibevoice_asr
