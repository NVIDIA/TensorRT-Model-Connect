/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/engine_io.h"

#include <cstddef>
#include <stdexcept>

namespace trtmc::vibevoice_asr {
namespace {

std::string dims(const std::vector<std::int64_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
        text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + "]";
}

[[noreturn]] void mismatch(const char* engine, const std::string& name, const std::string& what) {
    throw std::runtime_error(std::string("VibeVoice-ASR ") + engine + " engine tensor " + name +
                             " " + what);
}

const EngineTensor& tensor(const EngineTensors& tensors, const char* engine,
                           const std::string& name, bool input, DType dtype) {
    const auto it = tensors.find(name);
    if (it == tensors.end() || it->second.input != input)
        mismatch(engine, name, input ? "is not an input" : "is not an output");
    if (it->second.dtype != dtype)
        mismatch(engine, name, "has another data type than the runtime uses");
    return it->second;
}

// A static tensor: its shape is exactly `expected`.
void static_shape(const EngineTensor& t, const char* engine, const std::string& name,
                  const std::vector<std::int64_t>& expected) {
    if (t.dynamic || t.shape != expected)
        mismatch(engine, name, "has shape " + dims(t.shape) + ", expected " + dims(expected));
}

// An output of shape `expected`, where -1 marks the dynamic first dimension.
void output_shape(const EngineTensor& t, const char* engine, const std::string& name,
                  const std::vector<std::int64_t>& expected) {
    bool ok = t.shape.size() == expected.size();
    for (std::size_t d = 0; ok && d < expected.size(); ++d)
        ok = expected[d] == -1 || t.shape[d] == expected[d];
    if (!ok)
        mismatch(engine, name,
                 "has shape " + dims(t.shape) + ", expected " + dims(expected) + " (-1: any)");
}

// A dynamic input: the profile fixes every dimension but `varying` to `fixed`
// and lets `varying` take every extent in [1, longest].
void dynamic_input(const EngineTensor& t, const char* engine, const std::string& name,
                   std::vector<std::int64_t> fixed, std::size_t varying, std::int64_t longest) {
    const std::size_t rank = fixed.size();
    bool ok = t.dynamic && t.profile_min.size() == rank && t.profile_max.size() == rank &&
              t.profile_min[varying] <= 1 && t.profile_max[varying] >= longest;
    for (std::size_t d = 0; ok && d < rank; ++d) {
        if (d != varying)
            ok = t.profile_min[d] == fixed[d] && t.profile_max[d] == fixed[d];
    }
    if (!ok) {
        fixed[varying] = longest;
        mismatch(engine, name,
                 "profile spans " + dims(t.profile_min) + " to " + dims(t.profile_max) +
                     "; the runtime feeds up to " + dims(fixed) + " with dimension " +
                     std::to_string(varying) + " from 1");
    }
}

} // namespace

void check_encoder_io(const EngineTensors& tensors, const RuntimeConfig& config) {
    const std::int64_t longest =
        (static_cast<std::int64_t>(config.encoder_chunk_frames) + config.encoder_context_frames) *
        config.hop_length;
    dynamic_input(tensor(tensors, "encoder", "audio", true, DType::kFloat32), "encoder", "audio",
                  {1, 1, 1, 0}, 3, longest);
    output_shape(tensor(tensors, "encoder", "audio_features", false, DType::kBFloat16), "encoder",
                 "audio_features", {-1, config.hidden_size});
}

void check_decoder_io(const EngineTensors& tensors, const RuntimeConfig& config,
                      std::int64_t max_rows) {
    for (const char* name : {"input_ids", "audio_mask"})
        dynamic_input(tensor(tensors, "decoder", name, true, DType::kInt32), "decoder", name, {0},
                      0, max_rows);
    dynamic_input(tensor(tensors, "decoder", "audio_embeds", true, DType::kBFloat16), "decoder",
                  "audio_embeds", {0, config.hidden_size}, 0, max_rows);
    static_shape(tensor(tensors, "decoder", "cache_write_index", true, DType::kInt32), "decoder",
                 "cache_write_index", {1});
    static_shape(tensor(tensors, "decoder", "logits", false, DType::kFloat32), "decoder", "logits",
                 {1, config.vocab_size});
}

} // namespace trtmc::vibevoice_asr
