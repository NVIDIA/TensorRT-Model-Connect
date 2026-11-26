/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/engine_io.h"

#include <cstddef>
#include <stdexcept>

namespace trtmc::vibevoice {
namespace {

std::string dims(const std::vector<std::int64_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
        text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + "]";
}

[[noreturn]] void mismatch(const char* engine, const std::string& name, const std::string& what) {
    throw std::runtime_error(std::string("vibevoice ") + engine + " engine tensor " + name + " " +
                             what);
}

DType model_type(const RuntimeConfig& config) {
    return config.bfloat16 ? DType::kBFloat16 : DType::kFloat32;
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

// A static input: its shape is exactly `expected`.
void static_input(const EngineTensor& t, const char* engine, const std::string& name,
                  const std::vector<std::int64_t>& expected) {
    if (t.dynamic || t.shape != expected)
        mismatch(engine, name, "has shape " + dims(t.shape) + ", expected " + dims(expected));
}

// An output, as the module resolves it at the profile's largest inputs.
void output_shape(const EngineTensor& t, const char* engine, const std::string& name,
                  const std::vector<std::int64_t>& expected) {
    if (t.shape != expected)
        mismatch(engine, name,
                 "has shape " + dims(t.shape) + " at the largest inputs, expected " +
                     dims(expected));
}

// A dynamic input: the profile fixes every dimension but `varying` to `fixed` and
// lets `varying` take every extent in [lowest, highest].
void dynamic_input(const EngineTensor& t, const char* engine, const std::string& name,
                   std::vector<std::int64_t> fixed, std::size_t varying, std::int64_t lowest,
                   std::int64_t highest) {
    const std::size_t rank = fixed.size();
    bool ok = t.dynamic && t.profile_min.size() == rank && t.profile_max.size() == rank &&
              t.profile_min[varying] <= lowest && t.profile_max[varying] >= highest;
    for (std::size_t d = 0; ok && d < rank; ++d) {
        if (d != varying)
            ok = t.profile_min[d] == fixed[d] && t.profile_max[d] == fixed[d];
    }
    if (!ok) {
        fixed[varying] = highest;
        mismatch(engine, name,
                 "profile spans " + dims(t.profile_min) + " to " + dims(t.profile_max) +
                     "; the runtime feeds " + dims(fixed) + " with dimension " +
                     std::to_string(varying) + " from " + std::to_string(lowest));
    }
}

std::string layer_name(const char* base, std::int32_t layer) {
    return std::string(base) + "_" + std::to_string(layer);
}

} // namespace

std::vector<std::string> lm_tensor_names(const RuntimeConfig& config) {
    std::vector<std::string> names = {"inputs_embeds", "rope_cos",       "rope_sin",
                                      "hidden",        "log_normalizer", "control_logits"};
    for (std::int32_t layer = 0; layer <= config.num_layers; ++layer) {
        for (const char* base : {"past_key", "past_value", "present_key", "present_value"})
            names.push_back(layer_name(base, layer));
    }
    return names;
}

std::vector<std::string> embed_tensor_names() {
    return {"token_ids", "embeds"};
}

std::vector<std::string> head_tensor_names() {
    return {"noisy_latent", "timestep", "condition", "velocity"};
}

std::vector<std::string> feedback_tensor_names() {
    return {"latent", "state_in", "audio", "next_embed", "state_out"};
}

std::vector<std::string> voice_tensor_names() {
    return {"audio", "noise", "noise_scale", "embeds"};
}

void check_lm_io(const EngineTensors& tensors, const RuntimeConfig& config, std::int64_t max_rows) {
    const char* engine = "language model";
    const auto type = model_type(config);
    const std::int64_t hidden = config.hidden_size;
    const std::int64_t head_dim = config.head_dim;
    const std::int64_t kv_heads = config.num_kv_heads;
    const EngineTensor embeds = tensor(tensors, engine, "inputs_embeds", true, type);
    dynamic_input(embeds, engine, "inputs_embeds", {1, 0, hidden}, 1, 1, max_rows);
    // The largest row count the profile admits; the outputs are resolved at it.
    const std::int64_t rows = embeds.profile_max[1];
    for (const char* name : {"rope_cos", "rope_sin"})
        dynamic_input(tensor(tensors, engine, name, true, DType::kFloat32), engine, name,
                      {0, head_dim}, 0, 1, max_rows);
    for (std::int32_t layer = 0; layer < config.num_layers; ++layer) {
        for (const char* base : {"past_key", "past_value"}) {
            const auto name = layer_name(base, layer);
            dynamic_input(tensor(tensors, engine, name, true, type), engine, name,
                          {1, 0, kv_heads, head_dim}, 1, 0, std::int64_t{config.capacity} - 1);
        }
        for (const char* base : {"present_key", "present_value"}) {
            const auto name = layer_name(base, layer);
            output_shape(tensor(tensors, engine, name, false, type), engine, name,
                         {1, rows, kv_heads, head_dim});
        }
    }
    const auto extra = layer_name("past_key", config.num_layers);
    if (tensors.count(extra) != 0)
        mismatch(engine, extra, "exists: the engine has more layers than runtime.json");
    output_shape(tensor(tensors, engine, "hidden", false, type), engine, "hidden", {1, hidden});
    output_shape(tensor(tensors, engine, "log_normalizer", false, DType::kFloat32), engine,
                 "log_normalizer", {1});
    output_shape(tensor(tensors, engine, "control_logits", false, DType::kFloat32), engine,
                 "control_logits", {1, static_cast<std::int64_t>(config.control_token_ids.size())});
}

void check_embed_io(const EngineTensors& tensors, const RuntimeConfig& config) {
    const char* engine = "embedding";
    const EngineTensor ids = tensor(tensors, engine, "token_ids", true, DType::kInt32);
    dynamic_input(ids, engine, "token_ids", {0}, 0, 1, config.prefill_chunk);
    output_shape(tensor(tensors, engine, "embeds", false, model_type(config)), engine, "embeds",
                 {ids.profile_max[0], config.hidden_size});
}

void check_head_io(const EngineTensors& tensors, const RuntimeConfig& config) {
    const char* engine = "diffusion head";
    // The runtime always runs the conditional and the unconditional row together.
    const EngineTensor noisy = tensor(tensors, engine, "noisy_latent", true, DType::kFloat32);
    dynamic_input(noisy, engine, "noisy_latent", {0, config.latent_size}, 0, 2, 2);
    dynamic_input(tensor(tensors, engine, "timestep", true, DType::kFloat32), engine, "timestep",
                  {0}, 0, 2, 2);
    dynamic_input(tensor(tensors, engine, "condition", true, model_type(config)), engine,
                  "condition", {0, config.hidden_size}, 0, 2, 2);
    output_shape(tensor(tensors, engine, "velocity", false, DType::kFloat32), engine, "velocity",
                 {noisy.profile_max[0], config.latent_size});
}

void check_feedback_io(const EngineTensors& tensors, const RuntimeConfig& config) {
    const char* engine = "feedback";
    const auto type = model_type(config);
    static_input(tensor(tensors, engine, "latent", true, DType::kFloat32), engine, "latent",
                 {1, config.latent_size});
    static_input(tensor(tensors, engine, "state_in", true, type), engine, "state_in",
                 {config.stream_state_size});
    output_shape(tensor(tensors, engine, "audio", false, DType::kFloat32), engine, "audio",
                 {1, config.hop_length});
    output_shape(tensor(tensors, engine, "next_embed", false, type), engine, "next_embed",
                 {1, config.hidden_size});
    output_shape(tensor(tensors, engine, "state_out", false, type), engine, "state_out",
                 {config.stream_state_size});
}

void check_voice_io(const EngineTensors& tensors, const RuntimeConfig& config) {
    const char* engine = "voice encoder";
    dynamic_input(tensor(tensors, engine, "audio", true, DType::kFloat32), engine, "audio",
                  {1, 1, 1, 0}, 3, config.hop_length, config.max_voice_samples);
    const EngineTensor noise = tensor(tensors, engine, "noise", true, DType::kFloat32);
    dynamic_input(noise, engine, "noise", {1, 0, config.latent_size}, 1, 1,
                  config.max_voice_samples / config.hop_length);
    static_input(tensor(tensors, engine, "noise_scale", true, DType::kFloat32), engine,
                 "noise_scale", {1});
    output_shape(tensor(tensors, engine, "embeds", false, model_type(config)), engine, "embeds",
                 {1, noise.profile_max[1], config.hidden_size});
}

} // namespace trtmc::vibevoice
