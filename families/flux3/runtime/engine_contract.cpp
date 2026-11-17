/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/flux3/runtime/engine_contract.h"

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>

namespace trtmc::flux3 {
namespace {

using Shape = std::vector<std::int64_t>;
using Names = std::set<std::string>;

constexpr std::int64_t kImageChannels = 3;

[[noreturn]] void fail(const char* engine, const std::string& message) {
    throw std::runtime_error(std::string("flux3 ") + engine + " engine: " + message);
}

std::string describe(const Shape& shape) {
    std::string text;
    for (const auto extent : shape)
        text += (text.empty() ? "" : ", ") + std::to_string(extent);
    return "[" + text + "]";
}

std::string describe(const Names& names) {
    std::string text;
    for (const auto& name : names)
        text += (text.empty() ? "" : ", ") + name;
    return "{" + text + "}";
}

const char* describe(DType dtype) {
    switch (dtype) {
    case DType::kFloat32:
        return "float32";
    case DType::kBFloat16:
        return "bfloat16";
    case DType::kInt32:
        return "int32";
    default:
        return "another dtype";
    }
}

Names tensor_names(const std::vector<TensorInfo>& tensors) {
    Names names;
    for (const auto& tensor : tensors)
        names.insert(tensor.name);
    return names;
}

void require_tensors(const ITrtModule& engine, const char* label, const Names& inputs,
                     const Names& outputs) {
    const auto actual_inputs = tensor_names(engine.input_info());
    const auto actual_outputs = tensor_names(engine.output_info());
    if (actual_inputs != inputs || actual_outputs != outputs)
        fail(label, "must have inputs " + describe(inputs) + " and outputs " + describe(outputs) +
                        ", got " + describe(actual_inputs) + " and " + describe(actual_outputs));
}

void require_dtype(const ITrtModule& engine, const char* label, const std::string& name,
                   DType dtype) {
    const auto actual = engine.tensor_dtype(name);
    if (actual != dtype)
        fail(label, "'" + name + "' must be " + describe(dtype) + ", got " + describe(actual));
}

void require_static(const ITrtModule& engine, const char* label, const std::string& name,
                    DType dtype, const Shape& shape) {
    require_dtype(engine, label, name, dtype);
    const auto actual = engine.tensor_shape(name);
    if (engine.input_is_dynamic(name) || actual != shape)
        fail(label, "'" + name + "' must have the static shape " + describe(shape) + ", got " +
                        describe(actual) + (engine.input_is_dynamic(name) ? " (dynamic)" : ""));
}

// A dynamic input whose optimization profile admits every shape from low to high.
void require_profile(const ITrtModule& engine, const char* label, const std::string& name,
                     DType dtype, const Shape& low, const Shape& high) {
    require_dtype(engine, label, name, dtype);
    const auto profile = engine.profile_idx();
    const auto minimum = engine.input_profile_shape(name, profile, ProfileShapeSelector::kMin);
    const auto maximum = engine.input_profile_shape(name, profile, ProfileShapeSelector::kMax);
    bool covered = engine.input_is_dynamic(name) && minimum.size() == low.size() &&
                   maximum.size() == high.size();
    for (std::size_t axis = 0; covered && axis < low.size(); ++axis)
        covered = minimum[axis] <= low[axis] && maximum[axis] >= high[axis];
    if (!covered)
        fail(label, "'" + name + "' must be dynamic with a profile covering " + describe(low) +
                        " to " + describe(high) + ", got " + describe(minimum) + " to " +
                        describe(maximum));
}

} // namespace

std::vector<std::int64_t> latent_tensor_shape(const RuntimeConfig& config) {
    return {1, static_cast<std::int64_t>(config.latent_height) * config.latent_width,
            config.latent_channels};
}

void validate_text_encoder(const ITrtModule& engine, const RuntimeConfig& config) {
    constexpr const char* label = "text encoder";
    const auto& text = config.text;
    require_tensors(engine, label, {"input_ids", "valid_length"}, {"text_embeds"});
    require_profile(engine, label, "input_ids", DType::kInt32, {1, text.pad_multiple},
                    {1, text.max_tokens});
    require_static(engine, label, "valid_length", DType::kInt32, {1});
    require_dtype(engine, label, "text_embeds", DType::kBFloat16);
    // The output buffer, allocated for the largest profile shape, is bound as
    // the denoiser's text_embeds input, so it must hold max_tokens rows.
    const auto shape = engine.tensor_shape("text_embeds");
    if (shape.size() != 3 || shape[0] < 1 || shape[1] < text.max_tokens ||
        shape[2] != text.embed_dim)
        fail(label, "'text_embeds' must hold [1, " + std::to_string(text.max_tokens) + ", " +
                        std::to_string(text.embed_dim) + "], got " + describe(shape));
    if (engine.device_ptr("text_embeds") == nullptr)
        fail(label, "has no device buffer for 'text_embeds'");
}

void validate_denoiser(const ITrtModule& engine, const RuntimeConfig& config) {
    constexpr const char* label = "denoiser";
    const auto& text = config.text;
    const auto latents = latent_tensor_shape(config);
    const Shape projection = {1, config.timestep_embedding.dim};
    require_tensors(engine, label,
                    {"latents", "text_embeds", "timestep_proj", "target_timestep_proj"},
                    {"velocity"});
    require_static(engine, label, "latents", DType::kFloat32, latents);
    require_profile(engine, label, "text_embeds", DType::kBFloat16,
                    {1, text.pad_multiple, text.embed_dim}, {1, text.max_tokens, text.embed_dim});
    require_static(engine, label, "timestep_proj", DType::kFloat32, projection);
    require_static(engine, label, "target_timestep_proj", DType::kFloat32, projection);
    require_static(engine, label, "velocity", DType::kFloat32, latents);
}

void validate_vae(const ITrtModule& engine, const RuntimeConfig& config) {
    constexpr const char* label = "VAE";
    require_tensors(engine, label, {"latents"}, {"image"});
    require_static(engine, label, "latents", DType::kFloat32, latent_tensor_shape(config));
    require_static(engine, label, "image", DType::kFloat32,
                   {1, kImageChannels, config.height, config.width});
}

} // namespace trtmc::flux3
