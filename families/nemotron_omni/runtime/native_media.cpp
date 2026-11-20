/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_media.h"

#include "audio_frontend.h"
#include "trtmc/runtime/trt_backend.h"
#include "visual_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trtmc::nemotron_omni {
namespace {
struct FloatInput {
    std::string name;
    const std::vector<float>& values;
    std::vector<int64_t> shape;
};
size_t elements(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() / count)
            throw std::invalid_argument("Invalid Omni media tensor dimensions");
        count *= static_cast<size_t>(dimension);
    }
    return count;
}
std::vector<float> execute(ITrtModule& module, const std::vector<FloatInput>& floats,
                           TensorMap inputs, const std::string& output,
                           const std::vector<int64_t>& expected) {
    std::map<std::string, std::vector<__half>> halves;
    for (const auto& input : floats) {
        if (input.values.size() != elements(input.shape))
            throw std::invalid_argument("Omni media input dimensions disagree: " + input.name);
        const auto dtype = module.tensor_dtype(input.name);
        void* data = const_cast<float*>(input.values.data());
        if (dtype == DType::kFloat16) {
            auto& half = halves[input.name];
            half.reserve(input.values.size());
            for (const auto value : input.values) {
                auto converted = __float2half_rn(value);
                if (!std::isfinite(__half2float(converted)))
                    throw std::invalid_argument("Nonrepresentable Omni media input");
                half.push_back(converted);
            }
            data = half.data();
        } else if (dtype != DType::kFloat32) {
            throw std::invalid_argument("Unsupported native Omni media input dtype");
        }
        inputs.emplace(input.name, TensorMap::mapped_type{data, input.shape, dtype});
    }
    // tensor_shape() describes the module's cached buffer shape. Dynamic
    // outputs must use the concrete shape returned by the synchronous API;
    // this path already needs host features for the owning media composition.
    const auto outputs = module.forward(inputs);
    const auto found = outputs.find(output);
    if (found == outputs.end() || found->second.shape != expected || !found->second.data)
        throw std::runtime_error("Omni media output shape differs from the request: " + output);
    const auto& tensor = found->second;
    std::vector<float> result(elements(expected));
    if (tensor.dtype == DType::kFloat32) {
        std::memcpy(result.data(), tensor.data, result.size() * sizeof(float));
    } else if (tensor.dtype == DType::kFloat16) {
        const auto* half = static_cast<const __half*>(tensor.data);
        std::transform(half, half + result.size(), result.begin(),
                       [](__half value) { return __half2float(value); });
    } else {
        throw std::runtime_error("Unsupported native Omni media output dtype");
    }
    if (std::any_of(result.begin(), result.end(),
                    [](float value) { return !std::isfinite(value); }))
        throw std::runtime_error("Native Omni media encoder produced nonfinite features");
    return result;
}
std::vector<float> asset(const FamilyContext& context, const char* name, size_t count) {
    const auto bytes = context.reader.read_section(name);
    if (bytes.size() != count * sizeof(float))
        throw std::invalid_argument("Invalid Omni media asset size");
    std::vector<float> result(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    if (std::any_of(result.begin(), result.end(),
                    [](float value) { return !std::isfinite(value); }))
        throw std::invalid_argument("Nonfinite Omni media asset");
    return result;
}
} // namespace

NativeMedia::NativeMedia(const FamilyContext& context) {
    const auto metadata = nlohmann::json::parse(context.reader.read_section("runtime.json"));
    if (metadata.at("format") != 1)
        throw std::invalid_argument("Invalid native Omni media format");
    const auto& config = metadata.at("config");
    hidden_ = config.at("llm_config").at("hidden_size");
    patch_ = config.at("patch_size");
    visual_hidden_ = config.at("vit_hidden_size");
    const auto& audio = config.at("sound_config");
    audio_hidden_ = audio.at("hidden_size");
    audio_heads_ = audio.at("num_attention_heads");
    position_side_ = metadata.at("visual_position_side");
    max_patches_ = metadata.at("max_visual_patches");
    max_audio_frames_ = metadata.at("max_audio_frames");
    if (hidden_ <= 0 || patch_ <= 0 || visual_hidden_ <= 0 || audio_hidden_ <= 0 ||
        audio_heads_ <= 0 || max_audio_frames_ < 2 || max_patches_ < 4 || max_patches_ % 4)
        throw std::invalid_argument("Invalid Omni media configuration dimensions");
    positions_ = asset(context, "visual_positions.f32",
                       elements({position_side_, position_side_, visual_hidden_}));
    mel_filters_ = asset(context, "mel_filters.f32", 128 * 257);
    auto load = [&](const char* name) {
        const auto plan = context.reader.read_section(std::string(name) + ".plan");
        auto module = context.backend.create_module(plan.data(), plan.size(), {});
        if (!module || !module->ok())
            throw std::runtime_error(std::string("Cannot load Omni ") + name);
        module->set_timing_label(std::string("omni_") + name);
        return module;
    };
    image_patch_ = load("image_patch");
    video_patch_ = load("video_patch");
    visual_ = load("visual");
    audio_ = load("audio");
}

std::vector<float> NativeMedia::visual(const std::vector<std::vector<float>>& frames, int height,
                                       int width) {
    if (height <= 0 || width <= 0 || height % patch_ || width % patch_ || (height / patch_) % 2 ||
        (width / patch_) % 2 || int64_t(height / patch_) * (width / patch_) > max_patches_ ||
        (frames.size() != 1 && frames.size() != 2))
        throw std::invalid_argument(
            "Omni visual request exceeds built capacity or has incomplete patches");
    const int rows = height / patch_, columns = width / patch_, count = rows * columns;
    auto pixels = patch_pixels(frames, height, width, patch_);
    auto& patch = frames.size() == 1 ? image_patch_ : video_patch_;
    auto embeddings = execute(
        *patch, {{"patch_pixels", pixels, {count, int64_t(frames.size()) * 3 * patch_ * patch_}}},
        {}, "patch_embeddings", {count, visual_hidden_});
    auto positions = visual_positions(positions_, position_side_, visual_hidden_, rows, columns);
    auto shuffle = shuffle_indices(rows, columns);
    TensorMap integer{{"shuffle_indices", {shuffle.data(), {count / 4, 4}, DType::kInt32}}};
    return execute(*visual_,
                   {{"patch_embeddings", embeddings, {count, visual_hidden_}},
                    {"position_embeddings", positions, {count, visual_hidden_}}},
                   std::move(integer), "image_features", {count / 4, hidden_});
}

std::vector<float> NativeMedia::audio(const std::vector<float>& pcm16k) {
    const auto mel = audio_features(pcm16k, mel_filters_, max_audio_frames_);
    int tokens = static_cast<int>(mel.frames);
    for (int stage = 0; stage < 3; ++stage)
        tokens = (tokens + 1) / 2;
    auto positions = audio_positions(tokens, audio_hidden_, audio_heads_);
    TensorMap integer{{"relative_indices",
                       {positions.indices.data(), {audio_heads_, tokens, tokens}, DType::kInt32}}};
    return execute(*audio_,
                   {{"mel_features", mel.values, {static_cast<int64_t>(mel.frames), 128}},
                    {"relative_positions", positions.values, {2 * tokens - 1, audio_hidden_}}},
                   std::move(integer), "sound_features", {tokens, hidden_});
}
} // namespace trtmc::nemotron_omni
