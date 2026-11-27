/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_vision.h"

#include "trtmc/runtime/trt_backend.h"
#include "visual_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace trtmc::nemotron_bolt {
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
            throw std::invalid_argument("Invalid Bolt media tensor dimensions");
        count *= static_cast<size_t>(dimension);
    }
    return count;
}
std::vector<float> execute(ITrtModule& module, const std::vector<FloatInput>& floats,
                           TensorMap inputs, const std::string& output,
                           const std::vector<int64_t>& expected) {
    std::map<std::string, std::vector<__nv_bfloat16>> halves;
    for (const auto& input : floats) {
        if (input.values.size() != elements(input.shape))
            throw std::invalid_argument("Bolt media input dimensions disagree: " + input.name);
        const auto dtype = module.tensor_dtype(input.name);
        void* data = const_cast<float*>(input.values.data());
        if (dtype == DType::kBFloat16) {
            auto& half = halves[input.name];
            half.reserve(input.values.size());
            for (const auto value : input.values) {
                auto converted = __float2bfloat16_rn(value);
                if (!std::isfinite(__bfloat162float(converted)))
                    throw std::invalid_argument("Nonrepresentable Bolt media input");
                half.push_back(converted);
            }
            data = half.data();
        } else if (dtype != DType::kFloat32) {
            throw std::invalid_argument("Unsupported native Bolt media input dtype");
        }
        inputs.emplace(input.name, TensorMap::mapped_type{data, input.shape, dtype});
    }
    // tensor_shape() describes the module's cached buffer shape. Dynamic
    // outputs must use the concrete shape returned by the synchronous API;
    // this path already needs host features for the owning media composition.
    const auto outputs = module.forward(inputs);
    const auto found = outputs.find(output);
    if (found == outputs.end() || found->second.shape != expected || !found->second.data)
        throw std::runtime_error("Bolt media output shape differs from the request: " + output);
    const auto& tensor = found->second;
    std::vector<float> result(elements(expected));
    if (tensor.dtype == DType::kFloat32) {
        std::memcpy(result.data(), tensor.data, result.size() * sizeof(float));
    } else if (tensor.dtype == DType::kBFloat16) {
        const auto* half = static_cast<const __nv_bfloat16*>(tensor.data);
        std::transform(half, half + result.size(), result.begin(),
                       [](__nv_bfloat16 value) { return __bfloat162float(value); });
    } else {
        throw std::runtime_error("Unsupported native Bolt media output dtype");
    }
    if (std::any_of(result.begin(), result.end(),
                    [](float value) { return !std::isfinite(value); }))
        throw std::runtime_error("Native Bolt media encoder produced nonfinite features");
    return result;
}
} // namespace

NativeVision::NativeVision(IBackend& backend, const NativeLanguage::PlanReader& read,
                           std::vector<float> positions, int position_side, int max_patches)
    : positions_(std::move(positions)), side_(position_side), maximum_(max_patches) {
    if (side_ != 128 || maximum_ < 4 || maximum_ > 13312 || maximum_ % 4 ||
        positions_.size() != elements({side_, side_, 1280}))
        throw std::invalid_argument("Invalid Bolt RADIO position table or patch capacity");
    auto load = [&](const char* name) {
        const auto plan = read(name);
        auto module = backend.create_module(plan.data(), plan.size(), {});
        if (!module || !module->ok())
            throw std::runtime_error(std::string("Cannot load Bolt ") + name);
        return module;
    };
    image_ = load("image_patch");
    tubelet_ = load("video_patch");
    visual_ = load("visual");
}

std::vector<float> NativeVision::encode(const std::vector<std::vector<float>>& frames, int height,
                                        int width) {
    constexpr int patch = 16, visual_width = 1280, language_width = 4096;
    if (height <= 0 || width <= 0 || height % (2 * patch) || width % (2 * patch) ||
        (frames.size() != 1 && frames.size() != 2))
        throw std::invalid_argument(
            "Bolt requires one image or a two-frame tubelet with complete merged patches");
    const int rows = height / patch, columns = width / patch;
    if (int64_t(rows) * columns > maximum_)
        throw std::invalid_argument("Bolt patch count exceeds built capacity");
    const int count = rows * columns;
    const auto pixels = patch_pixels(frames, height, width, patch);
    auto& front = frames.size() == 1 ? *image_ : *tubelet_;
    const auto patches = execute(
        front, {{"patch_pixels", pixels, {count, int64_t(frames.size()) * 3 * patch * patch}}}, {},
        "patch_embeddings", {count, visual_width});
    const auto pos = visual_positions(positions_, side_, visual_width, rows, columns);
    auto shuffle = shuffle_indices(rows, columns);
    TensorMap indices{{"shuffle_indices", {shuffle.data(), {count / 4, 4}, DType::kInt32}}};
    return execute(*visual_,
                   {{"patch_embeddings", patches, {count, visual_width}},
                    {"position_embeddings", pos, {count, visual_width}}},
                   std::move(indices), "image_features", {count / 4, language_width});
}
} // namespace trtmc::nemotron_bolt
