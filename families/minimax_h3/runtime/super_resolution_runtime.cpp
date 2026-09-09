/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/super_resolution_runtime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kChannels = 3;

std::size_t checked_frame_values(int32_t height, int32_t width) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("MiniMax-H3 super-resolution geometry must be positive");
    const auto area = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (area > std::numeric_limits<std::size_t>::max() / kChannels)
        throw std::overflow_error("MiniMax-H3 super-resolution geometry overflow");
    return area * kChannels;
}

std::size_t checked_video_values(int32_t num_frames, std::size_t frame_values) {
    if (num_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 super-resolution frame count must be positive");
    if (frame_values >
        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(num_frames)) {
        throw std::overflow_error("MiniMax-H3 super-resolution video size overflow");
    }
    return static_cast<std::size_t>(num_frames) * frame_values;
}

std::vector<int64_t> frame_shape(int64_t batch, int32_t height, int32_t width) {
    return {batch, height, width, kChannels};
}

void require_shape(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected,
                   const char* label) {
    if (actual != expected)
        throw std::runtime_error(std::string("MiniMax-H3 super-resolution ") + label +
                                 " shape is incompatible with the bundle ABI");
}

} // namespace

void validate_super_resolution_config(const SuperResolutionConfig& config) {
    if (!config.enabled)
        return;
    if (config.section != "video_super_resolution_plan" || config.input_name != "frames" ||
        config.output_name != "upscaled_frames" || config.source_height != 480 ||
        config.source_width != 864 || config.target_height != 720 || config.target_width != 1296 ||
        config.batch_min != 1 || config.batch_opt != 4 || config.batch_max != 8) {
        throw std::invalid_argument(
            "MiniMax-H3 super-resolution metadata does not match the native 480p-to-720p ABI");
    }
}

void validate_super_resolution_plan(ITrtModule& module, const SuperResolutionConfig& config) {
    validate_super_resolution_config(config);
    if (!config.enabled)
        throw std::invalid_argument("MiniMax-H3 super-resolution plan is not enabled");
    if (!module.ok() || module.optimization_profile_count() != 1 || module.profile_idx() != 0 ||
        module.input_info().size() != 1U || module.output_info().size() != 1U ||
        !module.has_input(config.input_name) || !module.has_output(config.output_name) ||
        module.tensor_dtype(config.input_name) != DType::kFloat32 ||
        module.tensor_dtype(config.output_name) != DType::kFloat32 ||
        module.input_rank(config.input_name) != 4 || !module.input_is_dynamic(config.input_name)) {
        throw std::runtime_error(
            "MiniMax-H3 super-resolution plan has an invalid FP32 NHWC I/O ABI");
    }

    // ITrtModule exposes the live optimum input shape and the maximum-sized
    // output allocation. The engine's dynamic declaration is verified by
    // input_is_dynamic plus the exact min/opt/max profile below.
    require_shape(module.tensor_shape(config.input_name),
                  frame_shape(config.batch_opt, config.source_height, config.source_width),
                  "input");
    require_shape(module.tensor_shape(config.output_name),
                  frame_shape(config.batch_max, config.target_height, config.target_width),
                  "output allocation");
    require_shape(module.input_profile_shape(config.input_name, 0, ProfileShapeSelector::kMin),
                  frame_shape(config.batch_min, config.source_height, config.source_width),
                  "minimum profile");
    require_shape(module.input_profile_shape(config.input_name, 0, ProfileShapeSelector::kOpt),
                  frame_shape(config.batch_opt, config.source_height, config.source_width),
                  "optimum profile");
    require_shape(module.input_profile_shape(config.input_name, 0, ProfileShapeSelector::kMax),
                  frame_shape(config.batch_max, config.source_height, config.source_width),
                  "maximum profile");
}

std::vector<float> run_super_resolution(ITrtModule& module, const std::vector<float>& pixels,
                                        int32_t num_frames, const SuperResolutionConfig& config) {
    validate_super_resolution_plan(module, config);
    const std::size_t source_frame_values =
        checked_frame_values(config.source_height, config.source_width);
    const std::size_t target_frame_values =
        checked_frame_values(config.target_height, config.target_width);
    if (pixels.size() != checked_video_values(num_frames, source_frame_values)) {
        throw std::invalid_argument(
            "MiniMax-H3 super-resolution source pixels do not match the declared geometry");
    }

    std::vector<float> upscaled(checked_video_values(num_frames, target_frame_values));
    module.reset_execution_context();
    for (int32_t frame = 0; frame < num_frames;) {
        const int32_t batch = std::min(config.batch_max, num_frames - frame);
        TensorMap inputs;
        inputs.emplace(config.input_name,
                       Tensor{const_cast<float*>(pixels.data()) +
                                  static_cast<std::size_t>(frame) * source_frame_values,
                              frame_shape(batch, config.source_height, config.source_width),
                              DType::kFloat32});
        const TensorMap outputs = module.forward(inputs);
        const auto it = outputs.find(config.output_name);
        if (it == outputs.end() || it->second.data == nullptr ||
            it->second.dtype != DType::kFloat32 ||
            it->second.shape != frame_shape(batch, config.target_height, config.target_width) ||
            it->second.numel() != static_cast<std::size_t>(batch) * target_frame_values) {
            throw std::runtime_error(
                "MiniMax-H3 super-resolution plan returned an invalid output batch");
        }
        std::memcpy(upscaled.data() + static_cast<std::size_t>(frame) * target_frame_values,
                    it->second.data, it->second.nbytes());
        frame += batch;
    }
    module.sync();
    return upscaled;
}

} // namespace trtmc::minimax_h3
