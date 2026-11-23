/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bevfusion/runtime/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::bevfusion {
namespace {

const std::vector<int64_t> kCameraShape{1, 80, 128, 128};
const std::vector<int64_t> kLidarShape{1, 256, 128, 128};
const std::vector<int64_t> kOutputShape{1, 6, 200, 200};

void validate(const Tensor& tensor, const std::vector<int64_t>& shape, const char* name) {
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.shape != shape)
        throw std::invalid_argument(std::string("BEVFusion requires contiguous FP32 ") + name +
                                    " with the engine's fixed NCHW shape");
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> module) : module_(std::move(module)) {
    if (!module_ || !module_->ok())
        throw std::invalid_argument("BEVFusion requires a valid engine module");
    if (module_->input_info().size() != 2 || module_->output_info().size() != 1)
        throw std::invalid_argument("BEVFusion engine requires two inputs and one output");
    for (const auto& entry :
         TensorMap{{"camera_bev", {nullptr, kCameraShape, DType::kFloat32}},
                   {"lidar_bev", {nullptr, kLidarShape, DType::kFloat32}},
                   {"map_probabilities", {nullptr, kOutputShape, DType::kFloat32}}}) {
        const auto& name = entry.first;
        const bool present =
            name == "map_probabilities" ? module_->has_output(name) : module_->has_input(name);
        if (!present || module_->tensor_shape(name) != entry.second.shape ||
            module_->tensor_dtype(name) != DType::kFloat32)
            throw std::invalid_argument("BEVFusion engine tensor contract mismatch: " + name);
    }
}

std::vector<float> Pipeline::forward(const Tensor& camera_bev, const Tensor& lidar_bev) {
    validate(camera_bev, kCameraShape, "camera_bev");
    validate(lidar_bev, kLidarShape, "lidar_bev");
    const auto outputs = module_->forward({{"camera_bev", camera_bev}, {"lidar_bev", lidar_bev}});
    const auto output = outputs.find("map_probabilities");
    if (output == outputs.end() || output->second.data == nullptr ||
        output->second.dtype != DType::kFloat32 || output->second.shape != kOutputShape)
        throw std::runtime_error("BEVFusion engine returned invalid map_probabilities");
    const auto* values = static_cast<const float*>(output->second.data);
    return {values, values + output->second.numel()};
}

internal::SpatialClassScoresResult
Pipeline::run(const internal::SpatialFeaturesToClassScoresRequest& request, internal::ConfigView) {
    if (request.inputs.size() != 2)
        throw std::invalid_argument("BEVFusion requires camera_bev and lidar_bev");
    Tensor camera{}, lidar{};
    for (const auto& input : request.inputs) {
        Tensor* target = input.name == "camera_bev"  ? &camera
                         : input.name == "lidar_bev" ? &lidar
                                                     : nullptr;
        const auto& shape = input.name == "camera_bev" ? kCameraShape : kLidarShape;
        if (target == nullptr || target->data != nullptr ||
            input.channels != static_cast<uint64_t>(shape[1]) || input.height != 128 ||
            input.width != 128 || input.values.size() != static_cast<size_t>(shape[1] * 128 * 128))
            throw std::invalid_argument(
                "BEVFusion requires fixed CHW camera_bev and lidar_bev operands");
        *target = {const_cast<float*>(input.values.data()), shape, DType::kFloat32};
    }
    return {forward(camera, lidar),
            6,
            200,
            200,
            {"drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"},
            internal::ScoreKind::Probability};
}

} // namespace trtmc::bevfusion
