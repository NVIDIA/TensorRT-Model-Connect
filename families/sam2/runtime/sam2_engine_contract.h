/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace trtmc::sam2 {

inline constexpr std::int32_t kFrameCount = 5;
inline constexpr std::int32_t kOriginalImageHeight = 1280;
inline constexpr std::int32_t kOriginalImageWidth = 1088;

inline constexpr std::string_view kImagePlanSection = "engine.plan";
inline constexpr std::string_view kPromptPlanSection = "prompt.plan";
inline constexpr std::array<std::string_view, 4> kRecurrentPlanSections = {
    "recurrent.1.plan",
    "recurrent.2.plan",
    "recurrent.3.plan",
    "recurrent.4.plan",
};
inline constexpr std::array<std::string_view, 6> kRequiredPlanSections = {
    kImagePlanSection,         kPromptPlanSection,        kRecurrentPlanSections[0],
    kRecurrentPlanSections[1], kRecurrentPlanSections[2], kRecurrentPlanSections[3],
};
inline constexpr std::string_view kConfigSection = "runtime.json";

enum class TensorDataType : std::uint8_t {
    kFloat32,
    kBFloat16,
};

struct TensorContract {
    std::string_view name;
    TensorDataType data_type;
    std::array<std::int32_t, 4> dimensions;
    std::uint8_t rank;
};

inline constexpr TensorContract kPixelValues{
    "pixel_values", TensorDataType::kFloat32, {1, 3, 1024, 1024}, 4};

inline constexpr std::array<TensorContract, 3> kTrackerFpn = {{
    {"tracker_fpn_0", TensorDataType::kBFloat16, {1, 256, 256, 256}, 4},
    {"tracker_fpn_1", TensorDataType::kBFloat16, {1, 256, 128, 128}, 4},
    {"tracker_fpn_2", TensorDataType::kFloat32, {1, 256, 64, 64}, 4},
}};

inline constexpr std::array<TensorContract, 6> kBboxMaps = {{
    {"bbox_cls_stride_8", TensorDataType::kBFloat16, {1, 2, 128, 128}, 4},
    {"bbox_cls_stride_16", TensorDataType::kBFloat16, {1, 2, 64, 64}, 4},
    {"bbox_cls_stride_32", TensorDataType::kBFloat16, {1, 2, 32, 32}, 4},
    {"bbox_reg_stride_8", TensorDataType::kBFloat16, {1, 4, 128, 128}, 4},
    {"bbox_reg_stride_16", TensorDataType::kBFloat16, {1, 4, 64, 64}, 4},
    {"bbox_reg_stride_32", TensorDataType::kBFloat16, {1, 4, 32, 32}, 4},
}};

inline constexpr TensorContract kBoxPrompt{
    "box_xyxy_1024", TensorDataType::kFloat32, {1, 4, 0, 0}, 2};
inline constexpr TensorContract kMaskLogits256{
    "mask_logits_256", TensorDataType::kFloat32, {1, 1, 256, 256}, 4};
inline constexpr TensorContract kObjectPointer{
    "object_pointer", TensorDataType::kFloat32, {1, 256, 0, 0}, 2};
inline constexpr TensorContract kMemoryFeatures{
    "memory_features", TensorDataType::kBFloat16, {1, 64, 64, 64}, 4};

constexpr TensorContract historyMemoryFeatures(std::int32_t history_frames) {
    return {"history_memory_features", TensorDataType::kBFloat16, {history_frames, 64, 64, 64}, 4};
}

constexpr TensorContract historyObjectPointers(std::int32_t history_frames) {
    return {"history_object_pointers", TensorDataType::kFloat32, {history_frames, 256, 0, 0}, 2};
}

} // namespace trtmc::sam2
