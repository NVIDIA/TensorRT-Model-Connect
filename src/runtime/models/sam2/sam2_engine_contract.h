/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace trtmc::sam2 {

inline constexpr std::string_view kStrategyName = "sam2_bbox_video_tracking";
inline constexpr std::uint32_t kEngineContractVersion = 5;
inline constexpr std::int32_t kFrameCount = 5;
inline constexpr std::int32_t kSelectedObjectCount = 1;
inline constexpr std::int32_t kModelImageSize = 1024;
inline constexpr std::int32_t kOriginalImageHeight = 1280;
inline constexpr std::int32_t kOriginalImageWidth = 1088;

inline constexpr std::string_view kTargetTensorRtVersion = "11.1.0.106";
inline constexpr std::string_view kTargetTensorRtAbi = "11.1";
inline constexpr std::string_view kTargetGpuName = "NVIDIA L4";
inline constexpr std::string_view kTargetComputeCapability = "8.9";
inline constexpr std::int32_t kTargetComputeCapabilityMajor = 8;
inline constexpr std::int32_t kTargetComputeCapabilityMinor = 9;
inline constexpr std::string_view kPlanProfilingVerbosity = "detailed";
inline constexpr std::string_view kBenchmarkExecutionContextNvtxVerbosity = "none";
inline constexpr std::string_view kImageAttentionMetadataPrefix = "trtmc.sam2.iattention.block.";
inline constexpr std::int32_t kImageAttentionMetadataIndexWidth = 2;
inline constexpr std::array<std::string_view, 16> kImageAttentionMetadata = {
    "trtmc.sam2.iattention.block.00", "trtmc.sam2.iattention.block.01",
    "trtmc.sam2.iattention.block.02", "trtmc.sam2.iattention.block.03",
    "trtmc.sam2.iattention.block.04", "trtmc.sam2.iattention.block.05",
    "trtmc.sam2.iattention.block.06", "trtmc.sam2.iattention.block.07",
    "trtmc.sam2.iattention.block.08", "trtmc.sam2.iattention.block.09",
    "trtmc.sam2.iattention.block.10", "trtmc.sam2.iattention.block.11",
    "trtmc.sam2.iattention.block.12", "trtmc.sam2.iattention.block.13",
    "trtmc.sam2.iattention.block.14", "trtmc.sam2.iattention.block.15",
};

inline constexpr std::string_view kCheckpointSha256 =
    "89fd676560809c8504411b574cea305c86db1f65bda790ec7fe16cedc6c6ff73";
inline constexpr std::string_view kConfigSha256 =
    "59488bb78c7cc48aaaebd966ea9d054014f683459d062b7a959a4aa501342656";
inline constexpr std::string_view kGoldenManifestSha256 =
    "c25251ee27da05afd75adc3c6869cbc2944b80c05c5d6e703b6ebbbba697a4f0";

inline constexpr std::string_view kImagePlanSection = "sam2_image_engine_plan";
inline constexpr std::string_view kPromptPlanSection = "sam2_prompt_engine_plan";
inline constexpr std::array<std::string_view, 4> kRecurrentPlanSections = {
    "sam2_recurrent_h1_engine_plan",
    "sam2_recurrent_h2_engine_plan",
    "sam2_recurrent_h3_engine_plan",
    "sam2_recurrent_h4_engine_plan",
};
inline constexpr std::array<std::string_view, 6> kRequiredPlanSections = {
    kImagePlanSection,         kPromptPlanSection,        kRecurrentPlanSections[0],
    kRecurrentPlanSections[1], kRecurrentPlanSections[2], kRecurrentPlanSections[3],
};
inline constexpr std::string_view kBuildReceiptSection = "sam2_build_receipt.json";
inline constexpr std::string_view kConfigSection = "config.json";

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
