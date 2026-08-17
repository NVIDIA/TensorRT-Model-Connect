/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "checkpoint_reader.h"
#include "runtime/models/sam2/sam2_engine_contract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace nvinfer1 {
class INetworkDefinition;
class ITensor;
} // namespace nvinfer1

namespace trtmc::sam2::native {

// The delivered five-frame workload has one prompt frame followed by four
// recurrent frames. Separate recurrent plans intentionally keep every history
// extent static; padding a shorter history changes SAM2's attention semantics.
enum class TrackerPlanKind : std::uint8_t {
    kPrompt,
    kRecurrent,
};

struct TrackerPlanSpec {
    TrackerPlanKind kind{TrackerPlanKind::kPrompt};
    std::int32_t frame_index{0};
    std::int32_t history_frames{0};
    std::string_view plan_section;
    std::vector<TensorContract> inputs;
    std::vector<TensorContract> outputs;

    // Exact source-side token topology before the two-way transformer.
    std::int32_t supplied_point_count{0};
    std::int32_t sparse_prompt_tokens{0};
    std::int32_t decoder_tokens{0};
    bool multimask_output{false};

    // Source frame order and temporal metadata used to assemble recurrent
    // memory. These vectors are empty for the prompt plan.
    std::vector<std::int32_t> memory_frame_order;
    std::vector<std::int32_t> memory_temporal_embedding_rows;
    std::vector<std::int32_t> object_pointer_frame_order;
    std::vector<std::int32_t> object_pointer_temporal_distances;
};

TrackerPlanSpec promptTrackerPlanSpec();
TrackerPlanSpec recurrentTrackerPlanSpec(std::int32_t history_frames);

struct TrackerWeightSpec {
    std::string name;
    DType dtype{DType::kFloat32};
    std::vector<std::int64_t> shape;
};

// Every tracker-side tensor in the exact delivered checkpoint is inventoried,
// including configuration-present tensors that the fixed five-frame path does
// not execute. This makes architecture/checkpoint drift fail closed.
const std::vector<TrackerWeightSpec>& trackerWeightInventory();
void validateTrackerCheckpoint(const CheckpointReader& checkpoint);

enum class TrackerLoweringState : std::uint8_t {
    kImplemented,
    kPendingExactImplementation,
};

struct TrackerLayerGroup {
    std::string_view name;
    std::string_view source_symbol;
    std::size_t checkpoint_tensor_count{0};
    bool prompt_plan{false};
    bool recurrent_plans{false};
    TrackerLoweringState lowering{TrackerLoweringState::kPendingExactImplementation};
    std::string_view exactness_requirement;
};

const std::array<TrackerLayerGroup, 13>& trackerLayerInventory();
bool trackerGraphEmissionComplete() noexcept;

struct Sam2TrackerNetworkOutputs {
    nvinfer1::ITensor* mask_logits_256{nullptr};
    nvinfer1::ITensor* object_pointer{nullptr};
    nvinfer1::ITensor* memory_features{nullptr};
    std::size_t referenced_tensor_count{0};
    std::int32_t added_layer_count{0};
};

// Direct native TensorRT graph construction. Keep this object and its
// CheckpointReader alive through plan serialization because TensorRT may retain
// host constant pointers. Each instance is single-use and requires an empty,
// strongly typed network.
class Sam2TrackerNetworkBuilder final {
  public:
    Sam2TrackerNetworkBuilder(nvinfer1::INetworkDefinition& network,
                              const CheckpointReader& checkpoint);
    ~Sam2TrackerNetworkBuilder();
    Sam2TrackerNetworkBuilder(Sam2TrackerNetworkBuilder&&) noexcept;
    Sam2TrackerNetworkBuilder& operator=(Sam2TrackerNetworkBuilder&&) noexcept;

    Sam2TrackerNetworkBuilder(const Sam2TrackerNetworkBuilder&) = delete;
    Sam2TrackerNetworkBuilder& operator=(const Sam2TrackerNetworkBuilder&) = delete;

    Sam2TrackerNetworkOutputs buildPrompt();
    Sam2TrackerNetworkOutputs buildRecurrent(std::int32_t history_frames);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::sam2::native
