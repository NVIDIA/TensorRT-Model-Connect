/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_tracker_network.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

const trtmc::sam2::native::TrackerWeightSpec& requireWeight(std::string_view name) {
    const auto& inventory = trtmc::sam2::native::trackerWeightInventory();
    const auto found = std::find_if(inventory.begin(), inventory.end(),
                                    [name](const auto& item) { return item.name == name; });
    check(found != inventory.end(), "tracker inventory contains required named tensor");
    return *found;
}

void checkTensor(const trtmc::sam2::TensorContract& actual,
                 const trtmc::sam2::TensorContract& expected, const char* message) {
    check(actual.name == expected.name && actual.data_type == expected.data_type &&
              actual.dimensions == expected.dimensions && actual.rank == expected.rank,
          message);
}

void testPromptPlan() {
    using namespace trtmc::sam2;
    using namespace trtmc::sam2::native;
    const TrackerPlanSpec plan = promptTrackerPlanSpec();
    check(plan.kind == TrackerPlanKind::kPrompt && plan.frame_index == 0 &&
              plan.history_frames == 0 && plan.plan_section == kPromptPlanSection,
          "prompt plan selects the exact frame-zero plan section");
    check(plan.inputs.size() == 4 && plan.outputs.size() == 3,
          "prompt plan has three image features plus one box and three state outputs");
    for (std::size_t index = 0; index < kTrackerFpn.size(); ++index)
        checkTensor(plan.inputs[index], kTrackerFpn[index], "prompt FPN contract stays exact");
    checkTensor(plan.inputs[3], kBoxPrompt, "prompt consumes the exact model-space box");
    checkTensor(plan.outputs[0], kMaskLogits256, "prompt emits exact low-resolution logits");
    checkTensor(plan.outputs[1], kObjectPointer, "prompt emits exact object pointer");
    checkTensor(plan.outputs[2], kMemoryFeatures, "prompt emits exact memory features");
    check(plan.supplied_point_count == 2 && plan.sparse_prompt_tokens == 3 &&
              plan.decoder_tokens == 9 && !plan.multimask_output,
          "box prompt token and single-mask semantics match the public source");
    check(plan.memory_frame_order.empty() && plan.object_pointer_frame_order.empty(),
          "prompt plan never fabricates recurrent history");
}

void testRecurrentPlans() {
    using namespace trtmc::sam2;
    using namespace trtmc::sam2::native;
    constexpr std::array<std::array<std::int32_t, 4>, 4> memory_rows = {{
        {{6, -1, -1, -1}},
        {{6, 0, -1, -1}},
        {{6, 1, 0, -1}},
        {{6, 2, 1, 0}},
    }};
    constexpr std::array<std::array<std::int32_t, 4>, 4> pointer_frames = {{
        {{0, -1, -1, -1}},
        {{0, 1, -1, -1}},
        {{0, 2, 1, -1}},
        {{0, 3, 2, 1}},
    }};
    for (std::int32_t history = 1; history <= 4; ++history) {
        const TrackerPlanSpec plan = recurrentTrackerPlanSpec(history);
        check(plan.kind == TrackerPlanKind::kRecurrent && plan.frame_index == history &&
                  plan.history_frames == history &&
                  plan.plan_section == kRecurrentPlanSections[history - 1],
              "recurrent plan binds history extent to frame and plan section");
        check(plan.inputs.size() == 5 && plan.outputs.size() == 3,
              "recurrent plan has exact image, history, and output arity");
        checkTensor(plan.inputs[3], historyMemoryFeatures(history),
                    "recurrent memory input has a static history extent");
        checkTensor(plan.inputs[4], historyObjectPointers(history),
                    "recurrent pointer input has a static history extent");
        check(plan.supplied_point_count == 0 && plan.sparse_prompt_tokens == 2 &&
                  plan.decoder_tokens == 8 && plan.multimask_output,
              "recurrent empty-prompt and multimask token semantics match source");
        check(plan.memory_frame_order.size() == static_cast<std::size_t>(history) &&
                  plan.memory_temporal_embedding_rows.size() == static_cast<std::size_t>(history) &&
                  plan.object_pointer_frame_order.size() == static_cast<std::size_t>(history) &&
                  plan.object_pointer_temporal_distances.size() ==
                      static_cast<std::size_t>(history),
              "recurrent plan materializes every history item exactly once");
        for (std::int32_t index = 0; index < history; ++index) {
            check(plan.memory_frame_order[index] == index &&
                      plan.memory_temporal_embedding_rows[index] ==
                          memory_rows[history - 1][index] &&
                      plan.object_pointer_frame_order[index] == pointer_frames[history - 1][index],
                  "recurrent plan preserves source memory and pointer ordering");
        }
        check(plan.object_pointer_temporal_distances.front() == history,
              "conditioning pointer uses signed distance from frame zero");
        for (std::int32_t index = 1; index < history; ++index)
            check(plan.object_pointer_temporal_distances[index] == index,
                  "non-conditioning pointers use nearest-to-farthest distances");
    }

    for (const std::int32_t invalid : {-1, 0, 5, 6}) {
        try {
            (void)recurrentTrackerPlanSpec(invalid);
        } catch (const std::invalid_argument&) {
            continue;
        }
        check(false, "recurrent plan rejects every unsupported history extent");
    }
}

void testCompleteWeightAndLayerInventory() {
    using namespace trtmc::sam2::native;
    const auto& weights = trackerWeightInventory();
    check(weights.size() == 309, "tracker inventory covers all 309 tracker-side tensors");
    std::set<std::string> names;
    for (const TrackerWeightSpec& weight : weights) {
        check(weight.dtype == DType::kFloat32 && !weight.shape.empty(),
              "every tracker checkpoint tensor has exact FP32 dtype and non-scalar shape");
        check(names.insert(weight.name).second, "tracker weight names are unique");
    }
    check(requireWeight("maskmem_tpos_enc").shape == std::vector<std::int64_t>({7, 1, 1, 64}),
          "temporal memory table shape is exact");
    check(requireWeight("memory_attention.layers.3.cross_attn_image.k_proj.weight").shape ==
              std::vector<std::int64_t>({256, 64}),
          "memory cross-attention key projection preserves compressed memory width");
    check(requireWeight("sam_mask_decoder.output_upscaling.0.weight").shape ==
              std::vector<std::int64_t>({256, 64, 2, 2}),
          "native transposed-convolution weight layout is checkpoint exact");
    check(requireWeight("mask_downsample.weight").shape == std::vector<std::int64_t>({1, 1, 4, 4}),
          "configuration-present mask pointer adapter is not silently omitted");
    check(requireWeight("obj_ptr_tpos_proj.weight").shape == std::vector<std::int64_t>({64, 256}),
          "object-pointer temporal projection shape is exact");

    const auto& layers = trackerLayerInventory();
    const std::size_t tensor_count =
        std::accumulate(layers.begin(), layers.end(), std::size_t{0},
                        [](std::size_t total, const TrackerLayerGroup& group) {
                            return total + group.checkpoint_tensor_count;
                        });
    check(tensor_count == weights.size(),
          "layer-group inventory accounts for every tracker checkpoint tensor");
    check(trackerGraphEmissionComplete(),
          "every layer group reachable in the fixed workload has native graph emission");
    const auto pending =
        std::count_if(layers.begin(), layers.end(), [](const TrackerLayerGroup& group) {
            return group.lowering == TrackerLoweringState::kPendingExactImplementation;
        });
    check(pending == 1,
          "only the configuration-present mask-input adapter remains outside the workload");
    check(std::all_of(layers.begin(), layers.end(),
                      [](const TrackerLayerGroup& group) {
                          return group.lowering == TrackerLoweringState::kImplemented ||
                                 (!group.prompt_plan && !group.recurrent_plans &&
                                  !group.exactness_requirement.empty());
                      }),
          "no reachable layer group can be hidden behind the unreachable-workload allowance");
}

} // namespace

int main() {
    testPromptPlan();
    testRecurrentPlans();
    testCompleteWeightAndLayerInventory();
    return 0;
}
