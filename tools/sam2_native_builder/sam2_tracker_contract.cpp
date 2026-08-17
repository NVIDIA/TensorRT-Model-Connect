/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_tracker_network.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::sam2::native {
namespace {

constexpr std::size_t kExpectedCheckpointTensors = 603U;
constexpr std::size_t kExpectedCheckpointStorages = 595U;
constexpr std::size_t kExpectedTrackerTensors = 309U;

void add(std::vector<TrackerWeightSpec>& result, std::string name,
         std::initializer_list<std::int64_t> shape) {
    result.push_back({std::move(name), DType::kFloat32, shape});
}

void addLinear(std::vector<TrackerWeightSpec>& result, const std::string& module,
               std::int64_t input_features, std::int64_t output_features) {
    add(result, module + ".weight", {output_features, input_features});
    add(result, module + ".bias", {output_features});
}

void addLayerNorm(std::vector<TrackerWeightSpec>& result, const std::string& module,
                  std::int64_t channels) {
    add(result, module + ".weight", {channels});
    add(result, module + ".bias", {channels});
}

void addAttention(std::vector<TrackerWeightSpec>& result, const std::string& module,
                  std::int64_t query_features, std::int64_t key_value_features,
                  std::int64_t internal_features, std::int64_t output_features) {
    addLinear(result, module + ".q_proj", query_features, internal_features);
    addLinear(result, module + ".k_proj", key_value_features, internal_features);
    addLinear(result, module + ".v_proj", key_value_features, internal_features);
    addLinear(result, module + ".out_proj", internal_features, output_features);
}

void addMemoryAttention(std::vector<TrackerWeightSpec>& result) {
    for (std::int32_t layer = 0; layer < 4; ++layer) {
        const std::string prefix = "memory_attention.layers." + std::to_string(layer);
        addAttention(result, prefix + ".self_attn", 256, 256, 256, 256);
        addAttention(result, prefix + ".cross_attn_image", 256, 64, 256, 256);
        addLinear(result, prefix + ".linear1", 256, 2048);
        addLinear(result, prefix + ".linear2", 2048, 256);
        addLayerNorm(result, prefix + ".norm1", 256);
        addLayerNorm(result, prefix + ".norm2", 256);
        addLayerNorm(result, prefix + ".norm3", 256);
    }
    addLayerNorm(result, "memory_attention.norm", 256);
}

void addMemoryEncoder(std::vector<TrackerWeightSpec>& result) {
    struct DownsampleLayer {
        std::int32_t index;
        std::int64_t input_channels;
        std::int64_t output_channels;
    };
    constexpr std::array<DownsampleLayer, 4> convolutions = {
        {{0, 1, 4}, {3, 4, 16}, {6, 16, 64}, {9, 64, 256}}};
    constexpr std::array<std::int32_t, 4> norms = {1, 4, 7, 10};
    for (std::size_t i = 0; i < convolutions.size(); ++i) {
        const auto& convolution = convolutions[i];
        const std::string conv =
            "memory_encoder.mask_downsampler.encoder." + std::to_string(convolution.index);
        add(result, conv + ".weight",
            {convolution.output_channels, convolution.input_channels, 3, 3});
        add(result, conv + ".bias", {convolution.output_channels});
        addLayerNorm(result, "memory_encoder.mask_downsampler.encoder." + std::to_string(norms[i]),
                     convolution.output_channels);
    }
    add(result, "memory_encoder.mask_downsampler.encoder.12.weight", {256, 256, 1, 1});
    add(result, "memory_encoder.mask_downsampler.encoder.12.bias", {256});
    add(result, "memory_encoder.pix_feat_proj.weight", {256, 256, 1, 1});
    add(result, "memory_encoder.pix_feat_proj.bias", {256});

    for (std::int32_t layer = 0; layer < 2; ++layer) {
        const std::string prefix = "memory_encoder.fuser.layers." + std::to_string(layer);
        add(result, prefix + ".gamma", {256});
        add(result, prefix + ".dwconv.weight", {256, 1, 7, 7});
        add(result, prefix + ".dwconv.bias", {256});
        addLayerNorm(result, prefix + ".norm", 256);
        addLinear(result, prefix + ".pwconv1", 256, 1024);
        addLinear(result, prefix + ".pwconv2", 1024, 256);
    }
    add(result, "memory_encoder.out_proj.weight", {64, 256, 1, 1});
    add(result, "memory_encoder.out_proj.bias", {64});
}

void addPromptEncoder(std::vector<TrackerWeightSpec>& result) {
    add(result, "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", {2, 128});
    for (std::int32_t embedding = 0; embedding < 4; ++embedding) {
        add(result, "sam_prompt_encoder.point_embeddings." + std::to_string(embedding) + ".weight",
            {1, 256});
    }
    add(result, "sam_prompt_encoder.not_a_point_embed.weight", {1, 256});
    add(result, "sam_prompt_encoder.mask_downscaling.0.weight", {4, 1, 2, 2});
    add(result, "sam_prompt_encoder.mask_downscaling.0.bias", {4});
    addLayerNorm(result, "sam_prompt_encoder.mask_downscaling.1", 4);
    add(result, "sam_prompt_encoder.mask_downscaling.3.weight", {16, 4, 2, 2});
    add(result, "sam_prompt_encoder.mask_downscaling.3.bias", {16});
    addLayerNorm(result, "sam_prompt_encoder.mask_downscaling.4", 16);
    add(result, "sam_prompt_encoder.mask_downscaling.6.weight", {256, 16, 1, 1});
    add(result, "sam_prompt_encoder.mask_downscaling.6.bias", {256});
    add(result, "sam_prompt_encoder.no_mask_embed.weight", {1, 256});
}

void addTwoWayTransformer(std::vector<TrackerWeightSpec>& result) {
    for (std::int32_t layer = 0; layer < 2; ++layer) {
        const std::string prefix = "sam_mask_decoder.transformer.layers." + std::to_string(layer);
        addAttention(result, prefix + ".self_attn", 256, 256, 256, 256);
        addLayerNorm(result, prefix + ".norm1", 256);
        addAttention(result, prefix + ".cross_attn_token_to_image", 256, 256, 128, 256);
        addLayerNorm(result, prefix + ".norm2", 256);
        addLinear(result, prefix + ".mlp.layers.0", 256, 2048);
        addLinear(result, prefix + ".mlp.layers.1", 2048, 256);
        addLayerNorm(result, prefix + ".norm3", 256);
        addLayerNorm(result, prefix + ".norm4", 256);
        addAttention(result, prefix + ".cross_attn_image_to_token", 256, 256, 128, 256);
    }
    addAttention(result, "sam_mask_decoder.transformer.final_attn_token_to_image", 256, 256, 128,
                 256);
    addLayerNorm(result, "sam_mask_decoder.transformer.norm_final_attn", 256);
}

void addMaskDecoder(std::vector<TrackerWeightSpec>& result) {
    addTwoWayTransformer(result);

    add(result, "sam_mask_decoder.iou_token.weight", {1, 256});
    add(result, "sam_mask_decoder.mask_tokens.weight", {4, 256});
    add(result, "sam_mask_decoder.obj_score_token.weight", {1, 256});

    add(result, "sam_mask_decoder.output_upscaling.0.weight", {256, 64, 2, 2});
    add(result, "sam_mask_decoder.output_upscaling.0.bias", {64});
    addLayerNorm(result, "sam_mask_decoder.output_upscaling.1", 64);
    add(result, "sam_mask_decoder.output_upscaling.3.weight", {64, 32, 2, 2});
    add(result, "sam_mask_decoder.output_upscaling.3.bias", {32});
    add(result, "sam_mask_decoder.conv_s0.weight", {32, 256, 1, 1});
    add(result, "sam_mask_decoder.conv_s0.bias", {32});
    add(result, "sam_mask_decoder.conv_s1.weight", {64, 256, 1, 1});
    add(result, "sam_mask_decoder.conv_s1.bias", {64});

    for (std::int32_t token = 0; token < 4; ++token) {
        const std::string prefix =
            "sam_mask_decoder.output_hypernetworks_mlps." + std::to_string(token) + ".layers.";
        addLinear(result, prefix + "0", 256, 256);
        addLinear(result, prefix + "1", 256, 256);
        addLinear(result, prefix + "2", 256, 32);
    }

    addLinear(result, "sam_mask_decoder.iou_prediction_head.layers.0", 256, 256);
    addLinear(result, "sam_mask_decoder.iou_prediction_head.layers.1", 256, 256);
    addLinear(result, "sam_mask_decoder.iou_prediction_head.layers.2", 256, 4);
    addLinear(result, "sam_mask_decoder.pred_obj_score_head.layers.0", 256, 256);
    addLinear(result, "sam_mask_decoder.pred_obj_score_head.layers.1", 256, 256);
    addLinear(result, "sam_mask_decoder.pred_obj_score_head.layers.2", 256, 1);
}

std::vector<TrackerWeightSpec> makeWeightInventory() {
    std::vector<TrackerWeightSpec> result;
    result.reserve(kExpectedTrackerTensors);

    add(result, "maskmem_tpos_enc", {7, 1, 1, 64});
    add(result, "no_mem_embed", {1, 1, 256});
    add(result, "no_mem_pos_enc", {1, 1, 256});
    add(result, "no_obj_ptr", {1, 256});
    add(result, "no_obj_embed_spatial", {1, 64});
    addMemoryAttention(result);
    addMemoryEncoder(result);
    addPromptEncoder(result);
    addMaskDecoder(result);
    add(result, "mask_downsample.weight", {1, 1, 4, 4});
    add(result, "mask_downsample.bias", {1});
    for (std::int32_t layer = 0; layer < 3; ++layer)
        addLinear(result, "obj_ptr_proj.layers." + std::to_string(layer), 256, 256);
    addLinear(result, "obj_ptr_tpos_proj", 256, 64);

    if (result.size() != kExpectedTrackerTensors)
        throw std::logic_error("internal SAM2 tracker weight inventory count drifted");
    return result;
}

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool isTrackerWeight(std::string_view name) {
    constexpr std::array<std::string_view, 12> prefixes = {
        "maskmem_tpos_enc",     "no_mem_embed",      "no_mem_pos_enc",  "no_obj_ptr",
        "no_obj_embed_spatial", "memory_attention.", "memory_encoder.", "sam_prompt_encoder.",
        "sam_mask_decoder.",    "mask_downsample.",  "obj_ptr_proj.",   "obj_ptr_tpos_proj.",
    };
    return std::any_of(prefixes.begin(), prefixes.end(),
                       [name](std::string_view prefix) { return startsWith(name, prefix); });
}

TensorContract historyMemory(std::int32_t history_frames) {
    return historyMemoryFeatures(history_frames);
}

TensorContract historyPointers(std::int32_t history_frames) {
    return historyObjectPointers(history_frames);
}

std::vector<TensorContract> commonInputs() {
    return {kTrackerFpn[0], kTrackerFpn[1], kTrackerFpn[2]};
}

std::vector<TensorContract> commonOutputs() {
    return {kMaskLogits256, kObjectPointer, kMemoryFeatures};
}

} // namespace

TrackerPlanSpec promptTrackerPlanSpec() {
    TrackerPlanSpec result;
    result.kind = TrackerPlanKind::kPrompt;
    result.frame_index = 0;
    result.plan_section = kPromptPlanSection;
    result.inputs = commonInputs();
    result.inputs.push_back(kBoxPrompt);
    result.outputs = commonOutputs();
    result.supplied_point_count = 2;
    result.sparse_prompt_tokens = 3; // Two box corners plus PromptEncoder padding.
    result.decoder_tokens = 9;       // Six output tokens plus three sparse prompts.
    result.multimask_output = false; // A box counts as two points; configured max is one.
    return result;
}

TrackerPlanSpec recurrentTrackerPlanSpec(std::int32_t history_frames) {
    if (history_frames < 1 || history_frames > 4)
        throw std::invalid_argument("SAM2 recurrent history must be in the closed range [1, 4]");

    TrackerPlanSpec result;
    result.kind = TrackerPlanKind::kRecurrent;
    result.frame_index = history_frames;
    result.history_frames = history_frames;
    result.plan_section = kRecurrentPlanSections[static_cast<std::size_t>(history_frames - 1)];
    result.inputs = commonInputs();
    result.inputs.push_back(historyMemory(history_frames));
    result.inputs.push_back(historyPointers(history_frames));
    result.outputs = commonOutputs();
    result.supplied_point_count = 0;
    result.sparse_prompt_tokens = 2; // Source inserts an empty point, then pads it.
    result.decoder_tokens = 8;       // Six output tokens plus two sparse prompts.
    result.multimask_output = true;

    // Spatial memories are concatenated in chronological source-frame order.
    // The conditioning frame uses temporal row 6. Existing non-conditioning
    // frames j use row (current_frame - j - 1).
    result.memory_frame_order.reserve(static_cast<std::size_t>(history_frames));
    result.memory_temporal_embedding_rows.reserve(static_cast<std::size_t>(history_frames));
    for (std::int32_t frame = 0; frame < history_frames; ++frame) {
        result.memory_frame_order.push_back(frame);
        result.memory_temporal_embedding_rows.push_back(frame == 0 ? 6
                                                                   : history_frames - frame - 1);
    }

    // Object pointers are concatenated as conditioning frame 0, then nearest
    // prior frame to farthest. Signed forward distance is positive here.
    result.object_pointer_frame_order.push_back(0);
    result.object_pointer_temporal_distances.push_back(history_frames);
    for (std::int32_t distance = 1; distance < history_frames; ++distance) {
        result.object_pointer_frame_order.push_back(history_frames - distance);
        result.object_pointer_temporal_distances.push_back(distance);
    }
    return result;
}

const std::vector<TrackerWeightSpec>& trackerWeightInventory() {
    static const std::vector<TrackerWeightSpec> inventory = makeWeightInventory();
    return inventory;
}

void validateTrackerCheckpoint(const CheckpointReader& checkpoint) {
    if (checkpoint.tensorCount() != kExpectedCheckpointTensors) {
        throw CheckpointError("SAM2 checkpoint has " + std::to_string(checkpoint.tensorCount()) +
                              " tensors; expected exactly " +
                              std::to_string(kExpectedCheckpointTensors));
    }
    if (checkpoint.storageCount() != kExpectedCheckpointStorages) {
        throw CheckpointError("SAM2 checkpoint has " + std::to_string(checkpoint.storageCount()) +
                              " storages; expected exactly " +
                              std::to_string(kExpectedCheckpointStorages));
    }

    std::set<std::string> expected_names;
    for (const TrackerWeightSpec& expected : trackerWeightInventory()) {
        if (!expected_names.insert(expected.name).second)
            throw std::logic_error("duplicate SAM2 tracker weight inventory entry: " +
                                   expected.name);
        const WeightView weight =
            checkpoint.requireTensor(expected.name, expected.dtype, expected.shape);
        if (!weight.contiguous)
            throw CheckpointError("SAM2 tracker tensor '" + expected.name + "' must be contiguous");
    }

    std::set<std::string> actual_names;
    for (const std::string& name : checkpoint.tensorNames()) {
        if (isTrackerWeight(name))
            actual_names.insert(name);
    }
    if (actual_names != expected_names) {
        std::ostringstream message;
        message << "SAM2 tracker tensor namespace drifted (expected " << expected_names.size()
                << ", found " << actual_names.size() << ')';
        throw CheckpointError(message.str());
    }
}

const std::array<TrackerLayerGroup, 13>& trackerLayerInventory() {
    static constexpr std::array<TrackerLayerGroup, 13> inventory = {{
        {"temporal_and_sentinel_embeddings", "SAM2Base.__init__", 5, true, true,
         TrackerLoweringState::kImplemented,
         "static no-memory path and plan-specific memory temporal rows"},
        {"memory_attention", "MemoryAttention.forward", 106, false, true,
         TrackerLoweringState::kImplemented,
         "four pre-norm blocks with axial RoPE and pointer-token exclusion"},
        {"memory_encoder", "MemoryEncoder.forward", 40, true, true,
         TrackerLoweringState::kImplemented,
         "plan-specific binary or scaled-sigmoid mask, four downsamplers, two ConvNeXt blocks"},
        {"prompt_encoder", "PromptEncoder.forward", 17, true, true,
         TrackerLoweringState::kImplemented,
         "box-as-points padding and random Fourier positional encoding"},
        {"two_way_transformer", "TwoWayTransformer.forward", 82, true, true,
         TrackerLoweringState::kImplemented,
         "two bidirectional attention blocks plus final token-to-image attention"},
        {"decoder_tokens", "MaskDecoder.predict_masks", 3, true, true,
         TrackerLoweringState::kImplemented, "object, IoU, and four mask tokens in source order"},
        {"decoder_upscale_and_high_res", "MaskDecoder.predict_masks", 10, true, true,
         TrackerLoweringState::kImplemented,
         "two native transposed convolutions with high-resolution skip features"},
        {"mask_hypernetworks", "MaskDecoder.predict_masks", 24, true, true,
         TrackerLoweringState::kImplemented,
         "four three-layer MLPs followed by batched mask projection"},
        {"iou_prediction", "MaskDecoder.predict_masks", 6, true, true,
         TrackerLoweringState::kImplemented,
         "prompt stability fallback and recurrent argmax over tokens 1 through 3"},
        {"object_score_prediction", "SAM2Base._forward_sam_heads", 6, true, true,
         TrackerLoweringState::kImplemented,
         "hard object-presence selection before mask and pointer emission"},
        {"mask_input_pointer_adapter", "SAM2Base._use_mask_as_output", 2, false, false,
         TrackerLoweringState::kPendingExactImplementation,
         "configuration-present but unreachable in the fixed bbox-only workload"},
        {"object_pointer_projection", "SAM2Base._forward_sam_heads", 6, true, true,
         TrackerLoweringState::kImplemented,
         "selected SAM token through three-layer MLP with fixed no-object pointer"},
        {"object_pointer_temporal_projection", "SAM2Base._prepare_memory_conditioned_features", 2,
         false, true, TrackerLoweringState::kImplemented,
         "signed sine distance divided by four, projected from 256 to 64"},
    }};
    return inventory;
}

bool trackerGraphEmissionComplete() noexcept {
    return std::all_of(trackerLayerInventory().begin(), trackerLayerInventory().end(),
                       [](const TrackerLayerGroup& group) {
                           return group.lowering == TrackerLoweringState::kImplemented ||
                                  (!group.prompt_plan && !group.recurrent_plans);
                       });
}

} // namespace trtmc::sam2::native
