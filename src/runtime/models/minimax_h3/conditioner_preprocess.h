/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace trtmc {

inline constexpr int32_t kMiniMaxH3ConditionerImageHeight = 768;
inline constexpr int32_t kMiniMaxH3ConditionerImageWidth = 1344;
inline constexpr int32_t kMiniMaxH3ConditionerPatchSize = 16;
inline constexpr int32_t kMiniMaxH3ConditionerMergeSize = 2;
inline constexpr int32_t kMiniMaxH3ConditionerTemporalPatchSize = 2;
inline constexpr int32_t kMiniMaxH3ConditionerGridHeight = 48;
inline constexpr int32_t kMiniMaxH3ConditionerGridWidth = 84;
inline constexpr int32_t kMiniMaxH3ConditionerPatchRows = 4032;
inline constexpr int32_t kMiniMaxH3ConditionerMergedRows = 1008;
inline constexpr int32_t kMiniMaxH3ConditionerPatchVector = 1536;
inline constexpr int32_t kMiniMaxH3ConditionerMaxSequenceRows = 4096;

struct MiniMaxH3ConditionerTokenIds {
    int32_t vision_start{151652};
    int32_t image_pad{151655};
    int32_t vision_end{151653};
};

using MiniMaxH3ConditionerTokenizer = std::function<std::vector<int32_t>(const std::string&)>;

struct MiniMaxH3ConditionerPresentation {
    int32_t sequence_rows{0};
    int32_t num_keyframes{0};
    int32_t next_mrope_position{0};
    std::vector<int32_t> input_ids;
    std::vector<int32_t> h3_token_tags;
    std::vector<int32_t> qwen_mm_token_type_ids;
    std::vector<int32_t> mrope_position_ids; // [3, sequence_rows], axis-major
    // Compact vision outputs are keyframe-major. These sequence row indices
    // scatter the main output and each of the three DeepStack outputs in the
    // same deterministic order.
    std::vector<int32_t> vision_scatter_indices;
    // Flat [sequence_rows] int32 storage; bind as [sequence_rows, 1]. Reuse
    // this selector for the main vision override and all DeepStack injections.
    std::vector<int32_t> vision_selector;
};

struct MiniMaxH3ConditionerVisionFeatures {
    int32_t sequence_rows{0};
    int32_t feature_dim{0};
    std::vector<float> vision_embeddings; // [sequence_rows, feature_dim]
    std::array<std::vector<float>, 3> deepstack_embeddings;
    std::vector<int32_t> vision_selector; // flat [sequence_rows], bind as [sequence_rows, 1]
};

// Build the exact FL2VA Qwen3-VL presentation. The tokenizer is invoked once
// per emitted label, followed by one invocation with the prompt verbatim.
MiniMaxH3ConditionerPresentation
minimax_h3_make_conditioner_presentation(const std::string& prompt, bool has_first_keyframe,
                                         bool has_last_keyframe,
                                         const MiniMaxH3ConditionerTokenizer& tokenize,
                                         const MiniMaxH3ConditionerTokenIds& token_ids = {});

// Convert one already-prepared 768x1344 HWC RGB float32 keyframe into the
// Qwen3-VL processor binding [4032, 1536]. Values are normalized by mean/std
// 0.5, duplicated across the temporal patch, and emitted in merge-group order.
std::vector<float> minimax_h3_preprocess_conditioner_keyframe(const MediaImageInput& keyframe);

// Scatter compact keyframe-major [num_keyframes * 1008, feature_dim] vision
// outputs into zero-filled sequence-aligned [sequence_rows, feature_dim]
// bindings. The same ordered rows and selector apply to the main output and
// all three DeepStack outputs.
MiniMaxH3ConditionerVisionFeatures minimax_h3_scatter_vision_features(
    const MiniMaxH3ConditionerPresentation& presentation,
    const std::vector<float>& compact_vision_embeddings,
    const std::array<std::vector<float>, 3>& compact_deepstack_embeddings, int32_t feature_dim);

} // namespace trtmc
