/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace trtmc {

inline constexpr int32_t kMiniMaxH3Ref2VAPatchSize = 16;
inline constexpr int32_t kMiniMaxH3Ref2VATemporalPatchSize = 2;
inline constexpr int32_t kMiniMaxH3Ref2VASpatialMergeSize = 2;
inline constexpr int32_t kMiniMaxH3Ref2VAPatchVectorSize = 1536;
inline constexpr int32_t kMiniMaxH3Ref2VAImageShortEdge = 2048;
inline constexpr int32_t kMiniMaxH3Ref2VAReferenceFps = 24;
inline constexpr int32_t kMiniMaxH3Ref2VAConditionerFps = 2;

struct MiniMaxH3Ref2VATokenIds {
    int32_t vision_start{151652};
    int32_t image_pad{151655};
    int32_t video_pad{151656};
    int32_t vision_end{151653};
};

using MiniMaxH3Ref2VATokenizer = std::function<std::vector<int32_t>(const std::string&)>;

// This is an in-memory boundary. Media paths, containers, and codecs are
// resolved before constructing it. reference_index points back into the
// already-validated request-order AudioVideoReference list.
struct MiniMaxH3PreparedReference {
    std::size_t reference_index{0};
    AudioVideoReferenceKind kind{AudioVideoReferenceKind::kImage};
    MediaImageInput image;
    MediaVideoInput video;
    int32_t qwen_grid_h{0};
    int32_t qwen_grid_w{0};
    int32_t qwen_patch_rows{0};
    // Present for a standalone audio reference or a prepared video soundtrack.
    std::optional<MultiChannelAudioResult> audio;
};

enum class MiniMaxH3Ref2VAVisionKind {
    kImage,
    kVideo,
};

// One invocation of the dynamic Ref2VA Qwen vision engine. Video inputs are
// split into temporal pairs, so grid_t is always one and each pair is a run.
struct MiniMaxH3Ref2VAVisionInput {
    std::size_t reference_index{0};
    MiniMaxH3Ref2VAVisionKind kind{MiniMaxH3Ref2VAVisionKind::kImage};
    int32_t modality_index{0}; // one-based Picture or Video number
    int32_t run_index{0};      // zero-based within this reference
    float timestamp_seconds{-1.0F};
    int32_t grid_t{1};
    int32_t grid_h{0};
    int32_t grid_w{0};
    std::vector<float> pixel_values;       // [grid_h * grid_w, 1536]
    std::vector<int32_t> position_indices; // [grid_h * grid_w, 4]
    // Bind exact ATen-linspace weights as FP32 [rows, 4]. The engine publishes
    // them to BF16 before reproducing upstream's BF16 products and sums.
    std::vector<float> position_weights;
    std::vector<int32_t> vision_position_ids; // [grid_h * grid_w, 2]
};

struct MiniMaxH3Ref2VAVisionScatter {
    std::size_t reference_index{0};
    MiniMaxH3Ref2VAVisionKind kind{MiniMaxH3Ref2VAVisionKind::kImage};
    int32_t run_index{0};
    int32_t grid_t{1};
    int32_t grid_h{0};
    int32_t grid_w{0};
    int32_t compact_row_begin{0};
    int32_t compact_row_count{0};
    int32_t sequence_row_begin{0};
};

struct MiniMaxH3Ref2VAAudioLabel {
    std::size_t reference_index{0};
    int32_t audio_index{0}; // one-based Audio number
    bool from_video_soundtrack{false};
};

struct MiniMaxH3Ref2VAConditionerPresentation {
    int32_t sequence_rows{0};
    int32_t next_mrope_position{0};
    int32_t mrope_position_delta{0};
    std::vector<int32_t> input_ids;
    std::vector<int32_t> h3_token_tags;
    std::vector<int32_t> qwen_mm_token_type_ids;
    std::vector<int32_t> mrope_position_ids; // [3, sequence_rows], axis-major
    std::vector<int32_t> vision_selector;    // [sequence_rows], bind as [L, 1]
    std::vector<int32_t> vision_scatter_indices;
    std::vector<MiniMaxH3Ref2VAVisionInput> vision_inputs;
    std::vector<MiniMaxH3Ref2VAVisionScatter> vision_scatter;
    std::vector<int32_t> vision_run_lengths;
    std::vector<int32_t> vision_run_reference_ids;
    std::vector<MiniMaxH3Ref2VAAudioLabel> audio_labels;
};

// Build the official Ref2VA conditioner presentation and all dynamic vision
// bindings from decoded, spatially prepared media. The two reference vectors
// must align one-for-one in semantic request order.
MiniMaxH3Ref2VAConditionerPresentation minimax_h3_build_ref2va_conditioner_presentation(
    const std::string& prompt, const std::vector<AudioVideoReference>& references,
    const std::vector<MiniMaxH3PreparedReference>& prepared_references,
    const MiniMaxH3Ref2VATokenizer& tokenize, const MiniMaxH3Ref2VATokenIds& token_ids = {});

} // namespace trtmc
