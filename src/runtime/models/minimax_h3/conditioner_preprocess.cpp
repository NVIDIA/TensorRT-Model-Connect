/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/conditioner_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {
namespace {

constexpr int32_t kTextTag = 1;
constexpr int32_t kVisionTag = 0;
constexpr int32_t kQwenTextType = 0;
constexpr int32_t kQwenImageType = 1;
constexpr int32_t kRgbChannels = 3;
constexpr int32_t kVocabularySize = 151936;
constexpr int32_t kMergedGridHeight =
    kMiniMaxH3ConditionerGridHeight / kMiniMaxH3ConditionerMergeSize;
constexpr int32_t kMergedGridWidth =
    kMiniMaxH3ConditionerGridWidth / kMiniMaxH3ConditionerMergeSize;

void validate_special_token_ids(const MiniMaxH3ConditionerTokenIds& token_ids) {
    std::array<int32_t, 3> ids = {
        token_ids.vision_start,
        token_ids.image_pad,
        token_ids.vision_end,
    };
    if (std::any_of(ids.begin(), ids.end(),
                    [](int32_t id) { return id < 0 || id >= kVocabularySize; }))
        throw std::invalid_argument("MiniMax-H3 conditioner special token ID is out of range");
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::invalid_argument("MiniMax-H3 conditioner special token IDs must be distinct");
}

bool is_special_token(int32_t id, const MiniMaxH3ConditionerTokenIds& token_ids) {
    return id == token_ids.vision_start || id == token_ids.image_pad || id == token_ids.vision_end;
}

void validate_text_tokens(const std::vector<int32_t>& tokens,
                          const MiniMaxH3ConditionerTokenIds& token_ids) {
    for (const int32_t token : tokens) {
        if (token < 0 || token >= kVocabularySize)
            throw std::invalid_argument("MiniMax-H3 conditioner tokenizer emitted an invalid ID");
        if (is_special_token(token, token_ids))
            throw std::invalid_argument(
                "MiniMax-H3 conditioner text emitted a reserved vision token");
    }
}

void require_append_capacity(std::size_t current, std::size_t additional) {
    const auto maximum = static_cast<std::size_t>(kMiniMaxH3ConditionerMaxSequenceRows);
    if (current > maximum || additional > maximum - current)
        throw std::invalid_argument("MiniMax-H3 conditioner presentation exceeds 4096 rows");
}

void append_text_tokens(MiniMaxH3ConditionerPresentation& result,
                        const std::vector<int32_t>& tokens,
                        const MiniMaxH3ConditionerTokenIds& token_ids) {
    validate_text_tokens(tokens, token_ids);
    require_append_capacity(result.input_ids.size(), tokens.size());
    result.input_ids.insert(result.input_ids.end(), tokens.begin(), tokens.end());
    result.h3_token_tags.insert(result.h3_token_tags.end(), tokens.size(), kTextTag);
    result.qwen_mm_token_type_ids.insert(result.qwen_mm_token_type_ids.end(), tokens.size(),
                                         kQwenTextType);
    result.vision_selector.insert(result.vision_selector.end(), tokens.size(), 0);
}

void append_vision_block(MiniMaxH3ConditionerPresentation& result,
                         const MiniMaxH3ConditionerTokenIds& token_ids) {
    constexpr std::size_t kVisionBlockRows =
        static_cast<std::size_t>(kMiniMaxH3ConditionerMergedRows) + 2U;
    require_append_capacity(result.input_ids.size(), kVisionBlockRows);
    result.input_ids.push_back(token_ids.vision_start);
    result.h3_token_tags.push_back(kVisionTag);
    result.qwen_mm_token_type_ids.push_back(kQwenTextType);
    result.vision_selector.push_back(0);

    const auto pad_start = static_cast<int32_t>(result.input_ids.size());
    result.input_ids.insert(result.input_ids.end(), kMiniMaxH3ConditionerMergedRows,
                            token_ids.image_pad);
    result.h3_token_tags.insert(result.h3_token_tags.end(), kMiniMaxH3ConditionerMergedRows,
                                kVisionTag);
    result.qwen_mm_token_type_ids.insert(result.qwen_mm_token_type_ids.end(),
                                         kMiniMaxH3ConditionerMergedRows, kQwenImageType);
    result.vision_selector.insert(result.vision_selector.end(), kMiniMaxH3ConditionerMergedRows, 1);
    for (int32_t offset = 0; offset < kMiniMaxH3ConditionerMergedRows; ++offset)
        result.vision_scatter_indices.push_back(pad_start + offset);

    result.input_ids.push_back(token_ids.vision_end);
    result.h3_token_tags.push_back(kVisionTag);
    result.qwen_mm_token_type_ids.push_back(kQwenTextType);
    result.vision_selector.push_back(0);
}

void set_mrope_position(MiniMaxH3ConditionerPresentation& result, std::size_t row, int32_t temporal,
                        int32_t height, int32_t width) {
    const auto rows = static_cast<std::size_t>(result.sequence_rows);
    result.mrope_position_ids[row] = temporal;
    result.mrope_position_ids[rows + row] = height;
    result.mrope_position_ids[2U * rows + row] = width;
}

void fill_text_mrope(MiniMaxH3ConditionerPresentation& result, std::size_t begin, std::size_t end,
                     int32_t& current_position) {
    for (std::size_t row = begin; row < end; ++row) {
        const int32_t position = current_position + static_cast<int32_t>(row - begin);
        set_mrope_position(result, row, position, position, position);
    }
    current_position += static_cast<int32_t>(end - begin);
}

void fill_image_mrope(MiniMaxH3ConditionerPresentation& result, std::size_t begin,
                      int32_t& current_position) {
    for (int32_t offset = 0; offset < kMiniMaxH3ConditionerMergedRows; ++offset) {
        set_mrope_position(result, begin + static_cast<std::size_t>(offset), current_position,
                           current_position + offset / kMergedGridWidth,
                           current_position + offset % kMergedGridWidth);
    }
    current_position += std::max(kMergedGridHeight, kMergedGridWidth);
}

void build_mrope_positions(MiniMaxH3ConditionerPresentation& result) {
    result.mrope_position_ids.assign(static_cast<std::size_t>(result.sequence_rows) * 3U, 0);
    std::size_t cursor = 0;
    int32_t current_position = 0;
    int32_t image_runs = 0;
    while (cursor < result.qwen_mm_token_type_ids.size()) {
        const int32_t modality = result.qwen_mm_token_type_ids[cursor];
        std::size_t end = cursor + 1U;
        while (end < result.qwen_mm_token_type_ids.size() &&
               result.qwen_mm_token_type_ids[end] == modality)
            ++end;
        if (modality == kQwenTextType) {
            fill_text_mrope(result, cursor, end, current_position);
        } else {
            if (modality != kQwenImageType ||
                end - cursor != static_cast<std::size_t>(kMiniMaxH3ConditionerMergedRows))
                throw std::logic_error("MiniMax-H3 conditioner has an invalid image-token run");
            fill_image_mrope(result, cursor, current_position);
            ++image_runs;
        }
        cursor = end;
    }
    if (image_runs != result.num_keyframes)
        throw std::logic_error("MiniMax-H3 conditioner image grids do not match keyframes");
    result.next_mrope_position = current_position;
}

void validate_keyframe(const MediaImageInput& keyframe) {
    if (keyframe.height != kMiniMaxH3ConditionerImageHeight ||
        keyframe.width != kMiniMaxH3ConditionerImageWidth)
        throw std::invalid_argument("MiniMax-H3 conditioner keyframe must be prepared at 768x1344");
    constexpr std::size_t kExpected = static_cast<std::size_t>(kMiniMaxH3ConditionerImageHeight) *
                                      kMiniMaxH3ConditionerImageWidth * kRgbChannels;
    if (keyframe.pixels.size() != kExpected)
        throw std::invalid_argument(
            "MiniMax-H3 conditioner keyframe buffer does not match HWC dimensions");
    const auto invalid =
        std::find_if(keyframe.pixels.begin(), keyframe.pixels.end(), [](float value) {
            return !std::isfinite(value) || value < 0.0F || value > 1.0F;
        });
    if (invalid != keyframe.pixels.end())
        throw std::invalid_argument("MiniMax-H3 conditioner keyframe has invalid RGB values");
}

float normalize_pixel(float value) {
    return (value - 0.5F) / 0.5F;
}

void copy_patch(const MediaImageInput& keyframe, int32_t patch_y, int32_t patch_x,
                int32_t patch_row, std::vector<float>& output) {
    const auto row_base = static_cast<std::size_t>(patch_row) * kMiniMaxH3ConditionerPatchVector;
    for (int32_t channel = 0; channel < kRgbChannels; ++channel) {
        for (int32_t temporal = 0; temporal < kMiniMaxH3ConditionerTemporalPatchSize; ++temporal) {
            for (int32_t y = 0; y < kMiniMaxH3ConditionerPatchSize; ++y) {
                for (int32_t x = 0; x < kMiniMaxH3ConditionerPatchSize; ++x) {
                    const auto source =
                        (static_cast<std::size_t>(patch_y * kMiniMaxH3ConditionerPatchSize + y) *
                             kMiniMaxH3ConditionerImageWidth +
                         patch_x * kMiniMaxH3ConditionerPatchSize + x) *
                            kRgbChannels +
                        channel;
                    const auto target = (((static_cast<std::size_t>(channel) *
                                               kMiniMaxH3ConditionerTemporalPatchSize +
                                           temporal) *
                                              kMiniMaxH3ConditionerPatchSize +
                                          y) *
                                             kMiniMaxH3ConditionerPatchSize +
                                         x);
                    output[row_base + target] = normalize_pixel(keyframe.pixels[source]);
                }
            }
        }
    }
}

std::size_t checked_feature_elements(std::size_t rows, int32_t feature_dim) {
    if (feature_dim <= 0)
        throw std::invalid_argument("MiniMax-H3 conditioner feature dimension must be positive");
    if (rows > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(feature_dim))
        throw std::invalid_argument("MiniMax-H3 conditioner feature buffer is too large");
    return rows * static_cast<std::size_t>(feature_dim);
}

void validate_scatter_shapes(const MiniMaxH3ConditionerPresentation& presentation) {
    if (presentation.sequence_rows <= 0)
        throw std::invalid_argument("MiniMax-H3 conditioner scatter needs presentation rows");
    const auto rows = static_cast<std::size_t>(presentation.sequence_rows);
    if (presentation.input_ids.size() != rows || presentation.h3_token_tags.size() != rows)
        throw std::invalid_argument("MiniMax-H3 conditioner presentation row vectors disagree");
    if (presentation.qwen_mm_token_type_ids.size() != rows ||
        presentation.vision_selector.size() != rows)
        throw std::invalid_argument("MiniMax-H3 conditioner selector rows disagree");
    if (presentation.mrope_position_ids.size() != rows * 3U)
        throw std::invalid_argument("MiniMax-H3 conditioner MRoPE rows disagree");
    if (presentation.num_keyframes < 0 || presentation.num_keyframes > 2)
        throw std::invalid_argument("MiniMax-H3 conditioner keyframe count is invalid");
    const auto expected_scatter =
        static_cast<std::size_t>(presentation.num_keyframes) * kMiniMaxH3ConditionerMergedRows;
    if (presentation.vision_scatter_indices.size() != expected_scatter)
        throw std::invalid_argument("MiniMax-H3 conditioner compact vision row count is invalid");
}

void validate_qwen_types(const MiniMaxH3ConditionerPresentation& presentation) {
    const auto invalid = std::find_if(
        presentation.qwen_mm_token_type_ids.begin(), presentation.qwen_mm_token_type_ids.end(),
        [](int32_t type) { return type != kQwenTextType && type != kQwenImageType; });
    if (invalid != presentation.qwen_mm_token_type_ids.end())
        throw std::invalid_argument("MiniMax-H3 conditioner Qwen type must be text or image");
}

void validate_scatter_order(const MiniMaxH3ConditionerPresentation& presentation) {
    std::size_t compact_row = 0;
    for (int32_t row = 0; row < presentation.sequence_rows; ++row) {
        const int32_t selected = presentation.vision_selector[static_cast<std::size_t>(row)];
        if (selected != 0 && selected != 1)
            throw std::invalid_argument("MiniMax-H3 conditioner selector must contain zero or one");
        const bool qwen_image =
            presentation.qwen_mm_token_type_ids[static_cast<std::size_t>(row)] == kQwenImageType;
        if (static_cast<bool>(selected) != qwen_image)
            throw std::invalid_argument(
                "MiniMax-H3 conditioner selector and Qwen image rows disagree");
        if (selected == 1) {
            if (compact_row >= presentation.vision_scatter_indices.size() ||
                presentation.vision_scatter_indices[compact_row] != row)
                throw std::invalid_argument(
                    "MiniMax-H3 conditioner scatter rows are not strictly sequence ordered");
            if (presentation.h3_token_tags[static_cast<std::size_t>(row)] != kVisionTag)
                throw std::invalid_argument(
                    "MiniMax-H3 conditioner active vision row has a text tag");
            ++compact_row;
        }
    }
    if (compact_row != presentation.vision_scatter_indices.size())
        throw std::invalid_argument(
            "MiniMax-H3 conditioner scatter rows do not cover the selector");
}

void validate_compact_features(const std::vector<float>& values, std::size_t compact_rows,
                               int32_t feature_dim, const char* label) {
    if (values.size() != checked_feature_elements(compact_rows, feature_dim))
        throw std::invalid_argument(std::string("MiniMax-H3 ") + label +
                                    " compact feature shape is invalid");
    if (std::find_if(values.begin(), values.end(),
                     [](float value) { return !std::isfinite(value); }) != values.end())
        throw std::invalid_argument(std::string("MiniMax-H3 ") + label +
                                    " compact features must be finite");
}

std::vector<float> scatter_one_feature(const MiniMaxH3ConditionerPresentation& presentation,
                                       const std::vector<float>& compact, int32_t feature_dim) {
    std::vector<float> result(checked_feature_elements(
        static_cast<std::size_t>(presentation.sequence_rows), feature_dim));
    for (std::size_t compact_row = 0; compact_row < presentation.vision_scatter_indices.size();
         ++compact_row) {
        const auto sequence_row =
            static_cast<std::size_t>(presentation.vision_scatter_indices[compact_row]);
        const auto source = compact_row * static_cast<std::size_t>(feature_dim);
        const auto target = sequence_row * static_cast<std::size_t>(feature_dim);
        std::copy_n(compact.begin() + static_cast<std::ptrdiff_t>(source), feature_dim,
                    result.begin() + static_cast<std::ptrdiff_t>(target));
    }
    return result;
}

} // namespace

MiniMaxH3ConditionerPresentation minimax_h3_make_conditioner_presentation(
    const std::string& prompt, bool has_first_keyframe, bool has_last_keyframe,
    const MiniMaxH3ConditionerTokenizer& tokenize, const MiniMaxH3ConditionerTokenIds& token_ids) {
    if (!tokenize)
        throw std::invalid_argument("MiniMax-H3 conditioner requires a tokenizer callback");
    validate_special_token_ids(token_ids);

    MiniMaxH3ConditionerPresentation result;
    result.num_keyframes =
        static_cast<int32_t>(has_first_keyframe) + static_cast<int32_t>(has_last_keyframe);
    result.input_ids.reserve(static_cast<std::size_t>(result.num_keyframes) *
                             (kMiniMaxH3ConditionerMergedRows + 8));
    for (int32_t index = 0; index < result.num_keyframes; ++index) {
        const std::string label = "<Picture " + std::to_string(index + 1) + ">: ";
        append_text_tokens(result, tokenize(label), token_ids);
        append_vision_block(result, token_ids);
    }
    append_text_tokens(result, tokenize(prompt), token_ids);
    if (result.input_ids.empty())
        throw std::invalid_argument("MiniMax-H3 conditioner presentation must not be empty");
    result.sequence_rows = static_cast<int32_t>(result.input_ids.size());
    build_mrope_positions(result);
    return result;
}

std::vector<float> minimax_h3_preprocess_conditioner_keyframe(const MediaImageInput& keyframe) {
    validate_keyframe(keyframe);
    std::vector<float> output(static_cast<std::size_t>(kMiniMaxH3ConditionerPatchRows) *
                              kMiniMaxH3ConditionerPatchVector);
    constexpr int32_t kGroupRows = kMiniMaxH3ConditionerGridHeight / kMiniMaxH3ConditionerMergeSize;
    constexpr int32_t kGroupColumns =
        kMiniMaxH3ConditionerGridWidth / kMiniMaxH3ConditionerMergeSize;
    int32_t patch_row = 0;
    for (int32_t group_y = 0; group_y < kGroupRows; ++group_y) {
        for (int32_t group_x = 0; group_x < kGroupColumns; ++group_x) {
            for (int32_t merge_y = 0; merge_y < kMiniMaxH3ConditionerMergeSize; ++merge_y) {
                for (int32_t merge_x = 0; merge_x < kMiniMaxH3ConditionerMergeSize; ++merge_x) {
                    copy_patch(keyframe, group_y * kMiniMaxH3ConditionerMergeSize + merge_y,
                               group_x * kMiniMaxH3ConditionerMergeSize + merge_x, patch_row,
                               output);
                    ++patch_row;
                }
            }
        }
    }
    return output;
}

MiniMaxH3ConditionerVisionFeatures minimax_h3_scatter_vision_features(
    const MiniMaxH3ConditionerPresentation& presentation,
    const std::vector<float>& compact_vision_embeddings,
    const std::array<std::vector<float>, 3>& compact_deepstack_embeddings, int32_t feature_dim) {
    validate_scatter_shapes(presentation);
    validate_qwen_types(presentation);
    validate_scatter_order(presentation);
    const auto compact_rows = presentation.vision_scatter_indices.size();
    validate_compact_features(compact_vision_embeddings, compact_rows, feature_dim, "main vision");
    for (std::size_t level = 0; level < compact_deepstack_embeddings.size(); ++level) {
        const std::string label = "DeepStack " + std::to_string(level);
        validate_compact_features(compact_deepstack_embeddings[level], compact_rows, feature_dim,
                                  label.c_str());
    }

    MiniMaxH3ConditionerVisionFeatures result;
    result.sequence_rows = presentation.sequence_rows;
    result.feature_dim = feature_dim;
    result.vision_selector = presentation.vision_selector;
    result.vision_embeddings =
        scatter_one_feature(presentation, compact_vision_embeddings, feature_dim);
    for (std::size_t level = 0; level < result.deepstack_embeddings.size(); ++level) {
        result.deepstack_embeddings[level] =
            scatter_one_feature(presentation, compact_deepstack_embeddings[level], feature_dim);
    }
    return result;
}

} // namespace trtmc
