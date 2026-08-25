/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/ref2va_conditioner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

constexpr int32_t kTextTag = 1;
constexpr int32_t kVisionTag = 0;
constexpr int32_t kQwenTextType = 0;
constexpr int32_t kQwenImageType = 1;
constexpr int32_t kQwenVideoType = 2;
constexpr int32_t kRgbChannels = 3;
constexpr int32_t kVocabularySize = 151936;
constexpr int32_t kMaxPresentationRows = 262144;
constexpr int32_t kPositionGridSide = 48;
constexpr int32_t kMinVisionPatchRows = 48 * 48;
constexpr int32_t kMaxImagePatchRows = 128 * 512;
constexpr int32_t kMaxVideoPatchRows = 4176;

struct VisionBlockSpec {
    std::size_t reference_index{0};
    MiniMaxH3Ref2VAVisionKind kind{MiniMaxH3Ref2VAVisionKind::kImage};
    int32_t modality_index{0};
    int32_t run_index{0};
    float timestamp_seconds{-1.0F};
    const float* first_frame{nullptr};
    const float* second_frame{nullptr};
    int32_t height{0};
    int32_t width{0};
};

std::size_t checked_product(std::initializer_list<int32_t> dimensions, const char* label) {
    std::size_t result = 1;
    for (const int32_t dimension : dimensions) {
        if (dimension <= 0 ||
            result > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension))
            throw std::invalid_argument(std::string(label) + " has invalid dimensions");
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

void validate_special_token_ids(const MiniMaxH3Ref2VATokenIds& token_ids) {
    std::array<int32_t, 4> ids = {token_ids.vision_start, token_ids.image_pad, token_ids.video_pad,
                                  token_ids.vision_end};
    if (std::any_of(ids.begin(), ids.end(),
                    [](int32_t id) { return id < 0 || id >= kVocabularySize; }))
        throw std::invalid_argument("MiniMax-H3 Ref2VA special token ID is out of range");
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::invalid_argument("MiniMax-H3 Ref2VA special token IDs must be distinct");
}

bool is_special_token(int32_t token, const MiniMaxH3Ref2VATokenIds& token_ids) {
    return token == token_ids.vision_start || token == token_ids.image_pad ||
           token == token_ids.video_pad || token == token_ids.vision_end;
}

void validate_text_tokens(const std::vector<int32_t>& tokens,
                          const MiniMaxH3Ref2VATokenIds& token_ids) {
    for (const int32_t token : tokens) {
        if (token < 0 || token >= kVocabularySize)
            throw std::invalid_argument("MiniMax-H3 Ref2VA tokenizer emitted an invalid ID");
        if (is_special_token(token, token_ids))
            throw std::invalid_argument("MiniMax-H3 Ref2VA text emitted a reserved vision token");
    }
}

void require_capacity(std::size_t current, std::size_t additional) {
    constexpr auto maximum = static_cast<std::size_t>(kMaxPresentationRows);
    if (current > maximum || additional > maximum - current)
        throw std::invalid_argument("MiniMax-H3 Ref2VA presentation exceeds 262144 rows");
}

void append_text(MiniMaxH3Ref2VAConditionerPresentation& result, const std::string& text,
                 const MiniMaxH3Ref2VATokenizer& tokenize,
                 const MiniMaxH3Ref2VATokenIds& token_ids) {
    const auto tokens = tokenize(text);
    validate_text_tokens(tokens, token_ids);
    require_capacity(result.input_ids.size(), tokens.size());
    result.input_ids.insert(result.input_ids.end(), tokens.begin(), tokens.end());
    result.h3_token_tags.insert(result.h3_token_tags.end(), tokens.size(), kTextTag);
    result.qwen_mm_token_type_ids.insert(result.qwen_mm_token_type_ids.end(), tokens.size(),
                                         kQwenTextType);
    result.vision_selector.insert(result.vision_selector.end(), tokens.size(), 0);
}

int64_t round_half_even_positive(double value) {
    const double floor_value = std::floor(value);
    const double fraction = value - floor_value;
    auto result = static_cast<int64_t>(floor_value);
    if (fraction > 0.5 || (fraction == 0.5 && result % 2 != 0))
        ++result;
    return result;
}

std::string timestamp_label(float seconds) {
    const int64_t tenths = round_half_even_positive(static_cast<double>(seconds) * 10.0);
    return "<" + std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " seconds>";
}

void validate_pixel_values(const std::vector<float>& pixels, std::size_t expected,
                           const char* label) {
    if (pixels.size() != expected)
        throw std::invalid_argument(std::string(label) + " buffer does not match its dimensions");
    const auto invalid = std::find_if(pixels.begin(), pixels.end(), [](float value) {
        return !std::isfinite(value) || value < 0.0F || value > 1.0F;
    });
    if (invalid != pixels.end())
        throw std::invalid_argument(std::string(label) + " contains invalid RGB values");
}

void validate_qwen_grid(const MiniMaxH3PreparedReference& reference, int32_t height,
                        int32_t width) {
    const int32_t grid_h = height / kMiniMaxH3Ref2VAPatchSize;
    const int32_t grid_w = width / kMiniMaxH3Ref2VAPatchSize;
    if (reference.qwen_grid_h != grid_h || reference.qwen_grid_w != grid_w ||
        reference.qwen_patch_rows != grid_h * grid_w)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA prepared Qwen patch metadata does not match the media");
}

void validate_image(const MiniMaxH3PreparedReference& reference) {
    const auto& image = reference.image;
    const auto expected =
        checked_product({image.height, image.width, kRgbChannels}, "MiniMax-H3 Ref2VA image");
    validate_pixel_values(image.pixels, expected, "MiniMax-H3 Ref2VA image");
    const int32_t short_edge = std::min(image.height, image.width);
    const double ratio = static_cast<double>(image.width) / image.height;
    if (short_edge != kMiniMaxH3Ref2VAImageShortEdge || ratio < 0.25 || ratio > 4.0 ||
        image.height % 32 != 0 || image.width % 32 != 0)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA image must have a 2048px short edge and 1:4..4:1 ratio");
    const int64_t patches = static_cast<int64_t>(image.height / kMiniMaxH3Ref2VAPatchSize) *
                            (image.width / kMiniMaxH3Ref2VAPatchSize);
    if (patches < kMinVisionPatchRows || patches > kMaxImagePatchRows)
        throw std::invalid_argument("MiniMax-H3 Ref2VA image patch grid is out of range");
    validate_qwen_grid(reference, image.height, image.width);
}

bool is_legal_video_grid(int32_t grid_h, int32_t grid_w) {
    const int32_t short_edge = std::min(grid_h, grid_w);
    const int32_t long_edge = std::max(grid_h, grid_w);
    const double ratio = static_cast<double>(grid_w) / grid_h;
    if (short_edge % 2 != 0 || long_edge % 2 != 0 || ratio < 0.25 || ratio > 4.0)
        return false;
    if (short_edge == 48 && long_edge >= 48 && long_edge <= 84)
        return true;
    constexpr double area = 48.0 * 84.0;
    const double lower =
        std::max({1.75, static_cast<double>((long_edge - 1) * (long_edge - 1)) / area,
                  area / static_cast<double>((short_edge + 1) * (short_edge + 1))});
    const double upper =
        std::min({4.0, static_cast<double>((long_edge + 1) * (long_edge + 1)) / area,
                  area / static_cast<double>((short_edge - 1) * (short_edge - 1))});
    return lower <= upper;
}

std::size_t validate_video(const MiniMaxH3PreparedReference& reference) {
    const auto& video = reference.video;
    if (video.fps != static_cast<float>(kMiniMaxH3Ref2VAReferenceFps))
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA prepared video must be normalized to 24 fps");
    const auto frame_elements =
        checked_product({video.height, video.width, kRgbChannels}, "MiniMax-H3 Ref2VA video frame");
    const auto expected = checked_product(
        {video.num_frames, video.height, video.width, kRgbChannels}, "MiniMax-H3 Ref2VA video");
    validate_pixel_values(video.pixels, expected, "MiniMax-H3 Ref2VA video");
    const int32_t grid_h = video.height / kMiniMaxH3Ref2VAPatchSize;
    const int32_t grid_w = video.width / kMiniMaxH3Ref2VAPatchSize;
    const int64_t patches = static_cast<int64_t>(grid_h) * grid_w;
    if (video.height % 32 != 0 || video.width % 32 != 0 || !is_legal_video_grid(grid_h, grid_w) ||
        patches < kMinVisionPatchRows || patches > kMaxVideoPatchRows)
        throw std::invalid_argument("MiniMax-H3 Ref2VA video must use a legal rounded 768p canvas");
    validate_qwen_grid(reference, video.height, video.width);
    return frame_elements;
}

void validate_reference_alignment(const std::vector<AudioVideoReference>& references,
                                  const std::vector<MiniMaxH3PreparedReference>& prepared) {
    if (references.empty() || references.size() != prepared.size())
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA prepared references must align with request order");
    for (std::size_t index = 0; index < references.size(); ++index) {
        const auto& source = references[index];
        const auto& material = prepared[index];
        if (material.reference_index != index || material.kind != source.kind)
            throw std::invalid_argument(
                "MiniMax-H3 Ref2VA prepared reference index or kind does not match the request");
        const bool source_audio =
            source.kind == AudioVideoReferenceKind::kAudio ||
            (source.kind == AudioVideoReferenceKind::kVideo && source.video.soundtrack.has_value());
        if (source_audio != material.audio.has_value())
            throw std::invalid_argument(
                "MiniMax-H3 Ref2VA prepared audio does not match the request reference");
    }
}

std::pair<std::array<int32_t, 2>, std::array<float, 2>> interpolation_axis(int32_t index,
                                                                           int32_t size) {
    const float source =
        size == 1 ? 0.0F
                  : (static_cast<float>(index) * static_cast<float>(kPositionGridSide - 1)) /
                        static_cast<float>(size - 1);
    const int32_t floor_value = static_cast<int32_t>(std::floor(source));
    std::array<int32_t, 2> taps = {
        std::clamp(floor_value, 0, kPositionGridSide - 1),
        std::clamp(floor_value + 1, 0, kPositionGridSide - 1),
    };
    std::array<float, 2> weights = {
        std::max(1.0F - std::abs(source - static_cast<float>(floor_value)), 0.0F),
        std::max(1.0F - std::abs(source - static_cast<float>(floor_value) - 1.0F), 0.0F),
    };
    return {taps, weights};
}

void append_position_binding(MiniMaxH3Ref2VAVisionInput& result, int32_t row, int32_t column) {
    const auto [h_taps, h_weights] = interpolation_axis(row, result.grid_h);
    const auto [w_taps, w_weights] = interpolation_axis(column, result.grid_w);
    for (int32_t h = 0; h < 2; ++h) {
        for (int32_t w = 0; w < 2; ++w) {
            result.position_indices.push_back(h_taps[static_cast<std::size_t>(h)] *
                                                  kPositionGridSide +
                                              w_taps[static_cast<std::size_t>(w)]);
            result.position_weights.push_back(h_weights[static_cast<std::size_t>(h)] *
                                              w_weights[static_cast<std::size_t>(w)]);
        }
    }
    result.vision_position_ids.push_back(row);
    result.vision_position_ids.push_back(column);
}

void make_position_bindings(MiniMaxH3Ref2VAVisionInput& result) {
    const int32_t group_rows = result.grid_h / kMiniMaxH3Ref2VASpatialMergeSize;
    const int32_t group_columns = result.grid_w / kMiniMaxH3Ref2VASpatialMergeSize;
    for (int32_t group_y = 0; group_y < group_rows; ++group_y) {
        for (int32_t group_x = 0; group_x < group_columns; ++group_x) {
            for (int32_t merge_y = 0; merge_y < kMiniMaxH3Ref2VASpatialMergeSize; ++merge_y) {
                for (int32_t merge_x = 0; merge_x < kMiniMaxH3Ref2VASpatialMergeSize; ++merge_x) {
                    append_position_binding(result,
                                            group_y * kMiniMaxH3Ref2VASpatialMergeSize + merge_y,
                                            group_x * kMiniMaxH3Ref2VASpatialMergeSize + merge_x);
                }
            }
        }
    }
}

void copy_patch_pair(const VisionBlockSpec& block, int32_t patch_y, int32_t patch_x,
                     int32_t patch_row, std::vector<float>& output) {
    const auto row_base = static_cast<std::size_t>(patch_row) * kMiniMaxH3Ref2VAPatchVectorSize;
    const std::array<const float*, 2> frames = {block.first_frame, block.second_frame};
    for (int32_t channel = 0; channel < kRgbChannels; ++channel) {
        for (int32_t temporal = 0; temporal < kMiniMaxH3Ref2VATemporalPatchSize; ++temporal) {
            for (int32_t y = 0; y < kMiniMaxH3Ref2VAPatchSize; ++y) {
                for (int32_t x = 0; x < kMiniMaxH3Ref2VAPatchSize; ++x) {
                    const auto source =
                        (static_cast<std::size_t>(patch_y * kMiniMaxH3Ref2VAPatchSize + y) *
                             block.width +
                         patch_x * kMiniMaxH3Ref2VAPatchSize + x) *
                            kRgbChannels +
                        channel;
                    const auto target =
                        (((static_cast<std::size_t>(channel) * kMiniMaxH3Ref2VATemporalPatchSize +
                           temporal) *
                              kMiniMaxH3Ref2VAPatchSize +
                          y) *
                             kMiniMaxH3Ref2VAPatchSize +
                         x);
                    output[row_base + target] =
                        (frames[static_cast<std::size_t>(temporal)][source] - 0.5F) / 0.5F;
                }
            }
        }
    }
}

MiniMaxH3Ref2VAVisionInput patchify_vision_block(const VisionBlockSpec& block) {
    MiniMaxH3Ref2VAVisionInput result;
    result.reference_index = block.reference_index;
    result.kind = block.kind;
    result.modality_index = block.modality_index;
    result.run_index = block.run_index;
    result.timestamp_seconds = block.timestamp_seconds;
    result.grid_h = block.height / kMiniMaxH3Ref2VAPatchSize;
    result.grid_w = block.width / kMiniMaxH3Ref2VAPatchSize;
    const int32_t patches = result.grid_h * result.grid_w;
    result.pixel_values.resize(static_cast<std::size_t>(patches) * kMiniMaxH3Ref2VAPatchVectorSize);
    int32_t patch_row = 0;
    for (int32_t group_y = 0; group_y < result.grid_h / 2; ++group_y) {
        for (int32_t group_x = 0; group_x < result.grid_w / 2; ++group_x) {
            for (int32_t merge_y = 0; merge_y < 2; ++merge_y) {
                for (int32_t merge_x = 0; merge_x < 2; ++merge_x) {
                    copy_patch_pair(block, group_y * 2 + merge_y, group_x * 2 + merge_x,
                                    patch_row++, result.pixel_values);
                }
            }
        }
    }
    make_position_bindings(result);
    return result;
}

std::vector<int32_t> sample_video_indices(int32_t num_frames) {
    constexpr double stride =
        static_cast<double>(kMiniMaxH3Ref2VAReferenceFps) / kMiniMaxH3Ref2VAConditionerFps;
    std::vector<int32_t> result;
    double cursor = 0.0;
    while (round_half_even_positive(cursor) < num_frames) {
        const auto index = static_cast<int32_t>(round_half_even_positive(cursor));
        if (result.empty() || index > result.back())
            result.push_back(index);
        cursor += stride;
    }
    if (result.size() < static_cast<std::size_t>(kMiniMaxH3Ref2VATemporalPatchSize))
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA video is too short for one 2 fps temporal pair");
    return result;
}

std::vector<MiniMaxH3Ref2VAVisionInput>
make_video_inputs(const MiniMaxH3PreparedReference& reference, int32_t modality_index) {
    const std::size_t frame_elements = validate_video(reference);
    auto indices = sample_video_indices(reference.video.num_frames);
    std::vector<float> timestamps(indices.size());
    for (std::size_t index = 0; index < timestamps.size(); ++index)
        timestamps[index] = static_cast<float>(index) / kMiniMaxH3Ref2VAConditionerFps;
    if (indices.size() % 2 != 0) {
        indices.push_back(indices.back());
        timestamps.push_back(timestamps.back());
    }
    std::vector<MiniMaxH3Ref2VAVisionInput> result;
    result.reserve(indices.size() / 2);
    for (std::size_t index = 0; index < indices.size(); index += 2) {
        const auto* first = reference.video.pixels.data() +
                            static_cast<std::size_t>(indices[index]) * frame_elements;
        const auto* second = reference.video.pixels.data() +
                             static_cast<std::size_t>(indices[index + 1]) * frame_elements;
        const VisionBlockSpec block{reference.reference_index,
                                    MiniMaxH3Ref2VAVisionKind::kVideo,
                                    modality_index,
                                    static_cast<int32_t>(index / 2),
                                    (timestamps[index] + timestamps[index + 1]) / 2.0F,
                                    first,
                                    second,
                                    reference.video.height,
                                    reference.video.width};
        result.push_back(patchify_vision_block(block));
    }
    return result;
}

MiniMaxH3Ref2VAVisionInput make_image_input(const MiniMaxH3PreparedReference& reference,
                                            int32_t modality_index) {
    validate_image(reference);
    const VisionBlockSpec block{reference.reference_index,
                                MiniMaxH3Ref2VAVisionKind::kImage,
                                modality_index,
                                0,
                                -1.0F,
                                reference.image.pixels.data(),
                                reference.image.pixels.data(),
                                reference.image.height,
                                reference.image.width};
    return patchify_vision_block(block);
}

int32_t merged_rows(const MiniMaxH3Ref2VAVisionInput& input) {
    return input.grid_h * input.grid_w /
           (kMiniMaxH3Ref2VASpatialMergeSize * kMiniMaxH3Ref2VASpatialMergeSize);
}

void append_vision(MiniMaxH3Ref2VAConditionerPresentation& result, MiniMaxH3Ref2VAVisionInput input,
                   const MiniMaxH3Ref2VATokenIds& token_ids) {
    const int32_t rows = merged_rows(input);
    require_capacity(result.input_ids.size(), static_cast<std::size_t>(rows) + 2U);
    result.input_ids.push_back(token_ids.vision_start);
    result.h3_token_tags.push_back(kVisionTag);
    result.qwen_mm_token_type_ids.push_back(kQwenTextType);
    result.vision_selector.push_back(0);

    const int32_t sequence_begin = static_cast<int32_t>(result.input_ids.size());
    const int32_t compact_begin = static_cast<int32_t>(result.vision_scatter_indices.size());
    const int32_t pad =
        input.kind == MiniMaxH3Ref2VAVisionKind::kImage ? token_ids.image_pad : token_ids.video_pad;
    const int32_t qwen_type =
        input.kind == MiniMaxH3Ref2VAVisionKind::kImage ? kQwenImageType : kQwenVideoType;
    result.input_ids.insert(result.input_ids.end(), rows, pad);
    result.h3_token_tags.insert(result.h3_token_tags.end(), rows, kVisionTag);
    result.qwen_mm_token_type_ids.insert(result.qwen_mm_token_type_ids.end(), rows, qwen_type);
    result.vision_selector.insert(result.vision_selector.end(), rows, 1);
    for (int32_t row = 0; row < rows; ++row)
        result.vision_scatter_indices.push_back(sequence_begin + row);

    result.input_ids.push_back(token_ids.vision_end);
    result.h3_token_tags.push_back(kVisionTag);
    result.qwen_mm_token_type_ids.push_back(kQwenTextType);
    result.vision_selector.push_back(0);
    result.vision_run_lengths.push_back(rows);
    result.vision_run_reference_ids.push_back(static_cast<int32_t>(input.reference_index));
    result.vision_scatter.push_back({input.reference_index, input.kind, input.run_index,
                                     input.grid_t, input.grid_h, input.grid_w, compact_begin, rows,
                                     sequence_begin});
    result.vision_inputs.push_back(std::move(input));
}

void append_image_reference(MiniMaxH3Ref2VAConditionerPresentation& result,
                            const MiniMaxH3PreparedReference& reference, int32_t image_index,
                            const MiniMaxH3Ref2VATokenizer& tokenize,
                            const MiniMaxH3Ref2VATokenIds& token_ids) {
    append_text(result, "<Picture " + std::to_string(image_index) + ">: ", tokenize, token_ids);
    append_vision(result, make_image_input(reference, image_index), token_ids);
}

void append_video_reference(MiniMaxH3Ref2VAConditionerPresentation& result,
                            const MiniMaxH3PreparedReference& reference, int32_t video_index,
                            const MiniMaxH3Ref2VATokenizer& tokenize,
                            const MiniMaxH3Ref2VATokenIds& token_ids) {
    append_text(result, "<Video " + std::to_string(video_index) + ">: ", tokenize, token_ids);
    auto inputs = make_video_inputs(reference, video_index);
    for (auto& input : inputs) {
        append_text(result, timestamp_label(input.timestamp_seconds), tokenize, token_ids);
        append_vision(result, std::move(input), token_ids);
    }
}

void append_audio_label(MiniMaxH3Ref2VAConditionerPresentation& result,
                        const MiniMaxH3PreparedReference& reference, int32_t audio_index,
                        const MiniMaxH3Ref2VATokenizer& tokenize,
                        const MiniMaxH3Ref2VATokenIds& token_ids) {
    append_text(result, "<Audio " + std::to_string(audio_index) + ">: ", tokenize, token_ids);
    result.audio_labels.push_back({reference.reference_index, audio_index,
                                   reference.kind == AudioVideoReferenceKind::kVideo});
}

struct ModalityCounts {
    int32_t images{0};
    int32_t videos{0};
    int32_t audio{0};
};

void append_reference(MiniMaxH3Ref2VAConditionerPresentation& result,
                      const MiniMaxH3PreparedReference& reference, ModalityCounts& counts,
                      const MiniMaxH3Ref2VATokenizer& tokenize,
                      const MiniMaxH3Ref2VATokenIds& token_ids) {
    if (reference.audio.has_value())
        append_audio_label(result, reference, ++counts.audio, tokenize, token_ids);
    if (reference.kind == AudioVideoReferenceKind::kImage) {
        append_image_reference(result, reference, ++counts.images, tokenize, token_ids);
    } else if (reference.kind == AudioVideoReferenceKind::kVideo) {
        append_video_reference(result, reference, ++counts.videos, tokenize, token_ids);
    }
}

void set_mrope(MiniMaxH3Ref2VAConditionerPresentation& result, std::size_t row, int32_t temporal,
               int32_t height, int32_t width) {
    const auto rows = static_cast<std::size_t>(result.sequence_rows);
    result.mrope_position_ids[row] = temporal;
    result.mrope_position_ids[rows + row] = height;
    result.mrope_position_ids[2U * rows + row] = width;
}

void fill_text_mrope(MiniMaxH3Ref2VAConditionerPresentation& result, std::size_t begin,
                     std::size_t end, int32_t& current) {
    for (std::size_t row = begin; row < end; ++row) {
        const int32_t position = current + static_cast<int32_t>(row - begin);
        set_mrope(result, row, position, position, position);
    }
    current += static_cast<int32_t>(end - begin);
}

void fill_vision_mrope(MiniMaxH3Ref2VAConditionerPresentation& result, std::size_t begin,
                       const MiniMaxH3Ref2VAVisionScatter& run, int32_t& current) {
    const int32_t merged_h = run.grid_h / kMiniMaxH3Ref2VASpatialMergeSize;
    const int32_t merged_w = run.grid_w / kMiniMaxH3Ref2VASpatialMergeSize;
    for (int32_t row = 0; row < run.compact_row_count; ++row)
        set_mrope(result, begin + static_cast<std::size_t>(row), current, current + row / merged_w,
                  current + row % merged_w);
    current += std::max(merged_h, merged_w);
}

void build_mrope(MiniMaxH3Ref2VAConditionerPresentation& result) {
    result.mrope_position_ids.assign(static_cast<std::size_t>(result.sequence_rows) * 3U, 0);
    std::size_t cursor = 0;
    std::size_t vision_run = 0;
    int32_t current = 0;
    while (cursor < result.qwen_mm_token_type_ids.size()) {
        const int32_t modality = result.qwen_mm_token_type_ids[cursor];
        std::size_t end = cursor + 1U;
        while (end < result.qwen_mm_token_type_ids.size() &&
               result.qwen_mm_token_type_ids[end] == modality)
            ++end;
        if (modality == kQwenTextType) {
            fill_text_mrope(result, cursor, end, current);
        } else {
            if (vision_run >= result.vision_scatter.size() ||
                static_cast<int32_t>(end - cursor) !=
                    result.vision_scatter[vision_run].compact_row_count)
                throw std::logic_error("MiniMax-H3 Ref2VA MRoPE vision runs are inconsistent");
            fill_vision_mrope(result, cursor, result.vision_scatter[vision_run++], current);
        }
        cursor = end;
    }
    if (vision_run != result.vision_scatter.size())
        throw std::logic_error("MiniMax-H3 Ref2VA MRoPE omitted a vision run");
    result.next_mrope_position = current;
    result.mrope_position_delta = current - result.sequence_rows;
}

} // namespace

MiniMaxH3Ref2VAConditionerPresentation minimax_h3_build_ref2va_conditioner_presentation(
    const std::string& prompt, const std::vector<AudioVideoReference>& references,
    const std::vector<MiniMaxH3PreparedReference>& prepared_references,
    const MiniMaxH3Ref2VATokenizer& tokenize, const MiniMaxH3Ref2VATokenIds& token_ids) {
    if (!tokenize)
        throw std::invalid_argument("MiniMax-H3 Ref2VA tokenizer is required");
    validate_special_token_ids(token_ids);
    validate_reference_alignment(references, prepared_references);

    MiniMaxH3Ref2VAConditionerPresentation result;
    ModalityCounts counts;
    for (const auto& reference : prepared_references)
        append_reference(result, reference, counts, tokenize, token_ids);
    if (result.vision_inputs.empty())
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA conditioner requires at least one image or video reference");
    append_text(result, prompt, tokenize, token_ids);
    if (result.input_ids.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA presentation must not be empty");
    result.sequence_rows = static_cast<int32_t>(result.input_ids.size());
    build_mrope(result);
    return result;
}

} // namespace trtmc
