/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/qwen_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kPatch = 16;
constexpr int32_t kMerge = 2;
constexpr int32_t kHead = 72;
constexpr int32_t kLearnedSide = 48;
constexpr int32_t kMaxPatches = 65536; // Released processor max_pixels / 16^2.

int32_t validate_grid(QwenImageGrid grid) {
    const int64_t rows = static_cast<int64_t>(grid.height) * grid.width;
    if (grid.height <= 0 || grid.width <= 0 || grid.height % kMerge || grid.width % kMerge ||
        rows > kMaxPatches)
        throw std::invalid_argument(
            "MiniMax-H3 Qwen grid must be 2-aligned and within the released pixel limit");
    return static_cast<int32_t>(rows);
}

struct InterpolationAxis {
    std::array<int32_t, 2> taps;
    std::array<float, 2> weights;
};

InterpolationAxis interpolation_axis(int32_t position, int32_t size) {
    // Match the pinned vectorized helper's FP32 multiply THEN divide, not a
    // float64 linspace or a pre-rounded BF16 interpolation of learned values.
    const float source = static_cast<float>(position) * (kLearnedSide - 1) /
                         static_cast<float>(std::max(1, size - 1));
    const auto floor = static_cast<int32_t>(std::floor(source));
    const float fraction = source - static_cast<float>(floor);
    return {{{std::min(floor, kLearnedSide - 1), std::min(floor + 1, kLearnedSide - 1)}},
            {{std::max(0.0F, 1.0F - std::abs(fraction)),
              std::max(0.0F, 1.0F - std::abs(fraction - 1.0F))}}};
}

} // namespace

QwenVisionMetadata make_qwen_vision_metadata(QwenImageGrid grid) {
    const auto rows = static_cast<std::size_t>(validate_grid(grid));
    QwenVisionMetadata result;
    result.grid = grid;
    result.position_indices.resize(rows * 4);
    result.position_weights.resize(rows * 4);
    result.rotary_cos.resize(rows * kHead);
    result.rotary_sin.resize(rows * kHead);
    std::array<float, kHead / 4> inverse{};
    for (std::size_t index = 0; index < inverse.size(); ++index)
        inverse[index] = 1.0F / std::pow(10000.0F, static_cast<float>(2 * index) / (kHead / 2));
    std::size_t row = 0;
    for (int32_t block_y = 0; block_y < grid.height / kMerge; ++block_y) {
        for (int32_t block_x = 0; block_x < grid.width / kMerge; ++block_x) {
            for (int32_t inner_y = 0; inner_y < kMerge; ++inner_y) {
                for (int32_t inner_x = 0; inner_x < kMerge; ++inner_x, ++row) {
                    const int32_t y = block_y * kMerge + inner_y;
                    const int32_t x = block_x * kMerge + inner_x;
                    const auto h = interpolation_axis(y, grid.height);
                    const auto w = interpolation_axis(x, grid.width);
                    for (int32_t h_tap = 0; h_tap < 2; ++h_tap) {
                        for (int32_t w_tap = 0; w_tap < 2; ++w_tap) {
                            const auto tap = static_cast<std::size_t>(h_tap * 2 + w_tap);
                            result.position_indices[row * 4 + tap] =
                                h.taps[h_tap] * kLearnedSide + w.taps[w_tap];
                            result.position_weights[row * 4 + tap] =
                                h.weights[h_tap] * w.weights[w_tap];
                        }
                    }
                    for (int32_t axis = 0; axis < 2; ++axis) {
                        for (std::size_t index = 0; index < inverse.size(); ++index) {
                            const float phase =
                                static_cast<float>(axis == 0 ? y : x) * inverse[index];
                            const auto column =
                                static_cast<std::size_t>(axis) * inverse.size() + index;
                            const auto offset = row * kHead + column;
                            result.rotary_cos[offset] = result.rotary_cos[offset + kHead / 2] =
                                std::cos(phase);
                            result.rotary_sin[offset] = result.rotary_sin[offset + kHead / 2] =
                                std::sin(phase);
                        }
                    }
                }
            }
        }
    }
    return result;
}

std::vector<float> pack_resized_qwen_image(const uint8_t* rgb, std::size_t bytes, int32_t height,
                                           int32_t width) {
    if (height <= 0 || width <= 0 || height % (kPatch * kMerge) || width % (kPatch * kMerge))
        throw std::invalid_argument(
            "MiniMax-H3 Qwen pixels require exact resized 32-aligned geometry");
    const auto patches = static_cast<std::size_t>(validate_grid({height / kPatch, width / kPatch}));
    const auto expected_bytes = static_cast<std::size_t>(height) * width * 3;
    if (rgb == nullptr || bytes != expected_bytes)
        throw std::invalid_argument(
            "MiniMax-H3 Qwen input must be an exact contiguous RGB uint8 image");
    std::array<float, 256> normalization{};
    for (std::size_t value = 0; value < normalization.size(); ++value)
        normalization[value] = (static_cast<float>(value) - 127.5F) / 127.5F;
    // Transformers 5.15 TorchvisionBackend fuses rescale+normalize by scaling
    // mean/std to 127.5, then subtracting and dividing raw FP32 byte values.
    constexpr std::size_t patch_dim = 3 * 2 * kPatch * kPatch;
    std::vector<float> result(patches * patch_dim);
    std::size_t row = 0;
    for (int32_t block_y = 0; block_y < height / (kPatch * kMerge); ++block_y) {
        for (int32_t block_x = 0; block_x < width / (kPatch * kMerge); ++block_x) {
            for (int32_t inner_y = 0; inner_y < kMerge; ++inner_y) {
                for (int32_t inner_x = 0; inner_x < kMerge; ++inner_x, ++row) {
                    for (int32_t channel = 0; channel < 3; ++channel) {
                        for (int32_t temporal = 0; temporal < 2; ++temporal) {
                            for (int32_t y = 0; y < kPatch; ++y) {
                                for (int32_t x = 0; x < kPatch; ++x) {
                                    const auto source_y = (block_y * kMerge + inner_y) * kPatch + y;
                                    const auto source_x = (block_x * kMerge + inner_x) * kPatch + x;
                                    const auto source =
                                        (static_cast<std::size_t>(source_y) * width + source_x) *
                                            3 +
                                        channel;
                                    const auto target =
                                        row * patch_dim +
                                        ((channel * 2 + temporal) * kPatch + y) * kPatch + x;
                                    result[target] = normalization[rgb[source]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

QwenImagePresentation make_qwen_image_presentation(const ITokenizer& tokenizer,
                                                   const std::string& prompt,
                                                   const std::vector<QwenImageGrid>& images,
                                                   int32_t max_rows) {
    if (images.empty() || images.size() > 9 || max_rows < 1)
        throw std::invalid_argument(
            "MiniMax-H3 image Ref2VA requires 1..9 references and a positive conditioner profile");
    for (const auto grid : images)
        (void)validate_grid(grid);
    const int32_t start = tokenizer.id_for_token("<|vision_start|>");
    const int32_t end = tokenizer.id_for_token("<|vision_end|>");
    const int32_t pad = tokenizer.id_for_token("<|image_pad|>");
    if (start != 151652 || end != 151653 || pad != 151655)
        throw std::invalid_argument(
            "MiniMax-H3 reference tokenizer special ids do not match its pinned Qwen checkpoint");
    QwenImagePresentation result;
    std::array<std::vector<int32_t>, 3> axes;
    int64_t next_position = 0;
    auto emit = [&](int32_t token, int32_t tag, bool visual, std::array<int64_t, 3> position) {
        if (result.input_ids.size() >= static_cast<std::size_t>(max_rows))
            throw std::length_error("MiniMax-H3 reference presentation exceeds its conditioner "
                                    "profile; no truncation or reference downsampling is allowed");
        if (token < 0 || token >= 151936)
            throw std::invalid_argument(
                "MiniMax-H3 reference token is outside the checkpoint vocabulary");
        if (visual)
            result.visual_rows.push_back(static_cast<int32_t>(result.input_ids.size()));
        result.input_ids.push_back(token);
        result.h3_token_tags.push_back(tag);
        result.visual_mask.push_back(visual ? 1 : 0);
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            if (position[axis] < 0 || position[axis] > std::numeric_limits<int32_t>::max())
                throw std::overflow_error("MiniMax-H3 Qwen position exceeds native int32 bounds");
            axes[axis].push_back(static_cast<int32_t>(position[axis]));
        }
    };
    auto text = [&](const std::string& value) {
        for (const auto token : tokenizer.encode(value)) {
            emit(token, 1, false, {next_position, next_position, next_position});
            ++next_position;
        }
    };
    for (std::size_t index = 0; index < images.size(); ++index) {
        text("<Picture " + std::to_string(index + 1) + ">: ");
        emit(start, 0, false, {next_position, next_position, next_position});
        ++next_position;
        const auto grid = images[index];
        for (int32_t y = 0; y < grid.height / kMerge; ++y)
            for (int32_t x = 0; x < grid.width / kMerge; ++x)
                emit(pad, 0, true, {next_position, next_position + y, next_position + x});
        next_position += std::max(grid.height, grid.width) / kMerge;
        emit(end, 0, false, {next_position, next_position, next_position});
        ++next_position;
    }
    text(prompt);
    result.rope_delta =
        static_cast<int32_t>(next_position - static_cast<int64_t>(result.input_ids.size()));
    result.position_ids.reserve(result.input_ids.size() * axes.size());
    for (const auto& axis : axes)
        result.position_ids.insert(result.position_ids.end(), axis.begin(), axis.end());
    return result;
}

} // namespace trtmc::minimax_h3
