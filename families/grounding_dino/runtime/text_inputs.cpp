/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/grounding_dino/runtime/text_inputs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace trtmc {
namespace {

constexpr std::int32_t kCls = 101;
constexpr std::int32_t kSep = 102;
constexpr std::int32_t kPeriod = 1012;
constexpr std::int32_t kQuestion = 1029;

constexpr double kTwoPi = 6.283185307179586;
constexpr double kTemperature = 10000.0;

} // namespace

bool grounding_dino_is_special_token(std::int32_t token) {
    return token == kCls || token == kSep || token == kPeriod || token == kQuestion;
}

void grounding_dino_phrase_structure(const std::vector<std::int32_t>& tokens,
                                     std::vector<std::uint8_t>& block_mask,
                                     std::vector<std::int32_t>& position_ids) {
    const auto length = static_cast<std::int32_t>(tokens.size());
    block_mask.assign(static_cast<std::size_t>(length) * length, 0);
    position_ids.assign(static_cast<std::size_t>(length), 0);
    if (length == 0)
        return;

    std::vector<std::int32_t> previous(static_cast<std::size_t>(length), -1);
    std::vector<std::int32_t> following(static_cast<std::size_t>(length), length);
    std::int32_t last = -1;
    for (std::int32_t i = 0; i < length; ++i) {
        if (grounding_dino_is_special_token(tokens[static_cast<std::size_t>(i)]))
            last = i;
        previous[static_cast<std::size_t>(i)] = last;
    }
    std::int32_t next = length;
    for (std::int32_t i = length - 1; i >= 0; --i) {
        if (grounding_dino_is_special_token(tokens[static_cast<std::size_t>(i)]))
            next = i;
        following[static_cast<std::size_t>(i)] = next;
    }

    std::vector<std::uint8_t> valid(static_cast<std::size_t>(length), 0);
    for (std::int32_t i = 0; i < length; ++i) {
        const std::int32_t value = following[static_cast<std::size_t>(i)];
        valid[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(value != 0 && value != length - 1 && value != length);
    }

    for (std::int32_t row = 0; row < length; ++row) {
        for (std::int32_t column = 0; column < length; ++column) {
            const bool same = following[static_cast<std::size_t>(row)] ==
                              following[static_cast<std::size_t>(column)];
            const bool keep = same && valid[static_cast<std::size_t>(column)] != 0;
            block_mask[static_cast<std::size_t>(row) * length + column] =
                static_cast<std::uint8_t>(keep || row == column);
        }
        if (valid[static_cast<std::size_t>(row)] != 0) {
            position_ids[static_cast<std::size_t>(row)] =
                std::max(0, row - previous[static_cast<std::size_t>(row)] - 1);
        }
    }
}

std::vector<float> grounding_dino_sine_positions(const std::vector<std::int32_t>& position_ids,
                                                 std::int32_t features) {
    if (features <= 0 || features % 2 != 0)
        throw std::invalid_argument("grounding_dino sine positions need an even feature count");
    std::vector<double> dim_t(static_cast<std::size_t>(features));
    for (std::int32_t k = 0; k < features; ++k) {
        dim_t[static_cast<std::size_t>(k)] =
            std::pow(kTemperature, 2.0 * std::floor(k / 2.0) / features);
    }
    std::vector<float> out(position_ids.size() * static_cast<std::size_t>(features));
    for (std::size_t i = 0; i < position_ids.size(); ++i) {
        const double value = static_cast<double>(position_ids[i]) * kTwoPi;
        float* row = out.data() + i * static_cast<std::size_t>(features);
        // Interleaved, not split halves: even slots take the sine of their own
        // frequency, odd slots the cosine of theirs.
        for (std::int32_t k = 0; k < features; k += 2) {
            row[k] = static_cast<float>(std::sin(value / dim_t[static_cast<std::size_t>(k)]));
            row[k + 1] =
                static_cast<float>(std::cos(value / dim_t[static_cast<std::size_t>(k + 1)]));
        }
    }
    return out;
}

GroundingDinoTextInputs grounding_dino_text_inputs(const std::vector<std::int32_t>& tokens,
                                                   std::int32_t length, std::int32_t features) {
    if (length <= 0)
        throw std::invalid_argument("grounding_dino prompt length must be positive");
    if (tokens.empty())
        throw std::invalid_argument("grounding_dino prompt tokenised to nothing");
    if (static_cast<std::int32_t>(tokens.size()) > length) {
        throw std::invalid_argument(
            "grounding_dino prompt is longer than the bundle's max_text_len");
    }

    const auto real = static_cast<std::int32_t>(tokens.size());
    std::vector<std::uint8_t> block;
    std::vector<std::int32_t> positions;
    grounding_dino_phrase_structure(tokens, block, positions);

    GroundingDinoTextInputs out;
    out.real_length = real;
    out.input_ids.assign(static_cast<std::size_t>(length), 0);
    out.token_type_ids.assign(static_cast<std::size_t>(length), 0);
    out.position_ids.assign(static_cast<std::size_t>(length), 0);
    out.key_mask.assign(static_cast<std::size_t>(length), kGroundingDinoMaskFill);
    out.self_attention_mask.assign(static_cast<std::size_t>(length) * length,
                                   kGroundingDinoMaskFill);

    for (std::int32_t i = 0; i < real; ++i) {
        out.input_ids[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(i)];
        out.position_ids[static_cast<std::size_t>(i)] = positions[static_cast<std::size_t>(i)];
        out.key_mask[static_cast<std::size_t>(i)] = 0.0F;
        for (std::int32_t j = 0; j < real; ++j) {
            if (block[static_cast<std::size_t>(i) * real + j] != 0)
                out.self_attention_mask[static_cast<std::size_t>(i) * length + j] = 0.0F;
        }
    }
    // A fully masked softmax row is NaN, so every padded position keeps its
    // own diagonal open. Those positions are discarded everywhere downstream.
    for (std::int32_t i = real; i < length; ++i)
        out.self_attention_mask[static_cast<std::size_t>(i) * length + i] = 0.0F;

    out.position_embedding = grounding_dino_sine_positions(out.position_ids, features);
    return out;
}

} // namespace trtmc
