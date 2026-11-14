/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/minicpmv4_7/runtime/image_preprocess.h"

#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::minicpmv4_7 {

inline constexpr const char* kImagePlaceholder = "<|image_pad|>";

enum class Role { System, User, Assistant };

// One conversation turn after the provider template's content rendering: text parts and image
// placeholders joined by newlines.
struct ChatMessage {
    Role role{Role::User};
    std::string content;
};

// Template render_content for structured content: text parts and image placeholders,
// newline-joined.
std::string join_parts(const std::vector<std::string>& parts);

struct TokenIds {
    int32_t image_pad{-1}, image_start{-1}, image_end{-1}, slice_start{-1}, slice_end{-1};
    int32_t image_id_start{-1}, image_id_end{-1}, newline{-1};
};

// Python str.strip() / Jinja trim: Unicode White_Space plus the ASCII information separators.
std::string strip(const std::string& text);

// Native formatter for the recipe extracted from the checkpoint chat template at build time.
std::string format_conversation(const nlohmann::json& recipe,
                                const std::vector<ChatMessage>& messages);

// Replace the i-th image placeholder with the processor expansion of image i (image id prefix,
// thumbnail tokens, newline-separated rows of slice tokens).
std::string expand_images(const std::string& prompt, const std::vector<ImageCrops>& images,
                          int32_t merge_factor);

// Canvas M-RoPE (T, H, W) coordinates for each prompt token and the first decode coordinate.
// `crop_grids` lists the (grid_h, grid_w) patch grid of every crop in prompt order.
struct CanvasPositions {
    std::vector<std::array<int32_t, 3>> positions;
    int32_t decode_position{0};
};
CanvasPositions canvas_positions(const std::vector<int32_t>& ids, const TokenIds& tokens,
                                 const std::vector<std::array<int32_t, 2>>& crop_grids,
                                 int32_t merge_factor);

// torch.linspace(0, canvas - 1, count) in FP32 rounded half to even, plus `offset`.
std::vector<int32_t> canvas_coordinates(int32_t canvas, int32_t count, int32_t offset);

} // namespace trtmc::minicpmv4_7
