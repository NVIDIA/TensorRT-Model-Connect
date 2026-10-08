/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/clef/runtime/record.h"
#include "trtmc/task.h"

#include <array>

namespace trtmc::clef {
struct VisionFrame {
    int grid_height{0}, grid_width{0};
    bool video{false};
    double timestamp{0};
    std::vector<float> patches;
};
struct MediaRecord {
    std::vector<VisionFrame> frames;
    std::vector<std::int32_t> tokens;
};

MediaRecord preprocess_media(const ITokenizer& tokenizer, const StructuredDecisionRequest& request,
                             const Json& document, const Json& processor);
std::vector<std::array<int, 3>> media_positions(const Record& record, const MediaRecord& media,
                                                int image_token, int video_token);
void vision_positions(const VisionFrame& frame, const std::vector<char>& embedding, int width,
                      int heads, int side, std::vector<float>& positions, std::vector<float>& cos,
                      std::vector<float>& sin);
} // namespace trtmc::clef
