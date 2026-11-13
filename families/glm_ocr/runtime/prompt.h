/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/language.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::glm_ocr {

// The checkpoint's chat template for system, user and assistant messages. User messages
// carry text and images; each image renders as <|begin_of_image|><|image|><|end_of_image|>.
// The generation prompt (<|assistant|> and a newline) is appended.
struct RenderedPrompt {
    std::string text;
    std::vector<internal::ImageView> images; // in prompt order
};

RenderedPrompt render_prompt(const internal::ImagesTextToTextRequest& request);

// Token ids with every image placeholder expanded to its image tokens, and the multimodal
// rotary positions of each token (temporal, height, width), as Transformers' get_rope_index.
struct PromptLayout {
    std::vector<std::int32_t> ids;
    std::vector<std::array<std::int32_t, 3>> positions;
    std::vector<std::size_t> image_offsets; // first token row of each image
    std::int32_t next_position{0};          // position of the first generated token
};

struct ImageGrid {
    int grid_h{0}; // patch rows before merging
    int grid_w{0};
};

PromptLayout layout_prompt(const std::vector<std::int32_t>& ids, std::int32_t image_token_id,
                           const std::vector<ImageGrid>& images, int merge_size);

// Text rotary table: [rows, head_dim / 2]; pair i reads the position of its mrope section.
void text_rotary(const std::vector<std::array<std::int32_t, 3>>& positions, int head_dim,
                 float theta, const std::array<int, 3>& sections, std::vector<float>& cos,
                 std::vector<float>& sin);

} // namespace trtmc::glm_ocr
