/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/language.h"

#include <string>
#include <string_view>
#include <vector>

namespace trtmc::gemma3n {

// Python's str.strip(): removes leading and trailing Unicode whitespace.
std::string strip(std::string_view text);

struct Prompt {
    std::string text;
    std::vector<internal::ImageView> images;
};

// The Gemma 3n chat template with the generation prompt, and every image part
// replaced by `image_sequence` as the reference processor expands it. Throws
// on what the template or processor would reject.
Prompt render_prompt(Span<const internal::ImagesTextMessage> messages,
                     std::string_view image_sequence);

// "\n\n<start_of_image>" + soft_tokens x "<image_soft_token>" + "<end_of_image>\n\n".
std::string image_sequence(int soft_tokens);

} // namespace trtmc::gemma3n
