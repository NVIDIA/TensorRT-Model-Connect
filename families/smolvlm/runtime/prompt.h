/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "media.h"
#include "trtmc/internal/language.h"

#include <functional>
#include <string>
#include <vector>

namespace trtmc::smolvlm {

// The text that stands for one processed image: one
// <fake_token_around_image><row_r_col_c><image>*N group per tile and a newline
// per tile row, then the global view; an unsplit image is the global view only.
std::string image_prompt(int rows, int cols, int image_seq_len);

// Render the checkpoint chat template with the generation prompt:
// "<|im_start|>" + per message "Role:" (an image first) or "Role: ", the parts
// in order and "<end_of_utterance>\n", then "Assistant:". `image_text` returns
// the expansion of each image part, in prompt order.
std::string render_chat(Span<const internal::ImagesTextMessage> messages,
                        const std::function<std::string(const internal::ImageView&)>& image_text);

} // namespace trtmc::smolvlm
