/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::minicpm_ocr {

enum class PromptRole { System, User, Assistant };

// One message part: an image placeholder or a text segment.
struct PromptPart {
    bool image{false};
    std::string text;
};

struct PromptMessage {
    PromptRole role{PromptRole::User};
    std::vector<PromptPart> parts;
};

// Placeholder spellings and token counts of one preprocessed image.
struct ImageTokenLayout {
    int32_t overview_tokens{0};
    int32_t grid_rows{0};
    int32_t grid_cols{0};
    std::vector<int32_t> slice_tokens; // grid_rows * grid_cols entries, row-major
};

struct PromptTokens {
    std::string image_pad{"<|image_pad|>"};
    std::string image_start{"<image>"};
    std::string image_end{"</image>"};
    std::string slice_start{"<slice>"};
    std::string slice_end{"</slice>"};
    std::string image_id_start{"<image_id>"};
    std::string image_id_end{"</image_id>"};
};

// Python str.strip(): Unicode White_Space plus the ASCII information separators.
std::string python_strip(const std::string& text);

// Python bytes.decode("utf-8", errors="replace"): every maximal ill-formed
// subsequence becomes U+FFFD, so text cut off inside a character stays valid UTF-8.
std::string replace_invalid_utf8(const std::string& bytes);

// The checkpoint chat template for text and image parts (no tools), ending with
// the generation prompt. Each image part renders as one image-pad placeholder;
// parts of one message are joined with a newline, as the template does.
std::string render_chat(const std::vector<PromptMessage>& messages, bool enable_thinking,
                        const PromptTokens& tokens);

// Replaces the image-pad placeholders of a rendered prompt, in order, with the
// processor expansion: <image_id>k</image_id><image>pads</image> followed by the
// slice rows (<slice>pads</slice> per column, rows joined by a newline).
std::string expand_images(const std::string& rendered, const std::vector<ImageTokenLayout>& images,
                          bool use_image_id, const PromptTokens& tokens);

} // namespace trtmc::minicpm_ocr
