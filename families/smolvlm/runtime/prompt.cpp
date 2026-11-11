/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Ports the image token expansion of SmolVLMProcessor (Transformers 5.2.0,
// processing_smolvlm.py) and renders the checkpoint chat template; see ../NOTICE.
#include "prompt.h"

#include <stdexcept>
#include <type_traits>
#include <variant>

namespace trtmc::smolvlm {
namespace {

constexpr const char* kImage = "<image>";
constexpr const char* kFakeImage = "<fake_token_around_image>";
constexpr const char* kGlobalImage = "<global-img>";

std::string repeated(const char* token, int count) {
    std::string result;
    for (int i = 0; i < count; ++i)
        result += token;
    return result;
}

// Jinja "capitalize" of the template's role names.
const char* role_label(internal::MessageRole role) {
    switch (role) {
    case internal::MessageRole::System:
        return "System";
    case internal::MessageRole::User:
        return "User";
    case internal::MessageRole::Assistant:
        return "Assistant";
    default:
        throw std::invalid_argument(
            "SmolVLM messages take the system, user and assistant roles only");
    }
}

} // namespace

std::string image_prompt(int rows, int cols, int image_seq_len) {
    if (rows < 0 || cols < 0 || (rows == 0) != (cols == 0) || image_seq_len <= 0)
        throw std::invalid_argument("Invalid SmolVLM image grid");
    const std::string tokens = repeated(kImage, image_seq_len);
    std::string result;
    if (rows > 0) {
        for (int row = 1; row <= rows; ++row) {
            for (int col = 1; col <= cols; ++col)
                result += std::string(kFakeImage) + "<row_" + std::to_string(row) + "_col_" +
                          std::to_string(col) + ">" + tokens;
            result += "\n";
        }
        result += "\n";
    }
    return result + kFakeImage + kGlobalImage + tokens + kFakeImage;
}

std::string render_chat(Span<const internal::ImagesTextMessage> messages,
                        const std::function<std::string(const internal::ImageView&)>& image_text) {
    if (messages.empty())
        throw std::invalid_argument("SmolVLM needs at least one message");
    std::string result = "<|im_start|>";
    for (const auto& message : messages) {
        if (message.parts.empty())
            throw std::invalid_argument("SmolVLM messages need at least one part");
        result += role_label(message.role);
        result += std::holds_alternative<internal::ImageView>(message.parts[0]) ? ":" : ": ";
        for (const auto& part : message.parts) {
            std::visit(
                [&](const auto& value) {
                    using Value = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<Value, internal::TextPartView>) {
                        // The processor would read a literal <image> as an image slot.
                        if (value.text.find(kImage) != std::string_view::npos)
                            throw std::invalid_argument(
                                "SmolVLM text parts must not contain the <image> token");
                        result += value.text;
                    } else if constexpr (std::is_same_v<Value, internal::ImageView>) {
                        result += image_text(value);
                    } else {
                        throw std::invalid_argument(
                            "SmolVLM messages take text and image parts only");
                    }
                },
                part);
        }
        result += "<end_of_utterance>\n";
    }
    return result + "Assistant:";
}

} // namespace trtmc::smolvlm
