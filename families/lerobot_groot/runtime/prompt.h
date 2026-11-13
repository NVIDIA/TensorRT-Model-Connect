/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::lerobot_groot {

// LeRobot's instruction when none is given.
inline constexpr std::string_view kDefaultInstruction = "Perform the task.";

// Bytes of instruction text allowed per instruction token. Longer text is refused before it is
// tokenized, as the byte-level merges take time quadratic in the length of a word; natural
// instructions have four to six bytes a token.
inline constexpr std::size_t kInstructionBytesPerToken = 16;

// prepare_n1_7_language_batch as LeRobot's groot preprocessor applies it twice: the default
// instruction for empty text, then, when `formalize`, Python's lower() and a filter keeping
// isalnum(), isspace() and '_'; text that formalises to nothing becomes the default instruction
// verbatim. Only ASCII text is accepted, formalised or not: other text would also need Python's
// Unicode lower() and isalnum() and the tokenizer's NFC normalization, and is rejected rather
// than approximated.
std::string formalize_instruction(std::string_view text, bool formalize);

// Token ids of the Qwen3-VL chat template's fixed pieces.
struct TemplateTokens {
    std::int32_t im_start{0};
    std::int32_t im_end{0};
    std::int32_t vision_start{0};
    std::int32_t vision_end{0};
    std::int32_t image_pad{0};
    std::vector<std::int32_t> user;    // "user\n"
    std::vector<std::int32_t> newline; // "\n" after the end of turn
};

// What the condition engine reads for one prompt of S tokens.
struct Prompt {
    std::vector<std::int32_t> token_ids;   // [S]
    std::vector<std::int32_t> image_index; // [S]: image token number, 0 for text
    std::vector<std::int32_t> image_mask;  // [S]: 1 for image tokens
    std::vector<std::int32_t> positions;   // [3, S]: temporal, height and width rotary positions
};

// "<|im_start|>user\n", then per camera "<|vision_start|>" image pads "<|vision_end|>", the
// instruction tokens and "<|im_end|>\n", with the multimodal rotary positions get_rope_index
// assigns: text counts up on all three rows; an image's tokens share the temporal position and
// count rows and columns of its `grid` x `grid` merged patches, then the count moves on by `grid`.
Prompt build_prompt(const std::vector<std::int32_t>& instruction, std::int32_t cameras,
                    std::int32_t grid, const TemplateTokens& tokens);

} // namespace trtmc::lerobot_groot
