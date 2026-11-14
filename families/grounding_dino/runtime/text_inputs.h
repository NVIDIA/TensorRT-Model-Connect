/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// Everything the engine needs about the prompt, padded to max_text_len.
//
// The engine cannot derive these itself: the phrase structure depends on the
// token ids, and the ids are only known at inference time. The numpy twin of
// this derivation lives in families/grounding_dino/text_builder.py, which is
// what the parity harness scores; keep the two in step.
struct GroundingDinoTextInputs {
    std::vector<std::int32_t> input_ids;      // [length]
    std::vector<std::int32_t> token_type_ids; // [length]
    std::vector<std::int32_t> position_ids;   // [length]
    std::vector<float> self_attention_mask;   // [1, length, length], additive
    std::vector<float> key_mask;              // [1, 1, length], additive
    std::vector<float> position_embedding;    // [1, length, features]
    std::int32_t real_length{0};
};

// Saturating but fp16-safe. The reference uses the dtype minimum; any value
// this far below the row maximum drives the softmax term to zero either way.
constexpr float kGroundingDinoMaskFill = -10000.0F;

// [CLS], [SEP], "." and "?" - the phrase delimiters.
bool grounding_dino_is_special_token(std::int32_t token);

// The phrase block mask and per-phrase position ids, over `tokens` alone.
//
// Tokens sharing the same *next* special token form a phrase and may attend to
// each other; every token also attends to itself. Phrases whose delimiter sits
// at the very start or end are excluded, which is what isolates [CLS]/[SEP].
// Positions restart at each phrase: "a cat. a remote control." gives
// [0, 0, 1, 0, 0, 1, 2, 0, 0], not 0..8.
void grounding_dino_phrase_structure(const std::vector<std::int32_t>& tokens,
                                     std::vector<std::uint8_t>& block_mask,
                                     std::vector<std::int32_t>& position_ids);

// Interleaved sin/cos over one scalar per token, the enhancer's q/k positions.
std::vector<float> grounding_dino_sine_positions(const std::vector<std::int32_t>& position_ids,
                                                 std::int32_t features);

// Pad `tokens` to `length` and build every text input the engine takes.
GroundingDinoTextInputs grounding_dino_text_inputs(const std::vector<std::int32_t>& tokens,
                                                   std::int32_t length, std::int32_t features);

} // namespace trtmc
