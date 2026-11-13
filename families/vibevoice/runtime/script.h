/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::vibevoice {

struct Turn {
    std::int32_t speaker{0};
    std::string text;
};

// A transcript is either plain text (one turn of speaker 0) or a script whose
// every non-empty line reads ``Speaker <id>: <text>``. Speaker ids written
// from 1 are renumbered from 0, as the model's original script parser does.
std::vector<Turn> parse_script(std::string_view text);

struct PromptFrame {
    std::string system_prompt;
    std::string voice_input_header;
    std::string text_input_header;
    std::string speech_output_header;
    std::string audio_bos_token;
    std::string audio_eos_token;
    std::string audio_token;
};

// The text the reference chat template renders for ``turns``, with
// ``voice_frames[i]`` placeholder tokens for the voice prompt of speaker i.
std::string render_prompt(const PromptFrame& frame, const std::vector<Turn>& turns,
                          const std::vector<std::int64_t>& voice_frames);

} // namespace trtmc::vibevoice
