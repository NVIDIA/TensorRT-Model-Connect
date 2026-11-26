/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/vibevoice_asr/runtime/runtime_config.h"
#include "families/vibevoice_asr/runtime/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::vibevoice_asr {

struct Prompt {
    std::vector<std::int32_t> ids;
    std::int32_t speech_offset{0}; // index of the first <|box_start|> placeholder
    std::int32_t speech_rows{0};   // ceil(samples / hop_length) placeholders
};

// User-turn text after the speech span, as VibeVoiceASRProcessor writes it.
std::string user_suffix(const RuntimeConfig& config, std::size_t samples,
                        std::string_view context_info);

// System turn plus user turn (no generation prompt): the token-exact prompt of
// VibeVoiceASRProcessor._process_single_audio for 24 kHz audio of `samples`.
Prompt build_prompt(const ITokenizer& tokenizer, const RuntimeConfig& config, std::size_t samples,
                    std::string_view context_info);

} // namespace trtmc::vibevoice_asr
