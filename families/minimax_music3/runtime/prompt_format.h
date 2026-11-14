/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The text contract the language model is given. Mirrors this family's
// prompt_format.py, which was differential-tested against the reference over
// 410 captions and 409 lyric strings with no disagreement.

#include <cstdint>
#include <string>

namespace trtmc::minimax_music3 {

//: Sampling constants, from the reference.
constexpr int32_t kAudioCfgTokenId = 151654;
constexpr int32_t kAudioEndTokenId = 151670;
constexpr int32_t kAudioCodeOffset = 151675;
constexpr int32_t kSemanticVocabSize = 16384;
constexpr int32_t kMaxPromptTokens = 5000;
constexpr int32_t kArSamplingTopK = 50;
constexpr int32_t kArCfgTopK = 50;
constexpr float kArCfgScale = 1.5F;

// Rewrite <|key value|> spans as "key is value" and strip the markdown the
// input contract accepts but the checkpoint was not trained on.
std::string clean_caption(const std::string& caption);

// Keep only the leading structure tags on a tagged line, lowercase them, put
// each on its own line, and prefix the result with [start].
std::string normalize_lyrics(const std::string& lyrics);

// The assembled prompt: caption and lyrics inside the checkpoint's structure
// tokens, ending at <|audio_start|> so the first generated token is audio.
std::string assemble_prompt(const std::string& caption, const std::string& lyrics);

} // namespace trtmc::minimax_music3
