/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-20
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-02
// Intent:         GLM-ASR host plan: audio embedding count and transcription prompt layout
// Preconditions:  GlmAsrConfig carrying the checkpoint's token ids and encoder geometry
// Postconditions: Embedding count matches GlmAsrProcessor, prompt matches the chat template
// =============================================================================

#include "runtime/models/glmasr/glmasr_prompt_plan.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

// Token ids and counts below are the observed output of GlmAsrProcessor for
// zai-org/GLM-ASR-Nano-2512 at revision 61ba4e0b3309b6656edea3e93e419f7bd5c61957.
constexpr int32_t kUser = 59253;
constexpr int32_t kAssistant = 59254;
constexpr int32_t kAudio = 59260;
constexpr int32_t kBeginAudio = 59261;
constexpr int32_t kEndAudio = 59262;
constexpr int32_t kNewline = 10;

// "Please transcribe this audio into text"
const std::vector<int32_t> kPromptTokens{14215, 1700, 8091, 643, 14812, 1636, 2815};

void test_audio_embedding_count_matches_processor() {
    // Three seconds at 16 kHz with hop 160 gives 300 mel frames, which the
    // processor expands into 37 audio placeholders.
    check(trtmc::glmasr::mel_frames_for_samples(16000 * 3, 160) == 300,
          "glmasr mel frame count follows the hop length");
    check(trtmc::glmasr::encoder_output_frames(300) == 150,
          "glmasr conv front-end halves 300 mel frames");
    check(trtmc::glmasr::audio_embedding_count(300, 4) == 37,
          "glmasr audio embedding count matches GlmAsrProcessor for three seconds");

    // A full 30 second chunk fills the encoder's declared position budget.
    check(trtmc::glmasr::encoder_output_frames(3000) == 1500,
          "glmasr conv front-end maps a full chunk onto 1500 encoder frames");
    check(trtmc::glmasr::audio_embedding_count(3000, 4) == 375,
          "glmasr audio embedding count saturates a full chunk at 375");

    check(trtmc::glmasr::audio_embedding_count(0, 4) == 0,
          "glmasr audio embedding count is zero for empty audio");
    check(trtmc::glmasr::audio_embedding_count(300, 0) == 0,
          "glmasr audio embedding count rejects a zero merge factor");
    check(trtmc::glmasr::audio_embedding_count(4, 4) == 0,
          "glmasr audio embedding count is zero below one merged frame");
}

void test_prompt_layout_matches_chat_template() {
    trtmc::GlmAsrConfig config;
    const auto plan = trtmc::glmasr::build_prompt_plan(config, kPromptTokens, 37);
    const auto& ids = plan.input_ids;

    // The processor emits 52 tokens for this prompt and three seconds of audio.
    check(ids.size() == 52, "glmasr prompt length matches GlmAsrProcessor");
    check(plan.num_audio_embeddings == 37, "glmasr prompt keeps the requested embedding count");
    check(plan.audio_offset == 3, "glmasr audio placeholders start after the opening markers");

    const std::vector<int32_t> expected_head{kUser, kNewline, kBeginAudio, kAudio, kAudio, kAudio};
    check(std::vector<int32_t>(ids.begin(), ids.begin() + 6) == expected_head,
          "glmasr prompt opens with user, newline, begin-of-audio, then placeholders");

    std::vector<int32_t> expected_tail{kAudio, kAudio, kEndAudio, kUser, kNewline};
    expected_tail.insert(expected_tail.end(), kPromptTokens.begin(), kPromptTokens.end());
    expected_tail.push_back(kAssistant);
    expected_tail.push_back(kNewline);
    check(std::vector<int32_t>(ids.end() - static_cast<long>(expected_tail.size()), ids.end()) ==
              expected_tail,
          "glmasr prompt closes with end-of-audio, the instruction, and the assistant marker");

    int32_t placeholders = 0;
    for (int32_t id : ids)
        if (id == kAudio)
            ++placeholders;
    check(placeholders == 37, "glmasr prompt carries one placeholder per audio embedding");

    for (int32_t index = 0; index < plan.num_audio_embeddings; ++index)
        if (ids[static_cast<std::size_t>(plan.audio_offset + index)] != kAudio) {
            check(false, "glmasr audio_offset addresses a contiguous placeholder run");
            break;
        }
}

void test_prompt_clamps_to_engine_capacity() {
    trtmc::GlmAsrConfig config;
    config.max_audio_embeddings = 375;
    const auto plan = trtmc::glmasr::build_prompt_plan(config, kPromptTokens, 4096);
    check(plan.num_audio_embeddings == 375,
          "glmasr prompt clamps the embedding count to what the engine emits");

    const auto empty = trtmc::glmasr::build_prompt_plan(config, kPromptTokens, -1);
    check(empty.num_audio_embeddings == 0, "glmasr prompt treats a negative count as no audio");
    // Eight fixed markers frame the prompt: user, newline, begin-of-audio,
    // end-of-audio, user, newline, assistant, newline.
    check(empty.input_ids.size() == kPromptTokens.size() + 8,
          "glmasr prompt without audio still carries every framing marker");
}

} // namespace

int main() {
    test_audio_embedding_count_matches_processor();
    test_prompt_layout_matches_chat_template();
    test_prompt_clamps_to_engine_capacity();

    if (g_failures != 0) {
        std::cerr << g_failures << " glmasr prompt plan test(s) failed\n";
        return 1;
    }
    return 0;
}
