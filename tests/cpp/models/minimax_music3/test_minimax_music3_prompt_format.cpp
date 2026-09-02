/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-MM3-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-MM3-01
// Intent:         MiniMax-Music3 prompt contract: caption cleaning, lyric
//                 normalisation, and the assembled structure the checkpoint
//                 was trained on
// Preconditions:  None; the functions are pure string transforms
// Postconditions: The assembled prompt matches what prompt_format.py produces,
//                 which was differential-tested against the reference over 410
//                 captions and 409 lyric strings
// =============================================================================

#include "runtime/models/minimax_music3/prompt_format.h"

#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_equal(const std::string& actual, const std::string& expected, const char* name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << name << "\n  expected: " << expected << "\n  actual:   " << actual
                  << '\n';
        ++g_failures;
    }
}

using trtmc::minimax_music3::assemble_prompt;
using trtmc::minimax_music3::clean_caption;
using trtmc::minimax_music3::normalize_lyrics;

void test_caption_rewrites_special_tags() {
    // <|key value|> becomes "key is value"; a span with no space keeps its text.
    check_equal(clean_caption("<|bpm 96|>"), "bpm is 96", "special tag with a value");
    check_equal(clean_caption("<|acoustic|>"), "acoustic", "special tag without a value");
    check_equal(clean_caption("Genre: pop."), "Genre: pop.", "plain caption is untouched");
}

void test_caption_strips_markdown() {
    check_equal(clean_caption("## Heading"), "Heading", "atx heading");
    check_equal(clean_caption("- item"), "item", "bullet");
    check_equal(clean_caption("**bold**"), "bold", "bold");
    check_equal(clean_caption("*italic*"), "italic", "italic");
    check_equal(clean_caption("a\n\n\nb"), "a\nb", "blank runs collapse");
}

void test_lyrics_keep_only_leading_tags() {
    // A line opening with structure tags keeps only those tags. Text sharing
    // the line is dropped -- the contract the model card warns about.
    check_equal(normalize_lyrics("[verse] sung words"), "[start]\n[verse]",
                "text beside a leading tag is dropped");
    check_equal(normalize_lyrics("plain line"), "[start]\nplain line",
                "an untagged line survives whole");
}

void test_lyrics_lowercase_their_tags() {
    check_equal(normalize_lyrics("[Verse]"), "[start]\n[verse]", "tags are lowercased");
    check_equal(normalize_lyrics("[CHORUS]"), "[start]\n[chorus]", "uppercase tags too");
}

void test_lyrics_always_open_with_start() {
    const auto result = normalize_lyrics("anything");
    check(result.rfind("[start]\n", 0) == 0, "normalised lyrics open with [start]");
}

void test_assembled_prompt_carries_the_structure() {
    const auto prompt = assemble_prompt("pop", "[verse]");

    // The order is the checkpoint's, and the prompt ends at <|audio_start|> so
    // the first generated token is audio rather than text.
    check_equal(prompt,
                "<|im_start|><|caption_start|>pop<|caption_end|>"
                "<|lyrics_start|>[start]\n[verse]<|lyrics_end|><|im_end|><|audio_start|>",
                "assembled prompt");
}

void test_assembled_prompt_survives_an_empty_caption() {
    // An empty caption means unconditioned on a description, not a broken prompt.
    const auto prompt = assemble_prompt("", "[verse]");
    check(prompt.find("<|caption_start|><|caption_end|>") != std::string::npos,
          "an empty caption leaves its markers adjacent");
    check(prompt.rfind("<|audio_start|>") == prompt.size() - std::string("<|audio_start|>").size(),
          "the prompt still ends at <|audio_start|>");
}

void test_sampling_constants_match_the_reference() {
    using namespace trtmc::minimax_music3;
    // These decide what a draw may land on. An unmasked draw over the whole
    // vocabulary produces audio that carries no words.
    check(kAudioCodeOffset == 151675, "audio code offset");
    check(kSemanticVocabSize == 16384, "semantic vocabulary size");
    check(kAudioEndTokenId == 151670, "audio end token");
    check(kAudioCfgTokenId == 151654, "classifier-free guidance token");
    check(kArCfgScale == 1.5F, "autoregressive guidance scale");
    check(kArSamplingTopK == 50, "sampling top-k");
    check(kMaxPromptTokens == 5000, "prompt budget");
}

} // namespace

int main() {
    test_caption_rewrites_special_tags();
    test_caption_strips_markdown();
    test_lyrics_keep_only_leading_tags();
    test_lyrics_lowercase_their_tags();
    test_lyrics_always_open_with_start();
    test_assembled_prompt_carries_the_structure();
    test_assembled_prompt_survives_an_empty_caption();
    test_sampling_constants_match_the_reference();

    if (g_failures != 0) {
        std::cerr << g_failures << " MiniMax-Music3 prompt format test(s) failed\n";
        return 1;
    }
    return 0;
}
