/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/glmasr/runtime/glmasr_config.h"

#include <iostream>
#include <vector>

int main() {
    int failures = 0;
    const auto check = [&](bool condition, const char* label) {
        if (!condition) {
            std::cerr << "FAIL: " << label << '\n';
            ++failures;
        }
    };

    trtmc::GlmAsrConfig config;
    config.eos_token_id = 17;
    check(config.is_eos_token(17), "legacy scalar stop token");
    check(!config.is_eos_token(59253), "legacy bundle has no implicit alternate stop token");

    // GLM-ASR-Nano-2512, revision 61ba4e0b3309b6656edea3e93e419f7bd5c61957.
    // Saved Native transcripts end at 59253, which the old scalar contract lost.
    config.eos_token_ids = {59246, 59253, 59255};
    check(config.is_eos_token(59246), "checkpoint end-of-text token");
    check(config.is_eos_token(59253), "checkpoint user role stop token");
    check(config.is_eos_token(59255), "checkpoint alternate role stop token");
    check(!config.is_eos_token(17), "explicit checkpoint list supersedes legacy scalar");
    check(!config.is_eos_token(10), "ordinary newline continues decoding");
    check(!config.is_eos_token(14215), "ordinary transcription token continues decoding");

    return failures == 0 ? 0 : 1;
}
