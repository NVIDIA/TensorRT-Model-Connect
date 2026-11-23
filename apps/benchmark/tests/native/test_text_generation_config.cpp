/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text_generation_config.h"

#include <iostream>
#include <nlohmann/json.hpp>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    using trtmc::benchmark_worker::text_generation_config;

    // A translation request (the AIPerf FLORES suite's NLLB fields) reaches the family's config.
    const auto translation = text_generation_config(nlohmann::json{
        {"prompt", "Hello."},
        {"max_new_tokens", 128},
        {"temperature", 0.0},
        {"source_language", "eng_Latn"},
        {"source_language_token_id", 256047},
        {"target_language", "fra_Latn"},
        {"forced_bos_token_id", 256057},
    });
    check(translation.source_language_token_id == 256047, "source language token id is passed");
    check(translation.forced_bos_token_id == 256057, "forced first target token is passed");
    check(translation.temperature == 0.0F, "sampling fields are still read");

    // Without them, both stay unset (-1), like the command line's defaults, and so do the others.
    const auto plain = text_generation_config(nlohmann::json{{"prompt", "Hello."}});
    check(plain.source_language_token_id == -1 && plain.forced_bos_token_id == -1,
          "language token ids default to unset");
    check(plain.max_new_tokens == 128 && plain.temperature == 1.0F && plain.top_k == 1 &&
              plain.enable_thinking && !plain.use_chat_template &&
              plain.text_generation_mode == "auto",
          "other defaults are unchanged");
    return failures;
}
