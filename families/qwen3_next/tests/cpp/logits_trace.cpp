/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/diagnostics.h"
#include "trtmc/runtime/family_loader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "usage: qwen3_next_logits_trace BUNDLE ROOT PREFIX PROMPT STEPS");
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* diagnostics = dynamic_cast<trtmc::qwen3_next::Diagnostics*>(task.get());
        auto* generation = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        if (!diagnostics || !generation)
            throw std::runtime_error("Qwen3Next family interfaces unavailable");
        const int steps = std::stoi(argv[5]);
        auto trace = diagnostics->trace_logits(argv[4], steps);
        trtmc::TextGenerationConfig config;
        config.max_new_tokens = steps;
        config.top_k = 1;
        config.seed = -1;
        const auto actual = generation->generate(argv[4], config);
        // Reuse the same loaded Task across requests; KV state must reset.
        const auto replay = generation->generate(argv[4], config);
        if (actual.token_ids != replay.token_ids)
            throw std::runtime_error("Qwen3Next repeated requests leaked sequence state");
        if (actual.token_ids != trace.generated_ids)
            throw std::runtime_error("Qwen3Next Task generation disagrees with diagnostic logits");
        std::ofstream metadata(std::string(argv[3]) + ".json");
        metadata << nlohmann::json{{"prompt_ids", trace.prompt_ids},
                                   {"generated_ids", actual.token_ids},
                                   {"text", actual.text},
                                   {"rows", trace.rows.size()},
                                   {"columns", trace.rows.front().size()},
                                   {"prefill_ms", actual.prefill_ms},
                                   {"decode_ms", actual.decode_ms},
                                   {"request_replay", true}};
        std::ofstream data(std::string(argv[3]) + ".f32", std::ios::binary);
        for (const auto& row : trace.rows)
            data.write(reinterpret_cast<const char*>(row.data()),
                       static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!metadata || !data)
            throw std::runtime_error("cannot write Qwen3Next oracle evidence");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
