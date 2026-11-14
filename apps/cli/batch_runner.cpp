/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standalone batch runner: loads one bundle ONCE, then calls
// ITextGeneration::generate() once per prompt from a file, writing one JSON
// line per prompt to stdout. Avoids the ~60s-per-invocation engine-reload
// cost of running `trtmc run` once per prompt via a fresh process.
//
// Usage: batch_runner BUNDLE RUNTIME_ROOT PROMPTS_FILE MAX_NEW_TOKENS

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: batch_runner BUNDLE RUNTIME_ROOT PROMPTS_FILE MAX_NEW_TOKENS\n";
        return 1;
    }
    const std::string bundle_path = argv[1];
    const std::string runtime_root = argv[2];
    const std::string prompts_path = argv[3];
    const int32_t max_new_tokens = std::stoi(argv[4]);

    auto task = trtmc::load_task(bundle_path, runtime_root);
    auto* gen = dynamic_cast<trtmc::ITextGeneration*>(task.get());
    if (!gen) {
        std::cerr << "bundle's task does not implement ITextGeneration\n";
        return 1;
    }

    std::ifstream prompts_file(prompts_path);
    if (!prompts_file) {
        std::cerr << "cannot open prompts file: " << prompts_path << "\n";
        return 1;
    }

    std::string line;
    int n = 0;
    while (std::getline(prompts_file, line)) {
        if (line.empty())
            continue;
        trtmc::TextGenerationConfig cfg;
        cfg.max_new_tokens = max_new_tokens;

        const auto t0 = std::chrono::steady_clock::now();
        trtmc::TextResult result = gen->generate(line, cfg);
        const auto t1 = std::chrono::steady_clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        nlohmann::json j;
        j["prompt"] = line;
        j["text"] = result.text;
        j["token_ids"] = result.token_ids;
        j["wall_ms"] = wall_ms;
        std::cout << j.dump() << "\n";
        std::cout.flush();

        ++n;
        std::cerr << "[" << n << "] done (" << wall_ms << " ms)\n";
    }

    std::cerr << "TOTAL: " << n << " prompts\n";
    return 0;
}
