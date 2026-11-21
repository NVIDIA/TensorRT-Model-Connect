/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "task_runtime.h"
#include "trtmc/control.hpp"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"
#include "trtmc/text.hpp"

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

static int run(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: batch_runner BUNDLE RUNTIME_ROOT PROMPTS_FILE MAX_NEW_TOKENS\n";
        return 1;
    }
    const std::string bundle_path = argv[1];
    const std::string runtime_root = argv[2];
    const std::string prompts_path = argv[3];
    const int32_t max_new_tokens = std::stoi(argv[4]);

    std::unique_ptr<trtmc::ITask> task;
    trtmc::ITextGeneration* gen = nullptr;
    std::optional<trtmc::TextContinuation> continuation;
    if (trtmc::app::uses_existing_task_runtime(trtmc::Bundle::open(bundle_path).info().task)) {
        task = trtmc::load_task(bundle_path, runtime_root);
        gen = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        if (!gen) {
            std::cerr << "bundle's task does not implement ITextGeneration\n";
            return 1;
        }
    } else {
        trtmc::LoadOptions options;
        options.runtime_root = runtime_root;
        continuation = trtmc::Model::load(bundle_path, options).task<trtmc::TextContinuation>();
    }

    std::ifstream prompts_file(prompts_path);
    if (!prompts_file) {
        std::cerr << "cannot open prompts file: " << prompts_path << "\n";
        return 1;
    }

    std::string line;
    int n = 0;
    const trtmc::Config config{{"max_new_tokens", std::int64_t{max_new_tokens}}};
    trtmc::TextGenerationConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    auto emit = [&](std::string_view text, const auto& tokens, double wall_ms) {
        nlohmann::json j;
        j["prompt"] = line;
        j["text"] = text;
        j["token_ids"] = nlohmann::json::array();
        for (const auto token : tokens)
            j["token_ids"].push_back(token);
        j["wall_ms"] = wall_ms;
        std::cout << j.dump() << "\n";
        std::cout.flush();

        ++n;
        std::cerr << "[" << n << "] done (" << wall_ms << " ms)\n";
    };
    while (std::getline(prompts_file, line)) {
        if (line.empty())
            continue;
        if (continuation) {
            const trtmc::TextContinuationRequest request{line};
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = continuation->run(request, config);
            const auto t1 = std::chrono::steady_clock::now();
            emit(result.text(), result.token_ids(),
                 std::chrono::duration<double, std::milli>(t1 - t0).count());
        } else {
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = gen->generate(line, cfg);
            const auto t1 = std::chrono::steady_clock::now();
            emit(result.text, result.token_ids,
                 std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    std::cerr << "TOTAL: " << n << " prompts\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
