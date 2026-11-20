/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Loads an HSTU bundle through the public Task API, ranks one record, and
// prints the per-task logits: the native side of the family's validation.
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5)
            throw std::invalid_argument(
                "usage: hstu_task_probe BUNDLE RUNTIME_ROOT RECORD [REPEATS]");
        trtmc::BundleReader reader(argv[1]);
        auto loaded = trtmc::load_task(reader, argv[2]);
        auto* model = dynamic_cast<trtmc::IStructuredDecision*>(loaded.get());
        if (!model)
            throw std::runtime_error("bundle does not expose structured_decision");
        std::ifstream file(argv[3]);
        trtmc::StructuredDecisionRequest request;
        request.document.assign(std::istreambuf_iterator<char>(file), {});
        const int repeats = argc > 4 ? std::stoi(argv[4]) : 2;
        if (repeats < 1)
            throw std::invalid_argument("repeats must be positive");
        nlohmann::json results = nlohmann::json::array();
        for (int i = 0; i < repeats; ++i) {
            const auto start = std::chrono::steady_clock::now();
            const auto result = model->decide(request);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            nlohmann::json scores = nlohmann::json::array();
            for (const auto& score : result.scores)
                scores.push_back({{"question_id", score.question_id},
                                  {"option_ids", score.option_ids},
                                  {"logits", score.logits}});
            results.push_back({{"response", nlohmann::json::parse(result.document)},
                               {"scores", scores},
                               {"public_task_call_wall_ms", ms}});
        }
        std::cout << nlohmann::json({{"results", results}}).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
