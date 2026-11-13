/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <set>

trtmc::StructuredDecisionRequest read_request(const char* path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read record");
    trtmc::StructuredDecisionRequest request;
    request.document = std::string((std::istreambuf_iterator<char>(f)), {});
    return request;
}

int main(int argc, char** argv) {
    try {
        if (argc < 4)
            throw std::invalid_argument("usage: task_probe BUNDLE RUNTIME_ROOT RECORD [ITERATIONS] "
                                        "[WARMUP], or --sequence RECORD...");
        trtmc::BundleReader reader(argv[1]);
        auto loaded = trtmc::load_task(reader, argv[2]);
        auto* model = dynamic_cast<trtmc::IStructuredDecision*>(loaded.get());
        if (!model)
            throw std::runtime_error("bundle does not expose structured_decision");
        const bool sequence = std::string(argv[3]) == "--sequence";
        if ((!sequence && argc > 6) || (sequence && argc < 5))
            throw std::invalid_argument("invalid task probe arguments");
        std::vector<trtmc::StructuredDecisionRequest> requests;
        if (sequence) {
            for (int i = 4; i < argc; ++i)
                requests.push_back(read_request(argv[i]));
        } else
            requests.push_back(read_request(argv[3]));
        const int iterations = sequence   ? static_cast<int>(requests.size())
                               : argc > 4 ? std::stoi(argv[4])
                                          : 1;
        const int warmup = sequence ? 0 : argc > 5 ? std::stoi(argv[5]) : 0;
        if (iterations < 1 || warmup < 0)
            throw std::invalid_argument("invalid iteration counts");
        for (int i = 0; i < warmup; ++i)
            model->decide(requests.front());
        nlohmann::json results = nlohmann::json::array();
        for (int i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto result = model->decide(requests[sequence ? i : 0]);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            nlohmann::json scores = nlohmann::json::array();
            for (const auto& score : result.scores)
                scores.push_back({{"question_id", score.question_id},
                                  {"option_ids", score.option_ids},
                                  {"logits", score.logits},
                                  {"probabilities", score.probabilities}});
            results.push_back({{"response", nlohmann::json::parse(result.document)},
                               {"scores", scores},
                               {"public_task_call_wall_ms", ms}});
        }
        std::ifstream maps("/proc/self/maps");
        std::set<std::string> libraries;
        std::string line;
        while (std::getline(maps, line)) {
            const auto slash = line.find('/');
            if (slash != std::string::npos && line.find(".so", slash) != std::string::npos)
                libraries.insert(std::filesystem::path(line.substr(slash)).filename().string());
        }
        std::cout << nlohmann::json({{"results", results}, {"loaded_libraries", libraries}}).dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
