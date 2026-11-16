/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/features.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>

// Public SDK consumer; request construction and serialization are outside timing.
nlohmann::json execute(const trtmc::Model& model, const nlohmann::json& input,
                       const std::string& output_path, int warmup, int iterations) {
    trtmc::TextPredictionPositionsToTokenScoresRequest request;
    request.token_ids = input.at("token_ids").get<std::vector<int64_t>>();
    request.attention_mask = input.value("attention_mask", std::vector<uint8_t>{});
    request.blocked_attention =
        input.value("blocked_attention",
                    std::vector<uint8_t>(request.token_ids.size() * request.token_ids.size(), 0));
    request.segment_ids = input.value("segment_ids", std::vector<int64_t>{});
    request.prediction_positions = input.value("prediction_positions", std::vector<uint64_t>{});
    if (!input.contains("prediction_positions")) {
        request.prediction_positions.resize(request.token_ids.size());
        std::iota(request.prediction_positions.begin(), request.prediction_positions.end(), 0);
    }
    trtmc::Config config{{"causal", input.value("causal", true)}};
    if (input.contains("position_ids"))
        config.add("position_ids", input.at("position_ids").get<std::vector<int64_t>>());
    auto task = model.task<trtmc::TextPredictionPositionsToTokenScores>();
    auto result = task.run(request, config);
    const auto first = result.logits();
    const std::vector<float> snapshot(first.values.begin(), first.values.end());
    // A/B/A detects accidental cross-call state or borrowed result storage.
    auto alternate = request;
    std::reverse(alternate.token_ids.begin(), alternate.token_ids.end());
    task.run(alternate, config);
    const auto repeat = task.run(request, config);
    if (result.logits().values.size() != snapshot.size() ||
        repeat.logits().values.size() != snapshot.size() ||
        !std::equal(snapshot.begin(), snapshot.end(), result.logits().values.begin()) ||
        !std::equal(snapshot.begin(), snapshot.end(), repeat.logits().values.begin()))
        throw std::runtime_error("forward result ownership or independent-call check failed");
    for (int i = 0; i < warmup; ++i)
        task.run(request, config);
    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        auto measured = task.run(request, config);
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        result = std::move(measured);
    }
    const auto values = result.logits();
    std::ofstream output(output_path, std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(reinterpret_cast<const char*>(values.values.data()),
                 static_cast<std::streamsize>(values.values.size() * sizeof(float)));
    output.close();
    std::sort(times.begin(), times.end());
    const double median = times.size() % 2
                              ? times[times.size() / 2]
                              : (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2;
    return nlohmann::json{
        {"rows", values.rows},
        {"columns", values.columns},
        {"warmup", warmup},
        {"iterations", iterations},
        {"median_ms", median},
        {"min_ms", times.front()},
        {"samples_ms", times},
        {"timing_boundary", "synchronous public Task run: mask/input preparation, H2D, full "
                            "forward, D2H, owned logits/result packing; excludes setup/build, "
                            "load, request construction, serialization and result release"},
        {"independent_calls_and_owned_results", true},
    };
}

int main(int argc, char** argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument("usage: llama_forward_runner BUNDLE RUNTIME_ROOT "
                                        "REQUEST.json OUTPUT.f32 WARMUP ITERATIONS");
        nlohmann::json input;
        std::ifstream(argv[3]) >> input;
        const int warmup = std::stoi(argv[5]), iterations = std::stoi(argv[6]);
        if (warmup < 0 || iterations < 1)
            throw std::invalid_argument("warmup must be nonnegative and iterations positive");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        nlohmann::json output;
        if (input.is_array()) {
            output = nlohmann::json::array();
            for (size_t index = 0; index < input.size(); ++index) {
                const auto& item = input[index];
                const auto path = std::string(argv[4]) + "." + std::to_string(index);
                try {
                    auto value = execute(model, item.at("request"), path, warmup, iterations);
                    if (item.value("expect_invalid", false))
                        throw std::runtime_error("invalid request was accepted");
                    value["output_path"] = path;
                    output.push_back(std::move(value));
                } catch (const trtmc::Error& error) {
                    if (!item.value("expect_invalid", false) ||
                        error.code() != TRTMC_INVALID_ARGUMENT)
                        throw;
                    output.push_back({{"rejected", true}});
                }
            }
        } else {
            output = execute(model, input, argv[4], warmup, iterations);
        }
        std::cout << output.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
