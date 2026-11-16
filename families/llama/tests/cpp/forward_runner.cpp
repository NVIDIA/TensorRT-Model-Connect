/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/pipeline.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "usage: llama_forward_runner PLAN IDS.json LOGITS.f32 WARMUP ITERATIONS");
        const int warmup = std::stoi(argv[4]), iterations = std::stoi(argv[5]);
        if (warmup < 0 || iterations < 1)
            throw std::invalid_argument("invalid iteration counts");
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::vector<char> plan(static_cast<size_t>(file.tellg()));
        file.seekg(0);
        file.read(plan.data(), static_cast<std::streamsize>(plan.size()));
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        auto module = backend->create_module(plan.data(), plan.size(), {});
        if (!module || !module->ok())
            throw std::runtime_error("could not load bringup engine");
        trtmc::LlamaTextGenConfig config;
        config.single_forward = true;
        config.vocab_size = static_cast<int32_t>(module->tensor_shape("logits").back());
        config.prefill_max_length = static_cast<int32_t>(
            module->input_profile_shape("token_id", 0, trtmc::ProfileShapeSelector::kMax).front());
        trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), nullptr, config, nullptr,
                                                    nullptr);
        nlohmann::json input;
        std::ifstream(argv[2]) >> input;
        const auto ids = input.get<std::vector<int32_t>>();
        auto result = pipeline.generate_ids(ids, {});
        const auto snapshot = result.logits;
        auto alternate = ids;
        std::reverse(alternate.begin(), alternate.end());
        pipeline.generate_ids(alternate, {});
        if (result.logits != snapshot || pipeline.generate_ids(ids, {}).logits != snapshot)
            throw std::runtime_error(
                "single-forward calls must be independent and own their logits");
        for (int i = 0; i < warmup; ++i)
            pipeline.generate_ids(ids, {});
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto measured = pipeline.generate_ids(ids, {});
            times.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
            result = std::move(measured);
        }
        std::ofstream output(argv[3], std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(reinterpret_cast<const char*>(result.logits.data()),
                     static_cast<std::streamsize>(result.logits.size() * sizeof(float)));
        std::sort(times.begin(), times.end());
        const auto n = times.size();
        std::cout
            << nlohmann::
                   json{{"median_ms", (times[(n - 1) / 2] + times[n / 2]) / 2},
                        {"warmup", warmup},
                        {"iterations", iterations},
                        {"samples_ms", times},
                        {"rows", ids.size()},
                        {"columns", config.vocab_size},
                        {"timing_boundary",
                         "synchronous internal pipeline call: input preparation, H2D, full "
                         "forward, D2H and logits copy; excludes build/load and serialization"}}
                       .dump()
            << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
