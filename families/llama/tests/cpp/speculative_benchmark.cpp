/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/pipeline.h"
#include "trtmc/runtime/trt_backend.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

namespace {
void synchronize() {
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument("usage: llama_speculative_benchmark BUNDLE INPUT_IDS COUNT "
                                        "WARMUP REPEATS OUTPUT_JSON");
        std::ifstream input(argv[2]);
        if (!input)
            throw std::invalid_argument("cannot open input token fixture");
        nlohmann::json fixture;
        input >> fixture;
        const auto ids = fixture.get<std::vector<std::int32_t>>();
        const int count = std::stoi(argv[3]), warmup = std::stoi(argv[4]),
                  repeats = std::stoi(argv[5]);
        if (count < 2 || warmup < 1 || repeats < 1)
            throw std::invalid_argument("require count >= 2, warmup >= 1 and repeats >= 1");
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        trtmc::BundleReader bundle(argv[1]);
        trtmc::llama::speculative::Pipeline pipeline({bundle, *backend});
        nlohmann::json report{{"input_tokens", ids.size()}, {"output_tokens", count},
                              {"warmup_per_mode", warmup},  {"repeats_per_mode", repeats},
                              {"cuda_graph", false},        {"samples", nlohmann::json::array()}};
        std::vector<std::int32_t> expected;
        const char* names[] = {"autoregressive", "eagle3_chain", "eagle3_tree"};
        // Rotate mode order to reduce correlation with clock and thermal drift.
        for (int iteration = -warmup; iteration < repeats; ++iteration) {
            for (int offset = 0; offset < 3; ++offset) {
                const int mode = (iteration + warmup + offset) % 3;
                synchronize();
                const auto start = std::chrono::steady_clock::now();
                const auto result =
                    pipeline.generate_ids(ids, count, mode != 0, true, mode == 2 ? 2 : 1);
                synchronize();
                const double wall_ms = std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - start)
                                           .count();
                if (expected.empty())
                    expected = result.token_ids;
                if (result.token_ids.size() != static_cast<std::size_t>(count) ||
                    result.token_ids != expected)
                    throw std::runtime_error("benchmark output mismatch");
                const auto& accepted = pipeline.accepted_lengths();
                if (iteration >= 0) {
                    report["samples"].push_back({
                        {"iteration", iteration},
                        {"mode", names[mode]},
                        {"wall_ms", wall_ms},
                        {"prefill_including_draft_ms", result.prefill_ms},
                        {"decode_ms", result.decode_ms},
                        {"verification_rounds", accepted.size()},
                        {"accepted_lengths", accepted},
                        {"accepted_draft_tokens",
                         std::accumulate(accepted.begin(), accepted.end(), 0)},
                    });
                }
                std::cout << (iteration < 0 ? "warmup " : "sample ") << iteration << ' '
                          << names[mode] << " wall_ms=" << wall_ms << std::endl;
            }
        }
        report["output_ids"] = expected;
        report["all_outputs_equal"] = true;
        std::ofstream output(argv[6]);
        if (!output)
            throw std::runtime_error("cannot open benchmark output");
        output << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
