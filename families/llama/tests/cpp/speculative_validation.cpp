/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/plugin_helpers.h"
#include "families/llama/runtime/speculative/pipeline.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("usage: llama_speculative_validation BUNDLE INPUT_IDS_JSON "
                                        "COUNT OUTPUT_JSON");
        std::ifstream input(argv[2]);
        if (!input)
            throw std::invalid_argument("cannot open input token fixture");
        nlohmann::json fixture;
        input >> fixture;
        const auto ids = fixture.get<std::vector<std::int32_t>>();
        const int count = std::stoi(argv[3]);
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        trtmc::BundleReader bundle(argv[1]);
        trtmc::llama::speculative::Pipeline pipeline({bundle, *backend});
        if (count < 1)
            throw std::invalid_argument("validation requires a positive output count");
        if (!pipeline.generate_ids(ids, 0, true, true).token_ids.empty())
            throw std::runtime_error("zero-output request emitted a token");
        bool rejected = false;
        try {
            pipeline.generate_ids(ids, -1, true, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("negative output count was accepted");
        const auto vanilla = pipeline.generate_ids(ids, count, false, true);
        const auto chain = pipeline.generate_ids(ids, count, true, true, 1);
        const auto chain_accepted = pipeline.accepted_lengths();
        const auto eagle = pipeline.generate_ids(ids, count, true, true, 2);
        const auto accepted = pipeline.accepted_lengths();
        // Exercise reset after speculative writes and retained rejected rows.
        const auto repeated = pipeline.generate_ids(ids, count, false, true);
        const bool equal = vanilla.token_ids.size() == static_cast<std::size_t>(count) &&
                           vanilla.token_ids == eagle.token_ids &&
                           vanilla.token_ids == chain.token_ids;
        const bool reset_equal = vanilla.token_ids == repeated.token_ids;
        // A 65-row prompt covers a short final chunk with the 64-row prefill
        // profile; a one-row prompt must still select the prefill context.
        bool short_prompt_equal = true;
        for (const std::size_t length : {std::size_t(1), std::min(ids.size(), std::size_t(65))}) {
            const std::vector<std::int32_t> short_ids(ids.begin(), ids.begin() + length);
            const auto short_ar = pipeline.generate_ids(short_ids, 8, false, true);
            const auto short_spec = pipeline.generate_ids(short_ids, 8, true, true, 1);
            short_prompt_equal &= short_ar.token_ids == short_spec.token_ids;
        }
        int truncation_checks = 0;
        {
            for (const auto length :
                 {ids.size(), std::size_t(1), std::min(ids.size(), std::size_t(65))}) {
                const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + length);
                for (const int outputs : {1, 2, 7, length == ids.size() ? count : 9}) {
                    const auto reference = pipeline.generate_ids(prompt, outputs, false, true);
                    for (int width : {1, 2}) {
                        const auto actual =
                            pipeline.generate_ids(prompt, outputs, true, true, width);
                        if (actual.token_ids != reference.token_ids)
                            throw std::runtime_error(
                                "EAGLE3 disagrees with autoregressive reference");
                        ++truncation_checks;
                    }
                    // Reuse contexts after GPU policy/compaction with a plain AR request.
                    if (pipeline.generate_ids(prompt, outputs, false, true).token_ids !=
                        reference.token_ids)
                        throw std::runtime_error("device policy reset differs from AR reference");
                }
            }
        }
        nlohmann::json sampling_checks = nlohmann::json::array();
        bool sampled_non_greedy = false;
        int sampled_accepted = 0;
        for (int seed : {7, 19})
            for (int policy : {0, 1}) {
                trtmc::LlamaSamplingParams params;
                params.seed = seed;
                params.temperature = policy == 0 ? 0.8F : 1.0F;
                params.top_k = policy == 0 ? 50 : 0;
                params.top_p = policy == 0 ? 0.95F : 0.9F;
                for (const std::size_t length : {std::size_t(1), ids.size()}) {
                    const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + length);
                    const auto target =
                        pipeline.generate_ids(prompt, count, false, true, 1, params);
                    const auto greedy = pipeline.generate_ids(prompt, count, false, true);
                    sampled_non_greedy |= target.token_ids != greedy.token_ids;
                    for (int width : {1, 2}) {
                        for (int repeat = 0; repeat < 2; ++repeat) {
                            const auto actual =
                                pipeline.generate_ids(prompt, count, true, true, width, params);
                            if (actual.token_ids != target.token_ids)
                                throw std::runtime_error(
                                    "sampled EAGLE3 changes seeded target tokens");
                            for (int accepted : pipeline.accepted_lengths())
                                sampled_accepted += accepted;
                        }
                        if (pipeline.generate_ids(prompt, count, false, true, width, params)
                                .token_ids != target.token_ids)
                            throw std::runtime_error("sampled reset changes target RNG or cache");
                        sampling_checks.push_back({{"seed", seed},
                                                   {"policy", policy},
                                                   {"input_tokens", length},
                                                   {"width", width},
                                                   {"token_ids", target.token_ids},
                                                   {"repeats", 2}});
                    }
                    // Stop on a genuinely sampled token, not a compiled greedy EOS.
                    params.eos_token_id =
                        target.token_ids[std::min<std::size_t>(2, target.token_ids.size() - 1)];
                    const auto stopped =
                        pipeline.generate_ids(prompt, count, false, false, 1, params);
                    const auto actual =
                        pipeline.generate_ids(prompt, count, true, false, 2, params);
                    if (actual.token_ids != stopped.token_ids ||
                        actual.token_ids.back() != params.eos_token_id)
                        throw std::runtime_error("sampled EAGLE3 ignores the effective EOS");
                    params.eos_token_id = -1;
                }
            }
        if (!sampled_non_greedy || sampled_accepted == 0)
            throw std::runtime_error(
                "sampling coverage lacks stochastic output or draft acceptance");
        // The last available cache slot must not trigger an out-of-profile draft.
        const auto manifest =
            nlohmann::json::parse(trtmc::llama::require_text_section(bundle, "speculative.json"));
        const int capacity = manifest.at("target").at("capacity");
        std::vector<std::int32_t> boundary(capacity - 2, ids.front());
        trtmc::LlamaSamplingParams boundary_params;
        boundary_params.top_k = 20;
        boundary_params.seed = 7;
        const auto boundary_target =
            pipeline.generate_ids(boundary, 2, false, true, 1, boundary_params);
        if (pipeline.generate_ids(boundary, 2, true, true, 2, boundary_params).token_ids !=
            boundary_target.token_ids)
            throw std::runtime_error("sampled EAGLE3 capacity-boundary output differs");

        nlohmann::json result{
            {"sampling_checks", sampling_checks},
            {"sampled_non_greedy", sampled_non_greedy},
            {"sampled_accepted", sampled_accepted},
            {"sampled_capacity_equal", true},
            {"input_tokens", ids.size()},
            {"requested_output_tokens", count},
            {"autoregressive_ids", vanilla.token_ids},
            {"eagle3_ids", eagle.token_ids},
            {"chain_ids", chain.token_ids},
            {"chain_accepted_lengths", chain_accepted},
            {"tokens_equal", equal},
            {"reset_equal", reset_equal},
            {"short_prompt_equal", short_prompt_equal},
            {"truncation_checks", truncation_checks},
            {"accepted_lengths", accepted},
            {"verification_rounds", accepted.size()},
            {"autoregressive_prefill_ms", vanilla.prefill_ms},
            {"autoregressive_decode_ms", vanilla.decode_ms},
            {"eagle3_prefill_ms", eagle.prefill_ms},
            {"eagle3_decode_ms", eagle.decode_ms},
            {"chain_prefill_ms", chain.prefill_ms},
            {"chain_decode_ms", chain.decode_ms},
            {"text", eagle.text},
        };
        std::ofstream output(argv[4]);
        if (!output)
            throw std::runtime_error("cannot open validation output");
        output << result.dump(2) << '\n';
        std::cout << "tokens_equal=" << equal << " reset_equal=" << reset_equal
                  << " verification_rounds=" << accepted.size() << '\n';
        return equal && reset_equal && short_prompt_equal ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
