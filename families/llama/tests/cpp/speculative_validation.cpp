/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/pipeline.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6)
            throw std::invalid_argument("usage: llama_speculative_validation BUNDLE INPUT_IDS_JSON "
                                        "COUNT OUTPUT_JSON [all-policies]");
        const bool all_policies = argc == 6 && std::string(argv[5]) == "all-policies";
        if (argc == 6 && !all_policies)
            throw std::invalid_argument("expected all-policies");
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
        int policy_checks = 0;
        if (all_policies) {
            using Policy = trtmc::llama::speculative::ExecutionPolicy;
            for (const auto length :
                 {ids.size(), std::size_t(1), std::min(ids.size(), std::size_t(65))}) {
                const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + length);
                for (const int outputs : {1, 2, 7, length == ids.size() ? count : 9}) {
                    const auto reference = pipeline.generate_ids(prompt, outputs, false, true);
                    for (int width : {1, 2}) {
                        const auto host = pipeline.generate_ids(prompt, outputs, true, true, width,
                                                                Policy::kHost);
                        const auto host_lengths = pipeline.accepted_lengths();
                        if (host.token_ids != reference.token_ids)
                            throw std::runtime_error(
                                "host policy disagrees with autoregressive reference");
                        for (auto policy : {Policy::kDeviceDraft, Policy::kDeviceFull}) {
                            const auto actual =
                                pipeline.generate_ids(prompt, outputs, true, true, width, policy);
                            if (actual.token_ids != reference.token_ids ||
                                pipeline.accepted_lengths() != host_lengths)
                                throw std::runtime_error(
                                    "device policy tokens or acceptance paths differ");
                            ++policy_checks;
                        }
                    }
                    // Reuse contexts after GPU policy/compaction with a plain AR request.
                    if (pipeline.generate_ids(prompt, outputs, false, true).token_ids !=
                        reference.token_ids)
                        throw std::runtime_error("device policy reset differs from AR reference");
                }
            }
        }
        nlohmann::json result{
            {"input_tokens", ids.size()},
            {"requested_output_tokens", count},
            {"autoregressive_ids", vanilla.token_ids},
            {"eagle3_ids", eagle.token_ids},
            {"chain_ids", chain.token_ids},
            {"chain_accepted_lengths", chain_accepted},
            {"tokens_equal", equal},
            {"reset_equal", reset_equal},
            {"short_prompt_equal", short_prompt_equal},
            {"device_policy_checks", policy_checks},
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
