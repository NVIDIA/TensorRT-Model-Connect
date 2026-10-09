/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gemma/runtime/sampler.h"
#include "trtmc/task.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "families/gemma/runtime/edge_llm/media.h"
#endif

#include <iostream>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static trtmc::GemmaSampleResult greedy_sample(const trtmc::GemmaSamplingParams& params,
                                              int32_t winning_token) {
    std::vector<float> logits(8, 0.0F);
    logits[static_cast<std::size_t>(winning_token)] = 1.0F;
    auto sampler = trtmc::create_gemma_sampler(params);
    return sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
}

static void test_any_default_eos_stops_generation() {
    trtmc::TextGenerationConfig config;
    const std::vector<int32_t> defaults{5, 7};
    const auto params = trtmc::gemma_sampling_params_from_config(config, defaults);

    check(params.eos_token_ids == defaults, "sampler: preserves all default EOS IDs");
    check(greedy_sample(params, 7).is_eos, "sampler: second default EOS stops generation");
}

static void test_request_eos_overrides_model_defaults() {
    trtmc::TextGenerationConfig config;
    config.eos_token_id = 3;
    const auto params =
        trtmc::gemma_sampling_params_from_config(config, std::vector<int32_t>{5, 7});

    check(params.eos_token_ids == std::vector<int32_t>({3}),
          "sampler: request EOS replaces model defaults");
    check(!greedy_sample(params, 7).is_eos,
          "sampler: overridden model EOS no longer stops generation");
    check(greedy_sample(params, 3).is_eos, "sampler: explicit request EOS stops generation");
}

int main() {
    test_any_default_eos_stops_generation();
    test_request_eos_overrides_model_defaults();
#ifdef TRTMC_HAS_EDGE_LLM
    trtmc::TextGenerationConfig config;
    auto request = trtmc::gemma::edge_llm::make_request("hello", config, 128);
    check(request.maxGenerateLength == 128 && request.topK == 1,
          "MTP preserves the default greedy request");
    config.use_chat_template = true;
    config.enable_thinking = false;
    const auto chat = trtmc::gemma::edge_llm::make_request("\xe2\x80\x83hello \n", config, 128);
    check(chat.applyChatTemplate && !chat.enableThinking &&
              chat.requests.front().messages.front().contents.front().content ==
                  "\xe2\x80\x83hello \n",
          "Edge provider template receives the original prompt and disabled thinking");
    config.enable_thinking = true;
    const auto thought = trtmc::gemma::edge_llm::make_request("hello", config, 128);
    check(thought.applyChatTemplate && thought.enableThinking &&
              thought.requests.front().messages.front().contents.front().content == "hello",
          "Edge provider template owns enabled thinking formatting");
    config.use_chat_template = false;
    const auto raw = trtmc::gemma::edge_llm::make_request(" hello ", config, 128);
    check(raw.requests.front().messages.front().contents.front().content == " hello ",
          "Raw Gemma4 text must remain unmodified");
    config.seed = 17;
    const auto seeded = trtmc::gemma::edge_llm::make_request("hello", config, 128);
    check(seeded.samplingSeed == 17, "Gemma4 forwards an explicit supported seed");
    config.top_k = 50;
    bool rejected = false;
    try {
        (void)trtmc::gemma::edge_llm::make_request("hello", config, 128);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "MTP retains its unexecuted sampled-mode restriction");
    for (const auto* variant : {"eagle3", "dflash"}) {
        check(trtmc::gemma::edge_llm::allows_sampling(variant),
              "Command-successful requests must not be blocked by failed activity validation");
        check(trtmc::gemma::edge_llm::sampling_uses_vanilla(variant),
              "Sampled EAGLE3/DFlash must disclose Edge vanilla fallback");
        const auto forwarded = trtmc::gemma::edge_llm::make_request(
            "hello", config, 128, trtmc::gemma::edge_llm::allows_sampling(variant));
        check(forwarded.topK == 50 && forwarded.temperature == config.temperature &&
                  forwarded.samplingSeed == 17,
              "Forward sampling without forcing a quality-failing request to greedy");
    }
    check(!trtmc::gemma::edge_llm::allows_sampling("mtp"),
          "Do not introduce an unexecuted MTP sampled mode");
    const auto sampled = trtmc::gemma::edge_llm::make_request("hello", config, 128, true);
    check(sampled.topK == 50 && sampled.temperature == config.temperature,
          "DSpark must preserve supported sampling controls");

    const auto media = trtmc::gemma::edge_llm::generation_config({}, true, 128);
    check(media.use_chat_template && media.max_new_tokens == 128,
          "Typed media defaults require the provider template and preserve the budget");
    const trtmc::internal::ConfigEntry raw_media[]{{"use_chat_template", false}};
    rejected = false;
    try {
        (void)trtmc::gemma::edge_llm::generation_config(raw_media, true, 128);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "Typed media must reject bypassing the provider template");
    const trtmc::internal::ConfigEntry oversized_seed[]{{"seed", std::int64_t{2147483648}}};
    rejected = false;
    try {
        (void)trtmc::gemma::edge_llm::generation_config(oversized_seed, false, 128);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "Typed seed mapping must reject int32 overflow rather than truncate");

#endif

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All Gemma sampler tests passed.\n";
    return 0;
}
