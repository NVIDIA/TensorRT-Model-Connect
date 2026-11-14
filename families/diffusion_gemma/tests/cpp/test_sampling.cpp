/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/diffusion_gemma/runtime/pipeline.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        using namespace trtmc::diffusion_gemma;
        RuntimeConfig token_config;
        token_config.vocab_size = 262144;
        token_config.unsupported_token_ids = {255999, 258880, 258882};
        const std::vector<std::int32_t> generated_canvas = {2, 255999, 258880, 258882, 1};
        validate_token_ids(generated_canvas, token_config, false);
        bool media_rejected = false;
        try {
            validate_token_ids(generated_canvas, token_config, true);
        } catch (const std::invalid_argument&) {
            media_rejected = true;
        }
        check(media_rejected, "external image operands remain unsupported by a text bundle");
        for (const bool external_prompt : {false, true}) {
            for (const std::int32_t invalid : {-1, 262144}) {
                bool rejected = false;
                try {
                    validate_token_ids({2, invalid}, token_config, external_prompt);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                check(rejected, "internal and external tokens must stay inside the vocabulary");
            }
        }
        const auto causal = attention_mask(3, 2, 3, false);
        check(causal.size() == 15, "causal mask shape");
        check(causal[0] == 0.0F && causal[2] == 0.0F && std::isinf(causal[3]),
              "causal prefill does not attend to future positions");
        check(std::isinf(causal[5]) && causal[6] == 0.0F && causal[8] == 0.0F,
              "sliding causal window has the correct strict left boundary");
        const auto decoder = attention_mask(3, 8, 3, true);
        for (const auto value : decoder)
            check(value == 0.0F, "decoder attends fully over retained prefix and whole canvas");
        float logits[] = {0.0F, std::log(3.0F), 0.0F, std::log(3.0F)};
        std::mt19937_64 first(1729), second(1729);
        const auto result = sample_canvas(logits, 2, 2, 1.0F, 0.1, first);
        const auto repeat = sample_canvas(logits, 2, 2, 1.0F, 0.1, second);
        check(result.argmax == std::vector<std::int32_t>({1, 1}), "argmax canvas");
        check(std::abs(result.mean_entropy - 0.5623351446) < 1e-7,
              "entropy matches categorical reference, not max-probability confidence");
        check(result.next_canvas == repeat.next_canvas, "native seed reproducibility");
        std::mt19937_64 restrictive_seed(0), loose_seed(0);
        const auto restrictive = sample_canvas(logits, 2, 2, 1.0F, 0.1, restrictive_seed);
        const auto loose = sample_canvas(logits, 2, 2, 1.0F, 10.0, loose_seed);
        check(restrictive.next_canvas == std::vector<std::int32_t>({0, 0}),
              "restrictive entropy bound replaces the second sampled token");
        check(loose.next_canvas == std::vector<std::int32_t>({0, 1}),
              "loose entropy bound retains both categorical samples");
        const auto cooler = sample_canvas(logits, 2, 2, 0.5F, 0.1, first);
        check(std::abs(cooler.mean_entropy - 0.3250829734) < 1e-7,
              "checkpoint temperature schedule shapes the entropy distribution");
        logits[0] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try {
            (void)sample_canvas(logits, 2, 2, 1.0F, 0.1, first);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "non-finite logits must fail the sampler");
        std::cout << "diffusion_gemma sampling contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
