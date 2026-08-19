/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-CPP-05
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-DIFF-01
// Intent:         Wan generation conditioning: mask/null IDs, text conditioning, RoPE skip,
// deterministic latents Preconditions:  Wan config with valid conditioning parameters
// Postconditions: Mask and null IDs built correctly, encoder failures propagated, latents
// deterministic by seed
// =============================================================================

#include "wan_generation_conditioning.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_conditioning_inputs_build_mask_and_null_ids() {
    trtmc::WanDiffusionConfig config;
    config.use_rope = false;

    trtmc::diffusion::WanLayout layout;
    layout.seq_len = 5;

    const auto inputs = trtmc::diffusion::make_wan_conditioning_inputs(config, layout, {11, 0, 13});

    check(inputs.null_ids.size() == 5, "wan conditioning builds null ids for sequence length");
    check(inputs.null_ids[0] == 1, "wan conditioning seeds null prompt EOS token");
    check(inputs.encoder_attn_mask.size() == 5,
          "wan conditioning builds attention mask when rope is off");
    check(inputs.encoder_attn_mask[0] == 0.0F, "wan conditioning keeps first token unmasked");
    check(inputs.encoder_attn_mask[1] == -10000.0F, "wan conditioning masks zero token positions");
    check(inputs.encoder_attn_mask[2] == 0.0F,
          "wan conditioning keeps active token positions unmasked");
}

void test_wan_t5_token_framing_matches_hf() {
    const auto legacy = trtmc::diffusion::normalize_wan_t5_token_ids({2, 289, 3735, 1}, 8, true);
    check(legacy == std::vector<int32_t>({289, 3735, 1}),
          "wan token framing removes native-only prefix");

    const auto unframed = trtmc::diffusion::normalize_wan_t5_token_ids({289, 3735}, 8, false);
    check(unframed == std::vector<int32_t>({289, 3735, 1}), "wan token framing appends HF EOS");

    const auto truncated =
        trtmc::diffusion::normalize_wan_t5_token_ids({2, 10, 11, 12, 13, 1}, 4, true);
    check(truncated == std::vector<int32_t>({10, 11, 12, 1}),
          "wan token framing preserves EOS when truncating");
}

void test_text_conditioning_uses_both_prompt_and_null_prompt() {
    trtmc::diffusion::WanConditioningInputs inputs;
    inputs.null_ids = {1, 0, 0};
    std::string error;
    int encoder_calls = 0;
    int projector_calls = 0;
    trtmc::diffusion::WanTextConditioning conditioning;

    const bool ok = trtmc::diffusion::build_wan_text_conditioning(
        {7, 8, 9}, inputs, 3, error,
        [&encoder_calls](const std::vector<int32_t>& ids, std::vector<float>& embeddings,
                         std::string&) {
            ++encoder_calls;
            embeddings.assign(ids.begin(), ids.end());
            return true;
        },
        [&projector_calls](const std::vector<float>& embeddings, int32_t seq_len,
                           std::vector<float>& projected) {
            ++projector_calls;
            projected = embeddings;
            projected.push_back(static_cast<float>(seq_len));
        },
        conditioning);

    check(ok, "wan text conditioning succeeds with stub encoder/projector");
    check(encoder_calls == 2, "wan text conditioning encodes prompt and null prompt");
    check(projector_calls == 2, "wan text conditioning projects both embedding sets");
    check(conditioning.text_projected.size() == 4,
          "wan text conditioning stores prompt projection");
    check(conditioning.null_text.size() == 4, "wan text conditioning stores null projection");
    check(conditioning.null_text[0] == 1.0F, "wan text conditioning uses configured null ids");
}

void test_conditioning_inputs_skip_attention_mask_when_rope_enabled() {
    trtmc::WanDiffusionConfig config;
    config.use_rope = true;

    trtmc::diffusion::WanLayout layout;
    layout.seq_len = 4;

    const auto inputs = trtmc::diffusion::make_wan_conditioning_inputs(config, layout, {});

    check(inputs.null_ids == std::vector<int32_t>({1, 0, 0, 0}),
          "wan rope conditioning still builds null ids");
    check(inputs.encoder_attn_mask.empty(), "wan rope conditioning omits encoder attention mask");
}

void test_text_conditioning_propagates_encoder_failures() {
    trtmc::diffusion::WanConditioningInputs inputs;
    inputs.null_ids = {1, 0, 0};
    std::string error;
    trtmc::diffusion::WanTextConditioning conditioning;

    const bool prompt_fail = trtmc::diffusion::build_wan_text_conditioning(
        {1, 2}, inputs, 2, error,
        [](const std::vector<int32_t>&, std::vector<float>&, std::string& err) {
            err = "prompt encode failed";
            return false;
        },
        [](const std::vector<float>&, int32_t, std::vector<float>&) {}, conditioning);

    check(!prompt_fail, "wan text conditioning propagates prompt encoder failure");
    check(error == "prompt encode failed", "wan text conditioning preserves prompt encoder error");

    error.clear();
    int encoder_calls = 0;
    const bool null_fail = trtmc::diffusion::build_wan_text_conditioning(
        {1, 2}, inputs, 2, error,
        [&encoder_calls](const std::vector<int32_t>& ids, std::vector<float>& embeddings,
                         std::string& err) {
            ++encoder_calls;
            if (encoder_calls == 2) {
                err = "null encode failed";
                return false;
            }
            embeddings.assign(ids.begin(), ids.end());
            return true;
        },
        [](const std::vector<float>& embeddings, int32_t seq_len, std::vector<float>& projected) {
            projected = embeddings;
            projected.push_back(static_cast<float>(seq_len));
        },
        conditioning);

    check(!null_fail, "wan text conditioning propagates null encoder failure");
    check(error == "null encode failed", "wan text conditioning preserves null encoder error");
}

void test_text_conditioning_rejects_non_finite_embeddings() {
    trtmc::diffusion::WanConditioningInputs inputs;
    inputs.null_ids = {1, 0};
    std::string error;
    trtmc::diffusion::WanTextConditioning conditioning;

    const bool ok = trtmc::diffusion::build_wan_text_conditioning(
        {7, 8}, inputs, 2, error,
        [](const std::vector<int32_t>&, std::vector<float>& embeddings, std::string&) {
            embeddings = {1.0F, std::numeric_limits<float>::quiet_NaN()};
            return true;
        },
        [](const std::vector<float>& embeddings, int32_t, std::vector<float>& projected) {
            projected = embeddings;
        },
        conditioning);

    check(!ok, "wan text conditioning rejects non-finite T5 embeddings");
    check(error.find("non-finite") != std::string::npos,
          "wan text conditioning explains non-finite T5 failure");
}

void test_initial_latents_are_deterministic_by_seed() {
    const auto a = trtmc::diffusion::make_wan_initial_latents(6, 42U);
    const auto b = trtmc::diffusion::make_wan_initial_latents(6, 42U);
    const auto c = trtmc::diffusion::make_wan_initial_latents(6, 7U);

    check(a == b, "wan initial latents are deterministic for identical seeds");
    check(a != c, "wan initial latents change for different seeds");
    check(std::fabs(a[0]) > 0.0F, "wan initial latents contain gaussian samples");
}

void test_initial_latents_handle_odd_sizes() {
    const auto latents = trtmc::diffusion::make_wan_initial_latents(5, 9U);
    check(latents.size() == 5, "wan initial latents preserve odd latent count");
    check(std::fabs(latents[4]) > 0.0F, "wan initial latents fill trailing odd element");
}

void test_initial_latents_honor_caller_bytes_and_size() {
    const std::vector<float> supplied = {0.25F, -0.5F, 0.75F, -1.0F};
    std::vector<float> latents;
    std::string error;

    check(trtmc::diffusion::resolve_wan_initial_latents(4, supplied, 99, latents, error),
          "wan initial latents accept a caller override");
    check(latents == supplied, "wan initial latents preserve caller bytes");

    error.clear();
    check(!trtmc::diffusion::resolve_wan_initial_latents(5, supplied, 99, latents, error),
          "wan initial latents reject a mismatched caller override");
    check(!error.empty(), "wan initial latent size mismatch reports an error");
}

void test_initial_latents_honor_requested_seed() {
    std::vector<float> seed_one;
    std::vector<float> seed_two;
    std::string error;

    check(trtmc::diffusion::resolve_wan_initial_latents(8, {}, 1, seed_one, error),
          "wan initial latents generate from the requested seed");
    check(trtmc::diffusion::resolve_wan_initial_latents(8, {}, 2, seed_two, error),
          "wan initial latents generate from a different requested seed");
    check(seed_one != seed_two, "wan requested seeds produce different latents");
}

} // namespace

int main() {
    test_conditioning_inputs_build_mask_and_null_ids();
    test_wan_t5_token_framing_matches_hf();
    test_text_conditioning_uses_both_prompt_and_null_prompt();
    test_conditioning_inputs_skip_attention_mask_when_rope_enabled();
    test_text_conditioning_propagates_encoder_failures();
    test_text_conditioning_rejects_non_finite_embeddings();
    test_initial_latents_are_deterministic_by_seed();
    test_initial_latents_handle_odd_sizes();
    test_initial_latents_honor_caller_bytes_and_size();
    test_initial_latents_honor_requested_seed();

    if (g_failures != 0) {
        std::cerr << g_failures << " wan generation conditioning test(s) failed\n";
        return 1;
    }
    return 0;
}
