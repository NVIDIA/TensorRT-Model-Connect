/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/flux/runtime/flux_clip_helpers.h"
#include "families/flux/runtime/flux_rope_helpers.h"
#include "families/flux/runtime/flux_text_helpers.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

class PadTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {}; }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view token) const override {
        if (token == "<pad>")
            return 11;
        return token == "<|endoftext|>" ? 151643 : -1;
    }
    std::string token_for_id(int32_t) const override { return {}; }
};

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_clip_padding_preserves_eos_when_truncated() {
    using trtmc::diffusion::flux_clip::pad_and_truncate_ids;

    check(pad_and_truncate_ids({10, 1, 11}, 5, 11, 11) == std::vector<int32_t>({10, 1, 11, 11, 11}),
          "CLIP short input pads with EOS");
    check(pad_and_truncate_ids({10, 1, 2, 3, 11}, 5, 11, 11) ==
              std::vector<int32_t>({10, 1, 2, 3, 11}),
          "CLIP exact input preserves EOS");
    check(pad_and_truncate_ids({10, 1, 2, 3, 4, 11}, 5, 11, 11) ==
              std::vector<int32_t>({10, 1, 2, 3, 11}),
          "CLIP truncated input restores EOS");
}

void test_flux2_text_padding_matches_tokenizer_contract() {
    using trtmc::diffusion::flux_text::clear_padding_rows;
    using trtmc::diffusion::flux_text::prepare_inputs;
    using trtmc::diffusion::flux_text::resolve_pad_token_id;

    PadTokenizer tokenizer;
    check(resolve_pad_token_id(&tokenizer, true) == 11, "FLUX.2 resolves tokenizer pad id");
    check(resolve_pad_token_id(&tokenizer, false) == 0, "FLUX.1 retains legacy pad id");

    const auto prepared = prepare_inputs({12, 0, 13}, 5, 11);
    check(prepared.input_ids == std::vector<int32_t>({12, 0, 13, 11, 11}),
          "FLUX.2 text uses tokenizer pad id");
    check(prepared.attention_mask == std::vector<float>({0.0F, 0.0F, 0.0F, -1e9F, -1e9F}),
          "FLUX.2 text mask follows input length rather than token value");

    std::vector<float> embeddings(10, 1.0F);
    clear_padding_rows(embeddings, {3}, 5, 2, true);
    check(std::all_of(embeddings.begin(), embeddings.end(),
                      [](float value) { return value == 1.0F; }),
          "FLUX.2 preserves padded query embeddings");
    clear_padding_rows(embeddings, {3}, 5, 2, false);
    check(embeddings ==
              std::vector<float>({1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F}),
          "FLUX.1 clears padded query embeddings");
}

void test_flux2_rope_coordinates_match_diffusers_contract() {
    using trtmc::diffusion::flux_rope::axis_position;

    check(axis_position(0, 0, 0, 0, 17) == 0, "FLUX.2 text temporal coordinate");
    check(axis_position(1, 0, 0, 0, 17) == 0, "FLUX.2 text height coordinate");
    check(axis_position(2, 0, 0, 0, 17) == 0, "FLUX.2 text width coordinate");
    check(axis_position(3, 0, 0, 0, 17) == 17, "FLUX.2 text sequence coordinate");

    check(axis_position(0, 0, 5, 7, 0) == 0, "FLUX.2 image temporal coordinate");
    check(axis_position(1, 0, 5, 7, 0) == 5, "FLUX.2 image height coordinate");
    check(axis_position(2, 0, 5, 7, 0) == 7, "FLUX.2 image width coordinate");
    check(axis_position(3, 0, 5, 7, 0) == 0, "FLUX.2 image sequence coordinate");
}

} // namespace

void test_flux2_klein_text_contract() {
    using trtmc::diffusion::flux_text::apply_flux2_template;
    using trtmc::diffusion::flux_text::resolve_pad_token_id;

    PadTokenizer tokenizer;
    check(resolve_pad_token_id(&tokenizer, true, "<|endoftext|>") == 151643,
          "FLUX.2-klein resolves the Qwen pad token");
    // Qwen3 apply_chat_template(add_generation_prompt=True, enable_thinking=False).
    check(apply_flux2_template("a cat", "qwen3_chat") ==
              "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "FLUX.2-klein Qwen3 chat template");
    check(apply_flux2_template("x", "").rfind("<s>[SYSTEM_PROMPT]", 0) == 0,
          "FLUX.2-dev keeps the Mistral template");
    bool threw = false;
    try {
        (void)apply_flux2_template("x", "unknown");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "unknown FLUX.2 template is rejected");
}

void test_flux2_true_cfg_matches_diffusers_formula() {
    using trtmc::diffusion::flux_text::combine_true_cfg;

    std::vector<float> positive{1.0F, -2.0F, 0.5F};
    const std::vector<float> negative{0.5F, 1.0F, 0.5F};
    combine_true_cfg(positive, negative, 4.0F);
    // neg + 4 * (pos - neg)
    check(positive == std::vector<float>({2.5F, -11.0F, 0.5F}), "true CFG combination");
}

int main() {
    test_clip_padding_preserves_eos_when_truncated();
    test_flux2_text_padding_matches_tokenizer_contract();
    test_flux2_rope_coordinates_match_diffusers_contract();
    test_flux2_klein_text_contract();
    test_flux2_true_cfg_matches_diffusers_formula();
    if (failures > 0) {
        std::cerr << failures << " flux host contract test(s) FAILED\n";
    }
    return failures;
}
