/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/flux/flux_clip_helpers.h"
#include "runtime/models/flux/flux_rope_helpers.h"
#include "runtime/models/flux/flux_text_helpers.h"
#include "runtime/models/flux/pipeline.h"

#include <algorithm>
#include <iostream>
#include <string>

namespace {

int failures = 0;

class PadTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {}; }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view token) const override {
        return token == "<pad>" ? 11 : -1;
    }
    std::string token_for_id(int32_t) const override { return {}; }
};

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_flux_construction() {
    trtmc::FluxDiffusionConfig cfg;
    trtmc::FluxPreprocessorWeights weights;

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, weights, nullptr, nullptr, "test-flux");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline", "FluxPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-flux", "FluxPipeline model_id");
}

void test_flux_with_custom_config() {
    trtmc::FluxDiffusionConfig cfg;
    cfg.video_height = 256;
    cfg.video_width = 256;
    cfg.scale_factor_spatial = 8;
    cfg.patch_size = {1, 2, 2};

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, trtmc::FluxPreprocessorWeights{},
                                 nullptr, nullptr, "test-flux-custom");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline",
          "FluxPipeline custom config pipeline_type");
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

int main() {
    test_flux_construction();
    test_flux_with_custom_config();
    test_clip_padding_preserves_eos_when_truncated();
    test_flux2_text_padding_matches_tokenizer_contract();
    test_flux2_rope_coordinates_match_diffusers_contract();
    if (failures > 0) {
        std::cerr << failures << " flux pipeline test(s) FAILED\n";
    }
    return failures;
}
