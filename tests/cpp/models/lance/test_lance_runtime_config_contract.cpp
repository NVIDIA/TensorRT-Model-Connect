/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for Lance's serialized VL preprocessing config.
//
// Lance is the only family whose preprocessing config is resolved from two
// bundle sections at once: plugin.cpp passes config.json and
// preprocessor_config.json to lance_parse_preprocess_config(), and the two
// sources do not merge in a single direction.
//
//   patch_size / merge_size / temporal_patch_size -> preprocessor_config.json wins
//   image_mean / image_std                        -> config.json wins
//
// The precedence is opposite for the two groups because the overrides are
// applied in a fixed order, and nothing pinned either direction. This test
// fixes both, plus the resample fallback that must stay suppressed whenever
// config.json states an interpolation of its own.

#include "runtime/models/lance/image_preprocessor.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

bool near(float actual, float expected) {
    return std::fabs(actual - expected) < 1e-6F;
}

} // namespace

int main() {
    bool ok = true;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ok = false;
        }
    };

    // Deliberately non-default values throughout, so a consumer that stopped
    // reading a key and fell back to the struct default fails here.
    const std::string config = R"({
        "model_type": "lance",
        "preprocessor_type": "pad_center_chw",
        "interpolation": "nearest",
        "image_token_id": 151655,
        "fixed_image_size": 392,
        "patch_size": 16,
        "merge_size": 3,
        "temporal_patch_size": 4,
        "num_image_pad_tokens": 81,
        "vision_output_dim": 2048,
        "image_token_str": "<|image_pad|>",
        "vl_prompt_template": "<|im_start|>user\\n{image_pads}{prompt}<|im_end|>\\n<|im_start|>assistant\\n",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25]
    })";

    // Same three geometry keys, different values, plus its own normalization
    // triplet that config.json is expected to overrule.
    const std::string preprocessor_config = R"({
        "patch_size": 28,
        "merge_size": 5,
        "temporal_patch_size": 6,
        "resample": 2,
        "image_mean": [0.1, 0.2, 0.3],
        "image_std": [0.9, 0.8, 0.7]
    })";

    // The family's own consumer: plugin.cpp builds its LancePreprocessConfig
    // with exactly this call over the bundle's config.json and
    // preprocessor_config.json sections.
    const auto cfg = trtmc::lance_parse_preprocess_config(config, preprocessor_config);

    // Keys that only config.json carries.
    check(cfg.preprocessor_type == "pad_center_chw", "preprocessor strategy");
    check(cfg.image_token_id == 151655, "image token id");
    check(cfg.fixed_image_size == 392, "fixed image size");
    check(cfg.num_image_pad_tokens == 81, "image pad token count");
    check(cfg.vision_output_dim == 2048, "vision output dim");
    check(cfg.image_token_str == "<|image_pad|>", "image placeholder token");

    // Escaped newlines in the prompt template are decoded once, at parse time.
    check(cfg.vl_prompt_template.find("\\n") == std::string::npos,
          "prompt template carries no literal backslash-n");
    check(cfg.vl_prompt_template ==
              "<|im_start|>user\n{image_pads}{prompt}<|im_end|>\n<|im_start|>assistant\n",
          "prompt template");

    // Direction one: preprocessor_config.json overrules config.json.
    check(cfg.patch_size == 28, "patch size comes from preprocessor_config.json");
    check(cfg.merge_size == 5, "merge size comes from preprocessor_config.json");
    check(cfg.temporal_patch_size == 6, "temporal patch size comes from preprocessor_config.json");

    // Direction two: config.json overrules preprocessor_config.json.
    check(near(cfg.image_mean[0], 0.5F) && near(cfg.image_mean[1], 0.5F) &&
              near(cfg.image_mean[2], 0.5F),
          "image mean comes from config.json");
    check(near(cfg.image_std[0], 0.25F) && near(cfg.image_std[1], 0.25F) &&
              near(cfg.image_std[2], 0.25F),
          "image std comes from config.json");

    // An interpolation stated in config.json suppresses the resample fallback,
    // so "resample": 2 must not turn "nearest" into "bilinear".
    check(cfg.interpolation == "nearest", "stated interpolation survives the resample fallback");

    // With no interpolation stated, resample resolves it: 0 nearest, 2
    // bilinear, 3 bicubic, anything else leaves the bicubic default.
    const std::string config_without_interpolation = R"({"model_type": "lance"})";
    const auto from_resample = [&](const char* preproc) {
        return trtmc::lance_parse_preprocess_config(config_without_interpolation, preproc)
            .interpolation;
    };
    check(from_resample(R"({"resample": 0})") == "nearest", "resample 0 resolves to nearest");
    check(from_resample(R"({"resample": 2})") == "bilinear", "resample 2 resolves to bilinear");
    check(from_resample(R"({"resample": 3})") == "bicubic", "resample 3 resolves to bicubic");
    check(from_resample(R"({"resample": 7})") == "bicubic", "unknown resample keeps the default");

    // A preprocessor_config.json that omits the geometry keys still overrides
    // them, resetting config.json's values to the override defaults rather
    // than preserving them. Pinned because it is the surprising half of
    // direction one, and a merge that only wrote present keys would change it.
    const auto reset = trtmc::lance_parse_preprocess_config(config, R"({"resample": 3})");
    check(reset.patch_size == 14, "an absent patch_size override resets to 14");
    check(reset.merge_size == 2, "an absent merge_size override resets to 2");
    check(reset.temporal_patch_size == 2, "an absent temporal_patch_size override resets to 2");

    // With no preprocessor_config.json section at all the overrides are
    // skipped entirely and config.json's geometry stands.
    const auto config_only = trtmc::lance_parse_preprocess_config(config, "");
    check(config_only.patch_size == 16, "config.json patch size stands without an override file");
    check(config_only.merge_size == 3, "config.json merge size stands without an override file");
    check(config_only.temporal_patch_size == 4,
          "config.json temporal patch size stands without an override file");

    return ok ? 0 : 1;
}
