/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lance/runtime/image_preprocessor.h"

#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void rejects(const std::string& text, const std::string& message) {
    try {
        (void)trtmc::lance_parse_preprocess_config(text);
        check(false, message);
    } catch (const std::exception&) {
    }
}

const std::string valid_runtime = R"JSON({
    "vision_config": {
      "preprocessor_type": "decoy",
      "image_token_id": 7,
      "fixed_image_size": 32,
      "patch_size": 8,
      "merge_size": 1,
      "temporal_patch_size": 1,
      "num_image_pad_tokens": 3,
      "vision_output_dim": 64,
      "vl_prompt_template": "decoy {prompt}",
      "image_token_str": "<decoy>",
      "interpolation": "bicubic",
      "image_mean": [9.0, 9.0, 9.0],
      "image_std": [8.0, 8.0, 8.0]
    },
    "preprocessor_type": "pad_center_chw",
    "image_token_id": 31415,
    "fixed_image_size": 448,
    "patch_size": 16,
    "merge_size": 3,
    "temporal_patch_size": 4,
    "num_image_pad_tokens": 42,
    "vision_output_dim": 2048,
    "vl_prompt_template": "prefix\n\"quoted\"\\ {image_pads} :: {prompt}",
    "image_token_str": "<image>",
    "interpolation": "nearest",
    "image_mean": [0.1, 0.2, 0.3],
    "image_std": [1.1, 1.2, 1.3]
  })JSON";

} // namespace

int main() {
    const auto document = nlohmann::json::parse(valid_runtime);
    const auto config = trtmc::lance_parse_preprocess_config(valid_runtime);

    check(config.preprocessor_type == "pad_center_chw", "top-level preprocessor type");
    check(config.image_token_id == 31415, "top-level image token ID");
    check(config.fixed_image_size == 448, "fixed image size");
    check(config.patch_size == 16, "patch size");
    check(config.merge_size == 3, "merge size");
    check(config.temporal_patch_size == 4, "temporal patch size");
    check(config.num_image_pad_tokens == 42, "top-level image pad token count");
    check(config.vision_output_dim == 2048, "top-level vision output dimension");
    check(config.vl_prompt_template == "prefix\n\"quoted\"\\ {image_pads} :: {prompt}",
          "escaped template is decoded");
    check(config.image_token_str == "<image>", "image token string");
    check(config.interpolation == "nearest", "interpolation");
    const float mean[] = {0.1F, 0.2F, 0.3F};
    const float stddev[] = {1.1F, 1.2F, 1.3F};
    for (int index = 0; index < 3; ++index) {
        check(std::fabs(config.image_mean[index] - mean[index]) < 1e-6F, "image mean component");
        check(std::fabs(config.image_std[index] - stddev[index]) < 1e-6F, "image std component");
    }

    auto format_config = config;
    format_config.num_image_pad_tokens = 2;
    check(trtmc::lance_format_prompt("go", format_config) ==
              "prefix\n\"quoted\"\\ <image><image> :: go",
          "escaped template formatting");

    // Every promoted preprocessing field is required at the runtime.json top level.
    for (const auto& item : document.items()) {
        if (item.key() == "vision_config")
            continue;
        auto missing = document;
        missing.erase(item.key());
        rejects(missing.dump(), "missing " + item.key());
        auto null_value = document;
        null_value[item.key()] = nullptr;
        rejects(null_value.dump(), "null " + item.key());
    }

    for (const auto* key : {"image_token_id", "fixed_image_size", "patch_size", "merge_size",
                            "temporal_patch_size", "num_image_pad_tokens", "vision_output_dim"}) {
        for (const auto& value :
             {nlohmann::json(1.5), nlohmann::json("16"), nlohmann::json(true)}) {
            auto invalid = document;
            invalid[key] = value;
            rejects(invalid.dump(), std::string("non-integer ") + key);
        }
    }
    for (const auto* key :
         {"preprocessor_type", "vl_prompt_template", "image_token_str", "interpolation"}) {
        for (const auto& value : {nlohmann::json(1), nlohmann::json(true),
                                  nlohmann::json::array({"not", "a", "string"})}) {
            auto invalid = document;
            invalid[key] = value;
            rejects(invalid.dump(), std::string("non-string ") + key);
        }
    }
    for (const auto* key : {"image_mean", "image_std"}) {
        for (const auto& value :
             {nlohmann::json::array({0.1, 0.2}), nlohmann::json::array({0.1, 0.2, 0.3, 0.4}),
              nlohmann::json::array({0.1, "0.2", 0.3})}) {
            auto invalid = document;
            invalid[key] = value;
            rejects(invalid.dump(), std::string("invalid triplet ") + key);
        }
    }

    for (const auto* key : {"fixed_image_size", "patch_size", "merge_size", "temporal_patch_size",
                            "vision_output_dim"}) {
        for (const int value : {0, -1}) {
            auto invalid = document;
            invalid[key] = value;
            rejects(invalid.dump(), std::string("non-positive ") + key);
        }
    }

    for (const auto* mode : {"nearest", "bilinear", "bicubic"}) {
        auto accepted = document;
        accepted["interpolation"] = mode;
        const auto parsed = trtmc::lance_parse_preprocess_config(accepted.dump());
        check(parsed.interpolation == mode, std::string("accepted interpolation ") + mode);
    }
    auto invalid_interpolation = document;
    invalid_interpolation["interpolation"] = "lanczos";
    rejects(invalid_interpolation.dump(), "unsupported interpolation");
    rejects("{", "malformed JSON");
    rejects("[]", "non-object JSON");

    if (failures == 0)
        std::cout << "PASS: Lance runtime preprocessing config contract\n";
    return failures == 0 ? 0 : 1;
}
