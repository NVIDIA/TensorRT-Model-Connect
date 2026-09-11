/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phi4_multimodal/runtime/image_preprocessor.h"

#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
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
        (void)trtmc::phi4_multimodal_parse_preprocess_config(text);
        check(false, message);
    } catch (const nlohmann::json::exception&) {
    } catch (const std::runtime_error&) {
    }
}
} // namespace

int main() {
    // runtime.json contains the promoted preprocessing fields, not the HF decoder config.
    // Keep nested decoys first to catch a first-occurrence scan of the JSON text.
    const std::string text = R"({
        "img_processor": {"image_token_id": 7, "vision_output_dim": 64,
                          "num_image_pad_tokens": 3},
        "preprocessor_type": "phi4_hd_chw",
        "image_token_id": 200010,
        "fixed_image_size": 336,
        "patch_size": 16,
        "merge_size": 3,
        "temporal_patch_size": 1,
        "num_image_pad_tokens": 721,
        "vision_output_dim": 3072,
        "vl_prompt_template": "<|user|>\n{image_pads}{prompt}\n<|end|><|assistant|>",
        "image_token_str": "<|endoftext10|>",
        "interpolation": "bilinear",
        "image_mean": [0.1, 0.2, 0.3],
        "image_std": [0.6, 0.7, 0.8]
    })";
    const auto cfg = trtmc::phi4_multimodal_parse_preprocess_config(text);
    check(cfg.preprocessor_type == "phi4_hd_chw", "preprocessor type");
    check(cfg.image_token_id == 200010, "top-level image token ID");
    check(cfg.fixed_image_size == 336, "fixed image size");
    check(cfg.patch_size == 16, "patch size");
    check(cfg.merge_size == 3, "merge size");
    check(cfg.temporal_patch_size == 1, "temporal patch size");
    check(cfg.num_image_pad_tokens == 721, "top-level image pad token count");
    check(cfg.vision_output_dim == 3072, "top-level vision output dimension");
    check(cfg.vl_prompt_template == "<|user|>\n{image_pads}{prompt}\n<|end|><|assistant|>",
          "template newlines are decoded");
    check(cfg.image_token_str == "<|endoftext10|>", "image token string");
    check(cfg.interpolation == "bilinear", "interpolation");
    const float mean[] = {0.1F, 0.2F, 0.3F};
    const float std[] = {0.6F, 0.7F, 0.8F};
    for (int i = 0; i < 3; ++i) {
        check(std::fabs(cfg.image_mean[i] - mean[i]) < 1e-6F, "image mean component");
        check(std::fabs(cfg.image_std[i] - std[i]) < 1e-6F, "image std component");
    }

    const auto document = nlohmann::json::parse(text);
    for (const auto& item : document.items()) {
        if (item.key() == "img_processor")
            continue;
        auto missing = document;
        missing.erase(item.key());
        rejects(missing.dump(), "missing " + item.key());
        auto wrong_type = document;
        wrong_type[item.key()] = nullptr;
        rejects(wrong_type.dump(), "null " + item.key());
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
    for (const auto* key : {"fixed_image_size", "patch_size", "merge_size", "temporal_patch_size",
                            "vision_output_dim"}) {
        for (int value : {0, -1}) {
            auto invalid = document;
            invalid[key] = value;
            rejects(invalid.dump(), std::string("non-positive ") + key);
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
    for (const auto* mode : {"nearest", "bilinear", "bicubic"}) {
        auto valid = document;
        valid["interpolation"] = mode;
        check(trtmc::phi4_multimodal_parse_preprocess_config(valid.dump()).interpolation == mode,
              std::string("accepted interpolation ") + mode);
    }
    auto invalid = document;
    invalid["interpolation"] = "lanczos";
    rejects(invalid.dump(), "unsupported interpolation");
    rejects("{", "malformed JSON");
    rejects("[]", "non-object JSON");
    if (failures == 0)
        std::cout << "PASS: Phi4 runtime preprocessing config contract\n";
    return failures == 0 ? 0 : 1;
}
