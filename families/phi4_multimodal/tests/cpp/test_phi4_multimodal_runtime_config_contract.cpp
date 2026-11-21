/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phi4_multimodal/runtime/image_preprocessor.h"
#include "families/phi4_multimodal/runtime/phi4_longrope.h"
#include "families/phi4_multimodal/runtime/sampler.h"

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
    trtmc::Phi4MultimodalSamplingParams sampling;
    sampling.eos_token_id = 199999;
    sampling.additional_eos_token_ids = {200020, 199999};
    check(sampling.is_eos(199999), "checkpoint EOS");
    check(sampling.is_eos(200020), "checkpoint end-of-turn EOS");
    check(!sampling.is_eos(7), "ordinary token is not EOS");
    sampling.eos_token_id = 7;
    sampling.additional_eos_token_ids.clear();
    check(sampling.is_eos(7) && !sampling.is_eos(200020),
          "explicit request EOS overrides defaults");
    for (const int seed : {-1, 19}) {
        trtmc::Phi4MultimodalSamplingParams params;
        params.seed = seed;
        params.top_k = seed < 0 ? 1 : 3;
        params.eos_token_id = 1;
        params.additional_eos_token_ids = {2};
        auto sampler = trtmc::create_phi4_multimodal_sampler(params);
        const float eos_logits[] = {-1000.0F, -1000.0F, 1000.0F};
        auto result = sampler->sample(eos_logits, 3, params);
        check(result.token_id == 2 && result.is_eos,
              "sampler recognizes additional checkpoint EOS");
        params.additional_eos_token_ids.clear();
        sampler->reset();
        result = sampler->sample(eos_logits, 3, params);
        check(result.token_id == 2 && !result.is_eos, "sampler honors explicit EOS override");
    }
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
    // LongRoPE: short factors up to the threshold, long factors beyond it.
    check(trtmc::phi4_multimodal_rope_long_factor(4096, 4096) == 0, "4096 tokens stay short");
    check(trtmc::phi4_multimodal_rope_long_factor(4097, 4096) == 1, "4097 tokens use long");
    check(trtmc::phi4_multimodal_rope_long_factor(100000, 0) == 0, "threshold 0 has one regime");
    if (failures == 0)
        std::cout << "PASS: Phi4 runtime preprocessing config contract\n";
    return failures == 0 ? 0 : 1;
}
