/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deepseek_ocr/runtime/image_preprocessor.h"
#include "families/deepseek_ocr/runtime/runtime_config.h"
#include "families/deepseek_ocr/runtime/tensor_names.h"

#include <iostream>
#include <string>

int main() {
    const std::string runtime = R"({
        "language_config": {"vocab_size": 1, "num_hidden_layers": 2},
        "vision_config": {"image_token_id": 3, "vision_output_dim": 4},
        "tensor_parallel_size": 1,
        "num_layers": 12,
        "max_cache_length": 768,
        "vocab_size": 129280,
        "id_bos": 0,
        "id_eos": 1,
        "preprocessor_type": "simple_chw",
        "image_token_id": 128815,
        "fixed_image_size": 768,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 1,
        "num_image_pad_tokens": 145,
        "vision_output_dim": 1280,
        "vl_prompt_template": "{image_pads}{prompt}",
        "image_token_str": "<image>",
        "interpolation": "bicubic",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25],
        "prefill_max_length": 768,
        "io_map": {
            "cache_k_pattern": "cache_k_{i}",
            "cache_v_pattern": "cache_v_{i}",
            "present_k_pattern": "present_k_{i}",
            "present_v_pattern": "present_v_{i}"
        }
    })";

    const auto config = trtmc::deepseek_ocr_parse_preprocess_config(runtime);
    const auto model = trtmc::deepseek_ocr_parse_runtime_config(runtime);
    bool ok = true;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ok = false;
        }
    };
    check(config.image_token_id == 128815, "top-level image token");
    check(config.vision_output_dim == 1280, "top-level vision width");
    check(config.num_image_pad_tokens == 145, "image pad count");
    check(config.fixed_image_size == 768, "fixed image size");
    check(config.preprocessor_type == "simple_chw", "preprocessor type");
    check(config.interpolation == "bicubic", "interpolation");
    check(config.temporal_patch_size == 1, "temporal patch size");
    check(config.image_token_str == "<image>", "image token string");
    check(model.tensor_parallel_size == 1, "tensor parallel size");
    check(model.max_cache_length == 768, "max cache length");
    check(model.model.vocab_size == 129280, "top-level vocabulary size");
    check(model.model.num_layers == 12, "top-level layer count");
    check(model.model.id_bos == 0, "BOS token");
    check(model.model.id_eos == 1, "EOS token");
    check(model.model.image_token_id == 128815, "runtime image token");
    check(model.model.vision_output_dim == 1280, "runtime vision width");
    check(model.model.prefill_max_length == 768, "prefill length");
    check(model.cache_k_pattern == "cache_k_{i}", "cache K pattern");
    check(model.model.present_v_pattern == "present_v_{i}", "present V pattern");
    check(trtmc::deepseek_ocr_expand_layer_name(model.cache_k_pattern, 3) == "cache_k_3",
          "cache K expansion");
    check(trtmc::deepseek_ocr_expand_layer_name(model.model.present_v_pattern, 3) == "present_v_3",
          "present V expansion");
    return ok ? 0 : 1;
}
