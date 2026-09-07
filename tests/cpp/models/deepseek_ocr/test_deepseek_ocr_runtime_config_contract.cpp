/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for DeepSeek-OCR's serialized decoder config.
//
// DeepSeek-OCR nests its decoder geometry under "language_config", so the
// family plugin promotes those fields to bundle scope via
// get_bundle_config_overrides(). The nested block, and the vision tower's own
// "vision_config" dimensions, remain in the bundle's config.json next to the
// promoted values, so nothing pinned that the runtime reads the decoder
// contract rather than either neighbour.
//
// This exercises parse_base_config(), the production parser whose BaseConfig
// plugin.cpp consumes. The VL keys the plugin reads inline while building its
// PipelineContext are not reachable without a bundle and engines, so they stay
// with the REQUIRES_TRT,REQUIRES_GPU pipeline test rather than being restated
// here as generic JSON lookups.

#include "runtime/models/deepseek_ocr/image_preprocessor.h"
#include "trtmc/runtime/pipeline_plugin.h"

#include <cmath>
#include <iostream>
#include <string>

int main() {
    // Top-level values are the promoted decoder contract. The nested
    // "language_config" and "vision_config" carry deliberately different
    // values so a consumer that reached into either would fail these checks.
    const std::string config = R"({
        "model_type": "deepseek_vl_v2",
        "language_config": {
            "vocab_size": 1,
            "hidden_size": 2,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 5
        },
        "vision_config": {
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "image_token_id": 999,
            "vision_output_dim": 777,
            "num_image_pad_tokens": 888
        },
        "vocab_size": 129280,
        "hidden_size": 1280,
        "num_hidden_layers": 12,
        "num_attention_heads": 10,
        "num_key_value_heads": 10,
        "head_dim": 128,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "max_position_embeddings": 4096,
        "image_token_id": 128815,
        "fixed_image_size": 768,
        "num_image_pad_tokens": 145,
        "vision_output_dim": 1280,
        "preprocessor_type": "simple_chw",
        "interpolation": "bicubic",
        "temporal_patch_size": 1,
        "image_token_str": "<image>",
        "has_vision_engine": 1,
        "runtime_strategy": "deepseek_ocr_vision_language"
    })";

    const auto parsed = trtmc::parse_base_config(config, 768);
    bool ok = true;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ok = false;
        }
    };

    check(parsed.runtime_strategy == "deepseek_ocr_vision_language", "runtime strategy");
    check(parsed.vocab_size == 129280, "vocabulary size");
    check(parsed.hidden_size == 1280, "hidden size");
    check(parsed.num_layers == 12, "layer count");
    check(parsed.num_heads == 10, "attention head count");
    check(parsed.num_kv_heads == 10, "KV head count");
    check(parsed.head_dim == 128, "head dimension");
    check(parsed.id_bos == 0, "BOS token");
    check(parsed.id_eos == 1, "EOS token");
    check(parsed.attention_size == 1280, "attention width");
    check(parsed.max_cache_length == 768, "max cache length override");

    // The VL contract, through the family's own consumer: plugin.cpp builds
    // its DeepseekOcrPreprocessConfig with exactly this call, so a regression
    // that stops reading a key, reads vision_config instead, or silently
    // falls back to a default fails here.
    const auto vl = trtmc::deepseek_ocr_parse_preprocess_config(config, "");
    check(vl.image_token_id == 128815, "image token id reaches the VL config");
    check(vl.vision_output_dim == 1280, "vision output dim reaches the VL config");
    check(vl.num_image_pad_tokens == 145, "image pad token count");
    check(vl.fixed_image_size == 768, "fixed image size");
    check(vl.preprocessor_type == "simple_chw", "preprocessor strategy");
    check(vl.interpolation == "bicubic", "interpolation");
    check(vl.temporal_patch_size == 1, "temporal patch size");
    check(vl.image_token_str == "<image>", "image placeholder token");

    return ok ? 0 : 1;
}
