/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for Phi-4-multimodal's serialized runtime config.
//
// Unlike the families that nest their decoder under "text_config", Phi-4
// multimodal is already flat at the top level, so it needs no
// get_bundle_config_overrides(). What it does carry is a nested
// "img_processor" block holding the image tower's own hidden size, head count
// and layer count. Those keys survive into the bundle's config.json, so pin
// that the decoder contract is read from the top level and the vision tower's
// dimensions never leak into it.

#include "runtime/models/phi4_multimodal/image_preprocessor.h"
#include "trtmc/runtime/pipeline_plugin.h"

#include <iostream>
#include <string>

int main() {
    // Shape follows the checkpoint config: flat decoder geometry at the top
    // level, image tower geometry nested under "img_processor". The nested
    // values are deliberately small so a consumer that picked them up instead
    // would fail these checks.
    const std::string config = R"({
        "model_type": "phi4mm",
        "img_processor": {
            "image_size": 448,
            "patch_size": 14,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "image_token_id": 999,
            "vision_output_dim": 777,
            "num_image_pad_tokens": 888
        },
        "vocab_size": 200064,
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "bos_token_id": 199999,
        "eos_token_id": 199999,
        "max_position_embeddings": 131072,
        "image_token_id": 200010,
        "fixed_image_size": 448,
        "patch_size": 14,
        "num_image_pad_tokens": 721,
        "vision_output_dim": 3072,
        "preprocessor_type": "phi4_hd_chw",
        "interpolation": "bilinear",
        "image_token_str": "<|endoftext10|>",
        "runtime_strategy": "phi4_multimodal_vision_language"
    })";

    const auto parsed = trtmc::parse_base_config(config, 768);
    bool ok = true;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ok = false;
        }
    };

    // Decoder geometry, read from the top level.
    check(parsed.runtime_strategy == "phi4_multimodal_vision_language", "runtime strategy");
    check(parsed.vocab_size == 200064, "vocabulary size");
    check(parsed.num_layers == 32, "layer count");
    check(parsed.num_kv_heads == 8, "KV head count");
    check(parsed.head_dim == 128, "head dimension");
    check(parsed.id_bos == 199999, "BOS token");
    check(parsed.id_eos == 199999, "EOS token");
    check(parsed.max_cache_length == 768, "max cache length override");

    // The image tower's own dimensions must not reach the decoder contract.
    check(parsed.hidden_size == 3072, "hidden size is the decoder's, not img_processor's");
    check(parsed.num_heads == 24, "head count is the decoder's, not img_processor's");
    check(parsed.attention_size == 3072, "attention width");

    // The VL contract, through the family's own consumer: plugin.cpp builds
    // its Phi4MultimodalPreprocessConfig with exactly this call, so a
    // regression that stops reading a key, reaches into img_processor
    // instead, or silently falls back to a default fails here.
    const auto vl = trtmc::phi4_multimodal_parse_preprocess_config(config, "");
    check(vl.image_token_id == 200010, "image token id reaches the VL config");
    check(vl.vision_output_dim == 3072, "vision output dim is the decoder width");
    check(vl.num_image_pad_tokens == 721, "image pad token count");
    check(vl.fixed_image_size == 448, "fixed image size");
    check(vl.patch_size == 14, "patch size");
    check(vl.preprocessor_type == "phi4_hd_chw", "preprocessor strategy");
    check(vl.interpolation == "bilinear", "interpolation");
    check(vl.image_token_str == "<|endoftext10|>", "image placeholder token");

    return ok ? 0 : 1;
}
