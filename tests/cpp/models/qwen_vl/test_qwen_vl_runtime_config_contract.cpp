/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for Qwen3-VL's serialized runtime config.

#include "trtmc/runtime/pipeline_plugin.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // The nested objects mirror Qwen/Qwen3-VL-2B-Instruct at the pinned
    // 89644892e4d85e24eaac8bacfd4f463576704203 revision. The duplicated
    // top-level decoder fields are the final bundle contract consumed by the
    // strict production C++ parser. The top-level EOS list comes from the
    // pinned generation_config.json and must not be replaced by the nested
    // scalar value.
    const std::string config = R"({
        "vocab_size": 151936,
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "bos_token_id": 151643,
        "eos_token_id": [151645, 151643],
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "max_position_embeddings": 262144,
            "rope_theta": 5000000,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20],
                "rope_type": "default"
            }
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [5, 11, 17]
        },
        "runtime_strategy": "qwen_vl_vision_language",
        "precision": "bf16",
        "tokenizer_add_special_tokens": 0
    })";

    const auto parsed = trtmc::parse_base_config(config, 256);
    bool ok = true;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ok = false;
        }
    };

    check(parsed.runtime_strategy == "qwen_vl_vision_language", "runtime strategy");
    check(parsed.precision == "bf16", "precision");
    check(parsed.vocab_size == 151936, "vocabulary size");
    check(parsed.hidden_size == 2048, "hidden size");
    check(parsed.num_layers == 28, "layer count");
    check(parsed.num_heads == 16, "attention head count");
    check(parsed.num_kv_heads == 8, "KV head count");
    check(parsed.head_dim == 128, "head dimension");
    check(parsed.attention_size == 2048, "attention width");
    check(parsed.num_kv_heads * parsed.head_dim == 1024, "KV cache width");
    check(parsed.max_cache_length == 256, "cache length override");
    check(parsed.id_bos == 151643, "BOS token");
    check(parsed.id_eos == 151645, "EOS token");
    check(parsed.id_eos_ids == std::vector<int32_t>({151645, 151643}), "EOS token list");
    check(parsed.tokenizer_add_special_tokens_present, "tokenizer flag presence");
    check(!parsed.tokenizer_add_special_tokens, "tokenizer flag value");
    return ok ? 0 : 1;
}
