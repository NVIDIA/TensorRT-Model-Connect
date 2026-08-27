/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for Qwen3.5's serialized runtime config.

#include "trtmc/runtime/pipeline_plugin.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // The nested objects mirror Qwen/Qwen3.5-9B at the catalog-pinned
    // c202236235762e1c871ad0ccb60c8ee5ba337b9a revision. The duplicated
    // top-level decoder fields are the final bundle contract consumed by the
    // strict production C++ parser.
    const std::string config = R"({
        "vocab_size": 248320,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "bos_token_id": -1,
        "eos_token_id": 248044,
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "eos_token_id": 248044,
            "max_position_embeddings": 262144,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25
            }
        },
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": []
        },
        "num_mamba_layers": 24,
        "num_attention_layers": 8,
        "d_inner": 4096,
        "mamba_d_state": 128,
        "mamba_d_conv": 4,
        "mamba_nheads": 32,
        "mamba_head_dim": 128,
        "conv_dim": 8192,
        "runtime_strategy": "qwen3_5_hybrid_mamba_attention",
        "precision": "fp16",
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

    check(parsed.runtime_strategy == "qwen3_5_hybrid_mamba_attention", "runtime strategy");
    check(parsed.precision == "fp16", "precision");
    check(parsed.vocab_size == 248320, "vocabulary size");
    check(parsed.hidden_size == 4096, "hidden size");
    check(parsed.num_layers == 32, "layer count");
    check(parsed.num_heads == 16, "attention head count");
    check(parsed.num_kv_heads == 4, "KV head count");
    check(parsed.head_dim == 256, "head dimension");
    check(parsed.attention_size == 4096, "attention width");
    check(parsed.num_kv_heads * parsed.head_dim == 1024, "KV cache width");
    check(parsed.max_cache_length == 256, "cache length override");
    check(parsed.id_bos == -1, "absent BOS token contract");
    check(parsed.id_eos == 248044, "EOS token");
    check(parsed.id_eos_ids == std::vector<int32_t>({248044}), "EOS token list");
    check(parsed.tokenizer_add_special_tokens_present, "tokenizer flag presence");
    check(!parsed.tokenizer_add_special_tokens, "tokenizer flag value");
    return ok ? 0 : 1;
}
