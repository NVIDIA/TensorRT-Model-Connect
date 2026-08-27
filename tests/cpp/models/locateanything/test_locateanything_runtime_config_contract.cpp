/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for LocateAnything's serialized runtime config.

#include "trtmc/runtime/pipeline_plugin.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // The nested objects mirror nvidia/LocateAnything-3B's Hugging Face config.
    // The duplicated top-level decoder fields are the final bundle contract
    // consumed by the strict production C++ parser.
    const std::string config = R"({
        "vocab_size": 152681,
        "hidden_size": 2048,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "model_type": "locateanything",
        "text_config": {
            "model_type": "qwen2",
            "vocab_size": 152681,
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0
        },
        "vision_config": {
            "model_type": "moonvit",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "patch_size": 14,
            "merge_kernel_size": [2, 2]
        },
        "runtime_strategy": "locateanything_vision_language",
        "precision": "fp16",
        "tokenizer_add_special_tokens": 0
    })";

    const auto parsed = trtmc::parse_base_config(config, 384);
    bool ok = true;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ok = false;
        }
    };

    check(parsed.runtime_strategy == "locateanything_vision_language", "runtime strategy");
    check(parsed.precision == "fp16", "precision");
    check(parsed.vocab_size == 152681, "vocabulary size");
    check(parsed.hidden_size == 2048, "hidden size");
    check(parsed.num_layers == 36, "layer count");
    check(parsed.num_heads == 16, "attention head count");
    check(parsed.num_kv_heads == 2, "KV head count");
    check(parsed.head_dim == 128, "head dimension");
    check(parsed.attention_size == 2048, "attention width");
    check(parsed.max_cache_length == 384, "cache length override");
    check(parsed.id_bos == 151643, "BOS token");
    check(parsed.id_eos == 151645, "EOS token");
    check(parsed.id_eos_ids == std::vector<int32_t>({151645}), "EOS token list");
    check(parsed.tokenizer_add_special_tokens_present, "tokenizer flag presence");
    check(!parsed.tokenizer_add_special_tokens, "tokenizer flag value");
    return ok ? 0 : 1;
}
