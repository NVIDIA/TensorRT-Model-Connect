/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for InternVL's serialized runtime config.

#include "trtmc/runtime/pipeline_plugin.h"

#include <iostream>
#include <string>

int main() {
    const std::string config = R"({
        "model_type": "internvl",
        "text_config": {
            "vocab_size": 151674,
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "bos_token_id": 151643
        },
        "vocab_size": 151674,
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "bos_token_id": 151643,
        "runtime_strategy": "internvl_vision_language"
    })";

    const auto parsed = trtmc::parse_base_config(config, 256);
    bool ok = true;
    const auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ok = false;
        }
    };

    check(parsed.runtime_strategy == "internvl_vision_language", "runtime strategy");
    check(parsed.vocab_size == 151674, "vocabulary size");
    check(parsed.hidden_size == 1536, "hidden size");
    check(parsed.num_layers == 28, "layer count");
    check(parsed.num_heads == 12, "attention head count");
    check(parsed.num_kv_heads == 2, "KV head count");
    check(parsed.head_dim == 128, "head dimension");
    check(parsed.id_bos == 151643, "BOS token");
    check(parsed.attention_size == 1536, "attention width");
    return ok ? 0 : 1;
}
