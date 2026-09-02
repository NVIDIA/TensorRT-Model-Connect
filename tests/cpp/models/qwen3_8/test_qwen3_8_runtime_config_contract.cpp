/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only consumer contract for Qwen3.8's serialized runtime config.

#include "trtmc/runtime/pipeline_plugin.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // Transcribed from the config.json section of a bundle built from
    // Qwen/Qwen3.8-27B at 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. Qwen3.8
    // keeps every decoder dimension under text_config, but the runtime reads
    // the bundle with a top-level nlohmann lookup, so the duplicated top-level
    // fields are the actual contract. Without them compute_kv_dim() returns 0
    // and the KV cache allocates zero-sized tensors.
    const std::string config = R"({
        "vocab_size": 248320,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-06,
        "bos_token_id": 248044,
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "bos_token_id": 248044,
            "eos_token_id": 248044,
            "output_gate_type": "swish",
            "max_position_embeddings": 262144,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
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
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": []
        },
        "num_mamba_layers": 48,
        "num_attention_layers": 16,
        "d_inner": 6144,
        "mamba_d_state": 128,
        "mamba_d_conv": 4,
        "mamba_nheads": 48,
        "mamba_head_dim": 128,
        "conv_dim": 10240,
        "eos_token_id": [248046, 248044],
        "runtime_strategy": "qwen3_8_hybrid_mamba_attention",
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

    check(parsed.runtime_strategy == "qwen3_8_hybrid_mamba_attention", "runtime strategy");
    check(parsed.precision == "fp16", "precision");
    check(parsed.vocab_size == 248320, "vocabulary size");
    check(parsed.hidden_size == 5120, "hidden size");
    check(parsed.num_layers == 64, "layer count");
    check(parsed.num_heads == 24, "attention head count");
    check(parsed.num_kv_heads == 4, "KV head count");
    check(parsed.head_dim == 256, "head dimension");
    check(parsed.attention_size == 6144, "attention width");
    check(parsed.num_kv_heads * parsed.head_dim == 1024, "KV cache width");
    check(parsed.max_cache_length == 256, "cache length override");
    check(parsed.id_bos == 248044, "BOS token");

    // Qwen3.8 diverges from Qwen3.5 here. text_config carries a single
    // eos_token_id (248044), but the checkpoint actually terminates on 248046,
    // which only appears in generation_config.json. The builder serializes that
    // full list, and the family deliberately does not republish the
    // text_config value as a bundle override, because overrides are merged last
    // and would collapse the list to 248044 alone -- leaving 248046 unmatched
    // and generation running to max_new_tokens.
    check(parsed.id_eos == 248046, "primary EOS token from generation config");
    check(parsed.id_eos_ids == std::vector<int32_t>({248046, 248044}),
          "full EOS token list survives override merge");

    check(parsed.tokenizer_add_special_tokens_present, "tokenizer flag presence");
    check(!parsed.tokenizer_add_special_tokens, "tokenizer flag value");
    return ok ? 0 : 1;
}
