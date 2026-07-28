/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_plugin.h"

#include "utils/json_helpers.h"

#include <algorithm>
#include <initializer_list>

namespace trtmc {

namespace {

// Extract the first non-zero int from a list of JSON key aliases.
int32_t first_nonzero_int(const std::string& text, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        int32_t v = extract_json_int(text, key, 0);
        if (v != 0)
            return v;
    }
    return 0;
}

void parse_model_dimensions(const std::string& config_text, BaseConfig& cfg) {
    cfg.vocab_size = extract_json_int(config_text, "vocab_size", 0);
    cfg.hidden_size =
        first_nonzero_int(config_text, {"hidden_size", "n_embd", "d_model", "n_embed", "dim"});

    cfg.num_layers = std::max(
        first_nonzero_int(config_text, {"num_hidden_layers", "n_layer", "num_layers", "n_layers"}),
        1);

    int32_t decoder_layers = extract_json_int(config_text, "decoder_layers", 0);
    if (decoder_layers > 0)
        cfg.num_layers = decoder_layers;

    cfg.num_heads = std::max(
        first_nonzero_int(config_text, {"num_attention_heads", "n_head", "attention_heads",
                                        "num_heads", "n_heads", "decoder_attention_heads"}),
        1);

    cfg.num_kv_heads =
        std::max(extract_json_int(config_text, "num_key_value_heads", cfg.num_heads), 1);
    cfg.head_dim = extract_json_int(config_text, "head_dim", cfg.hidden_size / cfg.num_heads);
    cfg.attention_size = cfg.num_heads * cfg.head_dim;
}

void parse_cache_and_tokens(const std::string& config_text, int32_t max_cache_length_override,
                            BaseConfig& cfg) {
    if (max_cache_length_override > 0) {
        cfg.max_cache_length = max_cache_length_override;
    } else {
        cfg.max_cache_length = extract_json_int(config_text, "max_position_embeddings", 32);
        if (cfg.max_cache_length > 4096)
            cfg.max_cache_length = 4096;
    }
    cfg.id_bos = extract_json_int_or_first_array(config_text, "bos_token_id", -1);
    cfg.id_eos = extract_json_int_or_first_array(config_text, "eos_token_id", -1);
    cfg.id_eos_ids = extract_json_int_array(config_text, "eos_token_id", 256);
    if (cfg.id_eos_ids.empty() && cfg.id_eos >= 0)
        cfg.id_eos_ids.push_back(cfg.id_eos);
}

void parse_strategy_and_tokenizer_flags(const std::string& config_text, BaseConfig& cfg) {
    cfg.runtime_strategy = extract_json_string(config_text, "runtime_strategy", "");

    cfg.precision = extract_json_string(config_text, "precision", "fp32");

    int32_t raw = extract_json_int(config_text, "tokenizer_add_special_tokens", -1);
    if (raw >= 0) {
        cfg.tokenizer_add_special_tokens = (raw != 0);
        cfg.tokenizer_add_special_tokens_present = true;
    }
}

// Extract a JSON object as raw text for the given key.
static std::string extract_json_object_text(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    auto pos = text.find(needle);
    if (pos == std::string::npos)
        return "";
    auto brace = text.find('{', pos + needle.size());
    if (brace == std::string::npos)
        return "";
    int depth = 0;
    for (std::size_t i = brace; i < text.size(); ++i) {
        if (text[i] == '{')
            ++depth;
        else if (text[i] == '}') {
            if (--depth == 0)
                return text.substr(brace, i - brace + 1);
        }
    }
    return "";
}

void parse_io_map(const std::string& config_text, BaseConfig& cfg) {
    std::string obj = extract_json_object_text(config_text, "io_map");
    if (obj.empty())
        return;
    auto get = [&](const std::string& key, std::string& out) {
        std::string v = extract_json_string(obj, key, "");
        if (!v.empty())
            out = v;
    };
    get("token_id", cfg.io_map.token_id);
    get("position_id", cfg.io_map.position_id);
    get("attention_mask", cfg.io_map.attention_mask);
    get("logits", cfg.io_map.logits);
    get("cache_k", cfg.io_map.cache_k_pattern);
    get("cache_v", cfg.io_map.cache_v_pattern);
    get("present_k", cfg.io_map.present_k_pattern);
    get("present_v", cfg.io_map.present_v_pattern);
}

} // namespace

BaseConfig parse_base_config(const std::string& config_text, int32_t max_cache_length_override) {
    BaseConfig cfg;
    parse_model_dimensions(config_text, cfg);
    parse_cache_and_tokens(config_text, max_cache_length_override, cfg);
    parse_strategy_and_tokenizer_flags(config_text, cfg);
    parse_io_map(config_text, cfg);
    return cfg;
}

} // namespace trtmc
