/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_helpers.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc {

namespace {

// Parse speech_text_prompt_ids from JSON (array of ints).
// Mirrors the logic from fast_path_config.cpp's parse_speech_text_prompt_ids.
std::vector<int32_t> parse_speech_text_prompt_ids(const std::string& config_text) {
    std::vector<int32_t> prompt_ids;
    const std::size_t pos = config_text.find("\"speech_text_prompt_ids\"");
    if (pos == std::string::npos)
        return prompt_ids;

    const std::size_t bracket = config_text.find('[', pos);
    const std::size_t end_bracket = config_text.find(']', bracket);
    if (bracket == std::string::npos || end_bracket == std::string::npos)
        return prompt_ids;

    std::string arr = config_text.substr(bracket + 1, end_bracket - bracket - 1);
    std::size_t p = 0;
    while (p < arr.size()) {
        const std::size_t num_start = arr.find_first_of("0123456789", p);
        if (num_start == std::string::npos)
            break;
        prompt_ids.push_back(std::stoi(arr.substr(num_start)));
        p = arr.find_first_of(",]", num_start);
        if (p == std::string::npos)
            break;
        ++p;
    }
    return prompt_ids;
}

} // namespace

std::unique_ptr<PersonaplexKvCache> make_coarse_kv_cache(const std::string& json,
                                                         const BaseConfig& base,
                                                         cudaStream_t stream, DType cache_dtype) {
    int32_t hidden = extract_json_int(json, "coarse_hidden_size", base.hidden_size);
    int32_t layers = extract_json_int(json, "coarse_num_layers", base.num_layers);
    int32_t heads = extract_json_int(json, "coarse_num_heads", base.num_heads);
    int32_t hd = (heads > 0) ? hidden / heads : 128;
    int32_t max_cache = extract_json_int(json, "coarse_max_cache_length", base.max_cache_length);
    return std::make_unique<PersonaplexKvCache>(layers, max_cache, heads * hd, stream, cache_dtype);
}

int32_t compute_kv_dim_kv_heads(const BaseConfig& base, int32_t default_dim) {
    if (base.num_kv_heads > 0 && base.head_dim > 0)
        return base.num_kv_heads * base.head_dim;
    if (base.attention_size > 0)
        return base.attention_size;
    return default_dim;
}

std::vector<std::unique_ptr<TrtModule>> load_depth_engines(IBackend* backend,
                                                           const BundleFile& bundle,
                                                           const ModuleCreateOptions& options) {
    std::vector<std::unique_ptr<TrtModule>> depth_engines;
    auto depth_plans = find_depth_engine_plans_in_codebook_order(bundle);
    if (!depth_plans.empty()) {
        for (std::size_t i = 0; i < depth_plans.size(); ++i) {
            auto m = extract_optional_module(
                backend, depth_plans[i], ("speech depth_" + std::to_string(i)).c_str(), options);
            if (m)
                depth_engines.push_back(std::move(m));
        }
    }
    if (depth_engines.empty()) {
        auto* fallback = find_section(bundle, "depth_engine_plan");
        auto m = extract_optional_module(backend, fallback, "speech depth", options);
        if (m)
            depth_engines.push_back(std::move(m));
    }
    return depth_engines;
}

std::vector<const std::vector<char>*>
find_depth_engine_plans_in_codebook_order(const BundleFile& bundle) {
    constexpr const char* prefix = "depth_engine_plan_";
    const auto matching_sections = find_sections_by_prefix(bundle, prefix);

    std::vector<const std::vector<char>*> ordered_plans;
    ordered_plans.reserve(matching_sections.size());
    // Prefix lookup is lexicographic, which places index 10 before index 2.
    // Resolve each contiguous codebook section by its exact numeric name.
    for (std::size_t codebook = 0; codebook < matching_sections.size(); ++codebook) {
        const auto* plan = find_section(bundle, prefix + std::to_string(codebook));
        if (plan == nullptr) {
            throw std::runtime_error(
                "PersonaPlex depth engine sections must use contiguous numeric indices");
        }
        ordered_plans.push_back(plan);
    }
    return ordered_plans;
}

SpeechConfig build_speech_config_from_bundle(const BundleFile& bundle, const std::string& json,
                                             const BaseConfig& base) {
    SpeechConfig sc;
    sc.sample_rate = extract_json_int(json, "sample_rate", 24000);
    sc.temporal_hidden_size = base.hidden_size;
    sc.temporal_num_layers = base.num_layers;
    sc.num_codebooks = extract_json_int(json, "num_codebooks", 8);
    sc.codebook_size = extract_json_int(json, "codebook_size", 2048);
    sc.frame_rate = extract_json_float(json, "frame_rate", 12.5F);
    sc.mimi_max_frames = base.max_cache_length;
    int32_t depth_num_layers = extract_json_int(json, "depth_num_layers", 6);
    sc.depth_num_layers = extract_json_int(json, "fine_num_layers", depth_num_layers);
    int32_t depth_hidden_size = extract_json_int(json, "depth_hidden_size", base.hidden_size);
    sc.depth_hidden_size = extract_json_int(json, "fine_hidden_size", depth_hidden_size);
    int32_t depth_num_heads = extract_json_int(json, "depth_num_attention_heads", base.num_heads);
    sc.depth_num_heads = extract_json_int(json, "fine_num_heads", depth_num_heads);
    sc.depth_num_kv_heads = extract_json_int(json, "depth_num_key_value_heads", sc.depth_num_heads);
    sc.delays = extract_json_int_array(json, "delays", 32);
    sc.text_initial_token_id = extract_json_int(json, "text_initial_token_id", 32000);
    sc.audio_initial_token_id = extract_json_int(json, "audio_initial_token_id", 2048);
    sc.text_padding_id = extract_json_int(json, "text_padding_id", 3);
    sc.depth_temperature = extract_json_float(json, "speech_depth_temperature", 0.0F);
    sc.depth_top_k = extract_json_int(json, "speech_depth_top_k", 0);
    sc.text_eos_token_id = base.id_eos;
    sc.text_prompt_ids = parse_speech_text_prompt_ids(json);
    if (!extract_json_string(json, "speech_system_prompt", "").empty() &&
        sc.text_prompt_ids.empty()) {
        throw std::runtime_error("PersonaPlex native runtime requires speech_text_prompt_ids when "
                                 "speech_system_prompt is configured");
    }
    sc.audio_embeddings = section_to_floats(find_section(bundle, "audio_embeddings"));
    sc.temporal_text_embedding = section_to_floats(find_section(bundle, "temporal_text_embedding"));
    sc.depth_text_embedding = section_to_floats(find_section(bundle, "depth_text_embedding"));
    sc.depth_audio_embeddings = section_to_floats(find_section(bundle, "depth_audio_embeddings"));
    sc.depth_projection = section_to_floats(find_section(bundle, "depth_projection"));
    return sc;
}

int32_t safe_embed_dim(const std::vector<float>& data, int32_t divisor) {
    return (divisor > 0 && !data.empty()) ? static_cast<int32_t>(data.size()) / divisor : 0;
}

void infer_speech_vocab_sizes(SpeechConfig& sc, const std::string& json, const BaseConfig& base) {
    const int32_t h = base.hidden_size;
    int32_t depth_hidden = extract_json_int(json, "depth_hidden_size", base.hidden_size);
    const int32_t dh = extract_json_int(json, "fine_hidden_size", depth_hidden);
    int32_t n_codebooks = extract_json_int(json, "num_codebooks", 8);
    sc.audio_vocab_size = safe_embed_dim(sc.audio_embeddings, n_codebooks * h);
    sc.temporal_text_vocab = safe_embed_dim(sc.temporal_text_embedding, h);
    sc.depth_text_vocab = safe_embed_dim(sc.depth_text_embedding, dh);
    sc.num_depformer_emb = safe_embed_dim(sc.depth_audio_embeddings, sc.audio_vocab_size * dh);
    sc.temporal_hidden_for_proj = (!sc.depth_projection.empty() && h > 0) ? h : 0;
}

BaseConfig make_depth_engine_config(const std::string& json, const BaseConfig& base) {
    // Resolve fine_* fields first (used as fallbacks for speech_depth_*)
    int32_t fine_num_layers =
        extract_json_int(json, "fine_num_layers", extract_json_int(json, "depth_num_layers", 6));
    int32_t fine_hidden_size = extract_json_int(
        json, "fine_hidden_size", extract_json_int(json, "depth_hidden_size", base.hidden_size));
    int32_t fine_num_heads =
        extract_json_int(json, "fine_num_heads",
                         extract_json_int(json, "depth_num_attention_heads", base.num_heads));

    BaseConfig dc;
    dc.num_layers = extract_json_int(json, "speech_depth_num_layers", fine_num_layers);
    if (dc.num_layers <= 0)
        dc.num_layers = fine_num_layers;
    dc.hidden_size = extract_json_int(json, "speech_depth_hidden_size", fine_hidden_size);
    if (dc.hidden_size <= 0)
        dc.hidden_size = fine_hidden_size;
    dc.num_heads = extract_json_int(json, "speech_depth_num_heads", fine_num_heads);
    if (dc.num_heads <= 0)
        dc.num_heads = fine_num_heads;
    dc.num_kv_heads =
        extract_json_int(json, "speech_depth_num_kv_heads",
                         extract_json_int(json, "depth_num_key_value_heads", dc.num_heads));
    if (dc.num_kv_heads <= 0)
        dc.num_kv_heads = dc.num_heads;
    dc.vocab_size = extract_json_int(json, "speech_codebook_size",
                                     extract_json_int(json, "codebook_size", 2048));
    dc.head_dim = dc.hidden_size / std::max(dc.num_heads, 1);
    dc.attention_size = dc.num_heads * dc.head_dim;
    int32_t n_codebooks =
        extract_json_int(json, "speech_num_codebooks", extract_json_int(json, "num_codebooks", 8));
    dc.max_cache_length = n_codebooks + 2;
    return dc;
}

} // namespace trtmc
