/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/glm/triattention_kv_cache.h"

#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/trt_module.h"
#ifdef TRTMC_HAS_CUDA_KERNELS
#include "runtime/models/glm/triattention_kernels.h"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

using json = nlohmann::json;

constexpr float kMaskedScore = -1.0e4F;
constexpr float kEps = 1.0e-6F;
constexpr float kAbsFloor = 1.0e-8F;

void require_ta(bool cond, const char* msg) {
    if (!cond)
        throw std::runtime_error(msg);
}

GlmTriAttentionScoreAggregation parse_score_aggregation(const std::string& value) {
    if (value == "mean")
        return GlmTriAttentionScoreAggregation::kMean;
    if (value == "max")
        return GlmTriAttentionScoreAggregation::kMax;
    throw std::runtime_error("Unsupported TriAttention score_aggregation: " + value);
}

const char* score_aggregation_name(GlmTriAttentionScoreAggregation value) {
    switch (value) {
    case GlmTriAttentionScoreAggregation::kMean:
        return "mean";
    case GlmTriAttentionScoreAggregation::kMax:
        return "max";
    }
    return "unknown";
}

GlmTriAttentionRopeStyle parse_rope_style(const std::string& value) {
    if (value == "interleaved")
        return GlmTriAttentionRopeStyle::kInterleaved;
    if (value.empty() || value == "half")
        return GlmTriAttentionRopeStyle::kHalf;
    throw std::runtime_error("Unsupported TriAttention rope_style: " + value);
}

std::vector<float> make_offsets(int32_t max_length) {
    std::vector<float> out;
    for (int32_t current = 1; current > 0 && current <= max_length; current *= 2)
        out.push_back(static_cast<float>(current));
    return out;
}

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    int32_t exponent = static_cast<int32_t>((bits >> 10) & 0x1FU);
    uint32_t mantissa = bits & 0x03FFU;

    uint32_t out_bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out_bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03FFU;
            out_bits =
                sign | (static_cast<uint32_t>(exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1FU) {
        out_bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        out_bits = sign | (static_cast<uint32_t>(exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float out = 0.0F;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

float bf16_to_float(uint16_t bits) {
    const uint32_t out_bits = static_cast<uint32_t>(bits) << 16;
    float out = 0.0F;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

std::vector<float> parse_float_array(const json& array_value, const char* label) {
    if (!array_value.is_array())
        throw std::runtime_error(std::string("TriAttention ") + label + " must be an array");
    std::vector<float> out;
    out.reserve(array_value.size());
    for (const auto& item : array_value)
        out.push_back(item.get<float>());
    return out;
}

std::vector<std::vector<float>> parse_float_matrix(const json& array_value, const char* label) {
    if (!array_value.is_array()) {
        throw std::runtime_error(std::string("TriAttention ") + label + " must be a 2D array");
    }
    std::vector<std::vector<float>> out;
    out.reserve(array_value.size());
    for (const auto& row : array_value)
        out.push_back(parse_float_array(row, label));
    return out;
}

void maybe_generate_default_names(int32_t num_layers, GlmKvCacheNames& names) {
    if (!names.cache_k.empty())
        return;
    names.cache_k.reserve(static_cast<std::size_t>(num_layers));
    names.cache_v.reserve(static_cast<std::size_t>(num_layers));
    names.present_k.reserve(static_cast<std::size_t>(num_layers));
    names.present_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        std::string suffix = "_" + std::to_string(i);
        names.cache_k.push_back("cache_k" + suffix);
        names.cache_v.push_back("cache_v" + suffix);
        names.present_k.push_back("present_k" + suffix);
        names.present_v.push_back("present_v" + suffix);
    }
}

std::string sampled_head_key(int32_t layer, int32_t head) {
    char key[32];
    std::snprintf(key, sizeof(key), "layer%02d_head%02d", layer, head);
    return std::string(key);
}

float complex_abs(float real, float imag) {
    return std::sqrt(std::max(real * real + imag * imag, kAbsFloor));
}

// --- Registry-backed value reader ------------------------------------------
//
// apply_layer_value_<T> overlays a value from the runtime config onto an
// out-param IFF the registry has a non-default layer value for the field
// (i.e. something above SchemaDefault contributed). When the source is
// SchemaDefault we leave the out-param alone so legacy bundle-JSON values
// keep precedence for fields the caller never touched from the CLI.
//
// Previously this file contained a cluster of std::getenv readers
// (triattention_debug_enabled, triattention_profile_enabled, etc.) plus
// override helpers that patched GlmTriAttentionConfig with
// TRTMC_TRIATTN_OVERRIDE_* values. All of them are deleted here — values
// now flow through the config registry exclusively.
template <typename T>
bool registry_has_value(const ::trtmc::config::ConfigBundle& bundle, const std::string& field) {
    try {
        return bundle.source_of("triattention", field) != ::trtmc::config::Layer::SchemaDefault;
    } catch (const std::exception&) {
        return false;
    }
}

template <typename T>
void apply_layer_value(const ::trtmc::config::ConfigBundle& bundle, const std::string& field,
                       T& out) {
    if (!registry_has_value<T>(bundle, field))
        return;
    try {
        out = bundle.get<T>("triattention", field);
    } catch (const std::exception&) { /* type mismatch — skip */
    }
}

void apply_aggregation_from_registry(const ::trtmc::config::ConfigBundle& bundle,
                                     const std::string& field,
                                     GlmTriAttentionScoreAggregation& out) {
    if (!registry_has_value<std::string>(bundle, field))
        return;
    try {
        out = parse_score_aggregation(bundle.get<std::string>("triattention", field));
    } catch (const std::exception&) { /* keep previous value */
    }
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

float score_full_stddev_or_one(const std::vector<float>& scores, float mean) {
    if (scores.size() <= 1U)
        return 1.0F;
    float var = 0.0F;
    for (const float value : scores) {
        const float delta = value - mean;
        var += delta * delta;
    }
    const float denom = static_cast<float>(scores.size() - 1U);
    const float stddev = std::sqrt(std::max(var / denom, 0.0F));
    return stddev < kEps ? 1.0F : stddev;
}

int32_t round_up_rows(int32_t value, int32_t bucket, int32_t maximum) {
    if (bucket <= 1)
        return std::min(std::max(value, 1), maximum);
    const int32_t rounded = ((std::max(value, 1) + bucket - 1) / bucket) * bucket;
    return std::min(rounded, maximum);
}

} // namespace

namespace {

// Overlay core-runtime fields from the registry on top of whatever the
// legacy JSON path produced. Session > Platform > BundleDefault > BuildTime
// win; SchemaDefault reads are skipped so JSON-only bundles keep their
// values.
void overlay_core_runtime_from_registry(GlmTriAttentionConfig& cfg,
                                        const ::trtmc::config::ConfigBundle& bundle) {
    apply_layer_value<bool>(bundle, "enabled", cfg.enabled);
    apply_layer_value<std::int32_t>(bundle, "kv_budget", cfg.kv_budget);
    apply_layer_value<std::int32_t>(bundle, "divide_length", cfg.divide_length);
    apply_layer_value<std::int32_t>(bundle, "recent_window", cfg.recent_window);
    apply_aggregation_from_registry(bundle, "score_aggregation", cfg.score_aggregation);
    apply_aggregation_from_registry(bundle, "per_layer_aggregation", cfg.per_layer_aggregation);
    apply_layer_value<bool>(bundle, "count_prompt_tokens", cfg.count_prompt_tokens);
    apply_layer_value<bool>(bundle, "protect_prefill", cfg.protect_prefill);
    apply_layer_value<bool>(bundle, "disable_mlr", cfg.disable_mlr);
    apply_layer_value<bool>(bundle, "disable_trig", cfg.disable_trig);
    apply_layer_value<std::string>(bundle, "stats_section", cfg.stats_section);
    apply_layer_value<std::int32_t>(bundle, "offset_max_length", cfg.offset_max_length);
}

// Populate the debug/profile fields from the registry. These have no
// legacy JSON representation — they previously came from
// TRTMC_TRIATTN_* env vars, which are now gone.
void fill_debug_from_registry(GlmTriAttentionConfig& cfg,
                              const ::trtmc::config::ConfigBundle& bundle) {
    apply_layer_value<bool>(bundle, "debug", cfg.debug);
    apply_layer_value<bool>(bundle, "profile", cfg.profile);
    apply_layer_value<std::int32_t>(bundle, "runtime_bucket_rows", cfg.runtime_bucket_rows);
    apply_layer_value<bool>(bundle, "disable_gpu_selection", cfg.disable_gpu_selection);
    apply_layer_value<bool>(bundle, "disable_gpu_compaction", cfg.disable_gpu_compaction);
    apply_layer_value<bool>(bundle, "disable_gpu_state", cfg.disable_gpu_state);
    apply_layer_value<bool>(bundle, "zero_tail", cfg.zero_tail);
    apply_layer_value<std::string>(bundle, "dump_keep_path", cfg.dump_keep_path);
    apply_layer_value<std::int32_t>(bundle, "dump_compaction_index", cfg.dump_compaction_index);
    apply_layer_value<bool>(bundle, "abort_after_dump", cfg.abort_after_dump);
    apply_layer_value<bool>(bundle, "dump_score_cache", cfg.dump_score_cache);
    apply_layer_value<bool>(bundle, "dump_score_values", cfg.dump_score_values);
}

static void fill_core_from_legacy_json(GlmTriAttentionConfig& cfg, const std::string& config_json,
                                       int32_t max_cache_length) {
    if (config_json.find("\"triattention\"") == std::string::npos)
        return;
    const json root = json::parse(config_json);
    const auto it = root.find("triattention");
    if (it == root.end() || !it->is_object())
        return;
    cfg.enabled = it->value("enabled", false);
    cfg.kv_budget = it->value("kv_budget", max_cache_length);
    cfg.divide_length = it->value("divide_length", 128);
    cfg.recent_window = it->value("recent_window", 128);
    cfg.score_aggregation =
        parse_score_aggregation(it->value("score_aggregation", std::string("mean")));
    cfg.per_layer_aggregation =
        parse_score_aggregation(it->value("per_layer_aggregation", std::string("mean")));
    cfg.count_prompt_tokens = it->value("count_prompt_tokens", true);
    cfg.protect_prefill = it->value("protect_prefill", true);
    cfg.disable_mlr = it->value("disable_mlr", false);
    cfg.disable_trig = it->value("disable_trig", false);
    cfg.stats_section = it->value("stats_section", std::string("triattention_stats.json"));
    cfg.offset_max_length = it->value("offset_max_length", 65536);
}

static void validate_triattention_config(const GlmTriAttentionConfig& cfg,
                                         int32_t max_cache_length) {
    if (cfg.kv_budget < 1)
        throw std::runtime_error("TriAttention kv_budget must be >= 1");
    if (cfg.kv_budget > max_cache_length)
        throw std::runtime_error("TriAttention kv_budget cannot exceed engine max_cache_length");
    if (cfg.divide_length < 1)
        throw std::runtime_error("TriAttention divide_length must be >= 1");
    if (cfg.recent_window < 0)
        throw std::runtime_error("TriAttention recent_window must be >= 0");
    if (cfg.offset_max_length < 1)
        throw std::runtime_error("TriAttention offset_max_length must be >= 1");
}

} // namespace

GlmTriAttentionConfig
glm_parse_triattention_bundle_config(const std::string& config_json, int32_t max_cache_length,
                                     const ::trtmc::config::ConfigBundle* runtime_config) {
    GlmTriAttentionConfig cfg;
    // Legacy bundle path: pull core fields from the root-level
    // "triattention" object. New bundles route the same values through
    // the generic `defaults:` block which becomes the BundleDefault layer
    // in runtime_config.
    fill_core_from_legacy_json(cfg, config_json, max_cache_length);
    // Overlay registry-supplied values. Session/platform/bundle-default
    // layers win; SchemaDefault reads are skipped. Debug fields come
    // exclusively from the registry (no legacy JSON path).
    if (runtime_config != nullptr) {
        overlay_core_runtime_from_registry(cfg, *runtime_config);
        fill_debug_from_registry(cfg, *runtime_config);
    }
    if (!cfg.enabled)
        return cfg;
    validate_triattention_config(cfg, max_cache_length);
    return cfg;
}

namespace {

void fill_stats_header(GlmTriAttentionStats& stats, const json& root, int32_t num_attention_heads,
                       int32_t num_key_value_heads, int32_t num_layers) {
    stats.head_dim = root.value("head_dim", 0);
    stats.rope_style = parse_rope_style(root.value("rope_style", std::string("half")));
    stats.rope_theta = root.value("rope_theta", 10000.0F);
    stats.num_attention_heads = std::max(root.value("num_attention_heads", num_attention_heads), 1);
    stats.num_key_value_heads = std::max(root.value("num_key_value_heads", num_key_value_heads), 1);
    stats.stats_head_count = std::max(root.value("stats_head_count", 0), 0);
    stats.num_layers = std::max(root.value("num_layers", num_layers), 1);
    require_ta(stats.head_dim > 0 && (stats.head_dim % 2) == 0,
               "TriAttention stats head_dim must be a positive even number");
}

void fill_stats_inv_freq(GlmTriAttentionStats& stats, const json& root) {
    if (root.contains("inv_freq")) {
        stats.inv_freq = parse_float_array(root["inv_freq"], "inv_freq");
    } else {
        const int32_t half_dim = stats.head_dim / 2;
        stats.inv_freq.reserve(static_cast<std::size_t>(half_dim));
        for (int32_t i = 0; i < half_dim; ++i) {
            const float exponent =
                (2.0F * static_cast<float>(i)) / static_cast<float>(stats.head_dim);
            stats.inv_freq.push_back(1.0F / std::pow(stats.rope_theta, exponent));
        }
    }
    require_ta(static_cast<int32_t>(stats.inv_freq.size()) == stats.head_dim / 2,
               "TriAttention inv_freq size does not match head_dim / 2");
}

void append_sampled_heads_from_array(GlmTriAttentionStats& stats, const json& sampled_root,
                                     int32_t head_upper_bound) {
    for (const auto& item : sampled_root) {
        require_ta(item.is_array() && item.size() == 2,
                   "TriAttention sampled_heads entries must be [layer, head]");
        const int32_t layer = item[0].get<int32_t>();
        const int32_t head = item[1].get<int32_t>();
        require_ta(layer >= 0 && layer < stats.num_layers,
                   "TriAttention sampled head layer is out of range");
        require_ta(head >= 0 && head < head_upper_bound,
                   "TriAttention sampled head index is out of range");
        stats.sampled_score_heads_by_layer[static_cast<std::size_t>(layer)].push_back(head);
    }
}

void populate_sampled_heads(GlmTriAttentionStats& stats, const json& root,
                            int32_t head_upper_bound) {
    stats.sampled_score_heads_by_layer.assign(static_cast<std::size_t>(stats.num_layers), {});
    if (root.contains("sampled_heads") && root["sampled_heads"].is_array())
        append_sampled_heads_from_array(stats, root["sampled_heads"], head_upper_bound);

    bool any_sampled = false;
    for (auto& heads : stats.sampled_score_heads_by_layer) {
        std::sort(heads.begin(), heads.end());
        heads.erase(std::unique(heads.begin(), heads.end()), heads.end());
        any_sampled = any_sampled || !heads.empty();
    }
    if (any_sampled)
        return;
    for (auto& heads : stats.sampled_score_heads_by_layer) {
        heads.resize(static_cast<std::size_t>(head_upper_bound));
        std::iota(heads.begin(), heads.end(), 0);
    }
}

int32_t infer_sampled_head_upper_bound(const json& root) {
    if (!root.contains("sampled_heads") || !root["sampled_heads"].is_array())
        return 0;
    int32_t upper_bound = 0;
    for (const auto& item : root["sampled_heads"]) {
        if (!item.is_array() || item.size() != 2)
            continue;
        upper_bound = std::max(upper_bound, item[1].get<int32_t>() + 1);
    }
    return upper_bound;
}

void copy_dense_head_row(GlmTriAttentionHeadStats& dst,
                         const std::vector<std::vector<float>>& q_mean_real,
                         const std::vector<std::vector<float>>& q_mean_imag,
                         const std::vector<std::vector<float>>& q_abs_mean,
                         const std::vector<std::vector<float>>& freq_scale_sq, int32_t score_head,
                         int32_t half_dim) {
    const std::size_t src_idx = static_cast<std::size_t>(score_head);
    require_ta(static_cast<int32_t>(q_mean_real[src_idx].size()) == half_dim &&
                   static_cast<int32_t>(q_mean_imag[src_idx].size()) == half_dim &&
                   static_cast<int32_t>(q_abs_mean[src_idx].size()) == half_dim &&
                   static_cast<int32_t>(freq_scale_sq[src_idx].size()) == half_dim,
               "TriAttention layer_stats frequency count does not match head_dim / 2");
    const auto base = src_idx * static_cast<std::size_t>(half_dim);
    for (int32_t d = 0; d < half_dim; ++d) {
        const auto dst_idx = base + static_cast<std::size_t>(d);
        dst.q_mean_real[dst_idx] = q_mean_real[src_idx][static_cast<std::size_t>(d)];
        dst.q_mean_imag[dst_idx] = q_mean_imag[src_idx][static_cast<std::size_t>(d)];
        dst.q_abs_mean[dst_idx] = q_abs_mean[src_idx][static_cast<std::size_t>(d)];
        dst.freq_scale_sq[dst_idx] = freq_scale_sq[src_idx][static_cast<std::size_t>(d)];
    }
}

bool fill_layer_stats_from_dense(GlmTriAttentionHeadStats& dst, const json& layer_node,
                                 int32_t& inferred_stats_heads, int32_t half_dim) {
    const auto q_mean_real =
        parse_float_matrix(layer_node["q_mean_real"], "layer_stats.q_mean_real");
    const auto q_mean_imag =
        parse_float_matrix(layer_node["q_mean_imag"], "layer_stats.q_mean_imag");
    const auto q_abs_mean = parse_float_matrix(layer_node["q_abs_mean"], "layer_stats.q_abs_mean");
    const int32_t row_count = static_cast<int32_t>(q_mean_real.size());
    if (inferred_stats_heads <= 0)
        inferred_stats_heads = row_count;
    std::vector<std::vector<float>> freq_scale_sq(
        static_cast<std::size_t>(inferred_stats_heads),
        std::vector<float>(static_cast<std::size_t>(half_dim), 1.0F));
    if (layer_node.contains("freq_scale_sq"))
        freq_scale_sq =
            parse_float_matrix(layer_node["freq_scale_sq"], "layer_stats.freq_scale_sq");
    require_ta(row_count == inferred_stats_heads &&
                   static_cast<int32_t>(q_mean_imag.size()) == inferred_stats_heads &&
                   static_cast<int32_t>(q_abs_mean.size()) == inferred_stats_heads &&
                   static_cast<int32_t>(freq_scale_sq.size()) == inferred_stats_heads,
               "TriAttention layer_stats head count is inconsistent");
    const auto flat_size =
        static_cast<std::size_t>(inferred_stats_heads) * static_cast<std::size_t>(half_dim);
    dst.q_mean_real.resize(flat_size);
    dst.q_mean_imag.resize(flat_size);
    dst.q_abs_mean.resize(flat_size);
    dst.freq_scale_sq.resize(flat_size);
    for (int32_t score_head = 0; score_head < inferred_stats_heads; ++score_head)
        copy_dense_head_row(dst, q_mean_real, q_mean_imag, q_abs_mean, freq_scale_sq, score_head,
                            half_dim);
    return true;
}

bool try_parse_dense_layer_stats(GlmTriAttentionStats& stats, const json& root, int32_t half_dim) {
    if (!root.contains("layer_stats") || !root["layer_stats"].is_object() ||
        root["layer_stats"].empty())
        return false;
    stats.layer_stats.resize(static_cast<std::size_t>(stats.num_layers));
    int32_t inferred_stats_heads = stats.stats_head_count;
    bool any_stats = false;
    for (int32_t layer = 0; layer < stats.num_layers; ++layer) {
        auto layer_it = root["layer_stats"].find(std::to_string(layer));
        if (layer_it == root["layer_stats"].end() || !layer_it->is_object())
            continue;
        fill_layer_stats_from_dense(stats.layer_stats[static_cast<std::size_t>(layer)], *layer_it,
                                    inferred_stats_heads, half_dim);
        any_stats = true;
    }
    if (!any_stats)
        return false;
    stats.stats_head_count = std::max(inferred_stats_heads, 1);
    populate_sampled_heads(stats, root, stats.stats_head_count);
    return true;
}

void init_layer_stats_empty(GlmTriAttentionStats& stats, int32_t half_dim) {
    stats.layer_stats.resize(static_cast<std::size_t>(stats.num_layers));
    const auto flat_size =
        static_cast<std::size_t>(stats.stats_head_count) * static_cast<std::size_t>(half_dim);
    for (auto& layer : stats.layer_stats) {
        layer.q_mean_real.assign(flat_size, 0.0F);
        layer.q_mean_imag.assign(flat_size, 0.0F);
        layer.q_abs_mean.assign(flat_size, 0.0F);
        layer.freq_scale_sq.assign(flat_size, 1.0F);
    }
}

void accumulate_sparse_entry(GlmTriAttentionStats& stats, const json& raw_stats, int32_t layer,
                             int32_t head, int32_t half_dim, std::vector<int32_t>& group_counts) {
    const std::string key = sampled_head_key(layer, head);
    auto stats_it = raw_stats.find(key);
    require_ta(stats_it != raw_stats.end() && stats_it->is_object(),
               "TriAttention stats payload is missing entry");
    const auto q_mean_real = parse_float_array((*stats_it)["q_mean_real"], "q_mean_real");
    const auto q_mean_imag = parse_float_array((*stats_it)["q_mean_imag"], "q_mean_imag");
    const auto q_abs_mean = parse_float_array((*stats_it)["q_abs_mean"], "q_abs_mean");
    require_ta(static_cast<int32_t>(q_mean_real.size()) == half_dim &&
                   static_cast<int32_t>(q_mean_imag.size()) == half_dim &&
                   static_cast<int32_t>(q_abs_mean.size()) == half_dim,
               "TriAttention sparse stats entry does not match head_dim / 2");
    auto& layer_stats = stats.layer_stats[static_cast<std::size_t>(layer)];
    const auto base = static_cast<std::size_t>(head) * static_cast<std::size_t>(half_dim);
    for (int32_t d = 0; d < half_dim; ++d) {
        const auto idx = base + static_cast<std::size_t>(d);
        layer_stats.q_mean_real[idx] += q_mean_real[static_cast<std::size_t>(d)];
        layer_stats.q_mean_imag[idx] += q_mean_imag[static_cast<std::size_t>(d)];
        layer_stats.q_abs_mean[idx] += q_abs_mean[static_cast<std::size_t>(d)];
    }
    ++group_counts[static_cast<std::size_t>(layer * stats.stats_head_count + head)];
}

bool finalize_sparse_stats(GlmTriAttentionStats& stats, const std::vector<int32_t>& group_counts,
                           int32_t half_dim) {
    bool any_stats = false;
    for (int32_t layer = 0; layer < stats.num_layers; ++layer) {
        auto& layer_stats = stats.layer_stats[static_cast<std::size_t>(layer)];
        for (int32_t score_head = 0; score_head < stats.stats_head_count; ++score_head) {
            const int32_t count =
                group_counts[static_cast<std::size_t>(layer * stats.stats_head_count + score_head)];
            if (count <= 0)
                continue;
            any_stats = true;
            const auto base =
                static_cast<std::size_t>(score_head) * static_cast<std::size_t>(half_dim);
            for (int32_t d = 0; d < half_dim; ++d) {
                const auto idx = base + static_cast<std::size_t>(d);
                layer_stats.q_mean_real[idx] /= static_cast<float>(count);
                layer_stats.q_mean_imag[idx] /= static_cast<float>(count);
                layer_stats.q_abs_mean[idx] /= static_cast<float>(count);
            }
        }
    }
    return any_stats;
}

void parse_sparse_stats(GlmTriAttentionStats& stats, const json& root, int32_t half_dim) {
    require_ta(root.contains("sampled_heads") && root["sampled_heads"].is_array(),
               "TriAttention stats payload is missing sampled_heads");
    require_ta(root.contains("stats") && root["stats"].is_object(),
               "TriAttention stats payload is missing stats object");
    require_ta(stats.num_attention_heads % stats.num_key_value_heads == 0,
               "TriAttention num_attention_heads must be divisible by num_key_value_heads");
    if (stats.stats_head_count <= 0) {
        stats.stats_head_count =
            std::max({infer_sampled_head_upper_bound(root), stats.num_key_value_heads, 1});
    }
    populate_sampled_heads(stats, root, stats.stats_head_count);
    init_layer_stats_empty(stats, half_dim);

    std::vector<int32_t> group_counts(
        static_cast<std::size_t>(stats.num_layers * stats.stats_head_count), 0);
    const auto& raw_stats = root["stats"];
    for (const auto& item : root["sampled_heads"]) {
        require_ta(item.is_array() && item.size() == 2,
                   "TriAttention sampled_heads entries must be [layer, head]");
        const int32_t layer = item.at(0).get<int32_t>();
        const int32_t head = item.at(1).get<int32_t>();
        if (layer < 0 || layer >= stats.num_layers)
            continue;
        if (head < 0 || head >= stats.stats_head_count)
            continue;
        accumulate_sparse_entry(stats, raw_stats, layer, head, half_dim, group_counts);
    }
    require_ta(finalize_sparse_stats(stats, group_counts, half_dim),
               "TriAttention stats payload has no usable sampled heads");
}

} // namespace

GlmTriAttentionStats glm_parse_triattention_stats_json(const std::string& stats_json,
                                                       int32_t num_attention_heads,
                                                       int32_t num_key_value_heads,
                                                       int32_t num_layers) {
    GlmTriAttentionStats stats;
    const json root = json::parse(stats_json);
    fill_stats_header(stats, root, num_attention_heads, num_key_value_heads, num_layers);
    fill_stats_inv_freq(stats, root);
    const int32_t half_dim = stats.head_dim / 2;
    if (try_parse_dense_layer_stats(stats, root, half_dim))
        return stats;
    parse_sparse_stats(stats, root, half_dim);
    return stats;
}

void GlmTriAttentionKvCache::validate_shapes() {
    require_ta(config_.kv_budget >= 1 && config_.kv_budget <= max_length_,
               "TriAttention kv_budget must be within [1, max_length]");
    require_ta(stats_.head_dim > 0 && (stats_.head_dim % 2) == 0,
               "TriAttention stats head_dim must be a positive even number");
    require_ta(num_kv_heads_ > 0, "TriAttention num_kv_heads must be positive");
    query_head_count_ = kv_dim_ / stats_.head_dim;
    require_ta(query_head_count_ > 0 && (kv_dim_ % stats_.head_dim) == 0,
               "TriAttention kv_dim must be divisible by head_dim");
    require_ta((query_head_count_ % num_kv_heads_) == 0,
               "TriAttention expanded cache heads must be divisible by kv heads");
    require_ta(stats_.num_attention_heads > 0 && stats_.num_key_value_heads > 0 &&
                   (stats_.num_attention_heads % stats_.num_key_value_heads) == 0,
               "TriAttention stats attention head count must be divisible by kv heads");
    query_group_size_ = query_head_count_ / num_kv_heads_;
    cache_head_count_ = num_kv_heads_;
    require_ta(stats_.stats_head_count > 0 && (stats_.stats_head_count % cache_head_count_) == 0,
               "TriAttention stats_head_count must be divisible by runtime kv heads");
    score_group_size_ = stats_.stats_head_count / cache_head_count_;
    require_ta(score_group_size_ > 0, "TriAttention score_group_size must be positive");
}

void GlmTriAttentionKvCache::normalize_sampled_heads() {
    if (stats_.sampled_score_heads_by_layer.size() != static_cast<std::size_t>(num_layers_))
        stats_.sampled_score_heads_by_layer.assign(static_cast<std::size_t>(num_layers_), {});

    const int32_t dense_head_upper_bound = stats_.stats_head_count;
    bool any_sampled_heads = false;
    for (auto& heads : stats_.sampled_score_heads_by_layer) {
        std::sort(heads.begin(), heads.end());
        heads.erase(std::unique(heads.begin(), heads.end()), heads.end());
        for (const int32_t head : heads) {
            if (head < 0 || head >= dense_head_upper_bound)
                throw std::runtime_error("TriAttention sampled score head index is out of range");
        }
        any_sampled_heads = any_sampled_heads || !heads.empty();
    }
    if (!any_sampled_heads) {
        for (auto& heads : stats_.sampled_score_heads_by_layer) {
            heads.resize(static_cast<std::size_t>(dense_head_upper_bound));
            std::iota(heads.begin(), heads.end(), 0);
        }
    }
}

void GlmTriAttentionKvCache::log_init_debug() const {
    if (!config_.debug)
        return;
    std::cerr << "[trtmc.triattention] init kv_budget=" << config_.kv_budget
              << " divide_length=" << config_.divide_length
              << " recent_window=" << config_.recent_window
              << " per_layer_aggregation=" << score_aggregation_name(config_.per_layer_aggregation)
              << " count_prompt_tokens=" << (config_.count_prompt_tokens ? 1 : 0)
              << " protect_prefill=" << (config_.protect_prefill ? 1 : 0)
              << " disable_mlr=" << (config_.disable_mlr ? 1 : 0)
              << " disable_trig=" << (config_.disable_trig ? 1 : 0) << " kv_heads=" << num_kv_heads_
              << " query_heads=" << query_head_count_ << " cache_heads=" << cache_head_count_
              << " score_group=" << score_group_size_
              << " layers_with_stats=" << stats_.layer_stats.size() << '\n';
}

void GlmTriAttentionKvCache::allocate_layer_tensors() {
    cache_k_.reserve(static_cast<std::size_t>(num_layers_));
    cache_v_.reserve(static_cast<std::size_t>(num_layers_));
    present_k_.reserve(static_cast<std::size_t>(num_layers_));
    present_v_.reserve(static_cast<std::size_t>(num_layers_));
    for (int32_t i = 0; i < num_layers_; ++i) {
        cache_k_.emplace_back(std::vector<int64_t>{max_length_, kv_dim_}, cache_dtype_, stream_);
        cache_v_.emplace_back(std::vector<int64_t>{max_length_, kv_dim_}, cache_dtype_, stream_);
        present_k_.emplace_back(std::vector<int64_t>{1, kv_dim_}, cache_dtype_, stream_);
        present_v_.emplace_back(std::vector<int64_t>{1, kv_dim_}, cache_dtype_, stream_);
    }
    mask_buf_.resize(static_cast<std::size_t>(max_length_) + 1U);
    cache_positions_.reserve(static_cast<std::size_t>(max_length_));
    cache_positions_per_head_.resize(static_cast<std::size_t>(cache_head_count_));
    for (auto& head_positions : cache_positions_per_head_)
        head_positions.reserve(static_cast<std::size_t>(max_length_));
}

GlmTriAttentionKvCache::GlmTriAttentionKvCache(int32_t num_layers, int32_t num_kv_heads,
                                               int32_t max_length, int32_t kv_dim,
                                               cudaStream_t stream, GlmTriAttentionConfig config,
                                               GlmTriAttentionStats stats, DType cache_dtype,
                                               GlmKvCacheNames names)
    : num_layers_(num_layers), num_kv_heads_(num_kv_heads), max_length_(max_length),
      kv_dim_(kv_dim), stream_(stream), cache_dtype_(cache_dtype),
      cache_element_size_(dtype_size(cache_dtype)), names_(std::move(names)),
      config_(std::move(config)), stats_(std::move(stats)),
      offsets_(make_offsets(config_.offset_max_length)) {
    validate_shapes();
    normalize_sampled_heads();
    profile_enabled_ = config_.profile;
    log_init_debug();
    maybe_generate_default_names(num_layers_, names_);
    allocate_layer_tensors();
#ifdef TRTMC_HAS_CUDA_KERNELS
    if (!config_.disable_gpu_state)
        initialize_gpu_state();
#endif
    reset();
}

#ifdef TRTMC_HAS_CUDA_KERNELS
void GlmTriAttentionKvCache::allocate_core_selection_buffers(int32_t half_dim) {
    candidate_indices_device_ = DeviceTensor({max_length_}, DType::kInt32, stream_);
    keep_indices_device_ = DeviceTensor({static_cast<int64_t>(cache_head_count_) * max_length_},
                                        DType::kInt32, stream_);
    positions_device_ = DeviceTensor({static_cast<int64_t>(cache_head_count_) * max_length_},
                                     DType::kInt32, stream_);
    inv_freq_device_ =
        DeviceTensor({static_cast<int64_t>(stats_.inv_freq.size())}, DType::kFloat32, stream_);
    cos_phase_device_ =
        DeviceTensor({static_cast<int64_t>(offsets_.size() * static_cast<std::size_t>(half_dim))},
                     DType::kFloat32, stream_);
    sin_phase_device_ =
        DeviceTensor({static_cast<int64_t>(offsets_.size() * static_cast<std::size_t>(half_dim))},
                     DType::kFloat32, stream_);
    scratch_k_device_ = DeviceTensor({max_length_, kv_dim_}, cache_dtype_, stream_);
    scratch_v_device_ = DeviceTensor({max_length_, kv_dim_}, cache_dtype_, stream_);
    if (inv_freq_device_.ok())
        inv_freq_device_.copy_from_host(stats_.inv_freq.data());
}

void GlmTriAttentionKvCache::build_layer_gpu_stats(int32_t layer, int32_t half_dim) {
    const auto& host = stats_.layer_stats[static_cast<std::size_t>(layer)];
    const auto& sampled_heads =
        stats_.sampled_score_heads_by_layer[static_cast<std::size_t>(layer)];
    auto& gpu = layer_gpu_stats_[static_cast<std::size_t>(layer)];
    gpu.score_head_count = static_cast<int32_t>(sampled_heads.size());
    gpu.host_cache_head_indices.clear();
    if (host.q_mean_real.empty() || host.q_mean_imag.empty() || host.q_abs_mean.empty())
        return;
    if (gpu.score_head_count <= 0)
        return;
    const auto expected_stats_size =
        static_cast<std::size_t>(stats_.stats_head_count) * static_cast<std::size_t>(half_dim);
    if (host.q_mean_real.size() != expected_stats_size ||
        host.q_mean_imag.size() != expected_stats_size ||
        host.q_abs_mean.size() != expected_stats_size ||
        host.freq_scale_sq.size() != expected_stats_size)
        return;

    const auto n_heads = static_cast<std::size_t>(gpu.score_head_count);
    const auto flat = n_heads * static_cast<std::size_t>(half_dim);
    std::vector<int32_t> head_offsets(n_heads);
    std::vector<int32_t> head_cache_indices(n_heads);
    std::vector<float> q_mean_real(flat), q_mean_imag(flat), q_abs_mean(flat), freq_scale_sq(flat);
    gpu.host_cache_head_indices.resize(n_heads);

    for (int32_t sampled_idx = 0; sampled_idx < gpu.score_head_count; ++sampled_idx) {
        const int32_t score_head = sampled_heads[static_cast<std::size_t>(sampled_idx)];
        const int32_t cache_head = std::min(cache_head_count_ - 1, score_head / score_group_size_);
        head_cache_indices[static_cast<std::size_t>(sampled_idx)] = cache_head;
        gpu.host_cache_head_indices[static_cast<std::size_t>(sampled_idx)] = cache_head;
        head_offsets[static_cast<std::size_t>(sampled_idx)] =
            cache_head * query_group_size_ * stats_.head_dim;
        const auto src_base =
            static_cast<std::size_t>(score_head) * static_cast<std::size_t>(half_dim);
        const auto dst_base =
            static_cast<std::size_t>(sampled_idx) * static_cast<std::size_t>(half_dim);
        std::copy_n(host.q_mean_real.begin() + static_cast<std::ptrdiff_t>(src_base), half_dim,
                    q_mean_real.begin() + static_cast<std::ptrdiff_t>(dst_base));
        std::copy_n(host.q_mean_imag.begin() + static_cast<std::ptrdiff_t>(src_base), half_dim,
                    q_mean_imag.begin() + static_cast<std::ptrdiff_t>(dst_base));
        std::copy_n(host.q_abs_mean.begin() + static_cast<std::ptrdiff_t>(src_base), half_dim,
                    q_abs_mean.begin() + static_cast<std::ptrdiff_t>(dst_base));
        std::copy_n(host.freq_scale_sq.begin() + static_cast<std::ptrdiff_t>(src_base), half_dim,
                    freq_scale_sq.begin() + static_cast<std::ptrdiff_t>(dst_base));
    }

    gpu.head_offsets = DeviceTensor({gpu.score_head_count}, DType::kInt32, stream_);
    gpu.head_cache_indices = DeviceTensor({gpu.score_head_count}, DType::kInt32, stream_);
    gpu.q_mean_real = DeviceTensor({static_cast<int64_t>(gpu.score_head_count) * half_dim},
                                   DType::kFloat32, stream_);
    gpu.q_mean_imag = DeviceTensor({static_cast<int64_t>(gpu.score_head_count) * half_dim},
                                   DType::kFloat32, stream_);
    gpu.q_abs_mean = DeviceTensor({static_cast<int64_t>(gpu.score_head_count) * half_dim},
                                  DType::kFloat32, stream_);
    gpu.freq_scale_sq = DeviceTensor({static_cast<int64_t>(gpu.score_head_count) * half_dim},
                                     DType::kFloat32, stream_);
    gpu.scores = DeviceTensor({static_cast<int64_t>(gpu.score_head_count) * max_length_},
                              DType::kFloat32, stream_);

    gpu.head_offsets.copy_from_host(head_offsets.data());
    gpu.head_cache_indices.copy_from_host(head_cache_indices.data());
    gpu.q_mean_real.copy_from_host(q_mean_real.data());
    gpu.q_mean_imag.copy_from_host(q_mean_imag.data());
    gpu.q_abs_mean.copy_from_host(q_abs_mean.data());
    gpu.freq_scale_sq.copy_from_host(freq_scale_sq.data());
}

void GlmTriAttentionKvCache::initialize_gpu_state() {
    const int32_t half_dim = stats_.head_dim / 2;
    if (half_dim <= 0 || offsets_.empty() || stats_.layer_stats.empty() || query_head_count_ <= 0)
        return;
    allocate_core_selection_buffers(half_dim);
    layer_gpu_stats_.resize(static_cast<std::size_t>(num_layers_));
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        if (layer >= static_cast<int32_t>(stats_.layer_stats.size()))
            break;
        build_layer_gpu_stats(layer, half_dim);
    }
    cudaStreamSynchronize(stream_);
}

bool GlmTriAttentionKvCache::core_selection_buffers_ready() const {
    return candidate_indices_device_.ok() && keep_indices_device_.ok() && positions_device_.ok() &&
           inv_freq_device_.ok() && cos_phase_device_.ok() && sin_phase_device_.ok() &&
           scratch_k_device_.ok() && scratch_v_device_.ok();
}

bool GlmTriAttentionKvCache::layer_gpu_stats_ready(const LayerGpuStats& layer) {
    if (layer.score_head_count == 0)
        return true;
    return layer.head_offsets.ok() && layer.head_cache_indices.ok() && layer.q_mean_real.ok() &&
           layer.q_mean_imag.ok() && layer.q_abs_mean.ok() && layer.freq_scale_sq.ok() &&
           layer.scores.ok();
}

bool GlmTriAttentionKvCache::can_use_gpu_selection() const {
    if (config_.disable_gpu_selection)
        return false;
    if (!core_selection_buffers_ready())
        return false;
    for (const auto& layer : layer_gpu_stats_) {
        if (!layer_gpu_stats_ready(layer))
            return false;
    }
    return true;
}
#endif

void GlmTriAttentionKvCache::build_attention_mask(std::vector<float>& mask) const {
    const auto width = static_cast<std::size_t>(max_length_) + 1U;
    mask.assign(width, kMaskedScore);
    for (int32_t i = 0; i < cache_length_; ++i)
        mask[static_cast<std::size_t>(i)] = 0.0F;
    mask.back() = 0.0F;
}

int32_t GlmTriAttentionKvCache::preferred_cache_rows() const {
    if (!dynamic_binding_enabled_)
        return max_length_;
    const int32_t bucket_rows = config_.runtime_bucket_rows;
    return round_up_rows(std::max(cache_length_, 1), bucket_rows, max_length_);
}

void GlmTriAttentionKvCache::prepare_step(TensorMap& inputs, int32_t /*seq_len*/) {
    if (has_position_input_) {
        pos_buf_ = absolute_position_;
        Tensor pos_t;
        pos_t.data = &pos_buf_;
        pos_t.shape = {1};
        pos_t.dtype = DType::kInt32;
        inputs[names_.position_id] = pos_t;
    }

    const int32_t cache_rows = dynamic_binding_enabled_ ? preferred_cache_rows() : max_length_;
    const int32_t mask_width = dynamic_binding_enabled_ ? (cache_rows + 1) : (max_length_ + 1);
    if (dynamic_binding_enabled_ && bound_module_ != nullptr && cache_rows != bound_cache_rows_) {
        const std::vector<int64_t> cache_shape{cache_rows, kv_dim_};
        for (int32_t i = 0; i < num_layers_; ++i) {
            const auto li = static_cast<std::size_t>(i);
            bound_module_->bind_external(names_.cache_k[li], cache_k_[li].data(), cache_shape);
            bound_module_->bind_external(names_.cache_v[li], cache_v_[li].data(), cache_shape);
        }
        bound_cache_rows_ = cache_rows;
    }

    std::fill(mask_buf_.begin(), mask_buf_.begin() + mask_width, kMaskedScore);
    for (int32_t i = 0; i < cache_length_; ++i)
        mask_buf_[static_cast<std::size_t>(i)] = 0.0F;
    mask_buf_[static_cast<std::size_t>(mask_width - 1)] = 0.0F;

    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = dynamic_binding_enabled_
                       ? std::vector<int64_t>{1, mask_width}
                       : std::vector<int64_t>{static_cast<int64_t>(mask_buf_.size())};
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void GlmTriAttentionKvCache::bind_to(TrtModule& module) {
    bound_module_ = &module;
    has_position_input_ = module.has_input(names_.position_id);
    dynamic_binding_enabled_ =
        !names_.cache_k.empty() && module.input_is_dynamic(names_.cache_k.front());
    bound_cache_rows_ = 0;
    const int32_t initial_cache_rows =
        dynamic_binding_enabled_ ? preferred_cache_rows() : max_length_;
    const std::vector<int64_t> cache_shape{initial_cache_rows, kv_dim_};

    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto li = static_cast<std::size_t>(i);
        if (dynamic_binding_enabled_) {
            module.bind_external(names_.cache_k[li], cache_k_[li].data(), cache_shape);
            module.bind_external(names_.cache_v[li], cache_v_[li].data(), cache_shape);
            bound_cache_rows_ = initial_cache_rows;
        } else {
            module.bind_external(names_.cache_k[li], cache_k_[li].data());
            module.bind_external(names_.cache_v[li], cache_v_[li].data());
        }
        module.bind_external(names_.present_k[li], present_k_[li].data());
        module.bind_external(names_.present_v[li], present_v_[li].data());
    }
}

std::vector<float>
GlmTriAttentionKvCache::copy_cache_rows_to_host(const DeviceTensor& tensor, int32_t rows,
                                                GlmTriAttentionCompactionProfile* profile) const {
    const auto count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(kv_dim_);
    std::vector<float> out(count, 0.0F);
    if (rows <= 0)
        return out;

    const auto raw_bytes = count * cache_element_size_;
    std::vector<uint8_t> raw(raw_bytes);
    const auto copy_start = Clock::now();
    cudaStreamSynchronize(stream_);
    cudaMemcpy(raw.data(), tensor.data(), raw_bytes, cudaMemcpyDeviceToHost);
    if (profile != nullptr) {
        profile->host_copy_ms += elapsed_ms(copy_start);
        profile->host_copy_bytes += raw_bytes;
    }

    if (cache_dtype_ == DType::kFloat32) {
        std::memcpy(out.data(), raw.data(), raw_bytes);
        return out;
    }

    const auto convert_start = Clock::now();
    const auto* raw_u16 = reinterpret_cast<const uint16_t*>(raw.data());
    for (std::size_t i = 0; i < count; ++i) {
        if (cache_dtype_ == DType::kFloat16)
            out[i] = fp16_to_float(raw_u16[i]);
        else
            out[i] = bf16_to_float(raw_u16[i]);
    }
    if (profile != nullptr)
        profile->host_convert_ms += elapsed_ms(convert_start);
    return out;
}

void GlmTriAttentionKvCache::sync_shared_positions_from_head0() {
    if (cache_positions_per_head_.empty()) {
        cache_positions_.clear();
        return;
    }
    cache_positions_ = cache_positions_per_head_.front();
}

int32_t GlmTriAttentionKvCache::count_prefix_rows() const {
    const int32_t prefix_limit =
        prompt_end_position_ > 0 ? prompt_end_position_ : planned_prompt_length_;
    if (prefix_limit <= 0)
        return 0;
    int32_t count = 0;
    for (int32_t pos : cache_positions_) {
        if (pos < prefix_limit)
            ++count;
    }
    return count;
}

std::vector<char> GlmTriAttentionKvCache::build_reserve_mask(int32_t total_tokens,
                                                             int32_t old_budget) const {
    std::vector<char> reserve_mask(static_cast<std::size_t>(total_tokens), 0);
    const int32_t reserve_recent =
        std::min({std::max(config_.recent_window, 0), total_tokens, old_budget});
    if (reserve_recent > 0) {
        for (int32_t i = total_tokens - reserve_recent; i < total_tokens; ++i)
            reserve_mask[static_cast<std::size_t>(i)] = 1;
    }
    const int32_t prefix_limit =
        prompt_end_position_ > 0 ? prompt_end_position_ : planned_prompt_length_;
    if ((config_.protect_prefill || !config_.count_prompt_tokens) && prefix_limit > 0) {
        for (int32_t i = 0; i < total_tokens; ++i) {
            if (cache_positions_[static_cast<std::size_t>(i)] < prefix_limit)
                reserve_mask[static_cast<std::size_t>(i)] = 1;
        }
    }
    return reserve_mask;
}

std::vector<int32_t> GlmTriAttentionKvCache::broadcast_indices_per_head(std::vector<int32_t> rows,
                                                                        int32_t row_count) const {
    std::sort(rows.begin(), rows.end());
    std::vector<int32_t> keep(static_cast<std::size_t>(cache_head_count_ * row_count));
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        std::copy(rows.begin(), rows.begin() + row_count,
                  keep.begin() + static_cast<std::ptrdiff_t>(cache_head * row_count));
    }
    return keep;
}

std::vector<int32_t>
GlmTriAttentionKvCache::select_keep_indices(int32_t keep_budget,
                                            GlmTriAttentionCompactionProfile* profile) {
    const int32_t total_tokens = static_cast<int32_t>(cache_positions_.size());
    const int32_t old_budget = std::min(std::max(keep_budget, 0), total_tokens);
    if (total_tokens <= old_budget) {
        std::vector<int32_t> identity(static_cast<std::size_t>(total_tokens));
        std::iota(identity.begin(), identity.end(), 0);
        return broadcast_indices_per_head(std::move(identity), total_tokens);
    }

    const auto reserve_mask = build_reserve_mask(total_tokens, old_budget);
    std::vector<int32_t> reserved, candidates;
    reserved.reserve(static_cast<std::size_t>(total_tokens));
    candidates.reserve(static_cast<std::size_t>(total_tokens));
    for (int32_t i = 0; i < total_tokens; ++i) {
        if (reserve_mask[static_cast<std::size_t>(i)] != 0)
            reserved.push_back(i);
        else
            candidates.push_back(i);
    }
    if (profile != nullptr) {
        profile->reserved_count = static_cast<int32_t>(reserved.size());
        profile->candidate_count = static_cast<int32_t>(candidates.size());
    }

    if (static_cast<int32_t>(reserved.size()) >= old_budget) {
        reserved.resize(static_cast<std::size_t>(old_budget));
        return broadcast_indices_per_head(std::move(reserved), old_budget);
    }
    if (candidates.empty())
        return broadcast_indices_per_head(std::move(reserved), old_budget);

#ifdef TRTMC_HAS_CUDA_KERNELS
    if (can_use_gpu_selection())
        return select_keep_indices_gpu(old_budget, reserved, candidates, profile);
#endif
    return select_keep_indices_host(old_budget, reserved, candidates, profile);
}

int32_t GlmTriAttentionKvCache::compaction_keep_budget(int32_t total_tokens) const {
    const int32_t logical_budget = std::min(std::max(config_.kv_budget, 0), max_length_);
    int32_t total_budget = logical_budget;
    if (!config_.count_prompt_tokens)
        total_budget = std::min(max_length_, logical_budget + count_prefix_rows());
    if (!config_.count_prompt_tokens || prompt_end_position_ > 0 ||
        planned_prompt_length_ <= absolute_position_) {
        return std::min(total_budget, total_tokens);
    }

    const int32_t remaining_prompt_tokens =
        std::max(planned_prompt_length_ - absolute_position_, 0);
    const int32_t slack_budget = std::clamp(max_length_ - remaining_prompt_tokens, 0, max_length_);
    return std::min(std::max(total_budget, slack_budget), total_tokens);
}

int32_t GlmTriAttentionKvCache::compaction_trigger_length() const {
    int32_t base_trigger = config_.kv_budget;
    int32_t slack_trigger = config_.kv_budget + std::max(config_.divide_length, 1);
    if (!config_.count_prompt_tokens) {
        const int32_t prefix_rows = count_prefix_rows();
        base_trigger += prefix_rows;
        slack_trigger += prefix_rows;
    }
    return std::min(max_length_, std::max(base_trigger, slack_trigger));
}

std::vector<int32_t> GlmTriAttentionKvCache::broadcast_reserved_for_empty_budget(
    int32_t keep_budget, const std::vector<int32_t>& reserved) const {
    std::vector<int32_t> keep(static_cast<std::size_t>(cache_head_count_ * keep_budget));
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head)
        std::copy(reserved.begin(), reserved.end(),
                  keep.begin() + static_cast<std::ptrdiff_t>(cache_head * keep_budget));
    return keep;
}

void GlmTriAttentionKvCache::precompute_trig_phases(
    std::vector<std::vector<float>>& cos_phase, std::vector<std::vector<float>>& sin_phase,
    int32_t half_dim, GlmTriAttentionCompactionProfile* profile) const {
    if (config_.disable_trig)
        return;
    const auto trig_start = Clock::now();
    cos_phase.assign(offsets_.size(), std::vector<float>(static_cast<std::size_t>(half_dim)));
    sin_phase.assign(offsets_.size(), std::vector<float>(static_cast<std::size_t>(half_dim)));
    // TriAttention scoring must use the true absolute decode position, not the
    // compacted cache length. Once earlier compactions have dropped rows,
    // total_tokens no longer matches the model's current RoPE position, and
    // reusing it here corrupts later-round scoring.
    const float round_start = static_cast<float>(absolute_position_);
    for (std::size_t o = 0; o < offsets_.size(); ++o) {
        for (int32_t d = 0; d < half_dim; ++d) {
            const float phase =
                (round_start + offsets_[o]) * stats_.inv_freq[static_cast<std::size_t>(d)];
            cos_phase[o][static_cast<std::size_t>(d)] = std::cos(phase);
            sin_phase[o][static_cast<std::size_t>(d)] = std::sin(phase);
        }
    }
    if (profile != nullptr)
        profile->trig_prep_ms += elapsed_ms(trig_start);
}

bool GlmTriAttentionKvCache::layer_stats_shapes_valid(const GlmTriAttentionHeadStats& layer_stats,
                                                      int32_t half_dim) const {
    if (layer_stats.q_mean_real.empty() || layer_stats.q_mean_imag.empty() ||
        layer_stats.q_abs_mean.empty() || layer_stats.freq_scale_sq.empty())
        return false;
    const auto expected_stats_size =
        static_cast<std::size_t>(stats_.stats_head_count) * static_cast<std::size_t>(half_dim);
    return layer_stats.q_mean_real.size() == expected_stats_size &&
           layer_stats.q_mean_imag.size() == expected_stats_size &&
           layer_stats.q_abs_mean.size() == expected_stats_size &&
           layer_stats.freq_scale_sq.size() == expected_stats_size;
}

void GlmTriAttentionKvCache::extract_k_rot(const float* row_ptr, int32_t head_offset, int32_t d,
                                           int32_t half_dim, float& k_rot_real,
                                           float& k_rot_imag) const {
    if (stats_.rope_style == GlmTriAttentionRopeStyle::kInterleaved) {
        k_rot_real = row_ptr[head_offset + (2 * d)];
        k_rot_imag = row_ptr[head_offset + (2 * d) + 1];
    } else {
        k_rot_real = row_ptr[head_offset + d];
        k_rot_imag = row_ptr[head_offset + half_dim + d];
    }
}

float GlmTriAttentionKvCache::reduce_trig_sums(const std::vector<float>& trig_sums) const {
    if (trig_sums.empty())
        return 0.0F;
    if (config_.score_aggregation == GlmTriAttentionScoreAggregation::kMax)
        return *std::max_element(trig_sums.begin(), trig_sums.end());
    const float sum = std::accumulate(trig_sums.begin(), trig_sums.end(), 0.0F);
    return sum / static_cast<float>(trig_sums.size());
}

float GlmTriAttentionKvCache::score_one_row(
    const float* row_ptr, const GlmTriAttentionHeadStats& layer_stats, std::size_t stats_base,
    int32_t head_offset, int32_t half_dim, const std::vector<std::vector<float>>& cos_phase,
    const std::vector<std::vector<float>>& sin_phase) const {
    float additive = 0.0F;
    std::vector<float> trig_sums(config_.disable_trig ? 0U : offsets_.size(), 0.0F);
    for (int32_t d = 0; d < half_dim; ++d) {
        float k_rot_real = 0.0F;
        float k_rot_imag = 0.0F;
        extract_k_rot(row_ptr, head_offset, d, half_dim, k_rot_real, k_rot_imag);
        const auto idx = stats_base + static_cast<std::size_t>(d);
        const float q_real = layer_stats.q_mean_real[idx];
        const float q_imag = layer_stats.q_mean_imag[idx];
        const float q_abs = layer_stats.q_abs_mean[idx];
        const float freq_scale_sq = layer_stats.freq_scale_sq[idx];
        const float q_mean_abs = complex_abs(q_real, q_imag);
        const float k_abs = complex_abs(k_rot_real, k_rot_imag);
        const float extra_coef = config_.disable_mlr ? q_abs : (q_abs - q_mean_abs);
        additive += k_abs * extra_coef * freq_scale_sq;
        if (config_.disable_trig)
            continue;
        const float prod_real = q_real * k_rot_real + q_imag * k_rot_imag;
        const float prod_imag = q_imag * k_rot_real - q_real * k_rot_imag;
        for (std::size_t o = 0; o < offsets_.size(); ++o) {
            trig_sums[o] += freq_scale_sq * (prod_real * cos_phase[o][static_cast<std::size_t>(d)] -
                                             prod_imag * sin_phase[o][static_cast<std::size_t>(d)]);
        }
    }
    return reduce_trig_sums(trig_sums) + additive;
}

void GlmTriAttentionKvCache::score_rows_for_head(
    std::vector<float>& scores, const std::vector<float>& layer_cache,
    const GlmTriAttentionHeadStats& layer_stats, int32_t score_head, int32_t cache_head,
    int32_t half_dim, int32_t total_tokens, const std::vector<std::vector<float>>& cos_phase,
    const std::vector<std::vector<float>>& sin_phase) const {
    const int32_t head_offset = cache_head * query_group_size_ * stats_.head_dim;
    const auto stats_base =
        static_cast<std::size_t>(score_head) * static_cast<std::size_t>(half_dim);
    for (int32_t row = 0; row < total_tokens; ++row) {
        const float* row_ptr =
            layer_cache.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(kv_dim_);
        scores[static_cast<std::size_t>(row)] = score_one_row(
            row_ptr, layer_stats, stats_base, head_offset, half_dim, cos_phase, sin_phase);
    }
}

void GlmTriAttentionKvCache::standardize_scores(std::vector<float>& scores) const {
    if (scores.empty())
        return;
    const float mean =
        std::accumulate(scores.begin(), scores.end(), 0.0F) / static_cast<float>(scores.size());
    const float stddev = score_full_stddev_or_one(scores, mean);
    for (float& value : scores)
        value = (value - mean) / stddev;
}

void GlmTriAttentionKvCache::accumulate_layer_fallback(
    const std::vector<std::vector<float>>& layer_scores, std::vector<float>& global_fallback_sum,
    int32_t& global_fallback_count, int32_t total_tokens) const {
    for (const auto& scores : layer_scores) {
        for (int32_t row = 0; row < total_tokens; ++row)
            global_fallback_sum[static_cast<std::size_t>(row)] +=
                scores[static_cast<std::size_t>(row)];
        ++global_fallback_count;
    }
}

void GlmTriAttentionKvCache::reduce_group_into_aggregate(
    int32_t cache_head, int32_t total_tokens, const std::vector<std::vector<float>>& layer_scores,
    const std::vector<int32_t>& sampled_group, std::vector<float>& aggregate,
    bool first_layer_for_cache_head, float* layer_dump) const {
    const bool use_max = config_.per_layer_aggregation == GlmTriAttentionScoreAggregation::kMax;
    (void)cache_head;
    for (int32_t row = 0; row < total_tokens; ++row) {
        float reduced = layer_scores[static_cast<std::size_t>(sampled_group.front())]
                                    [static_cast<std::size_t>(row)];
        for (std::size_t group_idx = 1; group_idx < sampled_group.size(); ++group_idx) {
            reduced =
                std::max(reduced, layer_scores[static_cast<std::size_t>(sampled_group[group_idx])]
                                              [static_cast<std::size_t>(row)]);
        }
        if (use_max) {
            aggregate[static_cast<std::size_t>(row)] =
                first_layer_for_cache_head
                    ? reduced
                    : std::max(aggregate[static_cast<std::size_t>(row)], reduced);
        } else {
            aggregate[static_cast<std::size_t>(row)] += reduced;
        }
        if (layer_dump != nullptr)
            layer_dump[static_cast<std::size_t>(row)] = reduced;
    }
}

void GlmTriAttentionKvCache::accumulate_layer_to_aggregate(
    int32_t layer, int32_t total_tokens, const std::vector<std::vector<float>>& layer_scores,
    const std::vector<std::vector<int32_t>>& sampled_by_cache_head,
    std::vector<std::vector<float>>& aggregated_scores,
    std::vector<int32_t>& contributing_layers_by_cache_head,
    std::vector<float>& layer_aggregate_dump) const {
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        const auto& sampled_group = sampled_by_cache_head[static_cast<std::size_t>(cache_head)];
        if (sampled_group.empty())
            continue;
        const bool first_layer_for_cache_head =
            contributing_layers_by_cache_head[static_cast<std::size_t>(cache_head)] == 0;
        float* layer_dump = nullptr;
        if (!layer_aggregate_dump.empty()) {
            layer_dump =
                layer_aggregate_dump.data() +
                (static_cast<std::size_t>(layer) * static_cast<std::size_t>(cache_head_count_) +
                 static_cast<std::size_t>(cache_head)) *
                    static_cast<std::size_t>(total_tokens);
        }
        reduce_group_into_aggregate(cache_head, total_tokens, layer_scores, sampled_group,
                                    aggregated_scores[static_cast<std::size_t>(cache_head)],
                                    first_layer_for_cache_head, layer_dump);
        ++contributing_layers_by_cache_head[static_cast<std::size_t>(cache_head)];
    }
}

std::vector<float>
GlmTriAttentionKvCache::compute_fallback_mean(const std::vector<float>& global_fallback_sum,
                                              int32_t global_fallback_count,
                                              int32_t total_tokens) const {
    std::vector<float> mean(static_cast<std::size_t>(total_tokens), 0.0F);
    if (global_fallback_count <= 0)
        return mean;
    const float inv_count = 1.0F / static_cast<float>(global_fallback_count);
    for (int32_t row = 0; row < total_tokens; ++row)
        mean[static_cast<std::size_t>(row)] =
            global_fallback_sum[static_cast<std::size_t>(row)] * inv_count;
    return mean;
}

void GlmTriAttentionKvCache::finalize_per_head_aggregate(
    std::vector<std::vector<float>>& aggregated_scores,
    const std::vector<int32_t>& contributing_layers_by_cache_head,
    const std::vector<float>& global_fallback_mean) const {
    const bool per_layer_max =
        config_.per_layer_aggregation == GlmTriAttentionScoreAggregation::kMax;
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        auto& scores = aggregated_scores[static_cast<std::size_t>(cache_head)];
        const int32_t head_layer_count =
            contributing_layers_by_cache_head[static_cast<std::size_t>(cache_head)];
        if (head_layer_count <= 0) {
            scores = global_fallback_mean;
            continue;
        }
        if (per_layer_max)
            continue;
        const float inv = 1.0F / static_cast<float>(head_layer_count);
        for (float& value : scores)
            value *= inv;
    }
}

void GlmTriAttentionKvCache::maybe_dump_score_values(
    const std::vector<std::vector<float>>& aggregated_scores,
    const std::vector<float>& layer_aggregate_dump, int32_t total_tokens) const {
    if (!config_.dump_score_values)
        return;
    const char* dump_path =
        config_.dump_keep_path.empty() ? nullptr : config_.dump_keep_path.c_str();
    if (dump_path == nullptr || dump_path[0] == '\0')
        return;
    std::vector<float> packed_aggregate(
        static_cast<std::size_t>(cache_head_count_) * static_cast<std::size_t>(total_tokens), 0.0F);
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        std::copy(aggregated_scores[static_cast<std::size_t>(cache_head)].begin(),
                  aggregated_scores[static_cast<std::size_t>(cache_head)].end(),
                  packed_aggregate.begin() + static_cast<std::size_t>(cache_head) *
                                                 static_cast<std::size_t>(total_tokens));
    }
    char path_buf[1024];
    std::snprintf(path_buf, sizeof(path_buf), "%s.agg.bin", dump_path);
    std::ofstream out_agg(path_buf, std::ios::binary);
    out_agg.write(reinterpret_cast<const char*>(packed_aggregate.data()),
                  static_cast<std::streamsize>(packed_aggregate.size() * sizeof(float)));
    if (layer_aggregate_dump.empty())
        return;
    std::snprintf(path_buf, sizeof(path_buf), "%s.layeragg.bin", dump_path);
    std::ofstream out_layer(path_buf, std::ios::binary);
    out_layer.write(reinterpret_cast<const char*>(layer_aggregate_dump.data()),
                    static_cast<std::streamsize>(layer_aggregate_dump.size() * sizeof(float)));
}

std::vector<int32_t> GlmTriAttentionKvCache::build_keep_indices_per_head(
    const std::vector<std::vector<float>>& aggregated_scores, const std::vector<int32_t>& reserved,
    const std::vector<int32_t>& candidates, int32_t keep_budget, int32_t need) const {
    std::vector<int32_t> keep(static_cast<std::size_t>(cache_head_count_ * keep_budget));
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        auto* out = keep.data() +
                    static_cast<std::size_t>(cache_head) * static_cast<std::size_t>(keep_budget);
        std::copy(reserved.begin(), reserved.end(), out);
        std::vector<std::pair<float, int32_t>> ranked;
        ranked.reserve(candidates.size());
        const auto& scores = aggregated_scores[static_cast<std::size_t>(cache_head)];
        for (const int32_t row : candidates)
            ranked.emplace_back(scores[static_cast<std::size_t>(row)], row);
        const int32_t top_n = std::min<int32_t>(need, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + top_n, ranked.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first)
                                  return a.first > b.first;
                              return a.second < b.second;
                          });
        for (int32_t i = 0; i < top_n; ++i)
            out[static_cast<std::size_t>(reserved.size() + i)] =
                ranked[static_cast<std::size_t>(i)].second;
        std::sort(out, out + keep_budget);
    }
    return keep;
}

bool GlmTriAttentionKvCache::process_layer_for_host_selection(
    int32_t layer, int32_t half_dim, int32_t total_tokens,
    const std::vector<std::vector<float>>& cos_phase,
    const std::vector<std::vector<float>>& sin_phase,
    std::vector<std::vector<float>>& aggregated_scores, std::vector<float>& global_fallback_sum,
    int32_t& global_fallback_count, std::vector<int32_t>& contributing_layers_by_cache_head,
    std::vector<float>& layer_aggregate_dump, GlmTriAttentionCompactionProfile* profile) const {
    if (layer < 0 || layer >= static_cast<int32_t>(stats_.layer_stats.size()))
        return false;
    const auto& layer_stats = stats_.layer_stats[static_cast<std::size_t>(layer)];
    const auto& sampled_heads =
        stats_.sampled_score_heads_by_layer[static_cast<std::size_t>(layer)];
    if (sampled_heads.empty() || !layer_stats_shapes_valid(layer_stats, half_dim))
        return false;

    const auto layer_cache =
        copy_cache_rows_to_host(cache_k_[static_cast<std::size_t>(layer)], cache_length_, profile);
    if (profile != nullptr) {
        profile->sampled_layers += 1;
        profile->sampled_heads += static_cast<int32_t>(sampled_heads.size());
    }

    std::vector<std::vector<float>> layer_scores(
        sampled_heads.size(), std::vector<float>(static_cast<std::size_t>(total_tokens), 0.0F));
    std::vector<std::vector<int32_t>> sampled_by_cache_head(
        static_cast<std::size_t>(cache_head_count_));
    for (int32_t sampled_idx = 0; sampled_idx < static_cast<int32_t>(sampled_heads.size());
         ++sampled_idx) {
        const int32_t score_head = sampled_heads[static_cast<std::size_t>(sampled_idx)];
        const int32_t cache_head = std::min(cache_head_count_ - 1, score_head / score_group_size_);
        sampled_by_cache_head[static_cast<std::size_t>(cache_head)].push_back(sampled_idx);
        auto& scores = layer_scores[static_cast<std::size_t>(sampled_idx)];
        score_rows_for_head(scores, layer_cache, layer_stats, score_head, cache_head, half_dim,
                            total_tokens, cos_phase, sin_phase);
        standardize_scores(scores);
    }

    accumulate_layer_fallback(layer_scores, global_fallback_sum, global_fallback_count,
                              total_tokens);
    accumulate_layer_to_aggregate(layer, total_tokens, layer_scores, sampled_by_cache_head,
                                  aggregated_scores, contributing_layers_by_cache_head,
                                  layer_aggregate_dump);
    return true;
}

std::vector<int32_t> GlmTriAttentionKvCache::select_keep_indices_host(
    int32_t keep_budget, const std::vector<int32_t>& reserved,
    const std::vector<int32_t>& candidates, GlmTriAttentionCompactionProfile* profile) const {
    const int32_t need = std::max(0, keep_budget - static_cast<int32_t>(reserved.size()));
    if (need <= 0)
        return broadcast_reserved_for_empty_budget(keep_budget, reserved);
    const int32_t half_dim = stats_.head_dim / 2;
    const int32_t total_tokens = static_cast<int32_t>(cache_positions_.size());
    if (half_dim <= 0 || total_tokens <= 0)
        return {};
    std::vector<std::vector<float>> cos_phase;
    std::vector<std::vector<float>> sin_phase;
    precompute_trig_phases(cos_phase, sin_phase, half_dim, profile);

    const auto score_start = Clock::now();
    std::vector<std::vector<float>> aggregated_scores(
        static_cast<std::size_t>(cache_head_count_),
        std::vector<float>(static_cast<std::size_t>(total_tokens), 0.0F));
    std::vector<float> layer_aggregate_dump;
    if (config_.dump_score_values) {
        layer_aggregate_dump.assign(static_cast<std::size_t>(num_layers_) *
                                        static_cast<std::size_t>(cache_head_count_) *
                                        static_cast<std::size_t>(total_tokens),
                                    std::numeric_limits<float>::quiet_NaN());
    }
    std::vector<float> global_fallback_sum(static_cast<std::size_t>(total_tokens), 0.0F);
    int32_t global_fallback_count = 0;
    std::vector<int32_t> contributing_layers_by_cache_head(
        static_cast<std::size_t>(cache_head_count_), 0);
    int32_t contributing_layers = 0;
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        if (process_layer_for_host_selection(
                layer, half_dim, total_tokens, cos_phase, sin_phase, aggregated_scores,
                global_fallback_sum, global_fallback_count, contributing_layers_by_cache_head,
                layer_aggregate_dump, profile))
            ++contributing_layers;
    }
    if (contributing_layers <= 0)
        return {};
    const auto global_fallback_mean =
        compute_fallback_mean(global_fallback_sum, global_fallback_count, total_tokens);
    finalize_per_head_aggregate(aggregated_scores, contributing_layers_by_cache_head,
                                global_fallback_mean);
    if (profile != nullptr)
        profile->score_ms += elapsed_ms(score_start);
    maybe_dump_score_values(aggregated_scores, layer_aggregate_dump, total_tokens);

    const auto combine_start = Clock::now();
    auto keep =
        build_keep_indices_per_head(aggregated_scores, reserved, candidates, keep_budget, need);
    if (profile != nullptr)
        profile->combine_ms += elapsed_ms(combine_start);
    return keep;
}

#ifdef TRTMC_HAS_CUDA_KERNELS
void GlmTriAttentionKvCache::standardize_score_rows(float* rows, int32_t num_rows,
                                                    int32_t total_tokens) const {
    for (int32_t r = 0; r < num_rows; ++r) {
        float* score_row = rows + static_cast<std::size_t>(r) * total_tokens;
        float mean = 0.0F;
        for (int32_t row = 0; row < total_tokens; ++row)
            mean += score_row[row];
        mean /= static_cast<float>(total_tokens);
        float var = 0.0F;
        for (int32_t row = 0; row < total_tokens; ++row) {
            const float delta = score_row[row] - mean;
            var += delta * delta;
        }
        const float denom = total_tokens > 1 ? static_cast<float>(total_tokens - 1) : 1.0F;
        const float stddev = std::sqrt(std::max(var / denom, 0.0F));
        const float std_safe = stddev < kEps ? 1.0F : stddev;
        for (int32_t row = 0; row < total_tokens; ++row)
            score_row[row] = (score_row[row] - mean) / std_safe;
    }
}

void GlmTriAttentionKvCache::accumulate_flat_fallback(const float* rows, int32_t num_rows,
                                                      int32_t total_tokens,
                                                      std::vector<float>& fallback_sum,
                                                      int32_t& fallback_count) const {
    for (int32_t r = 0; r < num_rows; ++r) {
        const float* score_row = rows + static_cast<std::size_t>(r) * total_tokens;
        for (int32_t row = 0; row < total_tokens; ++row)
            fallback_sum[static_cast<std::size_t>(row)] += score_row[row];
        ++fallback_count;
    }
}

void GlmTriAttentionKvCache::aggregate_gpu_layer_into_cache_heads(
    const std::vector<float>& host_scores, const LayerGpuStats& gpu, int32_t total_tokens,
    std::vector<float>& aggregated_scores,
    std::vector<int32_t>& contributing_layers_by_cache_head) const {
    const bool use_max = config_.per_layer_aggregation == GlmTriAttentionScoreAggregation::kMax;
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        std::vector<int32_t> sampled_group;
        sampled_group.reserve(static_cast<std::size_t>(gpu.score_head_count));
        for (int32_t score_head = 0; score_head < gpu.score_head_count; ++score_head) {
            if (gpu.host_cache_head_indices[static_cast<std::size_t>(score_head)] == cache_head)
                sampled_group.push_back(score_head);
        }
        if (sampled_group.empty())
            continue;
        float* aggregate_row =
            aggregated_scores.data() + static_cast<std::size_t>(cache_head) * total_tokens;
        const bool first_layer =
            contributing_layers_by_cache_head[static_cast<std::size_t>(cache_head)] == 0;
        for (int32_t row = 0; row < total_tokens; ++row) {
            float reduced =
                host_scores[static_cast<std::size_t>(sampled_group.front()) * total_tokens +
                            static_cast<std::size_t>(row)];
            for (std::size_t group_idx = 1; group_idx < sampled_group.size(); ++group_idx) {
                reduced = std::max(
                    reduced,
                    host_scores[static_cast<std::size_t>(sampled_group[group_idx]) * total_tokens +
                                static_cast<std::size_t>(row)]);
            }
            aggregate_row[row] =
                use_max ? (first_layer ? reduced : std::max(aggregate_row[row], reduced))
                        : aggregate_row[row] + reduced;
        }
        ++contributing_layers_by_cache_head[static_cast<std::size_t>(cache_head)];
    }
}

GlmTriAttentionKvCache::GpuLayerResult GlmTriAttentionKvCache::process_layer_for_gpu_selection(
    int32_t layer, int32_t total_tokens, int32_t num_offsets, std::vector<float>& aggregated_scores,
    std::vector<float>& global_fallback_sum, int32_t& global_fallback_count,
    std::vector<int32_t>& contributing_layers_by_cache_head,
    GlmTriAttentionCompactionProfile* profile) {
    if (layer < 0 || layer >= static_cast<int32_t>(layer_gpu_stats_.size()))
        return GpuLayerResult::kSkipped;
    auto& gpu = layer_gpu_stats_[static_cast<std::size_t>(layer)];
    if (gpu.score_head_count <= 0)
        return GpuLayerResult::kSkipped;
    if (profile != nullptr) {
        profile->sampled_layers += 1;
        profile->sampled_heads += gpu.score_head_count;
    }
    const bool launched = glm_triattention_score_candidates_gpu(
        cache_k_[static_cast<std::size_t>(layer)].data(), cache_dtype_, kv_dim_, stats_.head_dim,
        (stats_.rope_style == GlmTriAttentionRopeStyle::kInterleaved),
        static_cast<const int32_t*>(candidate_indices_device_.data()), total_tokens, nullptr,
        static_cast<const float*>(inv_freq_device_.data()),
        static_cast<const float*>(cos_phase_device_.data()),
        static_cast<const float*>(sin_phase_device_.data()), num_offsets,
        static_cast<const int32_t*>(gpu.head_offsets.data()),
        static_cast<const int32_t*>(gpu.head_cache_indices.data()),
        static_cast<const float*>(gpu.q_mean_real.data()),
        static_cast<const float*>(gpu.q_mean_imag.data()),
        static_cast<const float*>(gpu.q_abs_mean.data()),
        static_cast<const float*>(gpu.freq_scale_sq.data()), gpu.score_head_count,
        config_.disable_mlr, config_.disable_trig,
        config_.score_aggregation == GlmTriAttentionScoreAggregation::kMax,
        static_cast<float*>(gpu.scores.data()), stream_);
    if (!launched)
        return GpuLayerResult::kFailed;
    std::vector<float> host_scores(static_cast<std::size_t>(gpu.score_head_count) *
                                   static_cast<std::size_t>(total_tokens));
    const auto score_bytes = host_scores.size() * sizeof(float);
    if (cudaMemcpyAsync(host_scores.data(), gpu.scores.data(), score_bytes, cudaMemcpyDeviceToHost,
                        stream_) != cudaSuccess)
        return GpuLayerResult::kFailed;
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
        return GpuLayerResult::kFailed;
    standardize_score_rows(host_scores.data(), gpu.score_head_count, total_tokens);
    accumulate_flat_fallback(host_scores.data(), gpu.score_head_count, total_tokens,
                             global_fallback_sum, global_fallback_count);
    aggregate_gpu_layer_into_cache_heads(host_scores, gpu, total_tokens, aggregated_scores,
                                         contributing_layers_by_cache_head);
    return GpuLayerResult::kContributed;
}

bool GlmTriAttentionKvCache::upload_candidate_indices_identity(int32_t total_tokens) {
    std::vector<int32_t> score_rows(static_cast<std::size_t>(total_tokens));
    std::iota(score_rows.begin(), score_rows.end(), 0);
    const auto bytes = static_cast<std::size_t>(total_tokens) * sizeof(int32_t);
    return cudaMemcpyAsync(candidate_indices_device_.data(), score_rows.data(), bytes,
                           cudaMemcpyHostToDevice, stream_) == cudaSuccess;
}

bool GlmTriAttentionKvCache::upload_gpu_trig_phases(int32_t num_offsets, int32_t half_dim,
                                                    GlmTriAttentionCompactionProfile* profile) {
    if (config_.disable_trig)
        return true;
    const auto trig_start = Clock::now();
    std::vector<float> cos_phase(static_cast<std::size_t>(num_offsets) *
                                 static_cast<std::size_t>(half_dim));
    std::vector<float> sin_phase(static_cast<std::size_t>(num_offsets) *
                                 static_cast<std::size_t>(half_dim));
    const float round_start = static_cast<float>(absolute_position_);
    for (int32_t o = 0; o < num_offsets; ++o) {
        for (int32_t d = 0; d < half_dim; ++d) {
            const std::size_t idx =
                static_cast<std::size_t>(o) * static_cast<std::size_t>(half_dim) + d;
            const float phase = (round_start + offsets_[static_cast<std::size_t>(o)]) *
                                stats_.inv_freq[static_cast<std::size_t>(d)];
            cos_phase[idx] = std::cos(phase);
            sin_phase[idx] = std::sin(phase);
        }
    }
    const auto phase_bytes = cos_phase.size() * sizeof(float);
    const bool ok_cos = cudaMemcpyAsync(cos_phase_device_.data(), cos_phase.data(), phase_bytes,
                                        cudaMemcpyHostToDevice, stream_) == cudaSuccess;
    const bool ok_sin = cudaMemcpyAsync(sin_phase_device_.data(), sin_phase.data(), phase_bytes,
                                        cudaMemcpyHostToDevice, stream_) == cudaSuccess;
    if (profile != nullptr)
        profile->trig_prep_ms += elapsed_ms(trig_start);
    return ok_cos && ok_sin;
}

void GlmTriAttentionKvCache::finalize_flat_per_head_aggregate(
    std::vector<float>& aggregated_scores,
    const std::vector<int32_t>& contributing_layers_by_cache_head,
    const std::vector<float>& global_fallback_mean, int32_t total_tokens) const {
    const bool per_layer_max =
        config_.per_layer_aggregation == GlmTriAttentionScoreAggregation::kMax;
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        float* score_row =
            aggregated_scores.data() + static_cast<std::size_t>(cache_head) * total_tokens;
        const int32_t head_layer_count =
            contributing_layers_by_cache_head[static_cast<std::size_t>(cache_head)];
        if (head_layer_count <= 0) {
            std::copy(global_fallback_mean.begin(), global_fallback_mean.end(), score_row);
            continue;
        }
        if (per_layer_max)
            continue;
        const float inv = 1.0F / static_cast<float>(head_layer_count);
        for (int32_t row = 0; row < total_tokens; ++row)
            score_row[row] *= inv;
    }
}

std::vector<int32_t> GlmTriAttentionKvCache::build_keep_from_flat_aggregate(
    const std::vector<float>& aggregated_scores, const std::vector<int32_t>& reserved,
    const std::vector<int32_t>& candidates, int32_t keep_budget, int32_t need,
    int32_t total_tokens) const {
    std::vector<int32_t> keep(static_cast<std::size_t>(cache_head_count_ * keep_budget));
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        const float* score_row =
            aggregated_scores.data() + static_cast<std::size_t>(cache_head) * total_tokens;
        auto* out = keep.data() +
                    static_cast<std::size_t>(cache_head) * static_cast<std::size_t>(keep_budget);
        std::copy(reserved.begin(), reserved.end(), out);
        std::vector<std::pair<float, int32_t>> ranked;
        ranked.reserve(candidates.size());
        for (const int32_t row : candidates)
            ranked.emplace_back(score_row[row], row);
        const int32_t top_n = std::min<int32_t>(need, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + top_n, ranked.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first)
                                  return a.first > b.first;
                              return a.second < b.second;
                          });
        for (int32_t i = 0; i < top_n; ++i)
            out[static_cast<std::size_t>(reserved.size() + i)] =
                ranked[static_cast<std::size_t>(i)].second;
        std::sort(out, out + keep_budget);
    }
    return keep;
}

bool GlmTriAttentionKvCache::run_gpu_selection_over_layers(
    int32_t total_tokens, int32_t num_offsets, std::vector<float>& aggregated_scores,
    std::vector<float>& global_fallback_sum, int32_t& global_fallback_count,
    std::vector<int32_t>& contributing_layers_by_cache_head, int32_t& contributing_layers,
    GlmTriAttentionCompactionProfile* profile) {
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto status = process_layer_for_gpu_selection(
            layer, total_tokens, num_offsets, aggregated_scores, global_fallback_sum,
            global_fallback_count, contributing_layers_by_cache_head, profile);
        if (status == GpuLayerResult::kFailed)
            return false;
        if (status == GpuLayerResult::kContributed)
            ++contributing_layers;
    }
    return true;
}

std::vector<int32_t> GlmTriAttentionKvCache::select_keep_indices_gpu(
    int32_t keep_budget, const std::vector<int32_t>& reserved,
    const std::vector<int32_t>& candidates, GlmTriAttentionCompactionProfile* profile) {
    const int32_t need = std::max(0, keep_budget - static_cast<int32_t>(reserved.size()));
    if (need <= 0)
        return broadcast_reserved_for_empty_budget(keep_budget, reserved);
    if (candidates.empty())
        return select_keep_indices_host(keep_budget, reserved, candidates, profile);

    const int32_t total_tokens = static_cast<int32_t>(cache_positions_.size());
    if (total_tokens <= 0)
        return {};
    if (!upload_candidate_indices_identity(total_tokens))
        return select_keep_indices_host(keep_budget, reserved, candidates, profile);

    const int32_t num_offsets = static_cast<int32_t>(offsets_.size());
    const int32_t half_dim = stats_.head_dim / 2;
    if (!upload_gpu_trig_phases(num_offsets, half_dim, profile))
        return select_keep_indices_host(keep_budget, reserved, candidates, profile);

    const auto score_start = Clock::now();
    std::vector<float> aggregated_scores(
        static_cast<std::size_t>(cache_head_count_) * static_cast<std::size_t>(total_tokens), 0.0F);
    std::vector<float> global_fallback_sum(static_cast<std::size_t>(total_tokens), 0.0F);
    int32_t global_fallback_count = 0;
    std::vector<int32_t> contributing_layers_by_cache_head(
        static_cast<std::size_t>(cache_head_count_), 0);
    int32_t contributing_layers = 0;
    if (!run_gpu_selection_over_layers(
            total_tokens, num_offsets, aggregated_scores, global_fallback_sum,
            global_fallback_count, contributing_layers_by_cache_head, contributing_layers, profile))
        return select_keep_indices_host(keep_budget, reserved, candidates, profile);
    if (contributing_layers <= 0)
        return select_keep_indices_host(keep_budget, reserved, candidates, profile);

    const auto global_fallback_mean =
        compute_fallback_mean(global_fallback_sum, global_fallback_count, total_tokens);
    finalize_flat_per_head_aggregate(aggregated_scores, contributing_layers_by_cache_head,
                                     global_fallback_mean, total_tokens);
    if (profile != nullptr)
        profile->score_ms += elapsed_ms(score_start);

    const auto combine_start = Clock::now();
    auto keep = build_keep_from_flat_aggregate(aggregated_scores, reserved, candidates, keep_budget,
                                               need, total_tokens);
    if (profile != nullptr)
        profile->combine_ms += elapsed_ms(combine_start);
    return keep;
}
#endif

void GlmTriAttentionKvCache::dump_cache_rows_to_file(const DeviceTensor& tensor, int32_t rows,
                                                     const std::string& path,
                                                     std::vector<std::string>& out_files) {
    const auto host_cache = copy_cache_rows_to_host(tensor, rows, nullptr);
    std::vector<float> packed(static_cast<std::size_t>(rows) *
                                  static_cast<std::size_t>(cache_head_count_) *
                                  static_cast<std::size_t>(stats_.head_dim),
                              0.0F);
    for (int32_t row = 0; row < rows; ++row) {
        const float* row_ptr =
            host_cache.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(kv_dim_);
        for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
            const int32_t head_offset = cache_head * query_group_size_ * stats_.head_dim;
            float* dst = packed.data() + (static_cast<std::size_t>(row) *
                                              static_cast<std::size_t>(cache_head_count_) +
                                          static_cast<std::size_t>(cache_head)) *
                                             static_cast<std::size_t>(stats_.head_dim);
            std::copy_n(row_ptr + head_offset, stats_.head_dim, dst);
        }
    }
    std::ofstream score_out(path, std::ios::binary);
    score_out.write(reinterpret_cast<const char*>(packed.data()),
                    static_cast<std::streamsize>(packed.size() * sizeof(float)));
    out_files.emplace_back(path);
}

void GlmTriAttentionKvCache::maybe_dump_score_cache(int32_t rows, const char* k_pattern,
                                                    const char* v_pattern,
                                                    std::vector<std::string>& score_files,
                                                    std::vector<std::string>& value_files) {
    if (!config_.dump_score_cache)
        return;
    const char* dump_path =
        config_.dump_keep_path.empty() ? nullptr : config_.dump_keep_path.c_str();
    if (dump_path == nullptr || dump_path[0] == '\0')
        return;
    char path_buf[1024];
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        std::snprintf(path_buf, sizeof(path_buf), k_pattern, dump_path, layer);
        dump_cache_rows_to_file(cache_k_[static_cast<std::size_t>(layer)], rows, path_buf,
                                score_files);
        std::snprintf(path_buf, sizeof(path_buf), v_pattern, dump_path, layer);
        dump_cache_rows_to_file(cache_v_[static_cast<std::size_t>(layer)], rows, path_buf,
                                value_files);
    }
}

#ifdef TRTMC_HAS_CUDA_KERNELS
bool GlmTriAttentionKvCache::gpu_compaction_upload_keep(int32_t keep_count,
                                                        const std::vector<int32_t>& keep_indices) {
    if (config_.disable_gpu_compaction || keep_count <= 0 || !keep_indices_device_.ok())
        return false;
    if (!scratch_k_device_.ok() || !scratch_v_device_.ok())
        return false;
    const auto keep_bytes = static_cast<std::size_t>(keep_indices.size()) * sizeof(int32_t);
    return cudaMemcpyAsync(keep_indices_device_.data(), keep_indices.data(), keep_bytes,
                           cudaMemcpyHostToDevice, stream_) == cudaSuccess;
}

bool GlmTriAttentionKvCache::gpu_compact_one_layer(int32_t layer, int32_t keep_count,
                                                   std::size_t row_bytes) {
    const bool ok_k = glm_triattention_compact_rows_gpu(
        cache_k_[static_cast<std::size_t>(layer)].data(), scratch_k_device_.data(), cache_dtype_,
        kv_dim_, static_cast<const int32_t*>(keep_indices_device_.data()), keep_count,
        stats_.head_dim, cache_head_count_, query_group_size_, stream_);
    const bool ok_v = glm_triattention_compact_rows_gpu(
        cache_v_[static_cast<std::size_t>(layer)].data(), scratch_v_device_.data(), cache_dtype_,
        kv_dim_, static_cast<const int32_t*>(keep_indices_device_.data()), keep_count,
        stats_.head_dim, cache_head_count_, query_group_size_, stream_);
    if (!ok_k || !ok_v)
        return false;
    const auto bytes = static_cast<std::size_t>(keep_count) * row_bytes;
    const bool copy_k =
        cudaMemcpyAsync(cache_k_[static_cast<std::size_t>(layer)].data(), scratch_k_device_.data(),
                        bytes, cudaMemcpyDeviceToDevice, stream_) == cudaSuccess;
    const bool copy_v =
        cudaMemcpyAsync(cache_v_[static_cast<std::size_t>(layer)].data(), scratch_v_device_.data(),
                        bytes, cudaMemcpyDeviceToDevice, stream_) == cudaSuccess;
    return copy_k && copy_v;
}
#endif

bool GlmTriAttentionKvCache::compact_layer_on_gpu(int32_t layer,
                                                  const std::vector<int32_t>& keep_indices,
                                                  int32_t keep_count, std::size_t row_bytes,
                                                  int64_t& repack_calls,
                                                  std::size_t& repack_bytes) {
#ifdef TRTMC_HAS_CUDA_KERNELS
    if (!gpu_compaction_upload_keep(keep_count, keep_indices))
        return false;
    if (!gpu_compact_one_layer(layer, keep_count, row_bytes))
        return false;
    repack_calls += 4;
    repack_bytes += static_cast<std::size_t>(keep_count) * row_bytes * 2U;
    return true;
#else
    (void)layer;
    (void)keep_indices;
    (void)keep_count;
    (void)row_bytes;
    (void)repack_calls;
    (void)repack_bytes;
    return false;
#endif
}

void GlmTriAttentionKvCache::compact_layer_on_host(int32_t layer,
                                                   const std::vector<int32_t>& keep_indices,
                                                   int32_t keep_count, std::size_t row_bytes,
                                                   std::size_t head_block_bytes,
                                                   int32_t old_cache_length, int64_t& repack_calls,
                                                   std::size_t& repack_bytes) {
    auto* ck = static_cast<uint8_t*>(cache_k_[static_cast<std::size_t>(layer)].data());
    auto* cv = static_cast<uint8_t*>(cache_v_[static_cast<std::size_t>(layer)].data());
    for (int32_t dst = 0; dst < keep_count; ++dst) {
        const auto dst_offset = static_cast<std::size_t>(dst) * row_bytes;
        for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
            const int32_t src =
                keep_indices[static_cast<std::size_t>(cache_head * keep_count + dst)];
            const auto src_offset = static_cast<std::size_t>(src) * row_bytes +
                                    static_cast<std::size_t>(cache_head) * head_block_bytes;
            const auto head_offset =
                dst_offset + static_cast<std::size_t>(cache_head) * head_block_bytes;
            cudaMemcpyAsync(ck + head_offset, ck + src_offset, head_block_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(cv + head_offset, cv + src_offset, head_block_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
            repack_calls += 2;
            repack_bytes += head_block_bytes * 2U;
        }
    }
    if (config_.zero_tail && old_cache_length > keep_count) {
        const auto tail_offset = static_cast<std::size_t>(keep_count) * row_bytes;
        const auto tail_bytes = static_cast<std::size_t>(old_cache_length - keep_count) * row_bytes;
        cudaMemsetAsync(ck + tail_offset, 0, tail_bytes, stream_);
        cudaMemsetAsync(cv + tail_offset, 0, tail_bytes, stream_);
    }
}

void GlmTriAttentionKvCache::compact_existing_cache(bool reserve_slot_for_append) {
    const int32_t trigger_length = compaction_trigger_length();
    if (cache_length_ < trigger_length)
        return;

    GlmTriAttentionCompactionProfile profile;
    GlmTriAttentionCompactionProfile* profile_ptr = profile_enabled_ ? &profile : nullptr;
    const auto row_bytes = static_cast<std::size_t>(kv_dim_) * cache_element_size_;
    const auto head_block_bytes =
        static_cast<std::size_t>(query_group_size_ * stats_.head_dim) * cache_element_size_;
    const int32_t old_cache_length = cache_length_;
    int32_t keep_count = compaction_keep_budget(cache_length_);
    if (reserve_slot_for_append)
        keep_count = std::min(keep_count, std::max(0, max_length_ - 1));
    cudaEvent_t repack_start = nullptr;
    cudaEvent_t repack_stop = nullptr;
    if (profile_ptr != nullptr) {
        cudaEventCreate(&repack_start);
        cudaEventCreate(&repack_stop);
        cudaEventRecord(repack_start, stream_);
    }
    int64_t repack_calls = 0;
    std::size_t repack_bytes = 0;
    const auto select_start = Clock::now();
    std::vector<int32_t> keep_indices = select_keep_indices(keep_count, profile_ptr);
    if (profile_ptr != nullptr)
        profile_ptr->select_ms += elapsed_ms(select_start);
    if (static_cast<int32_t>(keep_indices.size()) != cache_head_count_ * keep_count) {
        throw std::runtime_error("TriAttention keep index shape mismatch during compaction");
    }
    std::vector<std::string> score_cache_files;
    std::vector<std::string> value_cache_files;
    std::vector<std::string> post_score_cache_files;
    std::vector<std::string> post_value_cache_files;
    maybe_dump_score_cache(static_cast<int32_t>(cache_positions_.size()), "%s.layer%02d.bin",
                           "%s.v.layer%02d.bin", score_cache_files, value_cache_files);
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        if (compact_layer_on_gpu(layer, keep_indices, keep_count, row_bytes, repack_calls,
                                 repack_bytes))
            continue;
        compact_layer_on_host(layer, keep_indices, keep_count, row_bytes, head_block_bytes,
                              old_cache_length, repack_calls, repack_bytes);
    }
    finalize_repack_profile(profile_ptr, repack_start, repack_stop, repack_calls, repack_bytes);
    maybe_dump_score_cache(keep_count, "%s.post.layer%02d.bin", "%s.post.v.layer%02d.bin",
                           post_score_cache_files, post_value_cache_files);

    auto new_positions_by_head = build_new_positions_by_head(keep_indices, keep_count);
    log_compact_debug(new_positions_by_head.front(), keep_indices, keep_count);
    ++compaction_count_;
    log_compact_profile(profile_ptr, keep_count);
    emit_keep_dump_json(keep_indices, keep_count, new_positions_by_head, profile_ptr,
                        score_cache_files, value_cache_files, post_score_cache_files,
                        post_value_cache_files);
    cache_positions_per_head_ = std::move(new_positions_by_head);
    sync_shared_positions_from_head0();
    cache_length_ = keep_count;
}

void GlmTriAttentionKvCache::finalize_repack_profile(GlmTriAttentionCompactionProfile* profile_ptr,
                                                     cudaEvent_t repack_start,
                                                     cudaEvent_t repack_stop, int64_t repack_calls,
                                                     std::size_t repack_bytes) const {
    if (profile_ptr == nullptr)
        return;
    cudaEventRecord(repack_stop, stream_);
    cudaEventSynchronize(repack_stop);
    float repack_ms = 0.0F;
    cudaEventElapsedTime(&repack_ms, repack_start, repack_stop);
    profile_ptr->repack_ms = static_cast<double>(repack_ms);
    profile_ptr->repack_calls = repack_calls;
    profile_ptr->repack_bytes = repack_bytes;
    cudaEventDestroy(repack_start);
    cudaEventDestroy(repack_stop);
}

std::vector<std::vector<int32_t>>
GlmTriAttentionKvCache::build_new_positions_by_head(const std::vector<int32_t>& keep_indices,
                                                    int32_t keep_count) const {
    std::vector<std::vector<int32_t>> out(static_cast<std::size_t>(cache_head_count_));
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        auto& head_out = out[static_cast<std::size_t>(cache_head)];
        const auto& old_positions = cache_positions_per_head_[static_cast<std::size_t>(cache_head)];
        head_out.reserve(static_cast<std::size_t>(keep_count));
        for (int32_t dst = 0; dst < keep_count; ++dst) {
            const int32_t idx =
                keep_indices[static_cast<std::size_t>(cache_head * keep_count + dst)];
            head_out.push_back(old_positions[static_cast<std::size_t>(idx)]);
        }
    }
    return out;
}

std::vector<int32_t>
GlmTriAttentionKvCache::collect_dropped_positions(const std::vector<int32_t>& keep_indices,
                                                  int32_t keep_count) const {
    std::vector<char> keep_mask(static_cast<std::size_t>(cache_length_), 0);
    for (int32_t dst = 0; dst < keep_count; ++dst) {
        const int32_t idx = keep_indices[static_cast<std::size_t>(dst)];
        keep_mask[static_cast<std::size_t>(idx)] = 1;
    }
    std::vector<int32_t> dropped;
    for (int32_t i = 0; i < cache_length_; ++i) {
        if (keep_mask[static_cast<std::size_t>(i)] == 0)
            dropped.push_back(cache_positions_[static_cast<std::size_t>(i)]);
    }
    return dropped;
}

int32_t GlmTriAttentionKvCache::count_prefix_positions(
    const std::vector<int32_t>& representative_positions) const {
    const int32_t prefix_limit =
        prompt_end_position_ > 0 ? prompt_end_position_ : planned_prompt_length_;
    if (prefix_limit <= 0)
        return 0;
    int32_t kept = 0;
    for (int32_t pos : representative_positions) {
        if (pos < prefix_limit)
            ++kept;
    }
    return kept;
}

void GlmTriAttentionKvCache::log_compact_debug(const std::vector<int32_t>& representative_positions,
                                               const std::vector<int32_t>& keep_indices,
                                               int32_t keep_count) const {
    if (!config_.debug)
        return;
    const int32_t kept_prefix = count_prefix_positions(representative_positions);
    const auto dropped_positions = collect_dropped_positions(keep_indices, keep_count);
    const int32_t first_pos =
        representative_positions.empty() ? -1 : representative_positions.front();
    const int32_t last_pos =
        representative_positions.empty() ? -1 : representative_positions.back();
    std::cerr << "[trtmc.triattention] compact abs_pos=" << absolute_position_
              << " old_rows=" << cache_length_ << " kept_rows=" << keep_count
              << " kept_prefix=" << kept_prefix << " first_pos=" << first_pos
              << " last_pos=" << last_pos;
    if (!dropped_positions.empty()) {
        std::cerr << " dropped_pos=";
        for (std::size_t i = 0; i < dropped_positions.size(); ++i) {
            if (i > 0)
                std::cerr << ',';
            std::cerr << dropped_positions[i];
        }
    }
    std::cerr << '\n';
}

void GlmTriAttentionKvCache::log_compact_profile(
    const GlmTriAttentionCompactionProfile* profile_ptr, int32_t keep_count) const {
    if (profile_ptr == nullptr)
        return;
    std::cerr << "[trtmc.triattention.profile] compact#" << compaction_count_
              << " abs_pos=" << absolute_position_ << " old_rows=" << cache_length_
              << " kept_rows=" << keep_count << " reserved=" << profile_ptr->reserved_count
              << " candidates=" << profile_ptr->candidate_count
              << " sampled_layers=" << profile_ptr->sampled_layers
              << " sampled_heads=" << profile_ptr->sampled_heads
              << " host_copy_ms=" << profile_ptr->host_copy_ms
              << " host_convert_ms=" << profile_ptr->host_convert_ms
              << " trig_prep_ms=" << profile_ptr->trig_prep_ms
              << " score_ms=" << profile_ptr->score_ms << " combine_ms=" << profile_ptr->combine_ms
              << " select_ms=" << profile_ptr->select_ms << " repack_ms=" << profile_ptr->repack_ms
              << " host_copy_mb="
              << (static_cast<double>(profile_ptr->host_copy_bytes) / (1024.0 * 1024.0))
              << " repack_mb="
              << (static_cast<double>(profile_ptr->repack_bytes) / (1024.0 * 1024.0))
              << " repack_calls=" << profile_ptr->repack_calls << '\n';
}

bool GlmTriAttentionKvCache::should_emit_keep_dump(
    const GlmTriAttentionCompactionProfile* profile_ptr) const {
    if (config_.dump_keep_path.empty())
        return false;
    const int32_t dump_compaction_index = config_.dump_compaction_index;
    if (dump_compaction_index <= 0)
        return true;
    return dump_compaction_index == (compaction_count_ + (profile_ptr == nullptr ? 1 : 0));
}

void GlmTriAttentionKvCache::emit_keep_dump_json(
    const std::vector<int32_t>& keep_indices, int32_t keep_count,
    const std::vector<std::vector<int32_t>>& new_positions_by_head,
    const GlmTriAttentionCompactionProfile* profile_ptr,
    const std::vector<std::string>& score_cache_files,
    const std::vector<std::string>& value_cache_files,
    const std::vector<std::string>& post_score_cache_files,
    const std::vector<std::string>& post_value_cache_files) {
    if (!should_emit_keep_dump(profile_ptr))
        return;
    json dump;
    dump["compaction_index"] = profile_ptr != nullptr ? compaction_count_ : (compaction_count_ + 1);
    dump["absolute_position"] = absolute_position_;
    dump["cache_length_before"] = static_cast<int32_t>(cache_positions_.size());
    dump["keep_count"] = keep_count;
    dump["prompt_end_position"] = prompt_end_position_;
    dump["planned_prompt_length"] = planned_prompt_length_;
    dump["protect_prefill"] = config_.protect_prefill;
    dump["recent_window"] = config_.recent_window;
    dump["kv_budget"] = config_.kv_budget;
    dump["count_prompt_tokens"] = config_.count_prompt_tokens;
    dump["cache_positions"] = cache_positions_;
    dump["cache_positions_per_head"] = cache_positions_per_head_;
    dump["new_positions_by_head"] = new_positions_by_head;
    std::vector<std::vector<int32_t>> keep_by_head(static_cast<std::size_t>(cache_head_count_));
    for (int32_t cache_head = 0; cache_head < cache_head_count_; ++cache_head) {
        auto begin = keep_indices.begin() + static_cast<std::ptrdiff_t>(cache_head * keep_count);
        keep_by_head[static_cast<std::size_t>(cache_head)] =
            std::vector<int32_t>(begin, begin + keep_count);
    }
    dump["keep_indices_by_head"] = keep_by_head;
    if (profile_ptr != nullptr) {
        dump["profile"] = {
            {"reserved_count", profile_ptr->reserved_count},
            {"candidate_count", profile_ptr->candidate_count},
            {"sampled_layers", profile_ptr->sampled_layers},
            {"sampled_heads", profile_ptr->sampled_heads},
            {"score_ms", profile_ptr->score_ms},
            {"combine_ms", profile_ptr->combine_ms},
            {"select_ms", profile_ptr->select_ms},
            {"repack_ms", profile_ptr->repack_ms},
        };
    }
    if (config_.dump_score_cache) {
        dump["score_cache_shape"] = {static_cast<int32_t>(cache_positions_.size()),
                                     cache_head_count_, stats_.head_dim};
        dump["score_cache_dtype"] = "float32";
        dump["score_cache_files"] = score_cache_files;
        dump["value_cache_files"] = value_cache_files;
        dump["post_score_cache_shape"] = {keep_count, cache_head_count_, stats_.head_dim};
        dump["post_score_cache_files"] = post_score_cache_files;
        dump["post_value_cache_files"] = post_value_cache_files;
    }
    std::ofstream out(config_.dump_keep_path);
    out << dump.dump(2);
    out << '\n';
    if (config_.abort_after_dump)
        throw std::runtime_error("TriAttention aborted after keep dump");
}

void GlmTriAttentionKvCache::advance(int32_t n_tokens) {
    assert(n_tokens == 1 && "GlmTriAttentionKvCache::advance only supports n_tokens==1");
    (void)n_tokens;

    if (cache_length_ >= max_length_)
        compact_existing_cache(true);

    const auto row_bytes = static_cast<std::size_t>(kv_dim_) * cache_element_size_;
    const auto offset = static_cast<std::size_t>(cache_length_) * row_bytes;
    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto li = static_cast<std::size_t>(i);
        cudaMemcpyAsync(static_cast<uint8_t*>(cache_k_[li].data()) + offset, present_k_[li].data(),
                        row_bytes, cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(static_cast<uint8_t*>(cache_v_[li].data()) + offset, present_v_[li].data(),
                        row_bytes, cudaMemcpyDeviceToDevice, stream_);
    }

    if (cache_length_ < max_length_)
        ++cache_length_;
    for (auto& head_positions : cache_positions_per_head_)
        head_positions.push_back(absolute_position_);
    sync_shared_positions_from_head0();
    ++absolute_position_;

    if (cache_length_ >= compaction_trigger_length())
        compact_existing_cache();
}

void GlmTriAttentionKvCache::set_prompt_length(int32_t prompt_length) {
    planned_prompt_length_ = std::max(prompt_length, 0);
}

void GlmTriAttentionKvCache::mark_prefill_complete() {
    prompt_end_position_ = std::max(absolute_position_, planned_prompt_length_);
}

void GlmTriAttentionKvCache::reset() {
    cache_length_ = 0;
    absolute_position_ = 0;
    planned_prompt_length_ = 0;
    prompt_end_position_ = 0;
    compaction_count_ = 0;
    cache_positions_.clear();
    for (auto& head_positions : cache_positions_per_head_)
        head_positions.clear();
    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto li = static_cast<std::size_t>(i);
        cudaMemsetAsync(cache_k_[li].data(), 0, cache_k_[li].nbytes(), stream_);
        cudaMemsetAsync(cache_v_[li].data(), 0, cache_v_[li].nbytes(), stream_);
        cudaMemsetAsync(present_k_[li].data(), 0, present_k_[li].nbytes(), stream_);
        cudaMemsetAsync(present_v_[li].data(), 0, present_v_[li].nbytes(), stream_);
    }
    cudaStreamSynchronize(stream_);
}

std::size_t GlmTriAttentionKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& t : cache_k_)
        total += t.nbytes();
    for (const auto& t : cache_v_)
        total += t.nbytes();
    for (const auto& t : present_k_)
        total += t.nbytes();
    for (const auto& t : present_v_)
        total += t.nbytes();
#ifdef TRTMC_HAS_CUDA_KERNELS
    total += candidate_indices_device_.nbytes();
    total += keep_indices_device_.nbytes();
    total += positions_device_.nbytes();
    total += inv_freq_device_.nbytes();
    total += cos_phase_device_.nbytes();
    total += sin_phase_device_.nbytes();
    total += scratch_k_device_.nbytes();
    total += scratch_v_device_.nbytes();
    for (const auto& layer : layer_gpu_stats_) {
        total += layer.head_offsets.nbytes();
        total += layer.head_cache_indices.nbytes();
        total += layer.q_mean_real.nbytes();
        total += layer.q_mean_imag.nbytes();
        total += layer.q_abs_mean.nbytes();
        total += layer.freq_scale_sq.nbytes();
        total += layer.scores.nbytes();
    }
#endif
    return total;
}

bool GlmTriAttentionKvCache::ok() const {
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_))
        return false;
    for (const auto& t : cache_k_) {
        if (!t.ok())
            return false;
    }
    for (const auto& t : cache_v_) {
        if (!t.ok())
            return false;
    }
    for (const auto& t : present_k_) {
        if (!t.ok())
            return false;
    }
    for (const auto& t : present_v_) {
        if (!t.ok())
            return false;
    }
    return true;
}

} // namespace trtmc
