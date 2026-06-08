#pragma once

#include "trtmc/config/config_bundle.h"

#include <cctype>
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>

namespace trtmc {

struct VoxCPM2Config {
    int32_t sample_rate{48000};
    int32_t reference_sample_rate{16000};
    int32_t max_text_steps{0};
    float cfg_value{2.0F};
    int32_t inference_timesteps{10};
    bool normalize{true};
    bool denoise{true};
    bool retry_badcase{true};
    int32_t retry_badcase_max_times{3};
    float retry_badcase_ratio_threshold{6.0F};
    int64_t seed{-1};
    int32_t patch_size{4};
    int32_t feat_dim{64};
};

namespace voxcpm2_detail {

inline bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

inline bool is_int_char(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '-';
}

inline bool is_float_char(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '+' || c == '.' ||
           c == 'e' || c == 'E';
}

inline std::size_t skip_space(const std::string& text, std::size_t pos) {
    while (pos < text.size() && is_space(text[pos])) {
        ++pos;
    }
    return pos;
}

inline std::size_t scan_while(const std::string& text, std::size_t pos, bool (*allowed)(char)) {
    std::size_t end = pos;
    while (end < text.size() && allowed(text[end])) {
        ++end;
    }
    return end;
}

inline bool key_matches_at(const std::string& text, std::size_t pos, const std::string& key) {
    if (pos >= text.size() || text[pos] != '"') {
        return false;
    }
    const std::size_t value_begin = pos + 1;
    const std::size_t value_end = value_begin + key.size();
    return value_end < text.size() && text.compare(value_begin, key.size(), key) == 0 &&
           text[value_end] == '"';
}

inline bool find_top_level_key_colon(const std::string& text, const std::string& key,
                                     std::size_t& colon) {
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            if (depth == 1 && key_matches_at(text, i, key)) {
                const std::size_t after_key = i + key.size() + 2;
                const std::size_t maybe_colon = skip_space(text, after_key);
                if (maybe_colon < text.size() && text[maybe_colon] == ':') {
                    colon = maybe_colon;
                    return true;
                }
            }
            in_string = true;
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}' && depth > 0) {
            --depth;
        }
    }
    return false;
}

inline int32_t extract_top_level_json_int(const std::string& text, const std::string& key,
                                          int32_t fallback) {
    std::size_t colon = 0;
    if (!find_top_level_key_colon(text, key, colon)) {
        return fallback;
    }
    const std::size_t pos = skip_space(text, colon + 1);
    const std::size_t end = scan_while(text, pos, is_int_char);
    if (end == pos) {
        return fallback;
    }
    try {
        return static_cast<int32_t>(std::stoi(text.substr(pos, end - pos)));
    } catch (const std::exception&) {
        return fallback;
    }
}

inline float extract_top_level_json_float(const std::string& text, const std::string& key,
                                          float fallback) {
    std::size_t colon = 0;
    if (!find_top_level_key_colon(text, key, colon)) {
        return fallback;
    }
    const std::size_t pos = skip_space(text, colon + 1);
    const std::size_t end = scan_while(text, pos, is_float_char);
    if (end == pos) {
        return fallback;
    }
    try {
        return std::stof(text.substr(pos, end - pos));
    } catch (const std::exception&) {
        return fallback;
    }
}

} // namespace voxcpm2_detail

inline VoxCPM2Config make_voxcpm2_config_from_json(const std::string& config_json) {
    VoxCPM2Config cfg;
    cfg.sample_rate =
        voxcpm2_detail::extract_top_level_json_int(config_json, "sample_rate", cfg.sample_rate);
    cfg.reference_sample_rate = voxcpm2_detail::extract_top_level_json_int(
        config_json, "reference_sample_rate", cfg.reference_sample_rate);
    cfg.max_text_steps = voxcpm2_detail::extract_top_level_json_int(
        config_json, "voxcpm2_max_text_steps", cfg.max_text_steps);
    cfg.cfg_value = voxcpm2_detail::extract_top_level_json_float(
        config_json, "voxcpm2_cfg_value", cfg.cfg_value);
    cfg.inference_timesteps = voxcpm2_detail::extract_top_level_json_int(
        config_json, "voxcpm2_inference_timesteps", cfg.inference_timesteps);
    cfg.patch_size = voxcpm2_detail::extract_top_level_json_int(
        config_json, "voxcpm2_patch_size", cfg.patch_size);
    cfg.feat_dim =
        voxcpm2_detail::extract_top_level_json_int(config_json, "voxcpm2_feat_dim", cfg.feat_dim);
    return cfg;
}

inline void apply_voxcpm2_registry_overlay(VoxCPM2Config& voxcpm2_cfg,
                                           const config::ConfigBundle* runtime_config) {
    if (runtime_config == nullptr)
        return;
    try {
        if (runtime_config->source_of("audio_voxcpm2", "cfg_value") != config::Layer::SchemaDefault)
            voxcpm2_cfg.cfg_value = runtime_config->get<float>("audio_voxcpm2", "cfg_value");
        if (runtime_config->source_of("audio_voxcpm2", "inference_timesteps") !=
            config::Layer::SchemaDefault)
            voxcpm2_cfg.inference_timesteps =
                runtime_config->get<std::int32_t>("audio_voxcpm2", "inference_timesteps");
        if (runtime_config->source_of("audio_voxcpm2", "normalize") != config::Layer::SchemaDefault)
            voxcpm2_cfg.normalize = runtime_config->get<bool>("audio_voxcpm2", "normalize");
        if (runtime_config->source_of("audio_voxcpm2", "denoise") != config::Layer::SchemaDefault)
            voxcpm2_cfg.denoise = runtime_config->get<bool>("audio_voxcpm2", "denoise");
        if (runtime_config->source_of("audio_voxcpm2", "retry_badcase") !=
            config::Layer::SchemaDefault)
            voxcpm2_cfg.retry_badcase = runtime_config->get<bool>("audio_voxcpm2", "retry_badcase");
        if (runtime_config->source_of("audio_voxcpm2", "retry_badcase_max_times") !=
            config::Layer::SchemaDefault)
            voxcpm2_cfg.retry_badcase_max_times =
                runtime_config->get<std::int32_t>("audio_voxcpm2", "retry_badcase_max_times");
        if (runtime_config->source_of("audio_voxcpm2", "retry_badcase_ratio_threshold") !=
            config::Layer::SchemaDefault)
            voxcpm2_cfg.retry_badcase_ratio_threshold =
                runtime_config->get<float>("audio_voxcpm2", "retry_badcase_ratio_threshold");
        if (runtime_config->source_of("audio_voxcpm2", "seed") != config::Layer::SchemaDefault)
            voxcpm2_cfg.seed = runtime_config->get<std::int64_t>("audio_voxcpm2", "seed");
    } catch (const std::exception&) {
        // Schema absent or type mismatch: keep bundle/default values.
    }
}

inline VoxCPM2Config make_voxcpm2_config(const std::string& config_json,
                                         const config::ConfigBundle* runtime_config,
                                         int32_t max_text_steps_override = 0) {
    auto cfg = make_voxcpm2_config_from_json(config_json);
    if (max_text_steps_override > 0)
        cfg.max_text_steps = max_text_steps_override;
    apply_voxcpm2_registry_overlay(cfg, runtime_config);
    return cfg;
}

inline std::string describe_voxcpm2_config(const VoxCPM2Config& cfg) {
    std::ostringstream os;
    os << "sample_rate=" << cfg.sample_rate
       << ", reference_sample_rate=" << cfg.reference_sample_rate
       << ", max_text_steps=" << cfg.max_text_steps << ", cfg_value=" << cfg.cfg_value
       << ", inference_timesteps=" << cfg.inference_timesteps
       << ", normalize=" << (cfg.normalize ? "true" : "false")
       << ", denoise=" << (cfg.denoise ? "true" : "false")
       << ", retry_badcase=" << (cfg.retry_badcase ? "true" : "false")
       << ", retry_badcase_max_times=" << cfg.retry_badcase_max_times
       << ", retry_badcase_ratio_threshold=" << cfg.retry_badcase_ratio_threshold
       << ", seed=" << cfg.seed << ", patch_size=" << cfg.patch_size
       << ", feat_dim=" << cfg.feat_dim;
    return os.str();
}

} // namespace trtmc
