#pragma once

#include "trtmc/config/config_bundle.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <exception>
#include <sstream>
#include <string>

namespace trtmc {

struct VoxCPM2Config {
    int32_t sample_rate{48000};
    int32_t reference_sample_rate{16000};
    float cfg_value{2.0F};
    int32_t inference_timesteps{10};
    bool normalize{true};
    bool denoise{true};
    bool retry_badcase{true};
    int32_t retry_badcase_max_times{3};
    float retry_badcase_ratio_threshold{6.0F};
    int64_t seed{-1};
};

inline VoxCPM2Config make_voxcpm2_config_from_json(const std::string& config_json) {
    VoxCPM2Config cfg;
    cfg.sample_rate = extract_json_int(config_json, "sample_rate", cfg.sample_rate);
    cfg.reference_sample_rate =
        extract_json_int(config_json, "reference_sample_rate", cfg.reference_sample_rate);
    cfg.cfg_value = extract_json_float(config_json, "voxcpm2_cfg_value", cfg.cfg_value);
    cfg.inference_timesteps =
        extract_json_int(config_json, "voxcpm2_inference_timesteps", cfg.inference_timesteps);
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
                                         const config::ConfigBundle* runtime_config) {
    auto cfg = make_voxcpm2_config_from_json(config_json);
    apply_voxcpm2_registry_overlay(cfg, runtime_config);
    return cfg;
}

inline std::string describe_voxcpm2_config(const VoxCPM2Config& cfg) {
    std::ostringstream os;
    os << "sample_rate=" << cfg.sample_rate
       << ", reference_sample_rate=" << cfg.reference_sample_rate << ", cfg_value=" << cfg.cfg_value
       << ", inference_timesteps=" << cfg.inference_timesteps
       << ", normalize=" << (cfg.normalize ? "true" : "false")
       << ", denoise=" << (cfg.denoise ? "true" : "false")
       << ", retry_badcase=" << (cfg.retry_badcase ? "true" : "false")
       << ", retry_badcase_max_times=" << cfg.retry_badcase_max_times
       << ", retry_badcase_ratio_threshold=" << cfg.retry_badcase_ratio_threshold
       << ", seed=" << cfg.seed;
    return os.str();
}

} // namespace trtmc
