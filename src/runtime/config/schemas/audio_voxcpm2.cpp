// Registration for the "audio_voxcpm2" namespace schema.
// Mirrors python/tensorrt_model_connect/runtime_config/schemas/audio_voxcpm2.py.

#include "trtmc/config/schemas/audio_voxcpm2.h"

#include <any>
#include <cstdint>
#include <set>

namespace trtmc::config::schemas {

namespace {
bool is_nonneg_float(const std::any& v) {
    if (v.type() == typeid(float))
        return std::any_cast<float>(v) >= 0.0F;
    if (v.type() == typeid(double))
        return std::any_cast<double>(v) >= 0.0;
    return false;
}

bool is_positive_float(const std::any& v) {
    if (v.type() == typeid(float))
        return std::any_cast<float>(v) > 0.0F;
    if (v.type() == typeid(double))
        return std::any_cast<double>(v) > 0.0;
    return false;
}

bool is_positive_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) > 0;
}

bool is_nonneg_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) >= 0;
}
} // namespace

Schema make_audio_voxcpm2_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "audio_voxcpm2",
        {
            ConfigField{"cfg_value", "float", std::any{2.0F}, session, is_nonneg_float},
            ConfigField{"inference_timesteps", "int32", std::any{std::int32_t{10}}, session,
                        is_positive_int32},
            ConfigField{"normalize", "bool", std::any{true}, session, nullptr},
            ConfigField{"denoise", "bool", std::any{true}, session, nullptr},
            ConfigField{"retry_badcase", "bool", std::any{true}, session, nullptr},
            ConfigField{"retry_badcase_max_times", "int32", std::any{std::int32_t{3}}, session,
                        is_nonneg_int32},
            ConfigField{"retry_badcase_ratio_threshold", "float", std::any{6.0F}, session,
                        is_positive_float},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_audio_voxcpm2_schema,
                                             make_audio_voxcpm2_schema);
} // namespace trtmc::config::schemas
