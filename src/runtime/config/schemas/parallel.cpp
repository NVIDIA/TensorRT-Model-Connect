// Registration for the "parallel" namespace schema.
//
// Mirrors tensorrt_model_connect/tensorrt_model_connect/runtime_config/schemas/parallel.py.

#include "trtmc/config/schemas/parallel.h"

#include <any>
#include <cstdint>
#include <set>
#include <string>

namespace trtmc::config::schemas {

namespace {

const std::set<Layer> kBuildAndSession = {
    Layer::BuildTime,
    Layer::BundleDefault,
    Layer::PlatformProfile,
    Layer::SessionRequest,
};

bool valid_mode(const std::any& v) {
    if (v.type() != typeid(std::string))
        return false;
    const auto& s = std::any_cast<const std::string&>(v);
    return s == "single" || s == "tensor_parallel";
}

bool valid_tp_size(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    const auto size = std::any_cast<std::int32_t>(v);
    return size == 1 || size == 2 || size == 4 || size == 8;
}

bool valid_rank(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) >= -1;
}

ConfigField bool_field(const std::string& name, bool default_value, const std::set<Layer>& layers) {
    return ConfigField{name, "bool", std::any{default_value}, layers, nullptr};
}

ConfigField int_field(const std::string& name, std::int32_t default_value,
                      const std::set<Layer>& layers,
                      std::function<bool(const std::any&)> validator = nullptr) {
    return ConfigField{name, "int32", std::any{default_value}, layers, std::move(validator)};
}

ConfigField str_field(const std::string& name, const std::string& default_value,
                      const std::set<Layer>& layers,
                      std::function<bool(const std::any&)> validator = nullptr) {
    return ConfigField{name, "string", std::any{default_value}, layers, std::move(validator)};
}

} // namespace

Schema make_parallel_schema() {
    return Schema{
        "parallel",
        {
            str_field("mode", "single", kBuildAndSession, valid_mode),
            int_field("tp_size", 1, kBuildAndSession, valid_tp_size),
            int_field("rank", -1, kBuildAndSession, valid_rank),
            bool_field("require_mpirun", true, kBuildAndSession),
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_parallel_schema, make_parallel_schema);

} // namespace trtmc::config::schemas
