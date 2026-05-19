// Registration for the "deployment" namespace schema.
// Mirrors tensorrt_model_connect/runtime_config/schemas/deployment.py.

#include "trtmc/config/schemas/deployment.h"

#include <any>
#include <set>
#include <string>

namespace trtmc::config::schemas {

namespace {
bool is_valid_provider(const std::any& v) {
    if (v.type() != typeid(std::string))
        return false;
    const auto& provider = std::any_cast<const std::string&>(v);
    return provider == "native_trt" || provider == "tvm_ffi" ||
           provider == "tensorrt-edge-llm";
}

bool is_positive_int(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) > 0;
}
} // namespace

Schema make_deployment_schema() {
    const std::set<Layer> build_and_session = {Layer::BuildTime, Layer::BundleDefault,
                                               Layer::PlatformProfile,
                                               Layer::SessionRequest};
    const std::set<Layer> session = {Layer::PlatformProfile, Layer::SessionRequest};
    return Schema{
        "deployment",
        {
            ConfigField{"target", "string", std::any{std::string{"generic"}},
                        build_and_session, nullptr},
            ConfigField{"provider", "string", std::any{std::string{"native_trt"}},
                        build_and_session, is_valid_provider},
            ConfigField{"variant", "string", std::any{std::string{}}, build_and_session,
                        nullptr},
            ConfigField{"force_fallback", "bool", std::any{false}, session, nullptr},
            ConfigField{"enable_ffi_attention", "bool", std::any{false}, build_and_session,
                        nullptr},
            ConfigField{"ffi_kernel_artifacts", "string", std::any{std::string{}},
                        build_and_session, nullptr},
            ConfigField{"edge_llm_workspace", "string", std::any{std::string{}},
                        build_and_session, nullptr},
            ConfigField{"edge_llm_engine_dir", "string", std::any{std::string{}},
                        build_and_session, nullptr},
            ConfigField{"edge_llm_export_tool", "string",
                        std::any{std::string{"tensorrt-edgellm-export-llm"}},
                        build_and_session, nullptr},
            ConfigField{"edge_llm_build_tool", "string", std::any{std::string{"llm_build"}},
                        build_and_session, nullptr},
            ConfigField{"edge_llm_export_device", "string", std::any{std::string{"cuda"}},
                        build_and_session, nullptr},
            ConfigField{"edge_llm_max_input_len", "int32", std::any{std::int32_t{1024}},
                        build_and_session, is_positive_int},
            ConfigField{"edge_llm_max_batch_size", "int32", std::any{std::int32_t{4}},
                        build_and_session, is_positive_int},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_deployment_schema, make_deployment_schema);
} // namespace trtmc::config::schemas
