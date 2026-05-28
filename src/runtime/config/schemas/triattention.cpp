// Registration for the "triattention" namespace schema.
//
// Mirrors python/tensorrt_model_connect/runtime_config/schemas/triattention.py.
// Only edit both sides together; the cross-language match test gates on it.

#include "trtmc/config/schemas/triattention.h"

#include <any>
#include <cstdint>
#include <set>
#include <string>

namespace trtmc::config::schemas {

namespace {

const std::set<Layer> kBundleAndSession = {
    Layer::BuildTime,
    Layer::BundleDefault,
    Layer::PlatformProfile,
    Layer::SessionRequest,
};
const std::set<Layer> kSessionOnly = {Layer::SessionRequest};
const std::set<Layer> kBuildOnly = {Layer::BuildTime, Layer::BundleDefault};

bool is_positive_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) > 0;
}

bool is_nonnegative_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) >= 0;
}

bool is_ge1_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) >= 1;
}

bool is_mean_or_max(const std::any& v) {
    if (v.type() != typeid(std::string))
        return false;
    const auto& s = std::any_cast<const std::string&>(v);
    return s == "mean" || s == "max";
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

Schema make_triattention_schema() {
    return Schema{
        "triattention",
        {
            // --- Core runtime config (formerly bundle + overrides) ------
            bool_field("enabled", false, kBundleAndSession),
            int_field("kv_budget", 4096, kBundleAndSession, is_positive_int32),
            int_field("divide_length", 128, kBundleAndSession, is_positive_int32),
            int_field("recent_window", 128, kBundleAndSession, is_nonnegative_int32),
            str_field("score_aggregation", "mean", kBundleAndSession, is_mean_or_max),
            str_field("per_layer_aggregation", "mean", kBundleAndSession, is_mean_or_max),
            bool_field("count_prompt_tokens", true, kBundleAndSession),
            bool_field("protect_prefill", true, kBundleAndSession),
            bool_field("disable_mlr", false, kBundleAndSession),
            bool_field("disable_trig", false, kBundleAndSession),
            int_field("offset_max_length", 65536, kBundleAndSession, is_positive_int32),
            str_field("stats_section", "triattention_stats.json", kBuildOnly),
            // --- Debug / profiling (session-only) -----------------------
            bool_field("debug", false, kSessionOnly),
            bool_field("profile", false, kSessionOnly),
            int_field("runtime_bucket_rows", 32, kSessionOnly, is_ge1_int32),
            bool_field("disable_gpu_selection", false, kSessionOnly),
            bool_field("disable_gpu_compaction", false, kSessionOnly),
            bool_field("disable_gpu_state", false, kSessionOnly),
            bool_field("zero_tail", false, kSessionOnly),
            str_field("dump_keep_path", "", kSessionOnly),
            int_field("dump_compaction_index", 0, kSessionOnly, is_nonnegative_int32),
            bool_field("abort_after_dump", false, kSessionOnly),
            bool_field("dump_score_cache", false, kSessionOnly),
            bool_field("dump_score_values", false, kSessionOnly),
        },
    };
}

} // namespace trtmc::config::schemas

namespace trtmc::config::schemas {
REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(register_triattention_schema,
                                             make_triattention_schema);
} // namespace trtmc::config::schemas
