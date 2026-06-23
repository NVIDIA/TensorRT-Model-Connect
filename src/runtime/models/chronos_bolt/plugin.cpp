// ChronosBoltPlugin: handles "chronos_bolt_trt" strategy.
//
// This is a numeric forecasting pipeline for Chronos-Bolt-style bundles.
// The C++ runtime keeps the implementation intentionally narrow: load the
// TRT engine, feed dense context tensors, and return the forecast tensor.

#include "plugin_helpers.h"
#include "runtime/models/chronos_bolt/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

int32_t tensor_parallel_rank_from_env() {
    for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK", "SLURM_PROCID"}) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0')
            return static_cast<int32_t>(std::strtol(value, nullptr, 10));
    }
    return 0;
}

} // namespace

class ChronosBoltPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        std::string engine_section = "engine_plan";
        if (tp_config.enabled) {
            engine_section = tp_engine_section_name(tensor_parallel_rank_from_env());
        }

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, engine_section), "chronos_bolt forecast", opts);

        const auto& json = ctx.config_json;
        int32_t context_length =
            extract_json_int(json, "context_length", ctx.config.max_cache_length);
        int32_t prediction_length = extract_json_int(json, "prediction_length",
                                                     extract_json_int(json, "forecast_length", 24));
        int32_t num_quantiles = extract_json_int(
            json, "num_quantiles",
            static_cast<int32_t>(extract_json_float_array(json, "quantiles").size()));
        if (num_quantiles <= 0)
            num_quantiles = 3;

        return std::make_unique<ChronosBoltPipeline>(std::move(loaded.module), context_length,
                                                     prediction_length, num_quantiles,
                                                     ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_chronos_bolt_plugin, ChronosBoltPlugin,
                                       "chronos_bolt_trt");

} // namespace trtmc
