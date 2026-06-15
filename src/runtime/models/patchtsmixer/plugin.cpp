// PatchTSMixerPlugin: handles "patchtsmixer_trt" strategy.
// Loads a single TRT engine and dispatches to PatchTSMixerPipeline.

#include "runtime/models/patchtsmixer/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <cstdlib>
#include <string>

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

class PatchTSMixerPlugin final : public IPipelinePlugin {
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
            ctx.backend, find_section(ctx.bundle, engine_section), "engine_plan", opts);
        auto cfg = parse_patchtsmixer_config(ctx.config_json, ctx.config.max_cache_length);
        return std::make_unique<PatchTSMixerPipeline>(std::move(loaded.module), std::move(cfg),
                                                      ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_patchtsmixer_plugin, PatchTSMixerPlugin,
                                       "patchtsmixer_trt");

} // namespace trtmc
