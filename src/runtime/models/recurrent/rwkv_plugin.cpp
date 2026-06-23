// RwkvPlugin: handles "rwkv_recurrent" strategy.
// RWKV models with 5 recurrent state vectors per layer.

#include "plugin_helpers.h"
#include "runtime/models/recurrent/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <string>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

struct TensorParallelRuntime {
    TensorParallelRuntimeConfig config;
    DistributedRuntimeGroup group;
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

} // namespace

class RwkvPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        TensorParallelRuntime tp_runtime;
        tp_runtime.config = parse_tensor_parallel_runtime_config(ctx.config_json);
        if (tp_runtime.config.enabled)
            tp_runtime.group = initialize_tensor_parallel_group(tp_runtime.config.tp_size);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;
        if (tp_runtime.config.enabled) {
            opts.distributed_communicator = tp_runtime.group.communicator;
            opts.distributed_owner = tp_runtime.group.owner;
        }

        const std::string engine_section = tp_runtime.config.enabled
                                               ? tp_engine_section_name(tp_runtime.group.rank)
                                               : std::string("engine_plan");
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, engine_section), engine_section.c_str(), opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.module->stream();

        std::vector<RecurrentState::TensorSpec> specs = {
            {"attn_state", {ctx.config.hidden_size}, "present_attn"},
            {"ff_state", {ctx.config.hidden_size}, "present_ff"},
            {"num_state", {ctx.config.hidden_size}, "present_num"},
            {"den_state", {ctx.config.hidden_size}, "present_den"},
            {"max_state", {ctx.config.hidden_size}, "present_max"}};

        auto state = std::make_unique<RecurrentState>(ctx.config.num_layers, specs, stream);
        auto rgc = make_recurrent_gen_config(ctx.config);
        apply_recurrent_chat_template_format(ctx.bundle, rgc);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(state), rgc,
                                                   stream, "RwkvPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_rwkv_plugin, RwkvPlugin, "rwkv_recurrent");

} // namespace trtmc
