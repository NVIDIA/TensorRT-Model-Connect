// SsmPlugin: handles "ssm_recurrent" strategy.
// Mamba/SSM models with conv_state + ssm_state recurrent state.

#include "runtime/pipelines/recurrent_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

namespace trtmc {

class SsmPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.module->stream();

        int32_t d_inner = extract_json_int(ctx.config_json, "intermediate_size", 0);
        if (d_inner == 0)
            d_inner = extract_json_int(ctx.config_json, "d_inner", ctx.config.hidden_size * 2);
        int32_t state_size = extract_json_int(ctx.config_json, "state_size", 16);
        int32_t conv_kernel = extract_json_int(ctx.config_json, "conv_kernel", 4);

        std::vector<RecurrentState::TensorSpec> specs = {
            {"conv_state", {d_inner * conv_kernel}, "present_conv"},
            {"ssm_state", {state_size * d_inner}, "present_ssm"}};

        auto state = std::make_unique<RecurrentState>(ctx.config.num_layers, specs, stream);
        auto rgc = make_recurrent_gen_config(ctx.config);
        apply_recurrent_chat_template_format(ctx.bundle, rgc);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(state), rgc,
                                                   stream, "MambaPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_ssm_plugin, SsmPlugin, "ssm_recurrent");

} // namespace trtmc
