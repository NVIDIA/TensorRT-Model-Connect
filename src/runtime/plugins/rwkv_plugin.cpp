// RwkvPlugin: handles "rwkv_recurrent" strategy.
// RWKV models with 5 recurrent state vectors per layer.

#include "runtime/pipelines/recurrent_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class RwkvPlugin final : public IPipelinePlugin {
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
