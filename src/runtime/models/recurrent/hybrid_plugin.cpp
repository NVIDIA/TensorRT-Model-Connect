// HybridPlugin: handles "hybrid_mamba_attention" strategy.
// Nemotron-H style models with interleaved attention and Mamba layers,
// using KvCache for attention layers and RecurrentState for SSM layers.

#include "runtime/models/recurrent/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/hybrid_state.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>

namespace trtmc {

class HybridPlugin final : public IPipelinePlugin {
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
        int32_t kv_dim = compute_kv_dim(ctx.config);

        int32_t num_attention_layers = extract_json_int(ctx.config_json, "num_attention_layers", 0);
        int32_t num_mamba_layers = extract_json_int(ctx.config_json, "num_mamba_layers", 0);
        int32_t d_inner = extract_json_int(ctx.config_json, "d_inner", ctx.config.hidden_size * 2);
        int32_t mamba_d_state = extract_json_int(ctx.config_json, "mamba_d_state", 128);
        int32_t mamba_d_conv = extract_json_int(ctx.config_json, "mamba_d_conv", 4);
        int32_t mamba_nheads = extract_json_int(ctx.config_json, "mamba_nheads", 0);
        int32_t mamba_head_dim = extract_json_int(ctx.config_json, "mamba_head_dim", 0);
        int32_t conv_dim = extract_json_int(ctx.config_json, "conv_dim", d_inner);

        // KvCache for the attention layers
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        auto cache = std::make_unique<KvCache>(num_attention_layers, ctx.config.max_cache_length,
                                               kv_dim, stream, cache_dtype);
        if (!cache->ok())
            throw std::runtime_error("Failed to create KvCache for hybrid model");

        // RecurrentState for the Mamba/SSM layers (conv_state + ssm_state)
        int32_t effective_conv_dim = (conv_dim > 0) ? conv_dim : d_inner;
        int64_t conv_elems = static_cast<int64_t>(effective_conv_dim) * mamba_d_conv;
        int64_t ssm_elems =
            static_cast<int64_t>(mamba_nheads) * std::max(mamba_head_dim, 1) * mamba_d_state;

        auto ssm = std::make_unique<RecurrentState>(
            num_mamba_layers,
            std::vector<RecurrentState::TensorSpec>{{"conv_state", {conv_elems}, "present_conv"},
                                                    {"ssm_state", {ssm_elems}, "present_ssm"}},
            stream);

        // HybridState combines KvCache + RecurrentState behind IInferenceState
        auto hybrid = std::make_unique<HybridState>(std::move(cache), std::move(ssm));
        auto rgc = make_recurrent_gen_config(ctx.config);
        rgc.has_position_input = loaded.module->has_input("position_id");
        apply_recurrent_chat_template_format(ctx.bundle, rgc);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(hybrid), rgc,
                                                   stream, "HybridPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_hybrid_plugin, HybridPlugin,
                                       "hybrid_mamba_attention");

} // namespace trtmc
