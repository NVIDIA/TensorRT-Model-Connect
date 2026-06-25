// Qwen35Plugin: handles the Qwen3.5-owned hybrid recurrent strategy.
// Qwen3.5 style models with interleaved attention and Mamba layers,
// using Qwen35KvCache for attention layers and Qwen35RecurrentState for SSM layers.

#include "plugin_helpers.h"
#include "runtime/models/qwen3_5/hybrid_state.h"
#include "runtime/models/qwen3_5/pipeline.h"
#include "runtime/models/qwen3_5/recurrent_state.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
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

class Qwen35Plugin final : public IPipelinePlugin {
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
        int32_t kv_dim = compute_kv_dim(ctx.config);

        int32_t num_attention_layers = extract_json_int(ctx.config_json, "num_attention_layers", 0);
        int32_t num_mamba_layers = extract_json_int(ctx.config_json, "num_mamba_layers", 0);
        int32_t d_inner = extract_json_int(ctx.config_json, "d_inner", ctx.config.hidden_size * 2);
        int32_t mamba_d_state = extract_json_int(ctx.config_json, "mamba_d_state", 128);
        int32_t mamba_d_conv = extract_json_int(ctx.config_json, "mamba_d_conv", 4);
        int32_t mamba_nheads = extract_json_int(ctx.config_json, "mamba_nheads", 0);
        int32_t mamba_head_dim = extract_json_int(ctx.config_json, "mamba_head_dim", 0);
        int32_t conv_dim = extract_json_int(ctx.config_json, "conv_dim", d_inner);

        // Qwen35KvCache for the attention layers
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        auto cache = std::make_unique<Qwen35KvCache>(
            num_attention_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!cache->ok())
            throw std::runtime_error("Failed to create Qwen35KvCache for hybrid model");

        // Qwen35RecurrentState for the Mamba/SSM layers (conv_state + ssm_state)
        int32_t effective_conv_dim = (conv_dim > 0) ? conv_dim : d_inner;
        int64_t conv_elems = static_cast<int64_t>(effective_conv_dim) * mamba_d_conv;
        int64_t ssm_elems =
            static_cast<int64_t>(mamba_nheads) * std::max(mamba_head_dim, 1) * mamba_d_state;

        auto ssm =
            std::make_unique<Qwen35RecurrentState>(num_mamba_layers,
                                                   std::vector<Qwen35RecurrentState::TensorSpec>{
                                                       {"conv_state", {conv_elems}, "present_conv"},
                                                       {"ssm_state", {ssm_elems}, "present_ssm"}},
                                                   stream);

        auto hybrid = std::make_unique<Qwen35HybridState>(std::move(cache), std::move(ssm));
        auto rgc = make_recurrent_gen_config(ctx.config);
        rgc.has_position_input = loaded.module->has_input("position_id");
        apply_recurrent_chat_template_format(ctx.bundle, rgc);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(hybrid), rgc,
                                                   stream, "HybridPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen3_5_plugin, Qwen35Plugin,
                                       "qwen3_5_hybrid_mamba_attention");

} // namespace trtmc
