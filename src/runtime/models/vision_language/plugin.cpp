// VLPlugin: handles "vision_language" strategy.
// Two-engine pipeline: vision encoder + text decoder with KV cache.
//
// Tensor parallelism support: when the bundle declares
// `tensor_parallel_mode=tensor_parallel` and `tensor_parallel_size>1`, the
// text decoder is loaded from the rank-local `engine_plan_tp_rank<N>`
// section and a NCCL communicator is bound to its execution context. The
// vision encoder is kept replicated (loaded once per rank from the shared
// `vision_engine_plan` section) — vision features are computed identically
// on every rank, so no all-reduce on the vision branch.

#include "runtime/core/trt_engine_lifecycle.h"
#include "runtime/domains/multimodal/image_preprocessor.h"
#include "runtime/models/vision_language/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <iostream>
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

class VLPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        auto shared_stream = std::make_shared<CudaStream>();
        if (!shared_stream->ok())
            throw std::runtime_error("VLPlugin: failed to create CUDA stream");

        // Detect tensor-parallel metadata and bring up the NCCL group when
        // requested. tp_runtime.group.communicator stays null on single-device.
        TensorParallelRuntime tp_runtime;
        tp_runtime.config = parse_tensor_parallel_runtime_config(ctx.config_json);
        if (tp_runtime.config.enabled)
            tp_runtime.group = initialize_tensor_parallel_group(tp_runtime.config.tp_size);

        ModuleCreateOptions opts;
        opts.stream = shared_stream->get();
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;
        opts.distributed_communicator = tp_runtime.group.communicator;
        opts.distributed_owner = tp_runtime.group.owner;

        // Build KvCacheNames from IoMap patterns.
        const auto& io = ctx.config.io_map;
        KvCacheNames kv_names;
        for (int32_t i = 0; i < ctx.config.num_layers; ++i) {
            kv_names.cache_k.push_back(expand_layer_name(io.cache_k_pattern, i));
            kv_names.cache_v.push_back(expand_layer_name(io.cache_v_pattern, i));
            kv_names.present_k.push_back(expand_layer_name(io.present_k_pattern, i));
            kv_names.present_v.push_back(expand_layer_name(io.present_v_pattern, i));
        }

        // Text decoder section is rank-local under TP; single `engine_plan`
        // otherwise.
        const std::string text_section_name = tp_runtime.config.enabled
                                                  ? tp_engine_section_name(tp_runtime.group.rank)
                                                  : std::string("engine_plan");
        auto loaded = load_trt_module_from_plan(ctx.backend,
                                                find_section(ctx.bundle, text_section_name),
                                                text_section_name.c_str(), opts);
        loaded.module->keep_alive(shared_stream);

        cudaStream_t stream = loaded.module->stream();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<IInferenceState> state =
            std::make_unique<KvCache>(ctx.config.num_layers, ctx.config.max_cache_length, kv_dim,
                                      stream, cache_dtype, std::move(kv_names));

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        VLConfig vlc;
        vlc.vocab_size = ctx.config.vocab_size;
        vlc.id_bos = ctx.config.id_bos;
        vlc.id_eos = ctx.config.id_eos;
        vlc.image_token_id = extract_json_int(ctx.config_json, "image_token_id", -1);
        vlc.vision_output_dim = extract_json_int(ctx.config_json, "vision_output_dim", 0);
        vlc.has_position_input = loaded.module->has_input("position_id");

        bool has_vision_engine = extract_json_int(ctx.config_json, "has_vision_engine", 0) != 0;

        // Vision encoder is replicated across TP ranks — single section, no
        // NCCL on its execution context.
        ModuleCreateOptions vision_opts;
        vision_opts.stream = shared_stream->get();
        vision_opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        vision_opts.cuda_graphs = ctx.cuda_graphs;
        std::unique_ptr<TrtModule> vision_module;
        auto vision_loaded = try_load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "vision_engine_plan",
            vision_opts);
        if (vision_loaded.module && vision_loaded.module->ok()) {
            vision_loaded.module->keep_alive(shared_stream);
            vision_module = std::move(vision_loaded.module);
            std::cerr << "[trtmc] Vision encoder loaded" << std::endl;
        } else if (has_vision_engine) {
            std::cerr << "[trtmc] WARNING: Bundle declares vision engine but "
                         "deserialization failed"
                      << std::endl;
        }

        // Build VL preprocessing config from bundle's config.json +
        // preprocessor_config.json sections.
        std::string config_text, preproc_text;
        const auto* config_sec = find_section(ctx.bundle, "config.json");
        if (config_sec && !config_sec->empty())
            config_text.assign(config_sec->begin(), config_sec->end());
        const auto* preproc_sec = find_section(ctx.bundle, "preprocessor_config.json");
        if (preproc_sec && !preproc_sec->empty())
            preproc_text.assign(preproc_sec->begin(), preproc_sec->end());
        auto vl_preprocess = parse_vl_preprocess_config(config_text, preproc_text);

        return std::make_unique<VLPipeline>(
            std::move(loaded.module), std::move(vision_module), std::move(state), vlc,
            vl_preprocess, stream, std::move(tokenizer), ctx.bundle.info.model_id,
            /*sampler=*/nullptr, std::move(tp_runtime.group.owner));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_vl_plugin, VLPlugin, "vision_language");

} // namespace trtmc
