// Qwen3OmniPlugin: handles "qwen3_omni_multimodal" strategy.
// Omni pipeline with thinker (MoE decoder), optional talker, and optional code2wav.

#include "plugin_helpers.h"
#include "runtime/models/qwen3_omni/pipeline.h"
#include "runtime/models/qwen3_omni/recurrent_state.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

namespace trtmc {

class Qwen3OmniPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto& json = ctx.config_json;

        // Thinker (MoE decoder) -- main engine plan
        auto thinker_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "omni thinker", opts);
        cudaStream_t stream = thinker_loaded.module->stream();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<Qwen3OmniInferenceState> thinker_state = std::make_unique<Qwen3OmniKvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!thinker_state->ok())
            throw std::runtime_error("OmniPipeline: failed to create thinker Qwen3OmniKvCache");

        int32_t omni_talker_hidden_size = extract_json_int(json, "omni_talker_hidden_size", 0);
        int32_t omni_talker_max_cache_length =
            extract_json_int(json, "omni_talker_max_cache_length", 1024);
        int32_t omni_talker_num_layers = extract_json_int(json, "omni_talker_num_layers", 0);

        // Talker (optional)
        std::unique_ptr<TrtModule> talker_module;
        std::unique_ptr<Qwen3OmniInferenceState> talker_state;
        auto talker_loaded = try_load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "talker_engine_plan"), "talker", opts);
        if (talker_loaded.module && talker_loaded.module->ok()) {
            talker_module = std::move(talker_loaded.module);
            if (talker_module->has_input("cache_k_0")) {
                int32_t talker_kv_dim = omni_talker_hidden_size;
                int32_t talker_cache_len = omni_talker_max_cache_length;
                int32_t talker_layers =
                    omni_talker_num_layers > 0 ? omni_talker_num_layers : ctx.config.num_layers;
                talker_state = std::make_unique<Qwen3OmniKvCache>(
                    talker_layers, talker_cache_len, talker_kv_dim, stream, cache_dtype);
            } else {
                talker_state = std::make_unique<Qwen3OmniRecurrentState>(
                    0, std::vector<Qwen3OmniRecurrentState::TensorSpec>{}, stream);
            }
        }

        // Code2Wav (optional)
        std::unique_ptr<TrtModule> code2wav_module;
        auto code2wav_loaded = try_load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "code2wav_engine_plan"), "code2wav", opts);
        if (code2wav_loaded.module && code2wav_loaded.module->ok())
            code2wav_module = std::move(code2wav_loaded.module);

        // Build OmniConfig
        OmniConfig omni_cfg;
        omni_cfg.sample_rate = extract_json_int(json, "audio_sample_rate", 24000);
        omni_cfg.thinker_hidden_size = ctx.config.hidden_size;
        omni_cfg.thinker_num_layers = ctx.config.num_layers;
        omni_cfg.thinker_num_heads = ctx.config.num_heads;
        omni_cfg.num_experts = extract_json_int(json, "num_local_experts", 8);
        omni_cfg.num_experts_per_tok = extract_json_int(json, "num_experts_per_tok", 2);
        omni_cfg.talker_hidden_size = omni_talker_hidden_size;
        omni_cfg.talker_num_layers = omni_talker_num_layers;
        omni_cfg.talker_n_codebooks = extract_json_int(json, "omni_n_codebooks", 8);
        omni_cfg.talker_codebook_size = extract_json_int(json, "omni_codebook_size", 2048);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<OmniPipeline>(
            std::move(thinker_loaded.module), std::move(thinker_state), std::move(talker_module),
            std::move(talker_state), std::move(code2wav_module), std::move(omni_cfg), stream,
            std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen3_omni_plugin, Qwen3OmniPlugin,
                                       "qwen3_omni_multimodal");

} // namespace trtmc
