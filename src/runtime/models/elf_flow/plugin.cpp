// ElfFlowPlugin: loads GitHub ELF denoiser/decoder bundles.

#include "runtime/models/elf_flow/pipeline.h"
#include "plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

namespace trtmc {

class ElfFlowPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "elf_flow engine", opts);
        std::unique_ptr<TrtModule> text_encoder;
        if (const auto* encoder_plan = find_section(ctx.bundle, "elf_text_encoder_plan")) {
            auto loaded_encoder =
                load_trt_module_from_plan(ctx.backend, encoder_plan, "elf_text_encoder_plan", opts);
            text_encoder = std::move(loaded_encoder.module);
        }

        int32_t max_length = extract_json_int(ctx.config_json, "elf_max_length", 0);
        if (max_length <= 0)
            max_length = extract_json_int(ctx.config_json, "max_length", 0);
        if (max_length <= 0)
            max_length = extract_json_int(ctx.config_json, "max_position_embeddings", 0);
        int32_t max_input_length = extract_json_int(ctx.config_json, "elf_max_input_length", 0);
        if (max_input_length <= 0)
            max_input_length = extract_json_int(ctx.config_json, "max_input_length", 0);

        int32_t input_dim = extract_json_int(ctx.config_json, "elf_input_dim", 0);
        int32_t text_dim = extract_json_int(ctx.config_json, "elf_text_encoder_dim", 0);
        int32_t vocab_size = extract_json_int(ctx.config_json, "vocab_size", 0);
        float denoiser_noise_scale =
            extract_json_float(ctx.config_json, "elf_denoiser_noise_scale", 1.0F);
        float denoiser_p_mean = extract_json_float(ctx.config_json, "elf_denoiser_p_mean", -1.5F);
        float denoiser_p_std = extract_json_float(ctx.config_json, "elf_denoiser_p_std", 0.8F);
        float t_eps = extract_json_float(ctx.config_json, "elf_t_eps", 5e-2F);
        float latent_mean = extract_json_float(ctx.config_json, "elf_latent_mean", 0.0F);
        float latent_std = extract_json_float(ctx.config_json, "elf_latent_std", 1.0F);
        int32_t encoder_pad_token_id =
            extract_json_int(ctx.config_json, "elf_encoder_pad_token_id", 0);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<ElfFlowPipeline>(
            std::move(loaded.module), max_length, max_input_length, input_dim, text_dim, vocab_size,
            denoiser_noise_scale, denoiser_p_mean, denoiser_p_std, t_eps, std::move(tokenizer),
            ctx.bundle.info.model_id, std::move(text_encoder), latent_mean, latent_std,
            encoder_pad_token_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_elf_flow_plugin, ElfFlowPlugin, "elf_flow");

} // namespace trtmc
