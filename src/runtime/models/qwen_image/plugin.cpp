// QwenImagePlugin: handles "diffusion_qwen_image" strategy.
// Qwen-Image diffusion pipeline with Qwen2.5-VL text encoder, MMDiT denoiser,
// AutoencoderKLQwenImage VAE decoder, and Qwen-Image-specific preprocessor
// weights (per-channel latents_mean / latents_std).
//
// Mirrors zimage_plugin.cpp — loads diffusion engines via the shared
// load_diffusion_parts helper, parses Qwen-Image config and preprocessor
// weights, then constructs QwenImagePipeline via its Construction struct.
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.

#include "runtime/domains/diffusion/qwen_image_types.h"
#include "runtime/models/qwen_image/pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class QwenImagePlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        // Load the standard diffusion engine triple (text_encoder_0_plan,
        // denoiser_plan, vae_decoder_plan) via the shared helper. T2I bundles
        // do not ship vision / vae_encoder engines, so those stay null on
        // Construction.
        auto parts = load_diffusion_parts(ctx.backend, ctx.bundle, ctx.config_json, opts);

        // Parse Qwen-Image-specific config (diffusion / text_encoder / denoiser
        // / vae / image / tokenizer sections) from the raw bundle config JSON.
        auto qi_config = QwenImageConfig::parse(ctx.config_json);

        // Parse Qwen-Image preprocessor weights (latents_mean / latents_std)
        // from the bundle's preprocessor_weights section. Falls back to an
        // empty struct (valid=false) if the section is missing — VAE decode
        // will fail downstream if the weights are actually needed.
        QwenImagePreprocessorWeights qi_preprocessor;
        const auto* pw_sec = find_section(ctx.bundle, "preprocessor_weights");
        if (pw_sec && !pw_sec->empty())
            qi_preprocessor = parse_qwen_image_preprocessor_weights(*pw_sec);

        // Extract the first (and only, for Qwen-Image T2I) text encoder.
        std::unique_ptr<ITrtModule> text_module;
        if (!parts.text_encoders.empty())
            text_module = std::move(parts.text_encoders[0].module);

        QwenImagePipeline::Construction c;
        c.text_engine = std::move(text_module);
        c.denoiser_engine = std::move(parts.denoiser.module);
        c.vae_decoder_engine = std::move(parts.vae.module);
        c.vision_engine = std::move(parts.vision.module);
        c.vae_encoder_engine = std::move(parts.vae_encoder.module);
        c.tokenizer = std::move(parts.tokenizer);
        c.config = std::move(qi_config);
        c.preprocessor = std::move(qi_preprocessor);
        c.model_id = ctx.bundle.info.model_id;
        c.bundle_path = ctx.bundle_path;
        return std::make_unique<QwenImagePipeline>(std::move(c));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen_image_plugin, QwenImagePlugin,
                                       "diffusion_qwen_image");

} // namespace trtmc
