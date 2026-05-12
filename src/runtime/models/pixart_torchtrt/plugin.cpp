// PixArtTorchTrtPlugin: handles "diffusion_pixart_torchtrt" strategy.
// Uses PixArtTorchTrtPipeline for torch-trt compiled PixArt diffusion models.
// Unlike PixArtPlugin, no preprocessor_weights are needed since the torch-trt
// engines include all preprocessing internally.

#include "runtime/models/pixart_torchtrt/pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class PixArtTorchTrtPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        // Load the three component engines
        auto te =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, "text_encoder_0_plan"),
                                      "text_encoder_0_plan", opts);
        auto denoiser = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "denoiser_plan"), "denoiser_plan", opts);
        auto vae = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vae_decoder_plan"), "vae_decoder_plan", opts);

        auto config = make_diffusion_config(ctx.config_json);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<PixArtTorchTrtPipeline>(
            std::move(te.module), std::move(denoiser.module), std::move(vae.module),
            std::move(config), std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_pixart_torchtrt_plugin, PixArtTorchTrtPlugin,
                                       "diffusion_pixart_torchtrt");

} // namespace trtmc
