// LTXVideoPlugin: handles native TensorRT LTX-Video bundles.

#include "runtime/models/ltx_video/pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace trtmc {

class LTXVideoPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto text_encoder =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, "text_encoder_0_plan"),
                                      "text_encoder_0_plan", opts);
        auto denoiser = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "denoiser_plan"), "denoiser_plan", opts);
        auto vae = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vae_decoder_plan"), "vae_decoder_plan", opts);

        auto config = make_diffusion_config(ctx.config_json);
        auto options = parse_ltx_video_options(ctx.config_json);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<LTXVideoPipeline>(
            std::move(text_encoder.module), std::move(denoiser.module), std::move(vae.module),
            std::move(config), std::move(options), std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_ltx_video_plugin, LTXVideoPlugin, "diffusion_ltx");

} // namespace trtmc
