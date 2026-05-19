// SANA-WM plugin: loads optional native TensorRT modules while preserving the
// bridge-only bundle contract until native plan construction is complete.

#include "bundle/bundle_view.h"
#include "runtime/models/sana_wm/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <memory>
#include <utility>

namespace trtmc {

namespace {

std::unique_ptr<ITrtModule> load_optional_sana_wm_module(const PipelineContext& ctx,
                                                         const char* section,
                                                         const ModuleCreateOptions& opts) {
    const auto* plan = find_section(ctx.bundle, section);
    if (!plan || plan->empty())
        return nullptr;
    auto loaded = load_trt_module_from_plan(ctx.backend, plan, section, opts);
    return std::move(loaded.module);
}

SanaWmNativeModules load_sana_wm_native_modules(const PipelineContext& ctx) {
    ModuleCreateOptions opts;
    opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
    opts.cuda_graphs = ctx.cuda_graphs;

    SanaWmNativeModules modules;
    modules.text_encoder = load_optional_sana_wm_module(ctx, "text_encoder_0_plan", opts);
    modules.stage1_denoiser = load_optional_sana_wm_module(ctx, "denoiser_plan", opts);
    modules.vae_encoder = load_optional_sana_wm_module(ctx, "sana_wm_vae_encoder_plan", opts);
    modules.vae_decoder = load_optional_sana_wm_module(ctx, "vae_decoder_plan", opts);
    modules.refiner_text_encoder =
        load_optional_sana_wm_module(ctx, "sana_wm_refiner_text_encoder_plan", opts);
    modules.refiner_denoiser =
        load_optional_sana_wm_module(ctx, "sana_wm_refiner_denoiser_plan", opts);
    modules.refiner_vae_decoder =
        load_optional_sana_wm_module(ctx, "sana_wm_refiner_vae_decoder_plan", opts);
    return modules;
}

} // namespace

class SanaWmPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto config = parse_sana_wm_config(ctx.config_json);
        auto native_modules = load_sana_wm_native_modules(ctx);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        return std::make_unique<SanaWmPipeline>(std::move(config), ctx.hf_python, nullptr,
                                                std::move(native_modules), std::move(tokenizer));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sana_wm_plugin, SanaWmPlugin, "diffusion_sana_wm");

} // namespace trtmc
