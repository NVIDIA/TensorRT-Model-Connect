// TimmVitPlugin: handles timm ViT image-classification bundles.

#include "runtime/models/timm_vit/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class TimmVitPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);
        return std::make_unique<ImageClassificationPipeline>(std::move(loaded.module),
                                                             ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_timm_vit_plugin, TimmVitPlugin,
                                       "image_classification");

} // namespace trtmc
