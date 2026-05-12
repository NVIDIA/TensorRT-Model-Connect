// PatchTSMixerPlugin: handles "patchtsmixer_torchtrt" strategy.
// Loads a single TRT engine and dispatches to PatchTSMixerPipeline.

#include "runtime/models/patchtsmixer/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class PatchTSMixerPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan");
        auto cfg = parse_patchtsmixer_config(ctx.config_json, ctx.config.max_cache_length);
        return std::make_unique<PatchTSMixerPipeline>(std::move(loaded.module), std::move(cfg),
                                                      ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_patchtsmixer_plugin, PatchTSMixerPlugin,
                                       "patchtsmixer_torchtrt");

} // namespace trtmc
