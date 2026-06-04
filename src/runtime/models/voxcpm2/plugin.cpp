// VoxCPM2Plugin: explicit runtime boundary for "text_to_audio_voxcpm2".
// Full support requires native LocEnc, TSLM, RALM, LocDiT, and AudioVAE engines.

#include "runtime/domains/audio/audio_bundle_validation.h"
#include "runtime/domains/audio/voxcpm2_component_loader.h"
#include "runtime/domains/audio/voxcpm2_config.h"
#include "runtime/domains/audio/voxcpm2_generation_plan.h"
#include "runtime/models/voxcpm2/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <memory>
#include <string>
#include <utility>

namespace trtmc {

class VoxCPM2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        runtime::builders::audio::validate_text_to_audio_bundle_sections(
            runtime::builders::audio::TextToAudioBundleKind::kVoxCpm2, ctx.bundle, ctx.bundle_path);

        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto components =
            runtime::builders::audio::load_voxcpm2_component_modules(ctx.backend, ctx.bundle, opts);
        const auto generation_cfg = make_voxcpm2_config(ctx.config_json, ctx.runtime_config);
        const auto generation_plan =
            runtime::builders::audio::make_voxcpm2_generation_plan(generation_cfg);

        return std::make_unique<VoxCPM2Pipeline>(std::move(components), generation_plan,
                                                 ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_voxcpm2_plugin, VoxCPM2Plugin,
                                       "text_to_audio_voxcpm2");

} // namespace trtmc
