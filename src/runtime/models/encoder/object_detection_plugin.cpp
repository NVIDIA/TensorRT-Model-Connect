// ObjectDetectionPlugin: handles "object_detection" strategy.
// Uses EncoderPipeline in "object_detection" mode (no tokenizer needed).

#include "runtime/models/encoder/pipeline.h"
#include "plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class ObjectDetectionPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);

        return std::make_unique<EncoderPipeline>(std::move(loaded.module), "object_detection",
                                                 nullptr, ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_object_detection_plugin, ObjectDetectionPlugin,
                                       "object_detection");

} // namespace trtmc
