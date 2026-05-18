// SANA-WM plugin: Python bridge for the official image-to-video contract.

#include "runtime/models/sana_wm/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <memory>
#include <utility>

namespace trtmc {

class SanaWmPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto config = parse_sana_wm_config(ctx.config_json);
        return std::make_unique<SanaWmPipeline>(std::move(config), ctx.hf_python);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sana_wm_plugin, SanaWmPlugin, "diffusion_sana_wm");

} // namespace trtmc
