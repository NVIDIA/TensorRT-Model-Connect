// Cosmos3Plugin: registers Cosmos3Pipeline for the "diffusion_cosmos3"
// runtime strategy.

#include "runtime/models/cosmos3/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <memory>
#include <string>
#include <utility>

namespace trtmc {

class Cosmos3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        // For now this throws so the Python plugin can land with a clear
        // error path; the actual TrtModule loading + KvCache wiring will
        // be filled in once the DM generator TRT graph builder (Phase 4)
        // produces concrete engine artifacts that the pipeline can
        // consume.
        throw std::runtime_error("Cosmos3Plugin::create is a scaffold — full pipeline wiring "
                                 "follows once the DM generator TRT graph builder is "
                                 "implemented (Phase 4) and a working bundle layout for "
                                 "Cosmos 3 is defined.");
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_cosmos3_plugin, Cosmos3Plugin, "diffusion_cosmos3");

} // namespace trtmc
