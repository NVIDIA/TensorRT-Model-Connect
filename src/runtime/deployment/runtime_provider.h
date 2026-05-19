#pragma once

#include "runtime/deployment/artifact_store.h"
#include "runtime/deployment/deployment_manifest.h"
#include "trtmc/pipeline.h"

#include <memory>
#include <string>

namespace trtmc::deployment {

std::unique_ptr<IPipeline> load_runtime_provider(const Variant& variant,
                                                 const ArtifactStore& artifacts,
                                                 const std::string& bundle_path);

} // namespace trtmc::deployment
