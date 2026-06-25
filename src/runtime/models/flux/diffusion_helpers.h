#pragma once
#include "plugin_helpers.h"
#include "preprocessor_weights_helpers.h"
#include "runtime/models/flux/flux_diffusion_types.h"
#include "utils/json_helpers.h"

namespace trtmc {

FluxDiffusionConfig make_diffusion_config(const std::string& json);

// Flux diffusion resources loaded once, then dispatched to the Flux factory.
struct DiffusionParts {
    LoadedModule denoiser;
    LoadedModule vae;
    LoadedModule vision;
    LoadedModule vae_encoder;
    std::vector<LoadedModule> text_encoders;
    FluxDiffusionConfig config;
    FluxPreprocessorWeights weights;
    std::shared_ptr<ITokenizer> tokenizer;
};

DiffusionParts load_diffusion_parts(IBackend* backend, const BundleFile& bundle,
                                    const std::string& json,
                                    const ModuleCreateOptions& options = {},
                                    const std::string& denoiser_section_name = "denoiser_plan",
                                    const ModuleCreateOptions* denoiser_options = nullptr);

} // namespace trtmc
