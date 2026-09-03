/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/boltz2/pipeline.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtmc {
namespace {

const std::vector<char>& requireSection(const BundleFile& bundle, std::string_view name) {
    const auto* section = find_section(bundle, std::string(name));
    if (section == nullptr || section->empty())
        throw std::invalid_argument("Boltz-2 bundle is missing nonempty section: " +
                                    std::string(name));
    return *section;
}

class ModuleLoader {
  public:
    explicit ModuleLoader(const PipelineContext& context) : context_(context) {
        if (context.backend == nullptr ||
            std::string(context.backend->name() == nullptr ? "" : context.backend->name()) !=
                "trt") {
            throw std::invalid_argument("Boltz-2 requires the standard TensorRT backend");
        }
    }

    std::unique_ptr<ITrtModule> load(std::string_view section) {
        const auto& plan = requireSection(context_.bundle, section);
        ModuleCreateOptions options;
        options.stream = stream_;
        options.cuda_graphs = false;
        auto module = context_.backend->create_module(plan.data(), plan.size(), options);
        if (module == nullptr || !module->ok())
            throw std::runtime_error("Boltz-2 failed to load TensorRT section: " +
                                     std::string(section));
        if (stream_ == nullptr)
            stream_ = module->stream();
        if (module->stream() == nullptr || module->stream() != stream_)
            throw std::runtime_error("Boltz-2 engines must share one CUDA stream");
        module->set_timing_label("boltz2 " + std::string(section));
        return module;
    }

  private:
    const PipelineContext& context_;
    cudaStream_t stream_{nullptr};
};

boltz2::EngineSet loadEngines(const PipelineContext& context) {
    ModuleLoader loader(context);
    boltz2::EngineSet result;
    result.input = loader.load("engine_plan");
    result.trunk_init = loader.load("boltz2_trunk_init_plan");
    result.msa = loader.load("boltz2_msa_plan");
    for (std::size_t index = 0; index < result.pairformer.size(); ++index)
        result.pairformer[index] = loader.load(boltz2::kPairformerSections[index]);
    result.conditioning = loader.load("boltz2_diffusion_conditioning_plan");
    result.score_input = loader.load("boltz2_diffusion_score_input_plan");
    for (std::size_t index = 0; index < result.score_token.size(); ++index)
        result.score_token[index] = loader.load(boltz2::kTokenSections[index]);
    result.score_output = loader.load("boltz2_diffusion_score_output_plan");
    result.confidence = loader.load("boltz2_confidence_plan");
    return result;
}

boltz2::BundleArtifacts loadArtifacts(const BundleFile& bundle) {
    const auto& feature_data = requireSection(bundle, "boltz2_features");
    const auto& request = requireSection(bundle, "boltz2_request.yaml");
    const auto& metadata = requireSection(bundle, "boltz2_structure_metadata.json");
    const auto& random_samples = requireSection(bundle, "boltz2_random_samples");
    (void)requireSection(bundle, "boltz2_msa.a3m");
    (void)requireSection(bundle, "boltz2_graph_manifest.json");
    return {
        boltz2::FeatureBundle::parse(feature_data.data(), feature_data.size()),
        std::string(request.begin(), request.end()),
        std::string(metadata.begin(), metadata.end()),
        boltz2::RandomSamples::parse(random_samples.data(), random_samples.size()),
    };
}

class Boltz2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        if (context.cuda_graphs)
            throw std::invalid_argument("Boltz-2 does not yet admit CUDA graph execution");
        if (context.config.runtime_strategy != boltz2::kStrategy)
            throw std::invalid_argument("Boltz-2 received the wrong runtime strategy");
        return std::make_unique<boltz2::Boltz2Pipeline>(
            loadEngines(context), loadArtifacts(context.bundle), context.bundle.info.model_id,
            context.hf_python);
    }
};

} // namespace

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_boltz2_plugin, Boltz2Plugin,
                                       "boltz2_structure_prediction");

} // namespace trtmc
