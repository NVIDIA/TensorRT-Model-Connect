/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/boltz2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtmc {
namespace {

std::vector<char> requireSection(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::invalid_argument("Boltz-2 bundle is missing nonempty section: " +
                                    std::string(name));
    return bundle.read_section(name);
}

class ModuleLoader {
  public:
    explicit ModuleLoader(const FamilyContext& context) : context_(context) {
        if (std::string(context.backend.name() == nullptr ? "" : context.backend.name()) != "trt") {
            throw std::invalid_argument("Boltz-2 requires the standard TensorRT backend");
        }
    }

    std::unique_ptr<ITrtModule> load(std::string_view section) {
        const auto plan = requireSection(context_.reader, section);
        ModuleCreateOptions options;
        options.stream = stream_;
        auto module = context_.backend.create_module(plan.data(), plan.size(), options);
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
    const FamilyContext& context_;
    cudaStream_t stream_{nullptr};
};

boltz2::EngineSet loadEngines(const FamilyContext& context) {
    ModuleLoader loader(context);
    boltz2::EngineSet result;
    result.input = loader.load("engine.plan");
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

boltz2::BundleArtifacts loadArtifacts(const BundleReader& bundle) {
    const auto feature_data = requireSection(bundle, "boltz2_features");
    const auto request = requireSection(bundle, "boltz2_request.yaml");
    const auto metadata = requireSection(bundle, "boltz2_structure_metadata.json");
    const auto random_samples = requireSection(bundle, "boltz2_random_samples");
    (void)requireSection(bundle, "runtime.json");
    (void)requireSection(bundle, "boltz2_msa.a3m");
    (void)requireSection(bundle, "boltz2_graph_manifest.json");
    return {
        boltz2::FeatureBundle::parse(feature_data.data(), feature_data.size()),
        std::string(request.begin(), request.end()),
        std::string(metadata.begin(), metadata.end()),
        boltz2::RandomSamples::parse(random_samples.data(), random_samples.size()),
    };
}

} // namespace
} // namespace trtmc

TRTMC_DEFINE_FAMILY_PLUGIN_V1("boltz2")

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("boltz2 does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::IStructurePrediction::kTask)
        throw std::invalid_argument("Boltz-2 requires task=structure_prediction");
    return new trtmc::boltz2::Boltz2Pipeline(trtmc::loadEngines(context),
                                             trtmc::loadArtifacts(context.reader));
}
