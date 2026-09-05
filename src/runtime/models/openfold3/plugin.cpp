/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/openfold3/pipeline.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace trtmc {
namespace {

const std::vector<char>& requireSection(const BundleFile& bundle, std::string_view name) {
    const auto* section = find_section(bundle, std::string(name));
    if (section == nullptr || section->empty())
        throw std::invalid_argument("OpenFold3 bundle is missing section: " + std::string(name));
    return *section;
}

class ModuleLoader {
  public:
    explicit ModuleLoader(const PipelineContext& context) : context_(context) {
        if (context.backend == nullptr ||
            std::string(context.backend->name() == nullptr ? "" : context.backend->name()) != "trt")
            throw std::invalid_argument("OpenFold3 requires the TensorRT backend");
    }

    std::unique_ptr<ITrtModule> load(std::string_view section) {
        const auto& plan = requireSection(context_.bundle, section);
        ModuleCreateOptions options;
        options.stream = stream_;
        options.cuda_graphs = false;
        auto module = context_.backend->create_module(plan.data(), plan.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("OpenFold3 failed to load section: " + std::string(section));
        if (stream_ == nullptr)
            stream_ = module->stream();
        if (module->stream() != stream_)
            throw std::runtime_error("OpenFold3 engines must share one CUDA stream");
        module->set_timing_label("openfold3 " + std::string(section));
        return module;
    }

  private:
    const PipelineContext& context_;
    cudaStream_t stream_{nullptr};
};

openfold3::EngineSet loadEngines(const PipelineContext& context) {
    ModuleLoader loader(context);
    openfold3::EngineSet result;
    result.input = loader.load("engine_plan");
    result.trunk_cycle = loader.load("openfold3_trunk_cycle_plan");
    for (std::size_t index = 0; index < result.pairformer.size(); ++index)
        result.pairformer[index] = loader.load(openfold3::kPairformerSections[index]);
    result.conditioning = loader.load("openfold3_diffusion_conditioning_plan");
    result.score_input = loader.load("openfold3_diffusion_score_input_plan");
    for (std::size_t index = 0; index < result.score_token.size(); ++index)
        result.score_token[index] = loader.load(openfold3::kTokenSections[index]);
    result.score_output = loader.load("openfold3_diffusion_score_output_plan");
    result.confidence = loader.load("openfold3_confidence_plan");
    return result;
}

openfold3::BundleArtifacts loadArtifacts(const BundleFile& bundle) {
    const auto& features = requireSection(bundle, "openfold3_features");
    const auto& request = requireSection(bundle, "openfold3_query.json");
    const auto& metadata = requireSection(bundle, "openfold3_structure.json");
    const auto& random = requireSection(bundle, "openfold3_random_samples");
    const auto& manifest_payload = requireSection(bundle, "openfold3_graph_manifest.json");
    nlohmann::json manifest;
    try {
        manifest = nlohmann::json::parse(manifest_payload.begin(), manifest_payload.end());
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("OpenFold3 graph manifest is invalid: " +
                                    std::string(error.what()));
    }
    const auto precision = manifest.value("precision", "");
    if (precision != "fp16-mixed" && precision != "bf16-mixed")
        throw std::invalid_argument("OpenFold3 graph manifest has an unsupported precision");
    return {
        openfold3::FeatureBundle::parse(features.data(), features.size()),
        std::string(request.begin(), request.end()),
        std::string(metadata.begin(), metadata.end()),
        openfold3::RandomSamples::parse(random.data(), random.size()),
        precision,
    };
}

class OpenFold3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        if (context.cuda_graphs)
            throw std::invalid_argument("OpenFold3 does not support CUDA graph execution");
        if (context.config.runtime_strategy != openfold3::kStrategy)
            throw std::invalid_argument("OpenFold3 received the wrong runtime strategy");
        return std::make_unique<openfold3::OpenFold3Pipeline>(
            loadEngines(context), loadArtifacts(context.bundle), context.bundle.info.model_id);
    }
};

} // namespace

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_openfold3_plugin, OpenFold3Plugin,
                                       "openfold3_structure_prediction");

} // namespace trtmc
