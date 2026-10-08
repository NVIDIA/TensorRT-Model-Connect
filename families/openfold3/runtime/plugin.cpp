/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/openfold3/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::openfold3 {
namespace {

std::vector<char> requireSection(const BundleReader& reader, std::string_view name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::invalid_argument("OpenFold3 bundle is missing section: " + std::string(name));
    return reader.read_section(name);
}

class ModuleLoader {
  public:
    explicit ModuleLoader(IBackend& backend) : backend_(backend) {}

    std::unique_ptr<ITrtModule> load(const BundleReader& reader, std::string_view section) {
        const auto plan = requireSection(reader, section);
        ModuleCreateOptions options{};
        options.stream = stream_;
        auto module = backend_.create_module(plan.data(), plan.size(), options);
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
    IBackend& backend_;
    cudaStream_t stream_{nullptr};
};

EngineSet loadEngines(const FamilyContext& context) {
    ModuleLoader loader(context.backend);
    EngineSet result;
    result.input = loader.load(context.reader, "engine.plan");
    result.trunk_cycle = loader.load(context.reader, "openfold3_trunk_cycle_plan");
    for (std::size_t index = 0; index < result.pairformer.size(); ++index)
        result.pairformer[index] = loader.load(context.reader, kPairformerSections[index]);
    result.conditioning = loader.load(context.reader, "openfold3_diffusion_conditioning_plan");
    result.score_input = loader.load(context.reader, "openfold3_diffusion_score_input_plan");
    for (std::size_t index = 0; index < result.score_token.size(); ++index)
        result.score_token[index] = loader.load(context.reader, kTokenSections[index]);
    result.score_output = loader.load(context.reader, "openfold3_diffusion_score_output_plan");
    result.confidence = loader.load(context.reader, "openfold3_confidence_plan");
    return result;
}

BundleArtifacts loadArtifacts(const BundleReader& reader) {
    const auto features = requireSection(reader, "openfold3_features");
    const auto request = requireSection(reader, "openfold3_query.json");
    const auto metadata = requireSection(reader, "openfold3_structure.json");
    const auto random = requireSection(reader, "openfold3_random_samples");
    const auto manifest_payload = requireSection(reader, "openfold3_graph_manifest.json");
    std::string precision;
    try {
        const auto manifest =
            nlohmann::json::parse(manifest_payload.begin(), manifest_payload.end());
        if (!manifest.is_object())
            throw std::invalid_argument("OpenFold3 graph manifest must be a JSON object");
        const auto found = manifest.find("precision");
        if (found == manifest.end() || !found->is_string())
            throw std::invalid_argument("OpenFold3 graph manifest precision must be a string");
        precision = found->get<std::string>();
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("OpenFold3 graph manifest is invalid: " +
                                    std::string(error.what()));
    }
    if (precision != "fp16-mixed" && precision != "bf16-mixed")
        throw std::invalid_argument("OpenFold3 graph manifest has an unsupported precision");
    return {
        FeatureBundle::parse(features.data(), features.size()),
        std::string(request.begin(), request.end()),
        std::string(metadata.begin(), metadata.end()),
        RandomSamples::parse(random.data(), random.size()),
        std::move(precision),
    };
}

} // namespace

ITask* create(const FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("openfold3 does not support --kv-cache-size");
    auto artifacts = loadArtifacts(context.reader);
    auto engines = loadEngines(context);
    return new OpenFold3Pipeline(std::move(engines), std::move(artifacts));
}

} // namespace trtmc::openfold3

TRTMC_DEFINE_FAMILY_PLUGIN_V1("openfold3")

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::openfold3::create(context);
}
