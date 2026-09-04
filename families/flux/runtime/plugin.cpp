/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/flux/runtime/diffusion_helpers.h"
#include "families/flux/runtime/distributed_runtime.h"
#include "families/flux/runtime/pipeline.h"
#include "families/flux/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::flux_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

struct Parallel {
    std::int32_t size;
    bool context_parallel;

    bool distributed() const { return size > 1; }
};

Parallel parse_parallel(const nlohmann::json& json) {
    const auto mode = json.at("parallel_mode").get<std::string>();
    const auto size = json.at("parallel_size").get<std::int32_t>();
    if (mode == "single") {
        if (size != 1)
            throw std::runtime_error("FLUX single runtime requires parallel_size=1");
        return {1, false};
    }
    if (mode != "tensor_parallel" && mode != "context_parallel")
        throw std::runtime_error("FLUX runtime.json has an unsupported parallel_mode");
    if (size != 2 && size != 4 && size != 8)
        throw std::runtime_error("FLUX runtime.json has invalid parallel settings");
    return {size, mode == "context_parallel"};
}

} // namespace
} // namespace trtmc::flux_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto& data = flux_factory::require_section(context.reader, "runtime.json");
    const std::string runtime(data.begin(), data.end());
    const auto document = nlohmann::json::parse(runtime);
    const auto parallel = flux_factory::parse_parallel(document);
    flux_runtime::DistributedGroup group;
    if (parallel.distributed())
        group = flux_runtime::initialize_group(parallel.size);
    const std::int32_t rank = parallel.distributed() ? group.rank : 0;
    const std::string denoiser = parallel.context_parallel ? "denoiser.cp.plan"
                                 : parallel.distributed()
                                     ? "denoiser.rank" + std::to_string(rank) + ".plan"
                                     : "denoiser.plan";
    ModuleCreateOptions options{};
    ModuleCreateOptions denoiser_options = options;
    const ModuleCreateOptions* distributed_options = nullptr;
    if (parallel.distributed()) {
        denoiser_options.distributed_communicator = group.communicator;
        denoiser_options.distributed_owner = group.owner;
        distributed_options = &denoiser_options;
    }
    auto parts = load_diffusion_parts(&context.backend, context.reader, runtime, options, denoiser,
                                      distributed_options);
    const auto& batch = document.at("max_batch_size");
    parts.config.max_batch_size.dit = batch.at("dit").get<std::int32_t>();
    parts.config.max_batch_size.text_encoder = batch.at("text_encoder").get<std::int32_t>();
    parts.config.max_batch_size.vae = batch.at("vae").get<std::int32_t>();
    if (!parts.weights.valid || !parts.tokenizer || parts.text_encoders.empty())
        throw std::runtime_error("FLUX bundle is missing required runtime assets");
    std::vector<std::unique_ptr<ITrtModule>> text_encoders;
    for (auto& encoder : parts.text_encoders)
        text_encoders.push_back(std::move(encoder.module));
    std::unique_ptr<ITokenizer> clip;
    if (parts.weights.vae_bn_mean.empty()) {
        clip = create_clip_tokenizer_from_bundle(context.reader);
        if (!clip)
            throw std::runtime_error("FLUX.1 bundle is missing its CLIP tokenizer");
    }
    return new FluxPipeline(std::move(text_encoders), std::move(parts.denoiser.module),
                            std::move(parts.vae.module), std::move(parts.config),
                            std::move(parts.weights), std::move(parts.tokenizer), std::move(clip),
                            "", std::move(group.owner), rank, parallel.size);
}
