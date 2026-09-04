/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan_t2v/runtime/diffusion_helpers.h"
#include "families/wan_t2v/runtime/distributed_runtime.h"
#include "families/wan_t2v/runtime/pipeline.h"
#include "families/wan_t2v/runtime/runtime_config.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::wan_t2v {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

} // namespace
} // namespace trtmc::wan_t2v

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto& data = wan_t2v::require_section(context.reader, "runtime.json");
    const std::string runtime(data.begin(), data.end());
    const auto document = nlohmann::json::parse(runtime);
    const auto parallel = wan_t2v::parse_parallel_runtime_config(runtime);
    const auto group = wan_t2v::initialize_parallel_group(parallel.size);
    const std::string denoiser = wan_t2v::denoiser_section_name(parallel, group.rank);
    ModuleCreateOptions options{};
    ModuleCreateOptions denoiser_options{};
    const ModuleCreateOptions* selected_denoiser_options = nullptr;
    if (parallel.distributed()) {
        denoiser_options.distributed_communicator = group.communicator;
        denoiser_options.distributed_owner = group.owner;
        selected_denoiser_options = &denoiser_options;
    }
    auto parts = load_diffusion_parts(&context.backend, context.reader, runtime, options, denoiser,
                                      selected_denoiser_options);
    const auto& batch = document.at("max_batch_size");
    parts.config.max_batch_size.dit = batch.at("dit").get<std::int32_t>();
    parts.config.max_batch_size.text_encoder = batch.at("text_encoder").get<std::int32_t>();
    parts.config.max_batch_size.vae = batch.at("vae").get<std::int32_t>();
    if (!parts.weights.valid || !parts.tokenizer || parts.text_encoders.size() != 1)
        throw std::runtime_error("Wan bundle is missing required runtime assets");
    return new WanPipeline(std::move(parts.text_encoders.front().module),
                           std::move(parts.denoiser.module), std::move(parts.vae.module),
                           std::move(parts.config), std::move(parts.weights),
                           std::move(parts.tokenizer), "", group.owner, group.rank,
                           group.world_size, std::move(parts.vae_first_frame.module));
}
