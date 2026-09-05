/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/pixart/runtime/diffusion_helpers.h"
#include "families/pixart/runtime/distributed_runtime.h"
#include "families/pixart/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::pixart_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

} // namespace
} // namespace trtmc::pixart_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("pixart does not support --kv-cache-size");
    using namespace trtmc;
    const auto& data = pixart_factory::require_section(context.reader, "runtime.json");
    const std::string runtime(data.begin(), data.end());
    const auto document = nlohmann::json::parse(runtime);
    const auto size = document.at("tensor_parallel_size").get<std::int32_t>();
    if (size <= 0)
        throw std::runtime_error("PixArt tensor_parallel_size must be positive");
    const auto group = pixart::initialize_tensor_parallel_group(size);
    const std::string denoiser =
        size == 1 ? "denoiser.plan" : "denoiser.rank" + std::to_string(group.rank) + ".plan";
    ModuleCreateOptions options{};
    ModuleCreateOptions denoiser_options{};
    const ModuleCreateOptions* selected_denoiser_options = nullptr;
    if (size > 1) {
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
        throw std::runtime_error("PixArt bundle is missing required runtime assets");
    return new PixArtPipeline(std::move(parts.text_encoders.front().module),
                              std::move(parts.denoiser.module), std::move(parts.vae.module),
                              std::move(parts.config), std::move(parts.weights),
                              std::move(parts.tokenizer), "", nullptr, group.rank, size);
}
