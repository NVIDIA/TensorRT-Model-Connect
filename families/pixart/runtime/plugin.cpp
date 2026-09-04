/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/pixart/runtime/diffusion_helpers.h"
#include "families/pixart/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdlib>
#include <dlfcn.h>
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

void require_nccl(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size <= 1)
        return;
    static void* const handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("tensor-parallel runtime requires NCCL: " +
                                 std::string(error == nullptr ? "unknown loader error" : error));
    }
}

std::pair<std::int32_t, std::int32_t> rank_and_size(const nlohmann::json& json) {
    const auto size = json.at("tensor_parallel_size").get<std::int32_t>();
    if (size <= 0)
        throw std::runtime_error("PixArt tensor_parallel_size must be positive");
    require_nccl(size);
    if (size == 1)
        return {0, 1};
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("PixArt TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= size)
        throw std::runtime_error("PixArt RANK is outside tensor_parallel_size");
    return {static_cast<std::int32_t>(rank), size};
}

} // namespace
} // namespace trtmc::pixart_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto& data = pixart_factory::require_section(context.reader, "runtime.json");
    const std::string runtime(data.begin(), data.end());
    const auto document = nlohmann::json::parse(runtime);
    const auto [rank, size] = pixart_factory::rank_and_size(document);
    const std::string denoiser =
        size == 1 ? "denoiser.plan" : "denoiser.rank" + std::to_string(rank) + ".plan";
    ModuleCreateOptions options{};
    auto parts =
        load_diffusion_parts(&context.backend, context.reader, runtime, options, denoiser, nullptr);
    const auto& batch = document.at("max_batch_size");
    parts.config.max_batch_size.dit = batch.at("dit").get<std::int32_t>();
    parts.config.max_batch_size.text_encoder = batch.at("text_encoder").get<std::int32_t>();
    parts.config.max_batch_size.vae = batch.at("vae").get<std::int32_t>();
    if (!parts.weights.valid || !parts.tokenizer || parts.text_encoders.size() != 1)
        throw std::runtime_error("PixArt bundle is missing required runtime assets");
    return new PixArtPipeline(std::move(parts.text_encoders.front().module),
                              std::move(parts.denoiser.module), std::move(parts.vae.module),
                              std::move(parts.config), std::move(parts.weights),
                              std::move(parts.tokenizer), "", nullptr, rank, size);
}
