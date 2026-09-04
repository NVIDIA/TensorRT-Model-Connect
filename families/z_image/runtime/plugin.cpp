/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/z_image/runtime/diffusion_helpers.h"
#include "families/z_image/runtime/pipeline.h"
#include "families/z_image/runtime/preprocessor_weights_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdlib>
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::z_image_factory {
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
        throw std::runtime_error("Z-Image tensor_parallel_size must be positive");
    require_nccl(size);
    if (size == 1)
        return {0, 1};
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("Z-Image TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= size)
        throw std::runtime_error("Z-Image RANK is outside tensor_parallel_size");
    return {static_cast<std::int32_t>(rank), size};
}

ZImagePreprocessorWeights parse_weights(const std::vector<char>& data) {
    ZImagePreprocessorWeights weights;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    const auto index =
        z_image_preprocessor_weights::extract_preprocessor_index(data, blob, blob_size);
#define LOAD_Z_IMAGE_WEIGHT(key, member)                                                           \
    z_image_preprocessor_weights::load_preprocessor_floats(index, blob, blob_size, key,            \
                                                           weights.member)
    LOAD_Z_IMAGE_WEIGHT("t_embedder.mlp.0.weight", t_embedder_mlp_0_weight);
    LOAD_Z_IMAGE_WEIGHT("t_embedder.mlp.0.bias", t_embedder_mlp_0_bias);
    LOAD_Z_IMAGE_WEIGHT("t_embedder.mlp.2.weight", t_embedder_mlp_2_weight);
    LOAD_Z_IMAGE_WEIGHT("t_embedder.mlp.2.bias", t_embedder_mlp_2_bias);
    LOAD_Z_IMAGE_WEIGHT("cap_embedder.proj.weight", cap_proj_weight);
    LOAD_Z_IMAGE_WEIGHT("cap_embedder.proj.bias", cap_proj_bias);
    LOAD_Z_IMAGE_WEIGHT("cap_embedder.norm.weight", cap_norm_weight);
    LOAD_Z_IMAGE_WEIGHT("cap_pad_token", cap_pad_token);
    LOAD_Z_IMAGE_WEIGHT("x_embedder.weight", x_embed_weight);
    LOAD_Z_IMAGE_WEIGHT("x_embedder.bias", x_embed_bias);
#undef LOAD_Z_IMAGE_WEIGHT
    if (weights.cap_proj_bias.empty() || weights.cap_norm_weight.empty() ||
        weights.t_embedder_mlp_0_bias.empty() || weights.x_embed_weight.empty() ||
        weights.t_embedder_mlp_0_weight.empty() || weights.cap_proj_weight.empty()) {
        throw std::runtime_error("Z-Image preprocessor.weights is incomplete");
    }
    weights.dit_dim = static_cast<std::int32_t>(weights.cap_proj_bias.size());
    weights.cap_dim = static_cast<std::int32_t>(weights.cap_norm_weight.size());
    weights.freq_dim = static_cast<std::int32_t>(weights.t_embedder_mlp_0_bias.size());
    weights.valid = true;
    return weights;
}

} // namespace
} // namespace trtmc::z_image_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto& data = z_image_factory::require_section(context.reader, "runtime.json");
    const std::string runtime(data.begin(), data.end());
    const auto document = nlohmann::json::parse(runtime);
    const auto [rank, size] = z_image_factory::rank_and_size(document);
    const std::string denoiser =
        size == 1 ? "denoiser.plan" : "denoiser.rank" + std::to_string(rank) + ".plan";
    ModuleCreateOptions options{};
    auto parts =
        load_diffusion_parts(&context.backend, context.reader, runtime, options, denoiser, nullptr);
    const auto& batch = document.at("max_batch_size");
    parts.config.max_batch_size.dit = batch.at("dit").get<std::int32_t>();
    parts.config.max_batch_size.text_encoder = batch.at("text_encoder").get<std::int32_t>();
    parts.config.max_batch_size.vae = batch.at("vae").get<std::int32_t>();
    if (!parts.tokenizer || parts.text_encoders.size() != 1)
        throw std::runtime_error("Z-Image bundle is missing required runtime assets");
    auto family_weights = z_image_factory::parse_weights(
        z_image_factory::require_section(context.reader, "preprocessor.weights"));
    return new ZImagePipeline(std::move(parts.text_encoders.front().module),
                              std::move(parts.denoiser.module), std::move(parts.vae.module),
                              std::move(parts.config), std::move(family_weights),
                              std::move(parts.tokenizer), "", nullptr, rank, size);
}
