/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx2/runtime/distributed_runtime.h"
#include "families/ltx2/runtime/pipeline.h"
#include "families/ltx2/runtime/runtime_config.h"
#include "families/ltx2/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::ltx2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("LTX-2.5 bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load(IBackend& backend, const BundleReader& bundle, const char* name,
                                 const ModuleCreateOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    auto plan = require_section(bundle, name);
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("failed to load LTX-2.5 engine ") + name);
    module->set_timing_label(name);
    std::cerr << "[trtmc.load_timing] label=\"" << name << "\" load_deserialize_ms="
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count()
              << " plan_bytes=" << plan.size() << '\n';
    return module;
}

// The tile plan and vae.plan come from one build; reject a bundle whose plan shapes disagree.
void require_tile_engine(const ITrtModule& vae, const LTX2Options& options) {
    const auto& plan = options.vae_tiling;
    const std::vector<int64_t> latents{
        1, int64_t(plan.tile_latent[0]) * plan.tile_latent[1] * plan.tile_latent[2],
        options.latent_channels};
    const std::vector<int64_t> frames{plan.tile_pixels[0], plan.tile_pixels[1], plan.tile_pixels[2],
                                      3};
    if (vae.tensor_shape("latents") != latents || vae.tensor_shape("frames") != frames ||
        vae.tensor_dtype("frames") != DType::kFloat16)
        throw std::runtime_error("LTX-2.5 vae.plan does not match the bundle's VAE tile plan");
}

} // namespace
} // namespace trtmc::ltx2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("ltx2 does not support --kv-cache-size");
    const auto runtime_data = ltx2::require_section(context.reader, "runtime.json");
    const std::string runtime(runtime_data.begin(), runtime_data.end());
    const auto parallel = ltx2::parse_parallel_runtime_config(runtime);
    // Binds this rank's CUDA device before any engine is deserialized.
    const auto group = ltx2::initialize_parallel_group(parallel.size);
    auto options = parse_ltx2_options(runtime, group.world_size);
    ModuleCreateOptions plain{};
    ModuleCreateOptions denoiser_options{};
    if (parallel.distributed()) {
        denoiser_options.distributed_communicator = group.communicator;
        denoiser_options.distributed_owner = group.owner;
    }
    auto text = ltx2::load(context.backend, context.reader, "text_encoder.plan", plain);
    auto denoiser = ltx2::load(context.backend, context.reader, "denoiser.plan", denoiser_options);
    // Rank 0 returns the media. With a tile plan every rank decodes video tiles and the audio
    // rank decodes the audio; otherwise worker ranks never load the decoders.
    const bool tiled = options.vae_tiling.enabled();
    std::unique_ptr<ITrtModule> vae;
    std::unique_ptr<ITrtModule> audio;
    if (group.rank == 0 || tiled)
        vae = ltx2::load(context.backend, context.reader, "vae.plan", plain);
    if (group.rank == options.audio_rank(group.world_size))
        audio = ltx2::load(context.backend, context.reader, "audio.plan", plain);
    if (tiled)
        ltx2::require_tile_engine(*vae, options);
    // Rank 0 sizes the received waveform from runtime.json; it must match audio.plan.
    if (audio && !options.audio_waveform_shape.empty() &&
        audio->tensor_shape("waveform") != options.audio_waveform_shape)
        throw std::runtime_error(
            "LTX-2.5 audio.plan does not match runtime.json audio_waveform_shape");
    const auto tokenizer_data = ltx2::require_section(context.reader, "tokenizer.json");
    std::shared_ptr<ITokenizer> tokenizer = CreateLtx2BpeTokenizer(
        tokenizer_data.data(), tokenizer_data.size(), /*add_special_tokens=*/false);
    if (!tokenizer)
        throw std::runtime_error("LTX-2.5 bundle tokenizer.json is not a supported BPE tokenizer");
    return new LTX2Pipeline(
        std::move(text), std::move(denoiser), std::move(vae), std::move(audio), std::move(options),
        std::move(tokenizer),
        LTX2DistributedContext{group.owner, group.rank, group.world_size, group.channel});
}
