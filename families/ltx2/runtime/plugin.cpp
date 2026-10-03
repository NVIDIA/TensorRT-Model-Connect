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
    auto options = parse_ltx2_options(runtime);
    ModuleCreateOptions plain{};
    ModuleCreateOptions denoiser_options{};
    if (parallel.distributed()) {
        denoiser_options.distributed_communicator = group.communicator;
        denoiser_options.distributed_owner = group.owner;
    }
    auto text = ltx2::load(context.backend, context.reader, "text_encoder.plan", plain);
    auto denoiser = ltx2::load(context.backend, context.reader, "denoiser.plan", denoiser_options);
    // Only rank 0 decodes and returns media; worker ranks never load the decoders.
    std::unique_ptr<ITrtModule> vae;
    std::unique_ptr<ITrtModule> audio;
    if (group.rank == 0) {
        vae = ltx2::load(context.backend, context.reader, "vae.plan", plain);
        audio = ltx2::load(context.backend, context.reader, "audio.plan", plain);
    }
    const auto tokenizer_data = ltx2::require_section(context.reader, "tokenizer.json");
    std::shared_ptr<ITokenizer> tokenizer = CreateLtx2BpeTokenizer(
        tokenizer_data.data(), tokenizer_data.size(), /*add_special_tokens=*/false);
    if (!tokenizer)
        throw std::runtime_error("LTX-2.5 bundle tokenizer.json is not a supported BPE tokenizer");
    return new LTX2Pipeline(std::move(text), std::move(denoiser), std::move(vae), std::move(audio),
                            std::move(options), std::move(tokenizer),
                            LTX2DistributedContext{group.owner, group.rank, group.world_size});
}
