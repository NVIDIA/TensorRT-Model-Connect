/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/flux3/runtime/engine_contract.h"
#include "families/flux3/runtime/pipeline.h"
#include "families/flux3/runtime/runtime_config.h"
#include "families/flux3/runtime/tokenizer.h"
#include "trtmc/internal/image.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::flux3 {
namespace {

using EngineCheck = void (*)(const ITrtModule&, const RuntimeConfig&);

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error(std::string("flux3 bundle section is missing or empty: ") + name);
    return bundle.read_section(name);
}

void require_text_to_image(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::ITextToImage::kTask)
        throw std::runtime_error(
            "flux3 runtime implements only text_to_image; the bundle declares '" + task + "'");
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle,
                                           const RuntimeConfig& config) {
    const auto data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size());
    if (!tokenizer)
        throw std::runtime_error("flux3 tokenizer.json is not a byte-level BPE tokenizer");
    if (tokenizer->token_for_id(config.text.pad_token_id).empty())
        throw std::runtime_error("flux3 runtime.json: text.pad_token_id " +
                                 std::to_string(config.text.pad_token_id) +
                                 " is not a token of tokenizer.json");
    return tokenizer;
}

// Holds one serialized plan at a time and validates each engine as soon as it
// is created, so a mismatched bundle fails before the next engine loads.
std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const BundleReader& bundle,
                                        const char* name, const RuntimeConfig& config,
                                        EngineCheck check) {
    std::unique_ptr<ITrtModule> engine;
    {
        const auto plan = require_section(bundle, name);
        engine = backend.create_module(plan.data(), plan.size(), ModuleCreateOptions{});
    }
    if (!engine || !engine->ok())
        throw std::runtime_error(std::string("flux3 could not create a TensorRT engine from ") +
                                 name);
    engine->set_timing_label(name);
    check(*engine, config);
    return engine;
}

} // namespace
} // namespace trtmc::flux3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    namespace flux3 = trtmc::flux3;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("flux3 does not support --kv-cache-size");
    flux3::require_text_to_image(context.reader);
    const auto runtime = flux3::require_section(context.reader, "runtime.json");
    auto config = flux3::parse_runtime_config({runtime.data(), runtime.size()});
    auto tokenizer = flux3::load_tokenizer(context.reader, config);
    auto& backend = context.backend;
    flux3::Engines engines;
    engines.text_encoder = flux3::load_engine(backend, context.reader, "text_encoder.plan", config,
                                              flux3::validate_text_encoder);
    engines.denoiser = flux3::load_engine(backend, context.reader, "denoiser.plan", config,
                                          flux3::validate_denoiser);
    engines.vae =
        flux3::load_engine(backend, context.reader, "vae.plan", config, flux3::validate_vae);
    return new flux3::Flux3Pipeline(std::move(config), std::move(tokenizer), std::move(engines));
}
