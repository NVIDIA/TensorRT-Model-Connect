/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image2_1/runtime/pipeline.h"
#include "families/qwen_image2_1/runtime/runtime_config.h"
#include "families/qwen_image2_1/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::qwen_image2_1 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const BundleReader& bundle,
                                        const char* name) {
    // The plan bytes are released as soon as the engine is deserialized.
    const auto plan = require_section(bundle, name);
    const auto begin = std::chrono::steady_clock::now();
    auto module = backend.create_module(plan.data(), plan.size(), ModuleCreateOptions{});
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    std::cerr << "[trtmc.load_timing] label=\"" << name << "\" load_deserialize_ms=" << elapsed
              << " plan_bytes=" << plan.size() << '\n';
    if (!module || !module->ok())
        throw std::runtime_error(std::string("failed to load ") + name);
    module->set_timing_label(name);
    return module;
}

} // namespace
} // namespace trtmc::qwen_image2_1

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen_image2_1 does not support --kv-cache-size");
    using namespace trtmc::qwen_image2_1;

    const auto runtime = require_section(context.reader, "runtime.json");
    auto config = parse_runtime_config(std::string(runtime.begin(), runtime.end()));

    const auto tokenizer_data = require_section(context.reader, "tokenizer.json");
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_data.data(), tokenizer_data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("Qwen-Image 2.1 tokenizer.json is not a BPE tokenizer");

    Pipeline::Engines engines;
    engines.text_encoder = load_engine(context.backend, context.reader, "text_encoder.plan");
    engines.denoiser = load_engine(context.backend, context.reader, "denoiser.plan");
    engines.vae = load_engine(context.backend, context.reader, "vae.plan");
    return new Pipeline(std::move(engines), std::move(tokenizer), std::move(config));
}
