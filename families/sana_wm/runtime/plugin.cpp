/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sana_wm/runtime/pipeline.h"
#include "families/sana_wm/runtime/plugin_helpers.h"
#include "families/sana_wm/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::sana_wm_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::shared_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle, const char* section) {
    const auto& data = require_section(bundle, section);
    auto tokenizer = CreateSanaWmBpeTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("SANA-WM tokenizer is invalid: " + std::string(section));
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

} // namespace
} // namespace trtmc::sana_wm_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("sana_wm does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime = sana_wm_factory::require_section(context.reader, "runtime.json");
    const std::string runtime_text(runtime.begin(), runtime.end());
    auto config = parse_sana_wm_config(runtime_text);

    ModuleCreateOptions options{};
    const auto load = [&](const char* name, cudaStream_t stream = nullptr) {
        auto module_options = options;
        module_options.stream = stream;
        const auto& plan = sana_wm_factory::require_section(context.reader, name);
        return load_trt_module_from_plan(&context.backend, &plan, name, module_options).module;
    };
    SanaWmNativeModules modules;
    modules.text_encoder = load("text_encoder.0.plan");
    const auto stream = modules.text_encoder->stream();
    modules.stage1_denoiser = load("denoiser.plan", stream);
    modules.vae_encoder = load("vae_encoder.plan", stream);
    modules.vae_decoder = load("vae.plan", stream);
    std::shared_ptr<ITokenizer> refiner_tokenizer;
    if (!config.no_refiner) {
        modules.refiner_text_encoder = load("refiner.text_encoder.plan", stream);
        modules.refiner_text_connector = load("refiner.text_connector.plan", stream);
        modules.refiner_denoiser = load("refiner.denoiser.plan", stream);
        modules.refiner_vae_decoder = load("refiner.vae.plan", stream);
        refiner_tokenizer =
            sana_wm_factory::load_tokenizer(context.reader, "refiner.tokenizer.json");
    }
    auto stage1_tokenizer =
        sana_wm_factory::load_tokenizer(context.reader, "stage1.tokenizer.json");
    return new SanaWmPipeline(std::move(config), std::move(modules), std::move(stage1_tokenizer),
                              std::move(refiner_tokenizer));
}
