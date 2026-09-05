/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx_video/runtime/diffusion_helpers.h"
#include "families/ltx_video/runtime/pipeline.h"
#include "families/ltx_video/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::ltx_video_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string section_text(const BundleReader& bundle, const char* name) {
    const auto& data = require_section(bundle, name);
    return std::string(data.begin(), data.end());
}

} // namespace
} // namespace trtmc::ltx_video_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("ltx_video does not support --kv-cache-size");
    using namespace trtmc;
    const auto runtime = ltx_video_factory::section_text(context.reader, "runtime.json");
    ModuleCreateOptions options{};
    const auto& text_plan =
        ltx_video_factory::require_section(context.reader, "text_encoder.0.plan");
    const auto& denoiser_plan = ltx_video_factory::require_section(context.reader, "denoiser.plan");
    const auto& vae_plan = ltx_video_factory::require_section(context.reader, "vae.plan");
    auto text =
        load_trt_module_from_plan(&context.backend, &text_plan, "text_encoder.0.plan", options);
    auto denoiser =
        load_trt_module_from_plan(&context.backend, &denoiser_plan, "denoiser.plan", options);
    auto vae = load_trt_module_from_plan(&context.backend, &vae_plan, "vae.plan", options);
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("LTX-Video bundle does not contain its required tokenizer");
    auto config = make_diffusion_config(runtime);
    const auto batch = nlohmann::json::parse(runtime).at("max_batch_size");
    config.max_batch_size.dit = batch.at("dit").get<std::int32_t>();
    config.max_batch_size.text_encoder = batch.at("text_encoder").get<std::int32_t>();
    config.max_batch_size.vae = batch.at("vae").get<std::int32_t>();
    return new LTXVideoPipeline(std::move(text.module), std::move(denoiser.module),
                                std::move(vae.module), std::move(config),
                                parse_ltx_video_options(runtime), std::move(tokenizer), "");
}
