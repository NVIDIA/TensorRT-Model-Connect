/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image/runtime/pipeline.h"
#include "families/qwen_image/runtime/plugin_helpers.h"
#include "families/qwen_image/runtime/qwen_image_types.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::qwen_image_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load(IBackend& backend, const BundleReader& bundle, const char* name) {
    const auto& plan = require_section(bundle, name);
    ModuleCreateOptions options{};
    auto loaded = load_trt_module_from_plan(&backend, &plan, name, options);
    return std::move(loaded.module);
}

} // namespace
} // namespace trtmc::qwen_image_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen_image does not support --kv-cache-size");
    using namespace trtmc;
    const auto& data = qwen_image_factory::require_section(context.reader, "runtime.json");
    const std::string runtime(data.begin(), data.end());
    const auto document = nlohmann::json::parse(runtime);
    if (document.at("tensor_parallel_size").get<std::int32_t>() != 1)
        throw std::runtime_error("Qwen-Image runtime requires tensor_parallel_size=1");
    auto config = QwenImageConfig::parse(document);
    const auto& batch = document.at("max_batch_size");
    config.max_batch_size.dit = batch.at("dit").get<std::int32_t>();
    config.max_batch_size.text_encoder = batch.at("text_encoder").get<std::int32_t>();
    config.max_batch_size.vae = batch.at("vae").get<std::int32_t>();
    if (config.max_batch_size.dit <= 0 || config.max_batch_size.text_encoder <= 0 ||
        config.max_batch_size.vae <= 0)
        throw std::runtime_error("Qwen-Image max_batch_size values must be positive");

    QwenImagePipeline::Construction construction;
    construction.text_engine =
        qwen_image_factory::load(context.backend, context.reader, "text_encoder.0.plan");
    construction.denoiser_engine =
        qwen_image_factory::load(context.backend, context.reader, "denoiser.plan");
    construction.vae_decoder_engine =
        qwen_image_factory::load(context.backend, context.reader, "vae.plan");
    if (config.task_mode == QwenImageTaskMode::Edit) {
        construction.vision_engine =
            qwen_image_factory::load(context.backend, context.reader, "vision.plan");
        construction.vae_encoder_engine =
            qwen_image_factory::load(context.backend, context.reader, "vae_encoder.plan");
    }
    construction.tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!construction.tokenizer)
        throw std::runtime_error("Qwen-Image bundle does not contain its required tokenizer");
    construction.preprocessor = parse_qwen_image_preprocessor_weights(
        qwen_image_factory::require_section(context.reader, "preprocessor.weights"));
    if (!construction.preprocessor.valid)
        throw std::runtime_error("Qwen-Image preprocessor.weights is invalid");
    construction.config = std::move(config);
    return new QwenImagePipeline(std::move(construction));
}
