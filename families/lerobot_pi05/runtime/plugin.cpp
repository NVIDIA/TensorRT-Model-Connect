/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_pi05/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::lerobot_pi05 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

Normalization parse_normalization(const nlohmann::json& json) {
    Normalization norm;
    const auto mode = json.at("mode").get<std::string>();
    if (mode == "IDENTITY")
        return norm;
    if (mode == "MEAN_STD")
        norm.mode = NormalizationMode::kMeanStd;
    else if (mode == "QUANTILES" || mode == "QUANTILE10" || mode == "MIN_MAX")
        norm.mode = NormalizationMode::kRange;
    else
        throw std::runtime_error("lerobot_pi05 runtime.json has an unsupported normalization mode");
    norm.eps = json.at("eps").get<float>();
    norm.offset = json.at("offset").get<std::vector<float>>();
    norm.scale = json.at("scale").get<std::vector<float>>();
    return norm;
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    if (json.at("precision").get<std::string>() != "bf16" &&
        json.at("precision").get<std::string>() != "fp32")
        throw std::runtime_error("lerobot_pi05 runtime.json has an unsupported precision");
    RuntimeConfig config;
    config.cameras = json.at("cameras").get<std::vector<std::string>>();
    config.empty_cameras = json.at("empty_cameras").get<std::int32_t>();
    config.camera_height = json.at("camera_height").get<std::int32_t>();
    config.camera_width = json.at("camera_width").get<std::int32_t>();
    config.tokens_per_image = json.at("tokens_per_image").get<std::int32_t>();
    config.prompt_tokens = json.at("prompt_tokens").get<std::int32_t>();
    config.prefix_tokens = json.at("prefix_tokens").get<std::int32_t>();
    config.chunk_size = json.at("chunk_size").get<std::int32_t>();
    config.state_dim = json.at("state_dim").get<std::int32_t>();
    config.action_dim = json.at("action_dim").get<std::int32_t>();
    config.max_action_dim = json.at("max_action_dim").get<std::int32_t>();
    config.num_inference_steps = json.at("num_inference_steps").get<std::int32_t>();
    if (json.at("state_bins").get<std::int32_t>() != 256)
        throw std::runtime_error(
            "lerobot_pi05 runtime.json has an unsupported state discretization");
    config.state_normalization = parse_normalization(json.at("state_normalization"));
    config.action_normalization = parse_normalization(json.at("action_normalization"));
    const auto& range = json.at("action_range");
    if (!range.is_null()) {
        config.action_min = range.at("min").get<std::vector<float>>();
        config.action_max = range.at("max").get<std::vector<float>>();
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error(std::string("lerobot_pi05 ") + what + " engine failed to load");
    engine->set_timing_label(std::string("lerobot_pi05 ") + what);
    return engine;
}

} // namespace
} // namespace trtmc::lerobot_pi05

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lerobot_pi05 does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::IImagesStateTextToActionChunk::kTask)
        throw std::runtime_error("lerobot_pi05 bundle declares an unsupported task: " + task);

    auto config = trtmc::lerobot_pi05::parse_config(
        trtmc::lerobot_pi05::require_section(context.reader, "runtime.json"));
    const auto asset = trtmc::lerobot_pi05::require_section(context.reader, "tokenizer.bpe");
    trtmc::lerobot_pi05::PaligemmaBpeTokenizer tokenizer(
        trtmc::lerobot_pi05::parse_paligemma_bpe_asset(
            std::string_view(asset.data(), asset.size())));
    auto prefix = trtmc::lerobot_pi05::load_engine(
        context.backend, trtmc::lerobot_pi05::require_section(context.reader, "prefix.plan"),
        "prefix");
    auto denoise = trtmc::lerobot_pi05::load_engine(
        context.backend, trtmc::lerobot_pi05::require_section(context.reader, "denoise.plan"),
        "denoise");
    return new trtmc::lerobot_pi05::Pipeline(std::move(prefix), std::move(denoise),
                                             std::move(tokenizer), std::move(config), task);
}
