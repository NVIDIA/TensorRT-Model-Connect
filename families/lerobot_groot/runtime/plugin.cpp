/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_groot/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::lerobot_groot {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    const auto precision = json.at("precision").get<std::string>();
    if (precision != "bf16" && precision != "fp32")
        throw std::runtime_error("lerobot_groot runtime.json has an unsupported precision");
    RuntimeConfig config;
    config.cameras = json.at("cameras").get<std::vector<std::string>>();
    config.camera_size = json.at("camera_size").get<std::int32_t>();
    config.crop = json.at("crop").get<std::int32_t>();
    config.chunk_size = json.at("chunk_size").get<std::int32_t>();
    config.action_horizon = json.at("action_horizon").get<std::int32_t>();
    config.state_dim = json.at("state_dim").get<std::int32_t>();
    config.action_dim = json.at("action_dim").get<std::int32_t>();
    config.max_state_dim = json.at("max_state_dim").get<std::int32_t>();
    config.max_action_dim = json.at("max_action_dim").get<std::int32_t>();
    config.num_inference_steps = json.at("num_inference_steps").get<std::int32_t>();
    config.max_instruction_tokens = json.at("max_instruction_tokens").get<std::int32_t>();
    config.formalize_language = json.at("formalize_language").get<bool>();
    config.state_min = json.at("state_min").get<std::vector<float>>();
    config.state_max = json.at("state_max").get<std::vector<float>>();
    config.clip_state = json.at("clip_state").get<bool>();
    config.action_min = json.at("action_min").get<std::vector<float>>();
    config.action_max = json.at("action_max").get<std::vector<float>>();
    config.clip_action = json.at("clip_action").get<bool>();
    config.libero_gripper = json.at("libero_gripper").get<bool>();
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error(std::string("lerobot_groot ") + what + " engine failed to load");
    engine->set_timing_label(std::string("lerobot_groot ") + what);
    return engine;
}

} // namespace
} // namespace trtmc::lerobot_groot

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lerobot_groot does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::IImagesStateTextToActionChunk::kTask)
        throw std::runtime_error("lerobot_groot bundle declares an unsupported task: " + task);

    auto config = trtmc::lerobot_groot::parse_config(
        trtmc::lerobot_groot::require_section(context.reader, "runtime.json"));
    const auto vocabulary = trtmc::lerobot_groot::require_section(context.reader, "tokenizer.json");
    auto tokenizer = trtmc::lerobot_groot::CreateBpeTokenizer(vocabulary.data(), vocabulary.size());
    auto condition = trtmc::lerobot_groot::load_engine(
        context.backend, trtmc::lerobot_groot::require_section(context.reader, "condition.plan"),
        "condition");
    auto denoise = trtmc::lerobot_groot::load_engine(
        context.backend, trtmc::lerobot_groot::require_section(context.reader, "denoise.plan"),
        "denoise");
    return new trtmc::lerobot_groot::Pipeline(std::move(condition), std::move(denoise),
                                              std::move(tokenizer), std::move(config), task);
}
