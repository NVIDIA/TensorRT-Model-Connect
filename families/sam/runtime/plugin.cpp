/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam/runtime/distributed_runtime.h"
#include "families/sam/runtime/plugin_helpers.h"
#include "families/sam/runtime/sam_pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::sam_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

SamConfig parse_config(const std::vector<char>& data, std::int32_t& tp_size) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    SamConfig config;
    config.image_size = json.at("image_size").get<std::int32_t>();
    config.image_embedding_size = json.at("image_embedding_size").get<std::int32_t>();
    config.decoder_hidden_size = json.at("decoder_hidden_size").get<std::int32_t>();
    config.num_mask_outputs = json.at("num_mask_outputs").get<std::int32_t>();
    config.num_multimask_outputs = json.at("num_multimask_outputs").get<std::int32_t>();
    config.image_mean = json.at("image_mean").get<std::vector<float>>();
    config.image_std = json.at("image_std").get<std::vector<float>>();
    config.point_embed_bg = json.at("point_embed_bg").get<std::vector<float>>();
    config.point_embed_fg = json.at("point_embed_fg").get<std::vector<float>>();
    config.not_a_point_embed = json.at("not_a_point_embed").get<std::vector<float>>();
    config.shared_image_pe = json.at("shared_image_pe").get<std::vector<float>>();
    tp_size = json.at("tensor_parallel_size").get<std::int32_t>();
    const auto hidden = static_cast<std::size_t>(config.decoder_hidden_size);
    if (config.image_size <= 0 || config.image_embedding_size <= 0 || hidden == 0 ||
        config.num_mask_outputs <= 0 || config.num_multimask_outputs <= 0 ||
        config.image_mean.size() != 3 || config.image_std.size() != 3 ||
        config.point_embed_bg.size() != hidden || config.point_embed_fg.size() != hidden ||
        config.not_a_point_embed.size() != hidden || config.shared_image_pe.empty() ||
        tp_size <= 0) {
        throw std::runtime_error("SAM runtime.json does not match its runtime contract");
    }
    return config;
}

} // namespace
} // namespace trtmc::sam_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("sam does not support --kv-cache-size");
    const auto& config_data = trtmc::sam_factory::require_section(context.reader, "runtime.json");
    std::int32_t tp_size = 0;
    auto config = trtmc::sam_factory::parse_config(config_data, tp_size);
    const auto group = trtmc::sam::initialize_tensor_parallel_group(tp_size);
    const std::string encoder_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(group.rank) + ".plan";
    const auto& encoder_plan =
        trtmc::sam_factory::require_section(context.reader, encoder_section.c_str());
    const auto& decoder_plan = trtmc::sam_factory::require_section(context.reader, "decoder.plan");
    trtmc::ModuleCreateOptions encoder_options{};
    if (tp_size > 1) {
        encoder_options.distributed_communicator = group.communicator;
        encoder_options.distributed_owner = group.owner;
    }
    auto encoder = trtmc::load_trt_module_from_plan(&context.backend, &encoder_plan,
                                                    encoder_section.c_str(), encoder_options);
    trtmc::ModuleCreateOptions decoder_options{};
    decoder_options.stream = encoder.module->stream();
    auto decoder = trtmc::load_trt_module_from_plan(&context.backend, &decoder_plan, "decoder.plan",
                                                    decoder_options);
    return new trtmc::SamPipeline(std::move(encoder.module), std::move(decoder.module),
                                  std::move(config), "");
}
