/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/segformer/runtime/distributed_runtime.h"
#include "families/segformer/runtime/plugin_helpers.h"
#include "families/segformer/runtime/segment_pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::segformer {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

SegformerPreprocessConfig parse_config(const std::vector<char>& data, std::int32_t& tp_size) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    SegformerPreprocessConfig config;
    config.num_classes = json.at("num_classes").get<std::int32_t>();
    config.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.output_h = json.at("output_h").get<std::int32_t>();
    config.output_w = json.at("output_w").get<std::int32_t>();
    config.image_mean = json.at("image_mean").get<std::vector<float>>();
    config.image_std = json.at("image_std").get<std::vector<float>>();
    tp_size = json.at("tensor_parallel_size").get<std::int32_t>();
    if (config.num_classes <= 0 || config.input_image_h <= 0 || config.input_image_w <= 0 ||
        config.output_h <= 0 || config.output_w <= 0 || config.image_mean.size() != 3 ||
        config.image_std.size() != 3 || tp_size <= 0) {
        throw std::runtime_error("SegFormer runtime.json does not match its runtime contract");
    }
    return config;
}

} // namespace
} // namespace trtmc::segformer

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("segformer does not support --kv-cache-size");
    const auto& config_data = trtmc::segformer::require_section(context.reader, "runtime.json");
    std::int32_t tp_size = 0;
    auto config = trtmc::segformer::parse_config(config_data, tp_size);
    const auto group = trtmc::segformer::initialize_tensor_parallel_group(tp_size);
    const std::string section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(group.rank) + ".plan";
    const auto& plan = trtmc::segformer::require_section(context.reader, section.c_str());
    trtmc::ModuleCreateOptions options{};
    if (tp_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    auto loaded =
        trtmc::load_trt_module_from_plan(&context.backend, &plan, section.c_str(), options);
    return new trtmc::SegmentPipeline(std::move(loaded.module), std::move(config));
}
