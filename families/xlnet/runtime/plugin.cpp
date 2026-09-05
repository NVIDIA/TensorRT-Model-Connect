/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/xlnet/runtime/distributed_runtime.h"
#include "families/xlnet/runtime/pipeline.h"
#include "families/xlnet/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::xlnet_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

nlohmann::json require_config(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "runtime.json");
    const auto config = nlohmann::json::parse(data.begin(), data.end());
    if (!config.is_object())
        throw std::runtime_error("XLNet runtime.json must be an object");
    return config;
}

std::int32_t require_tensor_parallel_size(const nlohmann::json& config) {
    if (!config.contains("tensor_parallel_size") ||
        !config.at("tensor_parallel_size").is_number_integer()) {
        throw std::runtime_error("XLNet runtime.json requires tensor_parallel_size");
    }
    const auto value = config.at("tensor_parallel_size").get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("XLNet tensor_parallel_size is invalid");
    return static_cast<std::int32_t>(value);
}

std::string require_task(const BundleInfo& info) {
    if (info.task == IEncoding::kTask || info.task == IEmbedding::kTask ||
        info.task == IReranking::kTask) {
        return info.task;
    }
    throw std::runtime_error("XLNet does not implement task: " + info.task);
}

} // namespace
} // namespace trtmc::xlnet_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("xlnet does not support --kv-cache-size");
    const auto config = trtmc::xlnet_factory::require_config(context.reader);
    const auto tp_size = trtmc::xlnet_factory::require_tensor_parallel_size(config);
    const auto group = trtmc::xlnet::initialize_tensor_parallel_group(tp_size);
    const std::string section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(group.rank) + ".plan";
    const auto& plan = trtmc::xlnet_factory::require_section(context.reader, section.c_str());

    trtmc::ModuleCreateOptions options{};
    if (tp_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    auto loaded =
        trtmc::load_trt_module_from_plan(&context.backend, &plan, section.c_str(), options);
    auto tokenizer = trtmc::create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("XLNet bundle does not contain its required tokenizer");
    const auto task = trtmc::xlnet_factory::require_task(context.reader.info());
    return new trtmc::EncoderPipeline(std::move(loaded.module), task, std::move(tokenizer));
}
