/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/eagle_vlm/runtime/pipeline.h"
#include "families/eagle_vlm/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::eagle_vlm_factory {
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
        throw std::runtime_error("Eagle-VLM runtime.json must be an object");
    return config;
}

std::int32_t require_tensor_parallel_size(const nlohmann::json& config) {
    if (!config.contains("tensor_parallel_size") ||
        !config.at("tensor_parallel_size").is_number_integer()) {
        throw std::runtime_error("Eagle-VLM runtime.json requires tensor_parallel_size");
    }
    const auto value = config.at("tensor_parallel_size").get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("Eagle-VLM tensor_parallel_size is invalid");
    return static_cast<std::int32_t>(value);
}

void require_nccl(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size <= 1)
        return;
    static void* const handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("tensor-parallel runtime requires NCCL: " +
                                 std::string(error == nullptr ? "unknown loader error" : error));
    }
}

std::int32_t require_rank(std::int32_t tensor_parallel_size) {
    require_nccl(tensor_parallel_size);
    if (tensor_parallel_size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("Eagle-VLM TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= tensor_parallel_size)
        throw std::runtime_error("Eagle-VLM RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

std::string require_task(const BundleInfo& info) {
    if (info.task == IEncoding::kTask || info.task == IEmbedding::kTask ||
        info.task == IReranking::kTask) {
        return info.task;
    }
    throw std::runtime_error("Eagle-VLM does not implement task: " + info.task);
}

} // namespace
} // namespace trtmc::eagle_vlm_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    const auto config = trtmc::eagle_vlm_factory::require_config(context.reader);
    const auto tp_size = trtmc::eagle_vlm_factory::require_tensor_parallel_size(config);
    const auto rank = trtmc::eagle_vlm_factory::require_rank(tp_size);
    const std::string section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& plan = trtmc::eagle_vlm_factory::require_section(context.reader, section.c_str());

    trtmc::ModuleCreateOptions options{};
    auto loaded =
        trtmc::load_trt_module_from_plan(&context.backend, &plan, section.c_str(), options);
    auto tokenizer = trtmc::create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("Eagle-VLM bundle does not contain its required tokenizer");
    const auto task = trtmc::eagle_vlm_factory::require_task(context.reader.info());
    std::string pooling = "last";
    if (task == trtmc::IReranking::kTask) {
        if (!config.contains("pooling") || !config.at("pooling").is_string())
            throw std::runtime_error("Eagle-VLM reranking runtime.json requires pooling");
        pooling = config.at("pooling").get<std::string>();
    }
    return new trtmc::EncoderPipeline(std::move(loaded.module), task, std::move(tokenizer), "",
                                      std::move(pooling));
}
