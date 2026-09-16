/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/cli.h"

#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {
template <class Interface>
Interface& require_interface(trtmc::ITask& task) {
    auto* result = dynamic_cast<Interface*>(&task);
    if (result == nullptr)
        throw std::invalid_argument("BERT bundle does not implement the requested interface");
    return *result;
}

nlohmann::json execute(const std::string& handler, const nlohmann::json& values,
                       const char* default_runtime_root) {
    if (handler != "encode" && handler != "embed" && handler != "rerank")
        throw std::invalid_argument("unknown BERT CLI handler: " + handler);
    // Check identity before loading: a BERT command must not execute another
    // family's bundle just because it implements the same public interface.
    const trtmc::BundleReader reader(values.at("bundle").get<std::string>());
    if (reader.info().family != "bert")
        throw std::invalid_argument("BERT CLI requires a bert bundle");
    auto task =
        trtmc::load_task(reader, values.value("runtime_root", std::string(default_runtime_root)));
    if (handler == "rerank") {
        const auto score = require_interface<trtmc::IReranking>(*task).rerank(
            values.at("query").get<std::string>(), values.at("document").get<std::string>());
        if (!std::isfinite(score))
            throw std::runtime_error("BERT reranking returned a non-finite score");
        return {{"score", score}};
    }
    const auto text = values.at("text").get<std::string>();
    const auto result = handler == "encode"
                            ? require_interface<trtmc::IEncoding>(*task).encode(text)
                            : require_interface<trtmc::IEmbedding>(*task).embed(text);
    for (const auto value : result.data) {
        if (!std::isfinite(value))
            throw std::runtime_error("BERT embedding returned a non-finite value");
    }
    return {{"dim", result.dim}, {"values", result.data}};
}
} // namespace

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    try {
        const auto result =
            execute(handler, nlohmann::json::parse(values_json), default_runtime_root).dump() +
            '\n';
        output(context, result.data(), result.size());
        return 0;
    } catch (const std::exception& exception) {
        const std::string message = "Error: " + std::string(exception.what()) + '\n';
        error(context, message.data(), message.size());
        return 1;
    }
}
