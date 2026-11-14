/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/cli.h"

#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    try {
        if (std::string(handler) != "decide")
            throw std::invalid_argument("unknown Laya command");
        const auto values = nlohmann::json::parse(values_json);
        trtmc::BundleReader reader(values.at("bundle").get<std::string>());
        if (reader.info().family != "laya")
            throw std::invalid_argument("Laya command requires a laya bundle");
        auto loaded = trtmc::load_task(
            reader, values.value("runtime_root", std::string(default_runtime_root)));
        auto* model = dynamic_cast<trtmc::IStructuredDecision*>(loaded.get());
        if (!model)
            throw std::runtime_error("Laya bundle lacks the structured-decision interface");
        std::ifstream file(values.at("record").get<std::string>());
        if (!file)
            throw std::invalid_argument("cannot read Laya record");
        trtmc::StructuredDecisionRequest request;
        request.document = std::string(std::istreambuf_iterator<char>(file), {});
        const auto response = model->decide(request).document + '\n';
        output(context, response.data(), response.size());
        return 0;
    } catch (const std::exception& exception) {
        const auto message = std::string("Error: ") + exception.what() + '\n';
        error(context, message.data(), message.size());
        return 1;
    }
}
