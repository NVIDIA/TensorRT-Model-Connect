/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_moe/runtime/pipeline.h"
#include "trtmc/runtime/family_loader.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int rank() {
    for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "RANK"}) {
        const char* value = std::getenv(name);
        if (value != nullptr && *value != '\0')
            return std::stoi(value);
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "usage: qwen_moe_logits_trace BUNDLE ROOT PREFIX PROMPT MAX");
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* pipeline = dynamic_cast<trtmc::QwenMoeTextGenerationPipeline*>(task.get());
        if (pipeline == nullptr)
            throw std::runtime_error("bundle does not create QwenMoeTextGenerationPipeline");
        trtmc::TextGenerationConfig config;
        config.max_new_tokens = std::stoi(argv[5]);
        config.temperature = 1.0F;
        config.top_k = 1;
        const auto trace = pipeline->trace_logits(argv[4], config);
        if (trace.rows.empty() || trace.rows.front().empty())
            throw std::runtime_error("Qwen MoE logits trace is empty");
        const std::size_t columns = trace.rows.front().size();
        const std::string prefix = std::string(argv[3]) + ".rank" + std::to_string(rank());
        std::ofstream shape(prefix + ".shape");
        std::ofstream data(prefix + ".f32", std::ios::binary);
        shape << trace.rows.size() << ' ' << columns << '\n';
        for (const auto& row : trace.rows) {
            if (row.size() != columns)
                throw std::runtime_error("Qwen MoE logits trace rows have different widths");
            data.write(reinterpret_cast<const char*>(row.data()),
                       static_cast<std::streamsize>(row.size() * sizeof(float)));
        }
        if (!shape || !data)
            throw std::runtime_error("failed to write Qwen MoE logits trace");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
