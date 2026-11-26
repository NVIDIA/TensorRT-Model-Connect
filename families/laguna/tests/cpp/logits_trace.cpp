/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Writes the Laguna decode-path logits trace of one greedy run:
//   PREFIX.ids          prompt token IDs, one per line
//   PREFIX.prefill.f32  float32 prefill-context logits of the last prompt token
//   PREFIX.shape        "<rows> <vocab>"
//   PREFIX.f32          row-major float32 decode-context logits

#include "families/laguna/runtime/pipeline.h"
#include "trtmc/runtime/family_loader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 8)
            throw std::invalid_argument("usage: laguna_logits_trace BUNDLE ROOT PREFIX PROMPT "
                                        "MAX_NEW_TOKENS USE_CHAT_TEMPLATE(0|1) THINKING(0|1)");
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* pipeline = dynamic_cast<trtmc::laguna::Pipeline*>(task.get());
        if (pipeline == nullptr)
            throw std::runtime_error("bundle does not create a Laguna pipeline");
        trtmc::TextGenerationConfig config;
        config.max_new_tokens = std::stoi(argv[5]);
        config.temperature = 1.0F;
        config.top_k = 1;
        config.use_chat_template = std::string(argv[6]) == "1";
        config.enable_thinking = std::string(argv[7]) == "1";
        const auto trace = pipeline->trace_logits(argv[4], config);
        if (trace.rows.empty() || trace.rows.front().empty())
            throw std::runtime_error("Laguna logits trace is empty");
        const std::string prefix = argv[3];
        std::ofstream ids(prefix + ".ids");
        for (const auto id : trace.prompt_ids)
            ids << id << '\n';
        const std::size_t columns = trace.rows.front().size();
        if (trace.prefill_last.size() != columns)
            throw std::runtime_error("Laguna prefill logits differ in width");
        std::ofstream prefill(prefix + ".prefill.f32", std::ios::binary);
        prefill.write(reinterpret_cast<const char*>(trace.prefill_last.data()),
                      static_cast<std::streamsize>(columns * sizeof(float)));
        std::ofstream shape(prefix + ".shape");
        shape << trace.rows.size() << ' ' << columns << '\n';
        std::ofstream data(prefix + ".f32", std::ios::binary);
        for (const auto& row : trace.rows) {
            if (row.size() != columns)
                throw std::runtime_error("Laguna logits trace rows differ in width");
            data.write(reinterpret_cast<const char*>(row.data()),
                       static_cast<std::streamsize>(row.size() * sizeof(float)));
        }
        if (!ids || !shape || !data || !prefill)
            throw std::runtime_error("failed to write the Laguna logits trace");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
