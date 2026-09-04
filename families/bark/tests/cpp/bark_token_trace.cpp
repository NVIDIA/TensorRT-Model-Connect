/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bark/runtime/pipeline.h"
#include "trtmc/runtime/family_loader.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int process_rank() {
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
        if (argc != 7) {
            throw std::invalid_argument(
                "usage: bark_token_trace BUNDLE RUNTIME_ROOT PREFIX PROMPT MAX_TOKENS SEED");
        }
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* pipeline = dynamic_cast<trtmc::BarkPipeline*>(task.get());
        if (pipeline == nullptr)
            throw std::runtime_error("bundle does not create BarkPipeline");
        const std::string prefix = std::string(argv[3]) + ".rank" + std::to_string(process_rank());
        pipeline->set_token_trace_path(prefix);
        trtmc::AudioGenerationConfig config;
        config.max_new_tokens = std::stoi(argv[5]);
        config.seed = std::stoi(argv[6]);
        const auto audio = pipeline->generate_audio(argv[4], config);
        if (audio.samples.empty() || audio.sample_rate <= 0)
            throw std::runtime_error("Bark token trace generation produced no audio");
        std::cout << prefix << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
