/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Writes every debug_hidden_<i> window of a Qwen bundle for one greedy request:
// OUTDIR/<name>.prefill.f32 (prompt rows), OUTDIR/<name>.step<k>.f32 (decoder
// pass k, which consumes generated_ids[k-1]) and OUTDIR/index.json. Row-major
// float32. The prompt uses the chat template without thinking, as
// `trtmc run --use-chat-template true --enable-thinking false`.

#include "families/qwen/runtime/pipeline.h"
#include "trtmc/runtime/family_loader.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DebugWindowValues = trtmc::QwenTextGenerationPipeline::DebugWindowValues;

std::string json_ids(const std::vector<int32_t>& ids) {
    std::string text = "[";
    for (std::size_t i = 0; i < ids.size(); ++i)
        text += (i == 0 ? "" : ", ") + std::to_string(ids[i]);
    return text + "]";
}

// Write one window's values and return its index.json entry.
std::string write_window(const std::filesystem::path& outdir, const std::string& phase,
                         const DebugWindowValues& window) {
    const std::string file = window.name + "." + phase + ".f32";
    std::ofstream data(outdir / file, std::ios::binary);
    data.write(reinterpret_cast<const char*>(window.values.data()),
               static_cast<std::streamsize>(window.values.size() * sizeof(float)));
    data.close();
    if (!data)
        throw std::runtime_error("failed to write " + (outdir / file).string());
    return "{\"name\": \"" + window.name + "\", \"layer\": " + std::to_string(window.layer) +
           ", \"phase\": \"" + phase + "\", \"shape\": [" + std::to_string(window.shape[0]) + ", " +
           std::to_string(window.shape[1]) + "], \"file\": \"" + file + "\"}";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument(
                "usage: qwen_debug_windows_trace BUNDLE ROOT OUTDIR PROMPT MAX");
        }
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* pipeline = dynamic_cast<trtmc::QwenTextGenerationPipeline*>(task.get());
        if (pipeline == nullptr)
            throw std::runtime_error("bundle does not create QwenTextGenerationPipeline");
        trtmc::TextGenerationConfig config;
        config.max_new_tokens = std::stoi(argv[5]);
        config.temperature = 1.0F;
        config.top_k = 1;
        config.use_chat_template = true;
        config.enable_thinking = false;
        const auto trace = pipeline->trace_debug_windows(argv[4], config);

        const std::filesystem::path outdir(argv[3]);
        std::filesystem::create_directories(outdir);
        std::vector<std::string> entries;
        for (const auto& window : trace.prefill)
            entries.push_back(write_window(outdir, "prefill", window));
        for (std::size_t step = 0; step < trace.steps.size(); ++step) {
            for (const auto& window : trace.steps[step])
                entries.push_back(write_window(outdir, "step" + std::to_string(step + 1), window));
        }
        std::ofstream index(outdir / "index.json");
        index << "{\n  \"dtype\": \"float32\",\n  \"prompt_ids\": " << json_ids(trace.prompt_ids)
              << ",\n  \"generated_ids\": " << json_ids(trace.generated_ids)
              << ",\n  \"windows\": [";
        for (std::size_t i = 0; i < entries.size(); ++i)
            index << (i == 0 ? "\n    " : ",\n    ") << entries[i];
        index << (entries.empty() ? "]\n}\n" : "\n  ]\n}\n");
        index.close();
        if (!index)
            throw std::runtime_error("failed to write " + (outdir / "index.json").string());

        if (entries.empty())
            std::cout << "no debug_hidden_<i> outputs in this bundle\n";
        else
            std::cout << "wrote " << entries.size() << " debug windows to " << outdir.string()
                      << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
