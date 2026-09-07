/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void write_bundle(const std::filesystem::path& path) {
    const std::string runtime =
        R"({"max_input_length":1024,"max_cache_length":4096,"max_batch_size":1})";
    const std::string config = R"({"model":"qwen3"})";
    const std::string engine = "ENGINE";
    const nlohmann::json header{
        {"format", 1},
        {"family", "qwen"},
        {"task", "text_generation"},
        {"backend", "edge_llm"},
        {"sections",
         {{"edge_llm.json", {{"offset", 0}, {"length", runtime.size()}}},
          {"edge_llm/config.json", {{"offset", runtime.size()}, {"length", config.size()}}},
          {"edge_llm/llm.engine",
           {{"offset", runtime.size() + config.size()}, {"length", engine.size()}}}}},
    };
    const std::string encoded = header.dump();
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const std::uint64_t length = encoded.size();
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((length >> shift) & 0xffU));
    output.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
    output.write(runtime.data(), static_cast<std::streamsize>(runtime.size()));
    output.write(config.data(), static_cast<std::streamsize>(config.size()));
    output.write(engine.data(), static_cast<std::streamsize>(engine.size()));
}

bool rejects_unsupported_sampling(trtmc::ITextGeneration& text) {
    trtmc::TextGenerationConfig config;
    config.min_p = 0.1F;
    try {
        (void)text.generate("hello", config);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_qwen_edge_llm_runtime RUNTIME_ROOT\n";
        return 2;
    }
    const std::filesystem::path runtime_root(argv[1]);
    check(!std::filesystem::exists(runtime_root / "libtrtmc_backend_edge_llm.so"),
          "family-owned runtime does not require a fake Engine API backend");
    const std::filesystem::path bundle =
        runtime_root / ("qwen-edge-test-" + std::to_string(getpid()) + ".bundle");
    write_bundle(bundle);

    try {
        const auto info = trtmc::InspectBundle(bundle.string());
        check(info.family == "qwen", "bundle keeps Qwen family ownership");
        check(info.task == "text_generation", "bundle selects text Task API");
        check(info.backend == "edge_llm", "bundle explicitly selects Edge-LLM");

        auto task = trtmc::load_task(bundle.string(), runtime_root.string());
        auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        check(text != nullptr, "family returns ITextGeneration");
        if (text != nullptr) {
            trtmc::TextGenerationConfig config;
            config.max_new_tokens = 8;
            config.temperature = 0.0F;
            config.top_k = 1;
            const auto result = text->generate("hello", config);
            check(result.text == "fake:hello", "request reaches family-owned bridge");
            check(result.token_ids == std::vector<std::int32_t>({104, 101, 108, 108, 111}),
                  "bridge token IDs reach the Task result");
            check(rejects_unsupported_sampling(*text), "unsupported request fails closed");
        }

        auto second = trtmc::load_task(bundle.string(), runtime_root.string());
        check(dynamic_cast<trtmc::ITextGeneration*>(second.get()) != nullptr,
              "cached loader supports a second Edge-LLM task");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }

    std::error_code cleanup_error;
    std::filesystem::remove(bundle, cleanup_error);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
