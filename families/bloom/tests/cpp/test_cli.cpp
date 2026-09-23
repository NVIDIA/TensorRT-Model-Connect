/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/bundle.h"
#include "trtmc/internal/cli.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
int failures = 0, task_loads = 0;
std::string loaded_runtime, loaded_cache, prompt;
bool loaded_graphs = false;
trtmc::TextGenerationConfig captured_config;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
class FakeGeneration final : public trtmc::ITextGeneration {
  public:
    std::int32_t default_max_new_tokens() const override { return 128; }
    trtmc::TextResult generate(const std::string& text,
                               const trtmc::TextGenerationConfig& config) override {
        prompt = text;
        captured_config = config;
        auto result = trtmc::TextResult{"generated", {7, 11}, 1.5, 2.5};
        result.setup_ms = 0.5;
        return result;
    }
};
void write_bundle(const fs::path& path, const std::string& family) {
    const auto header = Json{{"format", 1},
                             {"family", family},
                             {"task", "text_generation"},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream output(path, std::ios::binary);
    output.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("PLAN", 4);
}
struct Capture {
    std::string output, error;
};
void capture_output(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->output.append(data, size);
}
void capture_error(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->error.append(data, size);
}
void contract(const fs::path& root) {
    const auto path = root / "bloom.bundle";
    write_bundle(path, "bloom");
    auto invoke = [&](const char* handler, const Json& values) {
        Capture captured;
        const auto status =
            trtmc_family_cli_v1(handler, values.dump().c_str(), "/installed/runtime", &captured,
                                capture_output, capture_error);
        return std::pair{status, captured};
    };
    Json values{{"bundle", path.string()}, {"prompt", "Hello"}};
    const auto first = invoke("generate", values);
    check(first.first == 0 && prompt == "Hello", "owner invokes text generation");
    check(Json::parse(first.second.output) == Json({{"text", "generated"},
                                                    {"token_ids", {7, 11}},
                                                    {"setup_ms", 0.5},
                                                    {"prefill_ms", 1.5},
                                                    {"decode_ms", 2.5}}),
          "owner preserves result fields");
    check(captured_config.max_new_tokens == 128 && captured_config.top_k == 1 &&
              captured_config.temperature == 1.0F && !captured_config.use_chat_template &&
              captured_config.enable_thinking && captured_config.seed == -1,
          "defaults preserve the runtime contract");
    values.update({{"max_new_tokens", 5},
                   {"temperature", 0.0},
                   {"top_k", 0},
                   {"top_p", 0.25},
                   {"min_p", 0.125},
                   {"seed", 0},
                   {"repetition_penalty", 1.5},
                   {"use_chat_template", true},
                   {"enable_thinking", false},
                   {"generation_mode", "ar"},
                   {"runtime_root", "/override/runtime"},
                   {"runtime_cache", "/cache/runtime"},
                   {"cuda_graphs", true}});
    check(invoke("generate", values).first == 0, "owner accepts supported overrides");
    check(captured_config.max_new_tokens == 5 && captured_config.temperature == 0.0F &&
              captured_config.top_k == 0 && captured_config.top_p == 0.25F &&
              captured_config.min_p == 0.125F && captured_config.seed == 0 &&
              captured_config.repetition_penalty == 1.5F && captured_config.use_chat_template &&
              !captured_config.enable_thinking && captured_config.text_generation_mode == "ar",
          "zero and false overrides survive transport");
    check(loaded_runtime == "/override/runtime" && loaded_cache == "/cache/runtime" &&
              loaded_graphs,
          "owner forwards backend cache and graph controls");
    for (const auto& invalid :
         {Json{{"max_new_tokens", 0}}, Json{{"max_new_tokens", INT64_C(2147483648)}},
          Json{{"top_k", -1}}, Json{{"temperature", -1.0}}, Json{{"top_p", 2.0}},
          Json{{"min_p", -1.0}}, Json{{"repetition_penalty", 0.0}}}) {
        auto broken = values;
        broken.update(invalid);
        const auto before = task_loads;
        check(invoke("generate", broken).first != 0 && task_loads == before,
              "invalid controls fail before loading a task");
    }
    write_bundle(path, "another_family");
    const auto before = task_loads;
    check(invoke("generate", values).first != 0 && task_loads == before,
          "owner rejects another family's bundle before loading it");
    check(invoke("unknown", values).first != 0 && task_loads == before,
          "owner rejects unknown handlers");
}
} // namespace
namespace trtmc {
std::unique_ptr<ITask> load_task(const BundleReader&, const std::string& runtime_root,
                                 std::uint64_t, const std::string& runtime_cache,
                                 bool cuda_graphs) {
    ++task_loads;
    loaded_runtime = runtime_root;
    loaded_cache = runtime_cache;
    loaded_graphs = cuda_graphs;
    return std::make_unique<FakeGeneration>();
}
} // namespace trtmc
int main() {
    const auto root = fs::temp_directory_path() / ("trtmc-bloom-cli-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        contract(root);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root);
    return failures == 0 ? 0 : 1;
}
