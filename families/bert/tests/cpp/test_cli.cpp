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
#include <vector>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
int failures = 0;
int task_loads = 0;
std::string loaded_runtime;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeBert final : public trtmc::IEncoding, public trtmc::IEmbedding, public trtmc::IReranking {
  public:
    const char* task() const noexcept override { return "encoding"; }
    trtmc::EmbeddingResult encode(const std::string& text) override {
        return {{static_cast<float>(text.size()), 1.0F}, 2};
    }
    trtmc::EmbeddingResult embed(const std::string& text) override {
        return {{static_cast<float>(text.size()), 2.0F}, 2};
    }
    float rerank(const std::string& query, const std::string& document) override {
        return query == document ? 1.0F : 0.0F;
    }
    std::vector<float> rerank_batch(const std::string&, const std::vector<std::string>&) override {
        throw std::logic_error("unexpected batch operation");
    }
};

void write_bundle(const fs::path& path, const std::string& family) {
    const auto header = Json{{"format", 1},
                             {"family", family},
                             {"task", "encoding"},
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

void bert_contract(const fs::path& root) {
    const auto path = root / "bert.bundle";
    write_bundle(path, "bert");
    auto invoke = [&](const char* handler, Json values) {
        Capture captured;
        const auto status =
            trtmc_family_cli_v1(handler, values.dump().c_str(), "/installed/runtime", &captured,
                                capture_output, capture_error);
        return std::pair{status, captured};
    };
    const auto encoded = invoke("encode", {{"bundle", path.string()}, {"text", "abc"}});
    check(encoded.first == 0 && Json::parse(encoded.second.output).at("values") == Json({3.0, 1.0}),
          "BERT owner converts the encoding request and preserves result fields");
    check(loaded_runtime == "/installed/runtime",
          "BERT owner uses the supplied installed runtime root");
    const auto embedded =
        invoke("embed",
               {{"bundle", path.string()}, {"text", "abc"}, {"runtime_root", "/override/runtime"}});
    check(embedded.first == 0 &&
              Json::parse(embedded.second.output).at("values") == Json({3.0, 2.0}),
          "BERT owner selects the embedding interface");
    check(loaded_runtime == "/override/runtime", "runtime_root override remains owned by BERT");
    const auto ranked =
        invoke("rerank", {{"bundle", path.string()}, {"query", "a"}, {"document", "a"}});
    check(ranked.first == 0 && Json::parse(ranked.second.output).at("score") == 1.0,
          "BERT owner selects the reranking interface");
    write_bundle(path, "another_family");
    const auto before = task_loads;
    check(invoke("encode", {{"bundle", path.string()}, {"text", "a"}}).first != 0 &&
              task_loads == before,
          "BERT rejects another family's bundle before loading it");
    check(invoke("unknown", {{"bundle", path.string()}}).first != 0 && task_loads == before,
          "BERT rejects unknown handlers before loading a task");
}

} // namespace

namespace trtmc {
std::unique_ptr<ITask> load_task(const BundleReader&, const std::string& runtime_root,
                                 std::uint64_t, const std::string&, bool) {
    ++task_loads;
    loaded_runtime = runtime_root;
    return std::make_unique<FakeBert>();
}
} // namespace trtmc

int main() {
    const auto root = fs::temp_directory_path() / ("trtmc-bert-cli-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        bert_contract(root);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root);
    return failures == 0 ? 0 : 1;
}
