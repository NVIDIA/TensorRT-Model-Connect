/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU contract of the token classifier and its bundle factory, with a fake
// engine and backend: what reaches the engine, what the Task returns, and which
// inputs, configurations, engines and bundles are refused.

#include "families/openai_privacy_filter/runtime/pipeline.h"
#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
using namespace trtmc;
using namespace trtmc::internal;
using trtmc::openai_privacy_filter::RuntimeConfig;
using trtmc::openai_privacy_filter::TokenClassifier;

constexpr std::int64_t kLabels = 3;

void check(bool condition, const char* what) {
    if (!condition)
        throw std::runtime_error(std::string("openai_privacy_filter task contract: ") + what);
}

template <class Error, class Function>
void rejects(Function&& function, const char* what) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(std::string("openai_privacy_filter accepted ") + what);
}

class Engine final : public ITrtModule {
  public:
    int calls{0};
    int fault{0};
    bool dynamic{true};
    std::int64_t capacity{4};
    std::int64_t labels{kLabels};
    std::vector<std::vector<std::int32_t>> inputs;
    std::vector<float> logits;

    TensorMap forward(const TensorMap& input) override {
        ++calls;
        if (fault == 1)
            throw std::runtime_error("engine failure");
        const auto& ids = input.at("input_ids");
        check(input.size() == 1 && ids.dtype == DType::kInt32 && ids.shape.size() == 1,
              "engine input");
        const auto* data = static_cast<const std::int32_t*>(ids.data);
        inputs.emplace_back(data, data + ids.numel());
        logits.clear();
        for (const auto id : inputs.back())
            for (std::int64_t label = 0; label < kLabels; ++label)
                logits.push_back(static_cast<float>(id) * static_cast<float>(label + 1) - 0.5f);
        Tensor output{logits.data(), {ids.shape[0], kLabels}, DType::kFloat32};
        if (fault == 2)
            return {};
        if (fault == 3)
            output.dtype = DType::kFloat16;
        if (fault == 4)
            output.shape[0] += 1;
        if (fault == 5)
            output.shape[1] = kLabels + 1;
        if (fault == 6)
            output.data = nullptr;
        return {{"logits", output}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string& name) const override {
        return name == "input_ids" ? DType::kInt32 : DType::kFloat32;
    }
    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        return name == "logits" ? std::vector<std::int64_t>{-1, labels}
                                : std::vector<std::int64_t>{-1};
    }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  ProfileShapeSelector) const override {
        return {capacity};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    std::int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return dynamic; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

std::vector<std::int32_t> encode(std::string_view text) {
    if (text == "empty")
        return {};
    if (text == "oversize")
        return {1, 2, 3, 4, 5};
    if (text == "invalid")
        return {100};
    return {1, 2};
}

constexpr RuntimeConfig kConfig{100, 4, kLabels};

void task_contract() {
    auto engine = std::make_unique<Engine>();
    auto* observed = engine.get();
    TokenClassifier model(std::move(engine), encode, kConfig);
    check(std::string(model.task()) == "text_to_head_scores", "primary task");
    auto bindings = model.task_bindings();
    check(bindings.size() == 1 && bindings[0].key.id == ITextToHeadScores::kTask &&
              bindings[0].fields.empty(),
          "one head-scores binding without configuration fields");
    auto* task = static_cast<ITextToHeadScores*>(bindings[0].implementation);

    const auto result = task->run({std::string_view("text")}, {});
    check(observed->inputs.back() == std::vector<std::int32_t>({1, 2}),
          "the tokenizer's ids reach the engine");
    check(result.shape == std::vector<std::uint64_t>({2, kLabels}), "shape is [tokens, labels]");
    check(result.values == std::vector<float>({0.5f, 1.5f, 2.5f, 1.5f, 3.5f, 5.5f}),
          "logits are returned as is");
    check(result.kind == ScoreKind::Logit && result.pooling == "none" &&
              result.normalization == "none",
          "raw per-token logits");

    std::vector<std::int32_t> ids{7, 99, 0, 3};
    const auto given = task->run({Span<const std::int32_t>(ids.data(), ids.size())}, {});
    check(observed->inputs.back() == ids && given.shape == std::vector<std::uint64_t>({4, kLabels}),
          "token ids are used as given");

    for (const auto* text : {"empty", "oversize", "invalid"}) {
        const auto before = observed->calls;
        rejects<std::invalid_argument>([&] { task->run({std::string_view(text)}, {}); }, text);
        check(before == observed->calls, "rejected text never reaches the engine");
    }
    for (const auto id : {-1, 100}) {
        ids = {id};
        rejects<std::invalid_argument>(
            [&] { task->run({Span<const std::int32_t>(ids.data(), ids.size())}, {}); },
            "an id outside the vocabulary");
    }
    rejects<std::invalid_argument>([&] { task->run({Span<const std::int32_t>{}}, {}); },
                                   "no token ids");
    ConfigEntry entry{"window", std::int64_t(1)};
    rejects<ConfigError>([&] { task->run({std::string_view("text")}, {&entry, 1}); },
                         "a configuration value");
    for (int fault = 1; fault <= 6; ++fault) {
        observed->fault = fault;
        rejects<std::runtime_error>([&] { task->run({std::string_view("text")}, {}); },
                                    "a faulty engine result");
    }
}

void engine_contract() {
    const auto engine_with = [](auto&& edit) {
        auto engine = std::make_unique<Engine>();
        edit(*engine);
        return engine;
    };
    rejects<std::invalid_argument>(
        [&] {
            TokenClassifier(engine_with([](Engine& e) { e.dynamic = false; }), encode, kConfig);
        },
        "a fixed-length engine");
    rejects<std::invalid_argument>(
        [&] { TokenClassifier(engine_with([](Engine& e) { e.capacity = 3; }), encode, kConfig); },
        "a profile below the bundle bound");
    rejects<std::invalid_argument>(
        [&] { TokenClassifier(engine_with([](Engine& e) { e.labels = 2; }), encode, kConfig); },
        "an engine with another label count");
    rejects<std::invalid_argument>(
        [&] { TokenClassifier(std::make_unique<Engine>(), nullptr, kConfig); },
        "a missing tokenizer");
}

class Backend final : public IBackend {
  public:
    int calls{0};
    std::unique_ptr<ITrtModule> create_module(const void*, std::size_t,
                                              const ModuleCreateOptions&) override {
        ++calls;
        return std::make_unique<Engine>();
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, std::size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        throw std::runtime_error("unexpected backend call");
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, std::size_t,
                                                          const ModuleCreateOptions&) override {
        throw std::runtime_error("unexpected backend call");
    }
    const char* name() const override { return "trt"; }
};

std::string byte_symbol(int byte) {
    const auto printable = [](int b) {
        return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int extra = 0;
    for (int b = 0; b < byte; ++b)
        extra += printable(b) ? 0 : 1;
    const unsigned cp =
        printable(byte) ? static_cast<unsigned>(byte) : 256U + static_cast<unsigned>(extra);
    if (cp < 0x80)
        return std::string(1, static_cast<char>(cp));
    return {static_cast<char>(0xC0 | (cp >> 6)), static_cast<char>(0x80 | (cp & 0x3F))};
}

// The 256 byte symbols and no merges: every byte is one token, its own id.
std::string byte_tokenizer_json() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int byte = 0; byte < 256; ++byte)
        vocab[byte_symbol(byte)] = byte;
    const std::string pattern =
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
        R"(|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
        R"(|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    const nlohmann::json json = {
        {"added_tokens", nlohmann::json::array()},
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "Split"},
             {"pattern", {{"Regex", pattern}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", false}}}}}},
        {"post_processor", nullptr},
        {"model",
         {{"type", "BPE"},
          {"ignore_merges", true},
          {"vocab", vocab},
          {"merges", nlohmann::json::array()}}},
    };
    return json.dump();
}

class BundleFile {
  public:
    BundleFile(const std::string& task,
               const std::vector<std::pair<std::string, std::string>>& sections) {
        char pattern[] = "/tmp/trtmc-openai-privacy-filter-XXXXXX";
        const int fd = mkstemp(pattern);
        if (fd < 0)
            throw std::runtime_error("cannot create a test bundle");
        close(fd);
        path_ = pattern;
        nlohmann::json index = nlohmann::json::object();
        std::uint64_t offset = 0;
        for (const auto& [name, data] : sections) {
            index[name] = {{"offset", offset}, {"length", data.size()}};
            offset += data.size();
        }
        const std::string header = nlohmann::json{
            {"format", 1},
            {"family", "openai_privacy_filter"},
            {"task", task},
            {"backend", "trt"},
            {"sections", index}}.dump();
        std::ofstream output(path_, std::ios::binary);
        output.write(reinterpret_cast<const char*>(kBundleMagic), 8);
        for (int shift = 0; shift < 64; shift += 8)
            output.put(static_cast<char>((header.size() >> shift) & 0xffU));
        output << header;
        for (const auto& section : sections)
            output << section.second;
    }
    ~BundleFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    const std::string& path() const { return path_; }

  private:
    std::string path_;
};

void factory_contract() {
    const std::string runtime =
        R"({"vocab_size":300,"labels":["O","B-x","I-x"],"max_sequence_length":4})";
    {
        BundleFile file("text_to_head_scores", {{"runtime.json", runtime},
                                                {"tokenizer.json", byte_tokenizer_json()},
                                                {"engine.plan", "PLAN"}});
        BundleReader reader(file.path());
        Backend backend;
        std::unique_ptr<ITask> task(trtmc_create_family({reader, backend}));
        auto* model = dynamic_cast<TokenClassifier*>(task.get());
        check(model != nullptr && backend.calls == 1, "the factory loads one engine");
        const auto result = model->run(TextToHeadScoresRequest{std::string_view("Hi!")}, {});
        check(result.shape == std::vector<std::uint64_t>({3, kLabels}),
              "the bundle tokenizer frames nothing");
        check(result.values[0] == 'H' * 1.0f - 0.5f && result.values[3] == 'i' * 1.0f - 0.5f &&
                  result.values[6] == '!' * 1.0f - 0.5f,
              "byte ids reach the engine");
        rejects<std::invalid_argument>(
            [&] { model->run(TextToHeadScoresRequest{std::string_view("12345")}, {}); },
            "text above the bound");
        rejects<std::invalid_argument>(
            [&] { model->run(TextToHeadScoresRequest{std::string_view("\xFF")}, {}); },
            "malformed UTF-8");
    }
    for (const auto* task : {"text_to_token_features", "text_to_pooled_features"}) {
        BundleFile file(task, {{"runtime.json", runtime}, {"engine.plan", "PLAN"}});
        BundleReader reader(file.path());
        Backend backend;
        rejects<std::runtime_error>(
            [&] { std::unique_ptr<ITask> made(trtmc_create_family({reader, backend})); },
            "a bundle of another task");
        check(backend.calls == 0, "an unsupported bundle never reaches the backend");
    }
    {
        BundleFile file("text_to_head_scores",
                        {{"runtime.json", R"({"vocab_size":300,"max_sequence_length":4})"},
                         {"tokenizer.json", byte_tokenizer_json()},
                         {"engine.plan", "PLAN"}});
        BundleReader reader(file.path());
        Backend backend;
        rejects<std::runtime_error>(
            [&] { std::unique_ptr<ITask> made(trtmc_create_family({reader, backend})); },
            "a runtime.json without labels");
        check(backend.calls == 0, "an invalid runtime.json never reaches the backend");
    }
    {
        BundleFile file(
            "text_to_head_scores",
            {{"runtime.json", R"({"vocab_size":200,"labels":["O"],"max_sequence_length":4})"},
             {"tokenizer.json", byte_tokenizer_json()},
             {"engine.plan", "PLAN"}});
        BundleReader reader(file.path());
        Backend backend;
        rejects<std::runtime_error>(
            [&] { std::unique_ptr<ITask> made(trtmc_create_family({reader, backend})); },
            "a tokenizer larger than the embedding table");
    }
}

} // namespace

int main() {
    try {
        task_contract();
        engine_contract();
        factory_contract();
        std::cout << "openai_privacy_filter task contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
