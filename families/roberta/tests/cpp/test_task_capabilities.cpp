/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/roberta/runtime/pipeline.h"
#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

class Tokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string&) const override { return {4, 5}; }
    std::string decode(const std::vector<std::int32_t>&) const override { return {}; }
    std::int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(std::int32_t) const override { return {}; }
};

class Encoder final : public trtmc::ITrtModule {
  public:
    int calls{0};
    std::vector<float> values{3.0F, 4.0F, 3.0F, 4.0F};

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        check(inputs.at("input_ids").shape == std::vector<std::int64_t>{2}, "input length");
        return {{"hidden_states", {values.data(), {2, 2}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"hidden_states", {2, 2}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<std::int64_t> tensor_shape(const std::string&) const override { return {2, 2}; }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  trtmc::ProfileShapeSelector) const override {
        return {2};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    std::int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

class Backend final : public trtmc::IBackend {
  public:
    int calls{0};
    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, std::size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        ++calls;
        return std::make_unique<Encoder>();
    }
    std::unique_ptr<trtmc::ITrtModule>
    create_module_prebound(const void*, std::size_t, const trtmc::ModuleCreateOptions&,
                           const std::vector<trtmc::ModuleExternalBinding>&) override {
        ++calls;
        throw std::runtime_error("unexpected backend call");
    }
    trtmc::BackendDualProfileModules
    create_dual_profile_modules(const void*, std::size_t,
                                const trtmc::ModuleCreateOptions&) override {
        ++calls;
        throw std::runtime_error("unexpected backend call");
    }
    const char* name() const override { return "trt"; }
};

void test_legacy_bundle_rejected_before_backend() {
    char pattern[] = "/tmp/trtmc-roberta-task-XXXXXX";
    const int fd = mkstemp(pattern);
    if (fd < 0)
        throw std::runtime_error("cannot create test bundle");
    close(fd);
    struct File {
        std::filesystem::path path;
        ~File() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } file{pattern};
    const std::string config = R"({"tensor_parallel_size":1,"tokenizer_add_special_tokens":true})";
    const std::string header =
        R"({"format":1,"family":"roberta","task":"reranking","backend":"trt","sections":{"runtime.json":{"offset":0,"length":)" +
        std::to_string(config.size()) + R"(},"engine.plan":{"offset":)" +
        std::to_string(config.size()) + R"(,"length":4}}})";
    {
        std::ofstream output(file.path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
        for (int shift = 0; shift < 64; shift += 8)
            output.put(static_cast<char>((header.size() >> shift) & 0xffU));
        output << header << config << "PLAN";
    }
    trtmc::BundleReader reader(file.path.string());
    Backend backend;
    try {
        std::unique_ptr<trtmc::ITask> task(trtmc_create_family({reader, backend}));
        check(false, "legacy reranking bundle must fail");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()) == "RoBERTa does not implement task: reranking",
              "factory reports the unsupported task");
    }
    check(backend.calls == 0, "unsupported bundle is rejected before engine deserialization");
}

} // namespace

int main() {
    test_legacy_bundle_rejected_before_backend();
    auto tokenizer = std::make_shared<Tokenizer>();
    check(!std::is_base_of<trtmc::IReranking, trtmc::EncoderPipeline>::value,
          "encoder must not expose a reranking interface without a scoring head");
    try {
        trtmc::EncoderPipeline unsupported(std::make_unique<Encoder>(), "reranking", tokenizer);
        check(false, "legacy reranking mode must be rejected");
    } catch (const std::invalid_argument& error) {
        check(std::string(error.what()).find("reranking") != std::string::npos,
              "unsupported mode identifies reranking");
    }

    for (const auto* mode : {"encoding", "embedding"}) {
        auto encoder = std::make_unique<Encoder>();
        auto* observed = encoder.get();
        trtmc::EncoderPipeline pipeline(std::move(encoder), mode, tokenizer);
        trtmc::ITask* task = &pipeline;
        check(dynamic_cast<trtmc::IReranking*>(task) == nullptr, "no false reranking capability");
        const auto result =
            std::string(mode) == "encoding" ? pipeline.encode("text") : pipeline.embed("text");
        check(observed->calls == 1, "supported mode executes its encoder once");
        check(result.dim == 2 && result.data.size() == 2, "supported result dimensions");
        if (result.data.size() == 2) {
            const float x = std::string(mode) == "encoding" ? 3.0F : 0.6F;
            const float y = std::string(mode) == "encoding" ? 4.0F : 0.8F;
            check(std::abs(result.data[0] - x) < 1e-6F && std::abs(result.data[1] - y) < 1e-6F,
                  "encoding and normalized embedding values are preserved");
        }
    }
    return failures == 0 ? 0 : 1;
}
