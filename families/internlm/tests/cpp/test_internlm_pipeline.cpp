/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression coverage for the one-token MMLU validation path. The
// batched prefill logits already supply that token, so touching the CUDA-graph
// decoder is both unnecessary and unsafe at the longest prompt shape.

#include "families/internlm/runtime/kv_cache.h"
#include "families/internlm/runtime/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class TestTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string&) const override { return {}; }
    std::string decode(const std::vector<std::int32_t>&) const override { return {}; }
    std::int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(std::int32_t) const override { return {}; }
};

struct ModuleStats {
    int32_t calls{0};
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
};

class CountingModule final : public trtmc::ITrtModule {
  public:
    CountingModule(std::shared_ptr<ModuleStats> stats, bool prefill, cudaStream_t stream)
        : stats_(std::move(stats)), prefill_(prefill), stream_(stream),
          present_k_(prefill ? trtmc::DeviceTensor::zeros({704, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}),
          present_v_(prefill ? trtmc::DeviceTensor::zeros({704, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        record(inputs);
        return {{"logits", trtmc::Tensor{logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { record(inputs); }
    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override { cuda_graph_active_ = true; }
    bool cuda_graph_active() const override { return cuda_graph_active_; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return prefill_ ? 0 : 1; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "attention_mask" ||
               name == "cache_k_0" || name == "cache_v_0";
    }
    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {704, 4};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return prefill_ ? 2 : 1; }
    void* device_ptr(const std::string& name) const override {
        if (name == "present_k_0")
            return const_cast<void*>(present_k_.data());
        if (name == "present_v_0")
            return const_cast<void*>(present_v_.data());
        return nullptr;
    }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>&) override {
        bind_external(name, pointer);
    }
    int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return prefill_; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return !prefill_ || (present_k_.ok() && present_v_.ok()); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    void record(const trtmc::TensorMap& inputs) {
        ++stats_->calls;
        stats_->shapes.clear();
        for (const auto& [name, tensor] : inputs)
            stats_->shapes[name] = tensor.shape;
    }

    std::shared_ptr<ModuleStats> stats_;
    bool prefill_{false};
    bool cuda_graph_active_{false};
    cudaStream_t stream_{nullptr};
    mutable trtmc::DeviceTensor present_k_;
    mutable trtmc::DeviceTensor present_v_;
    std::vector<float> logits_{0.1F, 0.2F, 0.9F, 0.3F};
};

void test_one_token_batched_prefill_does_not_touch_decoder() {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "FAIL: create CUDA stream\n";
        ++failures;
        return;
    }

    auto decode_stats = std::make_shared<ModuleStats>();
    auto prefill_stats = std::make_shared<ModuleStats>();
    auto decoder = std::make_unique<CountingModule>(decode_stats, false, stream);
    auto prefill = std::make_unique<CountingModule>(prefill_stats, true, stream);
    auto cache =
        std::make_unique<trtmc::InternlmKvCache>(1, 704, 4, stream, trtmc::DType::kFloat32);

    trtmc::InternlmTextGenConfig config;
    config.vocab_size = 4;
    config.id_eos = 2;
    config.num_layers = 1;
    config.prefill_max_length = 704;

    trtmc::InternlmTextGenerationPipeline pipeline(std::move(decoder), std::move(cache), config,
                                                   std::make_shared<TestTokenizer>(),
                                                   std::move(prefill));

    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 1;
    request.temperature = 0.0F;
    request.top_k = 1;
    // Keep the prefill argmax (token 2) non-EOS so this exercises the explicit
    // final-token stop instead of passing through the earlier EOS branch.
    request.eos_token_id = 3;

    std::vector<int32_t> prompt(700, 1);
    const auto result = pipeline.generate_ids(prompt, request);
    check(result.token_ids.size() == 701 && result.token_ids.back() == 2,
          "one non-EOS token comes from prefill logits");
    check(prefill_stats->calls == 1, "long batched prefill launches once");
    check(decode_stats->calls == 0, "one-token request does not prime decoder");
    check(prefill_stats->shapes["token_id"] == std::vector<int64_t>({700}),
          "prefill receives the complete prompt");

    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    test_one_token_batched_prefill_does_not_touch_decoder();
    return failures;
}
