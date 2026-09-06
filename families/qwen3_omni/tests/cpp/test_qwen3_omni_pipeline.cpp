/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/kv_cache.h"
#include "families/qwen3_omni/runtime/pipeline.h"
#include "families/qwen3_omni/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class FixedTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string&) const override { return {4, 5, 6}; }
    std::string decode(const std::vector<std::int32_t>&) const override { return "hello"; }
    std::int32_t id_for_token(std::string_view token) const override {
        return token == "<|endoftext|>" ? 7 : -1;
    }
    std::string token_for_id(std::int32_t id) const override { return id == 1 ? "hello" : ""; }
};

struct ModuleStats {
    std::int32_t launches{0};
    std::unordered_map<std::string, std::vector<std::int64_t>> input_shapes;
};

class FakeThinkerModule final : public trtmc::ITrtModule {
  public:
    FakeThinkerModule(bool prefill, std::shared_ptr<ModuleStats> stats, cudaStream_t stream)
        : prefill_(prefill), stats_(std::move(stats)), stream_(stream),
          present_k_(std::vector<std::int64_t>{32, 1}, trtmc::DType::kFloat32, stream),
          present_v_(std::vector<std::int64_t>{32, 1}, trtmc::DType::kFloat32, stream),
          logits_(8, -100.0F) {
        if (prefill_) {
            logits_[1] = 10.0F;
            logits_[2] = 10.0F;
        } else {
            logits_[3] = 10.0F;
        }
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++stats_->launches;
        stats_->input_shapes.clear();
        for (const auto& [name, tensor] : inputs)
            stats_->input_shapes[name] = tensor.shape;
        const auto rows = inputs.at("token_id").shape.at(0);
        present_host_.assign(static_cast<std::size_t>(rows), 0.0F);
        return {
            {"logits", trtmc::Tensor{logits_.data(), {1, 8}, trtmc::DType::kFloat32}},
            {"present_k_0", trtmc::Tensor{present_host_.data(), {rows, 1}, trtmc::DType::kFloat32}},
            {"present_v_0", trtmc::Tensor{present_host_.data(), {rows, 1}, trtmc::DType::kFloat32}},
        };
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }

    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "attention_mask" ||
               name == "cache_k_0" || name == "cache_v_0";
    }

    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }

    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == "position_id" ? trtmc::DType::kInt32 : trtmc::DType::kFloat32;
    }

    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {32, 1};
        if (name == "present_k_0" || name == "present_v_0")
            return {1, 1};
        if (name == "position_id")
            return {1};
        if (name == "attention_mask")
            return {1, 33};
        if (name == "logits")
            return {1, 8};
        return {};
    }

    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  trtmc::ProfileShapeSelector) const override {
        return {};
    }
    std::int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) const override {
        const auto bound = bindings_.find(name);
        if (bound != bindings_.end())
            return bound->second;
        if (name == "present_k_0")
            return const_cast<void*>(present_k_.data());
        if (name == "present_v_0")
            return const_cast<void*>(present_v_.data());
        return nullptr;
    }

    void bind_external(const std::string& name, void* pointer) override {
        bindings_[name] = pointer;
    }
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<std::int64_t>&) override {
        bind_external(name, pointer);
    }

    std::int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return present_k_.ok() && present_v_.ok(); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    bool prefill_{false};
    std::shared_ptr<ModuleStats> stats_;
    cudaStream_t stream_{nullptr};
    mutable std::unordered_map<std::string, void*> bindings_;
    mutable trtmc::DeviceTensor present_k_;
    mutable trtmc::DeviceTensor present_v_;
    std::vector<float> logits_;
    std::vector<float> present_host_;
};

struct PipelineStats {
    std::shared_ptr<ModuleStats> prefill = std::make_shared<ModuleStats>();
    std::shared_ptr<ModuleStats> decode = std::make_shared<ModuleStats>();
};

trtmc::Qwen3OmniRuntimeConfig make_config() {
    trtmc::Qwen3OmniRuntimeConfig config;
    config.precision = "bf16";
    config.thinker_num_layers = 1;
    config.thinker_num_key_value_heads = 1;
    config.thinker_head_dim = 1;
    config.thinker_vocab_size = 8;
    config.thinker_max_cache_length = 32;
    config.thinker_eos_token_id = 3;
    return config;
}

std::unique_ptr<trtmc::Qwen3OmniTextPipeline>
make_pipeline(cudaStream_t stream, PipelineStats& stats, bool omit_prefill = false) {
    auto prefill = omit_prefill ? std::unique_ptr<FakeThinkerModule>{}
                                : std::make_unique<FakeThinkerModule>(true, stats.prefill, stream);
    auto decode = std::make_unique<FakeThinkerModule>(false, stats.decode, stream);
    auto cache =
        std::make_unique<trtmc::Qwen3OmniKvCache>(1, 32, 1, stream, trtmc::DType::kFloat32);
    return std::make_unique<trtmc::Qwen3OmniTextPipeline>(std::move(prefill), std::move(decode),
                                                          std::move(cache), make_config(),
                                                          std::make_shared<FixedTokenizer>());
}

class StreamFixture {
  public:
    StreamFixture() { cudaStreamCreate(&stream); }
    ~StreamFixture() { cudaStreamDestroy(stream); }
    cudaStream_t stream{nullptr};
};

void test_pipeline_construction() {
    StreamFixture fixture;
    PipelineStats stats;
    auto pipeline = make_pipeline(fixture.stream, stats);
    check(std::string(pipeline->task()) == trtmc::ITextGeneration::kTask,
          "Qwen3-Omni implements the text-generation Task API");
    check(pipeline->default_max_new_tokens() == 128,
          "Qwen3-Omni preserves its default text token limit");
}

void test_validates_thinker() {
    StreamFixture fixture;
    PipelineStats stats;
    bool threw = false;
    try {
        auto pipeline = make_pipeline(fixture.stream, stats, true);
        (void)pipeline;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "Qwen3-Omni rejects a missing Thinker prefill component");
}

void test_batched_prefill_and_argmax() {
    StreamFixture fixture;
    PipelineStats stats;
    auto pipeline = make_pipeline(fixture.stream, stats);
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = 2;
    const auto result = pipeline->generate("question", config);
    check(result.text == "hello" && result.token_ids == std::vector<std::int32_t>({1}),
          "Qwen3-Omni keeps lowest-index argmax and stops on the decode EOS");
    check(stats.prefill->launches == 1 && stats.decode->launches == 1,
          "Qwen3-Omni uses one batched prefill and one decode launch");
    check(stats.prefill->input_shapes["token_id"] == std::vector<std::int64_t>({3}) &&
              stats.prefill->input_shapes["position_id"] == std::vector<std::int64_t>({3}) &&
              stats.prefill->input_shapes["attention_mask"] == std::vector<std::int64_t>({3, 35}),
          "Qwen3-Omni preserves batched prefill token, position, and causal-mask shapes");
}

} // namespace

int main() {
    test_pipeline_construction();
    test_validates_thinker();
    test_batched_prefill_and_argmax();
    if (failures != 0)
        std::cerr << failures << " Qwen3-Omni pipeline test(s) failed\n";
    return failures;
}
