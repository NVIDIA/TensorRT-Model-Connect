/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/kv_cache.h"
#include "runtime/models/qwen3_omni/omni_config.h"
#include "runtime/models/qwen3_omni/pipeline.h"
#include "runtime/models/qwen3_omni/talker_runtime.h"

#include <cstdint>
#include <cuda_runtime_api.h>
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

struct CountingOmniStats {
    int32_t launches{0};
    std::vector<std::unordered_map<std::string, std::vector<int64_t>>> shapes;
    std::vector<std::unordered_map<std::string, std::vector<int32_t>>> int_values;
};

class CountingOmniModule final : public trtmc::ITrtModule {
  public:
    CountingOmniModule(std::shared_ptr<CountingOmniStats> stats, cudaStream_t stream,
                       int32_t capacity)
        : stats_(std::move(stats)), stream_(stream), capacity_(capacity),
          device_logits_({1, 4}, trtmc::DType::kFloat32, stream) {
        device_logits_.copy_from_host(host_logits_.data());
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        record(inputs);
        return {{"logits", trtmc::Tensor{host_logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { record(inputs); }
    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "input_embed" ||
               name == "use_input_embed" || name == "cache_write_indices" ||
               name == "key_value_lengths" || name == "cache_k_0" || name == "cache_v_0";
    }
    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        if (name == "token_id" || name == "position_id" || name == "cache_write_indices" ||
            name == "key_value_lengths")
            return trtmc::DType::kInt32;
        if (name == "cache_k_0" || name == "cache_v_0" || name == "present_k_0" ||
            name == "present_v_0")
            return trtmc::DType::kBFloat16;
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_write_indices" || name == "key_value_lengths")
            return {1};
        if (name == "cache_k_0" || name == "cache_v_0" || name == "present_k_0" ||
            name == "present_v_0")
            return {1, 1, capacity_, 4};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        if (name == "logits")
            return const_cast<void*>(device_logits_.data());
        if (name == "cache_k_0" || name == "present_k_0")
            return cache_k_;
        if (name == "cache_v_0" || name == "present_v_0")
            return cache_v_;
        return nullptr;
    }
    void bind_external(const std::string& name, void* pointer) override {
        if (name == "cache_k_0")
            cache_k_ = pointer;
        else if (name == "cache_v_0")
            cache_v_ = pointer;
    }
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    bool ok() const override { return device_logits_.ok(); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    void record(const trtmc::TensorMap& inputs) {
        ++stats_->launches;
        std::unordered_map<std::string, std::vector<int64_t>> shapes;
        std::unordered_map<std::string, std::vector<int32_t>> values;
        for (const auto& [name, tensor] : inputs) {
            shapes[name] = tensor.shape;
            if (tensor.dtype == trtmc::DType::kInt32) {
                const auto* begin = static_cast<const int32_t*>(tensor.data);
                values[name] = std::vector<int32_t>(begin, begin + tensor.numel());
            }
        }
        stats_->shapes.push_back(std::move(shapes));
        stats_->int_values.push_back(std::move(values));
    }

    std::shared_ptr<CountingOmniStats> stats_;
    cudaStream_t stream_{nullptr};
    int32_t capacity_{0};
    std::vector<float> host_logits_{0.1F, 0.9F, 0.9F, 0.3F};
    mutable trtmc::DeviceTensor device_logits_;
    mutable void* cache_k_{nullptr};
    mutable void* cache_v_{nullptr};
};

std::unique_ptr<trtmc::OmniPipeline>
make_pipeline(cudaStream_t stream, int32_t capacity,
              const std::shared_ptr<CountingOmniStats>& decode_stats,
              const std::shared_ptr<CountingOmniStats>& prefill_stats) {
    auto decode = std::make_unique<CountingOmniModule>(decode_stats, stream, capacity);
    auto prefill = std::make_unique<CountingOmniModule>(prefill_stats, stream, capacity);
    auto cache =
        std::make_unique<trtmc::Qwen3OmniKvCache>(1, capacity, 4, stream, trtmc::DType::kBFloat16);
    trtmc::OmniConfig config;
    config.thinker_hidden_size = 4;
    config.thinker_vocab_size = 4;
    config.thinker_num_layers = 1;
    config.thinker_eos_token_id = 99;
    auto talker = std::make_unique<trtmc::Qwen3OmniTalkerRuntime>("unused", "unused", "", 1, 1);
    return std::make_unique<trtmc::OmniPipeline>(std::move(decode), std::move(cache), nullptr,
                                                 config, stream, nullptr, "test-omni",
                                                 std::move(prefill), std::move(talker));
}

void test_native_prefill_and_decode_scalars() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto decode_stats = std::make_shared<CountingOmniStats>();
    auto prefill_stats = std::make_shared<CountingOmniStats>();
    {
        auto pipeline = make_pipeline(stream, 8, decode_stats, prefill_stats);
        const auto output = pipeline->generate_thinker_ids({1, 2, 3}, 2);
        check(output == std::vector<int32_t>({1, 1}), "native output is deterministic");
        check(prefill_stats->launches == 1, "prompt uses one native prefill launch");
        check(decode_stats->launches == 1, "only one required decode launch runs");
        check(prefill_stats->shapes[0].count("attention_mask") == 0,
              "native prefill has no attention mask");
        check(prefill_stats->int_values[0].at("cache_write_indices") == std::vector<int32_t>{0},
              "prefill writes at cache offset zero");
        check(prefill_stats->int_values[0].at("key_value_lengths") == std::vector<int32_t>{3},
              "prefill exposes exactly the valid prefix");
        check(decode_stats->int_values[0].at("cache_write_indices") == std::vector<int32_t>{3},
              "decode appends after the prompt");
        check(decode_stats->int_values[0].at("key_value_lengths") == std::vector<int32_t>{4},
              "decode extends the valid prefix by one");
    }
    cudaStreamDestroy(stream);
}

void test_full_context_prompt_is_chunked_without_decode_fallback() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto decode_stats = std::make_shared<CountingOmniStats>();
    auto prefill_stats = std::make_shared<CountingOmniStats>();
    {
        constexpr int32_t capacity = 257;
        auto pipeline = make_pipeline(stream, capacity, decode_stats, prefill_stats);
        std::vector<int32_t> prompt(static_cast<std::size_t>(capacity), 1);
        (void)pipeline->generate_thinker_ids(prompt, 1);
        check(prefill_stats->launches == 2, "long prompt is split into two prefill launches");
        check(decode_stats->launches == 0, "long prompt never falls back to token decode");
        check(prefill_stats->shapes[0].at("token_id") == std::vector<int64_t>{256},
              "first prefill chunk matches profile maximum");
        check(prefill_stats->shapes[1].at("token_id") == std::vector<int64_t>{1},
              "final prefill chunk carries the remainder");
        check(prefill_stats->int_values[1].at("cache_write_indices") == std::vector<int32_t>{256},
              "second chunk writes at the prior logical length");
        check(prefill_stats->int_values[1].at("key_value_lengths") ==
                  std::vector<int32_t>{capacity},
              "full official capacity becomes visible");
    }
    cudaStreamDestroy(stream);
}

void test_prompt_beyond_capacity_is_rejected() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto decode_stats = std::make_shared<CountingOmniStats>();
    auto prefill_stats = std::make_shared<CountingOmniStats>();
    bool threw = false;
    {
        auto pipeline = make_pipeline(stream, 8, decode_stats, prefill_stats);
        try {
            (void)pipeline->generate_thinker_ids(std::vector<int32_t>(9, 1), 1);
        } catch (const std::runtime_error&) {
            threw = true;
        }
    }
    check(threw, "prompt beyond official capacity is rejected before execution");
    cudaStreamDestroy(stream);
}

void test_constructor_requires_both_split_modules() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto stats = std::make_shared<CountingOmniStats>();
    bool threw = false;
    try {
        auto decode = std::make_unique<CountingOmniModule>(stats, stream, 8);
        auto cache =
            std::make_unique<trtmc::Qwen3OmniKvCache>(1, 8, 4, stream, trtmc::DType::kBFloat16);
        trtmc::OmniConfig config;
        trtmc::OmniPipeline pipeline(std::move(decode), std::move(cache), nullptr, config, stream);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "native Omni pipeline requires a prefill module");
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_native_prefill_and_decode_scalars();
    test_full_context_prompt_is_chunked_without_decode_fallback();
    test_prompt_beyond_capacity_is_rejected();
    test_constructor_requires_both_split_modules();
    if (failures > 0)
        std::cerr << failures << " Qwen3-Omni pipeline test(s) FAILED\n";
    return failures;
}
