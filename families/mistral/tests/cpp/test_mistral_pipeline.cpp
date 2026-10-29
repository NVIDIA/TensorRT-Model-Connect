/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mistral/runtime/kv_cache.h"
#include "families/mistral/runtime/pipeline.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int32_t kVocabulary = 32;
constexpr int32_t kCacheLength = 64;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class TestTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {2, 3}; }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

class CountingModule final : public trtmc::ITrtModule {
  public:
    explicit CountingModule(bool prefill, bool graphs)
        : prefill_(prefill), graphs_supported_(graphs),
          present_k_(prefill ? trtmc::DeviceTensor::zeros({kCacheLength, 1}, trtmc::DType::kFloat32,
                                                          nullptr)
                             : trtmc::DeviceTensor{}),
          present_v_(prefill ? trtmc::DeviceTensor::zeros({kCacheLength, 1}, trtmc::DType::kFloat32,
                                                          nullptr)
                             : trtmc::DeviceTensor{}),
          logits_(kVocabulary, -10.0F) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        if (!prefill_ && cuda_graph_active())
            captured_ = true;
        const auto& ids = inputs.at("token_id");
        const auto* tokens = static_cast<const int32_t*>(ids.data);
        std::fill(logits_.begin(), logits_.end(), -10.0F);
        logits_[(tokens[ids.numel() - 1] + 1) % kVocabulary] = 10.0F;
        return {{"logits", {logits_.data(), {1, kVocabulary}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {
        ++prime_calls;
        captured_ = true;
    }
    void sync() override { require(cudaDeviceSynchronize() == cudaSuccess, "CUDA sync failed"); }
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override { enabled_ = true; }
    bool cuda_graph_active() const override { return graphs_supported_ && enabled_; }
    bool cuda_graph_captured() const override { return captured_; }
    int32_t profile_idx() const override { return 0; }
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
            return {kCacheLength, 1};
        if (name == "present_k_0" || name == "present_v_0")
            return {1, 1};
        if (name == "attention_mask")
            return {1, kCacheLength + 1};
        if (name == "position_id")
            return {1};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        if (name == "present_k_0")
            return const_cast<void*>(present_k_.data());
        if (name == "present_v_0")
            return const_cast<void*>(present_v_.data());
        return nullptr;
    }
    void bind_external(const std::string& name, void* pointer) override {
        if (name == "present_k_0" || name == "present_v_0")
            require(cudaMemsetAsync(pointer, 0, sizeof(float), nullptr) == cudaSuccess,
                    "initializing decoder K/V failed");
    }
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

    int calls{0};
    int prime_calls{0};

  private:
    bool prefill_, graphs_supported_, enabled_{false}, captured_{false};
    trtmc::DeviceTensor present_k_, present_v_;
    std::vector<float> logits_;
};

void exercise(bool graphs) {
    auto decoder = std::make_unique<CountingModule>(false, graphs);
    auto prefill = std::make_unique<CountingModule>(true, false);
    auto* decoder_view = decoder.get();
    auto* prefill_view = prefill.get();
    auto cache = std::make_unique<trtmc::MistralKvCache>(1, kCacheLength, 1, nullptr,
                                                         trtmc::DType::kFloat32);
    auto* cache_view = cache.get();
    trtmc::MistralTextGenConfig config;
    config.vocab_size = kVocabulary;
    config.id_eos = 31;
    config.num_layers = 1;
    config.prefill_max_length = kCacheLength;
    trtmc::MistralTextGenerationPipeline pipeline(std::move(decoder), std::move(cache), config,
                                                  std::make_shared<TestTokenizer>(),
                                                  std::move(prefill));
    trtmc::TextGenerationConfig request;
    request.temperature = 0.0F;
    request.top_k = 1;
    request.eos_token_id = 31;

    // A cold one-token request uses prefill logits without capturing a decoder graph.
    request.max_new_tokens = 1;
    require(pipeline.generate_ids({2, 3}, request).token_ids == std::vector<int32_t>({2, 3, 4}),
            "one-token request changed output");
    require(decoder_view->calls == 0 && decoder_view->prime_calls == 0,
            "one-token request launched unused decoder work");

    for (int count : {2, 20, 1, 20}) {
        const int before = decoder_view->calls;
        const int primes_before = decoder_view->prime_calls;
        const bool was_captured = decoder_view->cuda_graph_captured();
        request.max_new_tokens = count;
        std::vector<int32_t> expected{2, 3};
        for (int i = 0; i < count; ++i)
            expected.push_back(4 + i);
        require(pipeline.generate_ids({2, 3}, request).token_ids == expected,
                "request changed generated token IDs");
        require(decoder_view->calls - before == count - 1,
                "decoder computed logits after the final requested token");
        require(decoder_view->prime_calls - primes_before ==
                    (graphs && !was_captured && count > 1 ? 1 : 0),
                "request repeated an already captured decoder priming pass");
        require(cache_view->position() == 2 + count - 1,
                "request did not reset or advance the cache correctly");
    }

    // EOS still stops immediately, without a forward pass for the EOS token.
    request.max_new_tokens = 20;
    request.eos_token_id = 6;
    const int before = decoder_view->calls;
    require(pipeline.generate_ids({2, 3}, request).token_ids ==
                std::vector<int32_t>({2, 3, 4, 5, 6}),
            "EOS changed generated token IDs");
    require(decoder_view->calls - before == 2, "EOS triggered unused decoder work");

    request.max_new_tokens = 1;
    const int public_before = decoder_view->calls + decoder_view->prime_calls;
    require(pipeline.generate("prompt", request).token_ids == std::vector<int32_t>({4}),
            "public generation changed its token IDs");
    require(decoder_view->calls + decoder_view->prime_calls == public_before,
            "public one-token generation launched unused decoder work");

    request.max_new_tokens = 0;
    const int prefill_before = prefill_view->calls;
    require(pipeline.generate_ids({2, 3}, request).token_ids == std::vector<int32_t>({2, 3}),
            "zero-token request changed its prompt");
    require(prefill_view->calls == prefill_before, "zero-token request launched prefill");
    require(cudaDeviceSynchronize() == cudaSuccess, "pending KV copies failed");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "SKIP: Mistral pipeline regression requires CUDA\n";
        return 77;
    }
    try {
        exercise(true);
        exercise(false);
        std::cout << "Mistral decoder work and token parity checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
