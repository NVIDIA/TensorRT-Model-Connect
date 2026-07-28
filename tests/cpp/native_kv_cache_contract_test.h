/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::test {

struct NativeKvCall {
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;
    int32_t write_index{-1};
    int32_t kv_length{-1};
};

struct NativeKvTrace {
    std::vector<NativeKvCall> calls;
};

class NativeKvModuleStub final : public ITrtModule {
  public:
    NativeKvModuleStub(cudaStream_t stream, int32_t layers, int32_t capacity, int32_t kv_heads,
                       int32_t head_dim, DType dtype, bool native = true,
                       std::shared_ptr<NativeKvTrace> trace = nullptr, int32_t profile_limit = 4,
                       int32_t vocab_size = 16)
        : stream_(stream), native_(native), trace_(std::move(trace)), vocab_size_(vocab_size) {
        if (native_) {
            add("cache_write_indices", {1}, DType::kInt32, true);
            add("key_value_lengths", {1}, DType::kInt32, true);
        }
        add("position_id", {1}, DType::kInt32, true);
        if (trace_) {
            add("token_id", {profile_limit}, DType::kInt32, true);
            add("logits", {profile_limit, vocab_size}, DType::kFloat32, false);
        }
        for (int32_t i = 0; i < layers; ++i) {
            const auto suffix = "_" + std::to_string(i);
            const std::vector<int64_t> cache_shape =
                native_ ? std::vector<int64_t>{1, kv_heads, capacity, head_dim}
                        : std::vector<int64_t>{capacity, kv_heads * head_dim};
            const std::vector<int64_t> present_shape =
                native_ ? cache_shape : std::vector<int64_t>{1, kv_heads * head_dim};
            add("cache_k" + suffix, cache_shape, dtype, true);
            add("cache_v" + suffix, cache_shape, dtype, true);
            add("present_k" + suffix, present_shape, dtype, false);
            add("present_v" + suffix, present_shape, dtype, false);
        }
    }

    void set_tensor(const std::string& name, std::vector<int64_t> shape, DType dtype) {
        auto& entry = tensors_.at(name);
        entry.shape = std::move(shape);
        entry.dtype = dtype;
    }

    TensorMap forward(const TensorMap& inputs) override {
        if (!trace_)
            return {};
        NativeKvCall call{ints(inputs, "token_id"), ints(inputs, "position_id"),
                          scalar(inputs, "cache_write_indices"),
                          scalar(inputs, "key_value_lengths")};
        trace_->calls.push_back(call);
        const int32_t rows = static_cast<int32_t>(call.tokens.size());
        logits_.assign(static_cast<std::size_t>(rows * vocab_size_), -10.0F);
        for (int32_t row = 0; row < rows; ++row) {
            const int32_t token = call.positions[static_cast<std::size_t>(row)];
            logits_[static_cast<std::size_t>(row * vocab_size_ + token)] = 10.0F;
        }
        return {{"logits", Tensor{logits_.data(), {rows, vocab_size_}, DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return has(name, true); }
    bool has_output(const std::string& name) const override { return has(name, false); }
    DType tensor_dtype(const std::string& name) const override { return tensors_.at(name).dtype; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? std::vector<int64_t>{} : it->second.shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             ProfileShapeSelector) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        const auto it = bindings_.find(name);
        return it == bindings_.end() ? nullptr : it->second;
    }
    void bind_external(const std::string& name, void* ptr) override {
        if (tensors_.count(name) == 0)
            return;
        bindings_[name] = ptr;
        if (native_ && name.rfind("cache_", 0) == 0)
            bindings_["present_" + name.substr(6)] = ptr;
    }
    int32_t input_rank(const std::string& name) const override {
        return has_input(name) ? static_cast<int32_t>(tensor_shape(name).size()) : 0;
    }
    bool input_is_dynamic(const std::string&) const override { return false; }
    bool ok() const override { return stream_ != nullptr; }
    void keep_alive(std::shared_ptr<void> value) override {
        keep_alive_.push_back(std::move(value));
    }

  private:
    struct Entry {
        std::vector<int64_t> shape;
        DType dtype;
        bool input;
    };
    void add(std::string name, std::vector<int64_t> shape, DType dtype, bool input) {
        tensors_.emplace(std::move(name), Entry{std::move(shape), dtype, input});
    }
    bool has(const std::string& name, bool input) const {
        const auto it = tensors_.find(name);
        return it != tensors_.end() && it->second.input == input;
    }
    static std::vector<int32_t> ints(const TensorMap& inputs, const std::string& name) {
        const auto& tensor = inputs.at(name);
        const auto* values = static_cast<const int32_t*>(tensor.data);
        return {values, values + tensor.numel()};
    }
    static int32_t scalar(const TensorMap& inputs, const std::string& name) {
        return ints(inputs, name).at(0);
    }

    cudaStream_t stream_;
    bool native_;
    std::shared_ptr<NativeKvTrace> trace_;
    int32_t vocab_size_;
    std::vector<float> logits_;
    std::unordered_map<std::string, Entry> tensors_;
    std::unordered_map<std::string, void*> bindings_;
    std::vector<std::shared_ptr<void>> keep_alive_;
};

inline int32_t scalar(const TensorMap& inputs, const std::string& name) {
    return *static_cast<const int32_t*>(inputs.at(name).data);
}

template <typename Cache, typename Mutate>
bool rejects_native_contract(cudaStream_t stream, Mutate mutate) {
    Cache cache(1, 11, 2, stream, DType::kFloat16);
    NativeKvModuleStub module(stream, 1, 11, 1, 2, DType::kFloat16);
    mutate(module);
    try {
        cache.bind_cache_inputs(module);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

template <typename Pipeline, typename Cache, typename Config>
int run_native_kv_contract_tests(const char* model) {
    int failures = 0;
    const auto check = [&](bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "FAIL [" << model << "]: " << message << '\n';
            ++failures;
        }
    };
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;

    check(rejects_native_contract<Cache>(
              stream,
              [](auto& module) { module.set_tensor("key_value_lengths", {2}, DType::kInt32); }),
          "rejects a non-scalar key_value_lengths input");
    check(rejects_native_contract<Cache>(
              stream,
              [](auto& module) { module.set_tensor("cache_write_indices", {1}, DType::kFloat32); }),
          "rejects a non-int32 cache_write_indices input");
    check(rejects_native_contract<Cache>(
              stream,
              [](auto& module) { module.set_tensor("cache_k_0", {1, 1, 10, 2}, DType::kFloat16); }),
          "rejects a cache with the wrong capacity");
    check(rejects_native_contract<Cache>(
              stream,
              [](auto& module) { module.set_tensor("cache_k_0", {1, 1, 11, 2}, DType::kFloat32); }),
          "rejects a cache with the wrong dtype");

    {
        Cache cache(1, 11, 2, stream, DType::kFloat16);
        NativeKvModuleStub prefill(stream, 1, 11, 1, 2, DType::kFloat16);
        NativeKvModuleStub decode(stream, 1, 11, 1, 2, DType::kFloat16);
        cache.bind_cache_inputs(prefill);
        cache.bind_to(decode);
        check(cache.ok() && cache.device_memory_bytes() == 88,
              "allocates one state-owned FP16 K/V cache");
        check(prefill.device_ptr("cache_k_0") == cache.cache_k(0).data() &&
                  prefill.device_ptr("present_k_0") == cache.cache_k(0).data(),
              "prefill cache and present K share state storage");
        check(decode.device_ptr("cache_k_0") == prefill.device_ptr("cache_k_0") &&
                  decode.device_ptr("present_v_0") == prefill.device_ptr("cache_v_0"),
              "prefill and decode contexts share the same K/V storage");

        TensorMap inputs;
        cache.prepare_step(inputs, 4);
        check(inputs.count("attention_mask") == 0 && scalar(inputs, "cache_write_indices") == 0 &&
                  scalar(inputs, "key_value_lengths") == 4,
              "native metadata describes the current write without a dense mask");

        cache.set_position(10);
        inputs.clear();
        cache.prepare_step(inputs);
        check(scalar(inputs, "cache_write_indices") == 10 &&
                  scalar(inputs, "key_value_lengths") == 11,
              "the final cache row is usable");
        cache.advance();
        bool overflow = false;
        try {
            cache.prepare_step(inputs);
        } catch (const std::runtime_error&) {
            overflow = true;
        }
        check(overflow && cache.position() == 11,
              "an over-capacity write is rejected without logical progression");
    }

    {
        Cache cache(1, 11, 2, stream, DType::kFloat16);
        NativeKvModuleStub legacy(stream, 1, 11, 1, 2, DType::kFloat16, false);
        cache.bind_to(legacy);
        TensorMap inputs;
        cache.prepare_step(inputs);
        cache.advance();
        check(cache.needs_attention_mask() && inputs.count("attention_mask") == 1 &&
                  inputs.count("cache_write_indices") == 0 && cache.position() == 1,
              "legacy attention-mask cache path still advances normally");
    }

    const auto make_config = [] {
        Config config;
        config.vocab_size = 16;
        config.id_eos = 9;
        config.disable_cuda_graph = true;
        config.prefill_max_length = 4;
        config.num_layers = 1;
        config.kv_dim = 2;
        return config;
    };
    const auto make_request = [](int32_t max_new_tokens) {
        GenerateConfig request;
        request.max_new_tokens = max_new_tokens;
        request.temperature = 0.0F;
        request.top_k = 1;
        request.seed = -1;
        return request;
    };
    const std::vector<int32_t> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    {
        auto prefill_trace = std::make_shared<NativeKvTrace>();
        auto decode_trace = std::make_shared<NativeKvTrace>();
        auto prefill = std::make_unique<NativeKvModuleStub>(stream, 1, 11, 1, 2, DType::kFloat16,
                                                            true, prefill_trace);
        auto decoder = std::make_unique<NativeKvModuleStub>(stream, 1, 11, 1, 2, DType::kFloat16,
                                                            true, decode_trace);
        auto cache = std::make_unique<Cache>(1, 11, 2, stream, DType::kFloat16);
        Cache* cache_ptr = cache.get();
        std::vector<typename Pipeline::DecoderContext> decoders;
        decoders.push_back(typename Pipeline::DecoderContext{11, std::move(decoder)});
        Pipeline pipeline(std::move(decoders), std::move(cache), make_config(), stream, nullptr, "",
                          nullptr, std::move(prefill));
        const auto result = pipeline.generate_ids(prompt, make_request(1));

        check(prefill_trace->calls.size() == 3, "split prefill runs 4/4/2 chunks");
        const int32_t starts[] = {0, 4, 8};
        const int32_t sizes[] = {4, 4, 2};
        if (prefill_trace->calls.size() == 3) {
            for (int32_t i = 0; i < 3; ++i) {
                const auto& call = prefill_trace->calls[static_cast<std::size_t>(i)];
                check(static_cast<int32_t>(call.tokens.size()) == sizes[i] &&
                          call.write_index == starts[i] && call.kv_length == starts[i] + sizes[i],
                      "chunk " + std::to_string(i) + " has correct size and KV range");
                for (int32_t j = 0; j < sizes[i]; ++j)
                    check(call.tokens[static_cast<std::size_t>(j)] == 10 + starts[i] + j &&
                              call.positions[static_cast<std::size_t>(j)] == starts[i] + j,
                          "chunk token and absolute position progress together");
            }
        }
        check(result.token_ids.size() == 11 && result.token_ids.back() == 9 &&
                  cache_ptr->position() == 10 && decode_trace->calls.empty(),
              "final prefill row supplies EOS without an extra decode");
    }

    {
        auto prefill_trace = std::make_shared<NativeKvTrace>();
        auto decode_trace = std::make_shared<NativeKvTrace>();
        auto prefill = std::make_unique<NativeKvModuleStub>(stream, 1, 11, 1, 2, DType::kFloat16,
                                                            true, prefill_trace);
        auto decoder = std::make_unique<NativeKvModuleStub>(stream, 1, 11, 1, 2, DType::kFloat16,
                                                            true, decode_trace);
        auto cache = std::make_unique<Cache>(1, 11, 2, stream, DType::kFloat16);
        Cache* cache_ptr = cache.get();
        std::vector<typename Pipeline::DecoderContext> decoders;
        decoders.push_back(typename Pipeline::DecoderContext{11, std::move(decoder)});
        Pipeline pipeline(std::move(decoders), std::move(cache), make_config(), stream, nullptr, "",
                          nullptr, std::move(prefill));
        bool overflow = false;
        try {
            (void)pipeline.generate_ids(prompt, make_request(2));
        } catch (const std::runtime_error&) {
            overflow = true;
        }
        check(overflow && prefill_trace->calls.empty() && decode_trace->calls.empty() &&
                  cache_ptr->position() == 0,
              "request over capacity is rejected before any runtime progression");
    }

    cudaStreamDestroy(stream);
    return failures;
}

} // namespace trtmc::test
