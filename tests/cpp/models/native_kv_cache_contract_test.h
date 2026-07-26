/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

class NativeKvModuleStub final : public ITrtModule {
  public:
    NativeKvModuleStub(int32_t num_layers, int32_t capacity, int32_t num_kv_heads, int32_t head_dim,
                       DType dtype, bool native_contract = true)
        : stream_(nullptr) {
        cudaStreamCreate(&stream_);
        if (native_contract) {
            add_input("cache_write_indices", {1}, DType::kInt32);
            add_input("key_value_lengths", {1}, DType::kInt32);
        }
        add_input("position_id", {1}, DType::kInt32);
        for (int32_t i = 0; i < num_layers; ++i) {
            const std::string suffix = "_" + std::to_string(i);
            const std::vector<int64_t> shape{1, num_kv_heads, capacity, head_dim};
            add_input("cache_k" + suffix, shape, dtype);
            add_input("cache_v" + suffix, shape, dtype);
            add_output("present_k" + suffix, shape, dtype);
            add_output("present_v" + suffix, shape, dtype);
        }
    }

    ~NativeKvModuleStub() override { cudaStreamDestroy(stream_); }

    TensorMap forward(const TensorMap&) override { return {}; }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }

    bool has_input(const std::string& name) const override {
        const auto it = tensors_.find(name);
        return it != tensors_.end() && it->second.is_input;
    }

    bool has_output(const std::string& name) const override {
        const auto it = tensors_.find(name);
        return it != tensors_.end() && !it->second.is_input;
    }

    DType tensor_dtype(const std::string& name) const override {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? DType::kFloat32 : it->second.dtype;
    }

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
        if (name.rfind("cache_", 0) == 0) {
            const auto separator = name.find('_', 6);
            if (separator != std::string::npos) {
                const std::string present =
                    "present_" + name.substr(6, separator - 6) + name.substr(separator);
                if (tensors_.count(present) != 0)
                    bindings_[present] = ptr;
            }
        }
    }

    int32_t input_rank(const std::string& name) const override {
        return has_input(name) ? static_cast<int32_t>(tensor_shape(name).size()) : 0;
    }

    bool input_is_dynamic(const std::string&) const override { return false; }
    bool ok() const override { return stream_ != nullptr; }
    void keep_alive(std::shared_ptr<void> resource) override {
        keep_alive_.push_back(std::move(resource));
    }

  private:
    struct Entry {
        bool is_input;
        std::vector<int64_t> shape;
        DType dtype;
    };

    void add_input(std::string name, std::vector<int64_t> shape, DType dtype) {
        tensors_.emplace(std::move(name), Entry{true, std::move(shape), dtype});
    }

    void add_output(std::string name, std::vector<int64_t> shape, DType dtype) {
        tensors_.emplace(std::move(name), Entry{false, std::move(shape), dtype});
    }

    cudaStream_t stream_;
    std::unordered_map<std::string, Entry> tensors_;
    std::unordered_map<std::string, void*> bindings_;
    std::vector<std::shared_ptr<void>> keep_alive_;
};

inline int32_t scalar_input(const TensorMap& inputs, const std::string& name) {
    const auto it = inputs.find(name);
    if (it == inputs.end() || it->second.data == nullptr)
        throw std::runtime_error("missing scalar input " + name);
    return *static_cast<const int32_t*>(it->second.data);
}

template <typename Cache>
int run_native_kv_cache_contract_test(int32_t capacity, const char* model_name) {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL [" << model_name << "]: " << message << '\n';
            ++failures;
        }
    };

    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kNumKvHeads = 2;
    constexpr int32_t kHeadDim = 4;
    constexpr int32_t kKvDim = kNumKvHeads * kHeadDim;
    constexpr DType kDtype = DType::kFloat16;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "FAIL [" << model_name << "]: could not create CUDA stream\n";
        return 1;
    }

    {
        Cache cache(kNumLayers, capacity, kKvDim, stream, kDtype);
        NativeKvModuleStub prefill(kNumLayers, capacity, kNumKvHeads, kHeadDim, kDtype);
        cache.bind_cache_inputs(prefill);

        const std::size_t expected_bytes = static_cast<std::size_t>(kNumLayers) * 2U *
                                           static_cast<std::size_t>(capacity) *
                                           static_cast<std::size_t>(kKvDim) * 2U;
        check(cache.device_memory_bytes() == expected_bytes,
              "runtime allocates exactly the complete K/V capacity");
        check(!cache.needs_attention_mask(), "key_value_lengths replaces the dense attention mask");

        for (int32_t i = 0; i < kNumLayers; ++i) {
            const std::string suffix = "_" + std::to_string(i);
            check(prefill.device_ptr("cache_k" + suffix) == cache.cache_k(i).data(),
                  "prefill cache_k uses state-owned storage");
            check(prefill.device_ptr("present_k" + suffix) == cache.cache_k(i).data(),
                  "prefill present_k aliases cache_k");
            check(prefill.device_ptr("cache_v" + suffix) == cache.cache_v(i).data(),
                  "prefill cache_v uses state-owned storage");
            check(prefill.device_ptr("present_v" + suffix) == cache.cache_v(i).data(),
                  "prefill present_v aliases cache_v");
        }

        TensorMap prefill_inputs;
        cache.prepare_step(prefill_inputs, 5);
        check(prefill_inputs.count("attention_mask") == 0,
              "native prefill does not materialize a dense attention mask");
        check(scalar_input(prefill_inputs, "cache_write_indices") == 0,
              "prefill writes at cache row zero");
        check(scalar_input(prefill_inputs, "key_value_lengths") == 5,
              "prefill publishes its post-update active length");

        std::vector<const void*> present_k;
        std::vector<const void*> present_v;
        for (int32_t i = 0; i < kNumLayers; ++i) {
            const std::string suffix = "_" + std::to_string(i);
            present_k.push_back(prefill.device_ptr("present_k" + suffix));
            present_v.push_back(prefill.device_ptr("present_v" + suffix));
        }
        cache.write_prefill_kv(present_k, present_v, 5);
        check(cache.position() == 5, "prefill advances only the logical position");

        NativeKvModuleStub decode(kNumLayers, capacity, kNumKvHeads, kHeadDim, kDtype);
        cache.bind_to(decode);
        for (int32_t i = 0; i < kNumLayers; ++i) {
            const std::string suffix = "_" + std::to_string(i);
            check(decode.device_ptr("cache_k" + suffix) == prefill.device_ptr("cache_k" + suffix),
                  "prefill and decode share cache_k storage");
            check(decode.device_ptr("present_k" + suffix) == prefill.device_ptr("cache_k" + suffix),
                  "decode present_k aliases shared cache_k");
            check(decode.device_ptr("cache_v" + suffix) == prefill.device_ptr("cache_v" + suffix),
                  "prefill and decode share cache_v storage");
            check(decode.device_ptr("present_v" + suffix) == prefill.device_ptr("cache_v" + suffix),
                  "decode present_v aliases shared cache_v");
        }

        TensorMap decode_inputs;
        cache.prepare_step(decode_inputs);
        check(scalar_input(decode_inputs, "cache_write_indices") == 5,
              "decode writes at the current logical position");
        check(scalar_input(decode_inputs, "key_value_lengths") == 6,
              "decode publishes its post-update active length");
        cache.advance();
        check(cache.position() == 6, "native decode advances without a cache copy");
        check(cache.device_memory_bytes() == expected_bytes,
              "binding a second context does not allocate another cache");

        bool bidirectional_rejected = false;
        try {
            TensorMap inputs;
            cache.prepare_bidirectional_step(inputs, 2);
        } catch (const std::runtime_error&) {
            bidirectional_rejected = true;
        }
        check(bidirectional_rejected, "native causal KV mode rejects bidirectional block decoding");

        cache.set_position(capacity);
        bool overflow_rejected = false;
        try {
            TensorMap inputs;
            cache.prepare_step(inputs);
        } catch (const std::runtime_error&) {
            overflow_rejected = true;
        }
        check(overflow_rejected, "runtime rejects writes beyond fixed capacity");

        NativeKvModuleStub legacy(kNumLayers, capacity, kNumKvHeads, kHeadDim, kDtype, false);
        bool mixed_contract_rejected = false;
        try {
            cache.bind_to(legacy);
        } catch (const std::runtime_error&) {
            mixed_contract_rejected = true;
        }
        check(mixed_contract_rejected,
              "prefill and decode cannot mix native and legacy cache contracts");
    }

    cudaStreamDestroy(stream);
    return failures;
}

} // namespace trtmc::test
