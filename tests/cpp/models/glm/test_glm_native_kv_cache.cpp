/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/glm/kv_cache.h"
#include "runtime/models/glm/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class GlmNativeKvModuleStub final : public trtmc::ITrtModule {
  public:
    GlmNativeKvModuleStub(cudaStream_t stream, int32_t capacity, bool native = true)
        : stream_(stream), native_(native) {
        if (native_) {
            add("cache_write_indices", {1}, trtmc::DType::kInt32, true);
            add("key_value_lengths", {1}, trtmc::DType::kInt32, true);
        }
        add("position_id", {1}, trtmc::DType::kInt32, true);
        const std::vector<int64_t> shape{1, 1, capacity, 2};
        add("cache_k_0", shape, trtmc::DType::kBFloat16, true);
        add("cache_v_0", shape, trtmc::DType::kBFloat16, true);
        add("present_k_0", shape, trtmc::DType::kBFloat16, false);
        add("present_v_0", shape, trtmc::DType::kBFloat16, false);
    }

    void set_tensor(const std::string& name, std::vector<int64_t> shape, trtmc::DType dtype) {
        auto& tensor = tensors_.at(name);
        tensor.shape = std::move(shape);
        tensor.dtype = dtype;
    }

    void add_attention_mask() { add("attention_mask", {1, 1}, trtmc::DType::kFloat32, true); }

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return has(name, true); }
    bool has_output(const std::string& name) const override { return has(name, false); }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return tensors_.at(name).dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto found = tensors_.find(name);
        return found == tensors_.end() ? std::vector<int64_t>{} : found->second.shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        const auto found = bindings_.find(name);
        return found == bindings_.end() ? nullptr : found->second;
    }
    void bind_external(const std::string& name, void* pointer) override {
        if (tensors_.count(name) == 0)
            return;
        bindings_[name] = pointer;
        if (native_ && name.rfind("cache_", 0) == 0)
            bindings_["present_" + name.substr(6)] = pointer;
    }
    int32_t input_rank(const std::string& name) const override {
        return has_input(name) ? static_cast<int32_t>(tensor_shape(name).size()) : 0;
    }
    bool input_is_dynamic(const std::string&) const override { return false; }
    bool ok() const override { return stream_ != nullptr; }
    void keep_alive(std::shared_ptr<void> owner) override {
        keep_alive_.push_back(std::move(owner));
    }

  private:
    struct Entry {
        std::vector<int64_t> shape;
        trtmc::DType dtype;
        bool input;
    };

    void add(std::string name, std::vector<int64_t> shape, trtmc::DType dtype, bool input) {
        tensors_.emplace(std::move(name), Entry{std::move(shape), dtype, input});
    }

    bool has(const std::string& name, bool input) const {
        const auto found = tensors_.find(name);
        return found != tensors_.end() && found->second.input == input;
    }

    cudaStream_t stream_;
    bool native_;
    std::unordered_map<std::string, Entry> tensors_;
    std::unordered_map<std::string, void*> bindings_;
    std::vector<std::shared_ptr<void>> keep_alive_;
};

int32_t scalar(const trtmc::TensorMap& inputs, const std::string& name) {
    return *static_cast<const int32_t*>(inputs.at(name).data);
}

bool throws_runtime_error(const std::function<void()>& callable) {
    try {
        callable();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool throws_invalid_argument(const std::function<void()>& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL [GLM]: " << message << '\n';
            ++failures;
        }
    };

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;

    check(throws_invalid_argument(
              [&] { trtmc::GlmKvCache cache(1, 11, 2, stream, trtmc::DType::kFloat16); }),
          "rejects non-BF16 allocation");
    check(throws_invalid_argument(
              [&] { trtmc::GlmKvCache cache(0, 11, 2, stream, trtmc::DType::kBFloat16); }),
          "rejects invalid geometry");

    {
        trtmc::GlmKvCache cache(1, 11, 2, stream, trtmc::DType::kBFloat16);
        GlmNativeKvModuleStub legacy(stream, 11, false);
        check(throws_runtime_error([&] { cache.bind_to(legacy); }),
              "rejects an engine without native scalar controls");

        GlmNativeKvModuleStub legacy_mask(stream, 11);
        legacy_mask.add_attention_mask();
        check(throws_runtime_error([&] { cache.bind_to(legacy_mask); }),
              "rejects a legacy attention mask");
    }

    {
        trtmc::GlmKvCache cache(1, 11, 2, stream, trtmc::DType::kBFloat16);
        GlmNativeKvModuleStub bad_scalar(stream, 11);
        bad_scalar.set_tensor("cache_write_indices", {2}, trtmc::DType::kInt32);
        check(throws_runtime_error([&] { cache.bind_to(bad_scalar); }),
              "rejects a malformed native scalar");

        GlmNativeKvModuleStub bad_cache(stream, 11);
        bad_cache.set_tensor("cache_k_0", {1, 1, 10, 2}, trtmc::DType::kBFloat16);
        check(throws_runtime_error([&] { cache.bind_to(bad_cache); }),
              "rejects a cache smaller than model context");
    }

    {
        trtmc::GlmKvCache cache(1, 11, 2, stream, trtmc::DType::kBFloat16);
        GlmNativeKvModuleStub prefill(stream, 11);
        GlmNativeKvModuleStub decode(stream, 11);
        cache.bind_to(prefill);
        cache.bind_to(decode);
        check(cache.ok() && cache.device_memory_bytes() == 88,
              "allocates one full-capacity BF16 K/V buffer");
        check(prefill.device_ptr("cache_k_0") == cache.cache_k(0).data() &&
                  prefill.device_ptr("present_k_0") == cache.cache_k(0).data() &&
                  decode.device_ptr("cache_v_0") == cache.cache_v(0).data() &&
                  decode.device_ptr("present_v_0") == cache.cache_v(0).data(),
              "prefill, decode, cache, and present tensors alias one buffer");

        const std::vector<const void*> present_k{cache.cache_k(0).data()};
        const std::vector<const void*> present_v{cache.cache_v(0).data()};
        trtmc::TensorMap inputs;

        cache.prepare_step(inputs, 4);
        check(inputs.count("attention_mask") == 0 && scalar(inputs, "cache_write_indices") == 0 &&
                  scalar(inputs, "key_value_lengths") == 4,
              "first prefill chunk starts at zero without a mask");
        cache.append_prefill_kv(present_k, present_v, 4);

        inputs.clear();
        cache.prepare_step(inputs, 4);
        check(scalar(inputs, "cache_write_indices") == 4 &&
                  scalar(inputs, "key_value_lengths") == 8,
              "second prefill chunk advances native scalar inputs");
        cache.append_prefill_kv(present_k, present_v, 4);

        inputs.clear();
        cache.prepare_step(inputs, 2);
        check(scalar(inputs, "cache_write_indices") == 8 &&
                  scalar(inputs, "key_value_lengths") == 10,
              "final short chunk retains exact active length");
        cache.append_prefill_kv(present_k, present_v, 2);

        inputs.clear();
        cache.prepare_step(inputs);
        check(scalar(inputs, "cache_write_indices") == 10 &&
                  scalar(inputs, "key_value_lengths") == 11,
              "decode writes the final context row");
        cache.advance();
        check(cache.position() == 11 && throws_runtime_error([&] { cache.prepare_step(inputs); }),
              "overflow fails before cache progression");

        const std::vector<const void*> wrong_pointer{nullptr};
        cache.reset();
        check(throws_runtime_error([&] { cache.append_prefill_kv(wrong_pointer, present_v, 1); }),
              "prefill completion requires present/cache aliasing");
    }

    {
        auto decoder = std::make_unique<GlmNativeKvModuleStub>(stream, 11);
        auto prefill = std::make_unique<GlmNativeKvModuleStub>(stream, 11);
        auto cache = std::make_unique<trtmc::GlmKvCache>(1, 11, 2, stream, trtmc::DType::kBFloat16);
        trtmc::GlmTextGenConfig generation;
        generation.vocab_size = 16;
        generation.id_eos = 9;
        generation.disable_cuda_graph = true;
        generation.prefill_max_length = 4;
        generation.num_layers = 1;
        trtmc::GlmTextGenerationPipeline pipeline(std::move(decoder), std::move(prefill),
                                                  std::move(cache), std::move(generation), stream,
                                                  nullptr);

        trtmc::GenerateConfig unsupported;
        unsupported.max_new_tokens = 0;
        unsupported.text_generation_mode = "diffusion";
        check(throws_runtime_error([&] { (void)pipeline.generate_ids({1}, unsupported); }),
              "unsupported generation mode fails closed for a zero-token request");

        unsupported.max_new_tokens = 1;
        check(throws_runtime_error([&] { (void)pipeline.generate_ids({}, unsupported); }),
              "unsupported generation mode fails closed for an empty prompt");
    }

    cudaStreamDestroy(stream);
    return failures;
}
