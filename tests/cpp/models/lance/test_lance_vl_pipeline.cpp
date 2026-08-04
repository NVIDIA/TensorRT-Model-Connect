/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Lance native-KV contract and split text/VL prefill tests.

#include "runtime/models/lance/kv_cache.h"
#include "runtime/models/lance/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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

template <typename T>
std::vector<T> host_values(const trtmc::Tensor& tensor) {
    const auto* begin = static_cast<const T*>(tensor.data);
    return std::vector<T>(begin, begin + static_cast<std::ptrdiff_t>(tensor.numel()));
}

struct NativeTextStats {
    int32_t calls{0};
    std::vector<std::vector<int32_t>> token_history;
    std::vector<std::vector<int32_t>> position_history;
    std::vector<std::vector<int32_t>> mrope_history;
    std::vector<int32_t> write_index_history;
    std::vector<int32_t> kv_length_history;
    std::vector<std::vector<float>> mask_history;
    std::vector<std::vector<int64_t>> mask_shape_history;
    std::vector<std::vector<float>> embed_selector_history;
};

class NativeTextModule final : public trtmc::ITrtModule {
  public:
    NativeTextModule(std::shared_ptr<NativeTextStats> stats, bool prefill, cudaStream_t stream,
                     int32_t capacity = 16, int32_t prefill_limit = 8, bool mrope = false,
                     bool native_scalars = true)
        : stats_(std::move(stats)), prefill_(prefill), stream_(stream), capacity_(capacity),
          prefill_limit_(prefill_limit), mrope_(mrope), native_scalars_(native_scalars) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++stats_->calls;
        stats_->token_history.push_back(host_values<int32_t>(inputs.at("token_id")));
        stats_->position_history.push_back(host_values<int32_t>(inputs.at("position_id")));
        stats_->write_index_history.push_back(
            host_values<int32_t>(inputs.at("cache_write_indices")).front());
        stats_->kv_length_history.push_back(
            host_values<int32_t>(inputs.at("key_value_lengths")).front());
        if (const auto it = inputs.find("mrope_position_ids"); it != inputs.end())
            stats_->mrope_history.push_back(host_values<int32_t>(it->second));
        if (const auto it = inputs.find("attention_mask"); it != inputs.end()) {
            stats_->mask_history.push_back(host_values<float>(it->second));
            stats_->mask_shape_history.push_back(it->second.shape);
        }
        if (const auto it = inputs.find("use_input_embed"); it != inputs.end())
            stats_->embed_selector_history.push_back(host_values<float>(it->second));
        return {{"logits", trtmc::Tensor{logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }

    bool has_input(const std::string& name) const override {
        if (name == "attention_mask")
            return prefill_;
        if (name == "mrope_position_ids")
            return mrope_;
        if (name == "cache_write_indices" || name == "key_value_lengths")
            return native_scalars_;
        return name == "token_id" || name == "position_id" || name == "input_embed" ||
               name == "use_input_embed" || name == "cache_k_0" || name == "cache_v_0";
    }

    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }

    trtmc::DType tensor_dtype(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0" || name == "present_k_0" ||
            name == "present_v_0")
            return trtmc::DType::kBFloat16;
        if (name == "token_id" || name == "position_id" || name == "mrope_position_ids" ||
            name == "cache_write_indices" || name == "key_value_lengths")
            return trtmc::DType::kInt32;
        return trtmc::DType::kFloat32;
    }

    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0" || name == "present_k_0" ||
            name == "present_v_0")
            return {1, 1, capacity_, 4};
        if (name == "cache_write_indices" || name == "key_value_lengths")
            return {1};
        if (name == "attention_mask")
            return {-1, -1};
        if (name == "mrope_position_ids")
            return {3, -1};
        if (name == "position_id" || name == "token_id")
            return {-1};
        if (name == "input_embed")
            return {-1, 4};
        if (name == "use_input_embed")
            return {-1, 1};
        return {};
    }

    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        if (name == "token_id")
            return {prefill_ ? prefill_limit_ : 1};
        return tensor_shape(name);
    }

    int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) const override {
        const auto it = bindings_.find(name);
        return it == bindings_.end() ? nullptr : it->second;
    }

    void bind_external(const std::string& name, void* pointer) override {
        bindings_[name] = pointer;
        if (name == "cache_k_0")
            bindings_["present_k_0"] = pointer;
        if (name == "cache_v_0")
            bindings_["present_v_0"] = pointer;
    }

    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensor_shape(name).size());
    }
    bool input_is_dynamic(const std::string&) const override { return false; }
    bool ok() const override { return stream_ != nullptr; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::shared_ptr<NativeTextStats> stats_;
    bool prefill_{false};
    cudaStream_t stream_{nullptr};
    int32_t capacity_{16};
    int32_t prefill_limit_{8};
    bool mrope_{false};
    bool native_scalars_{true};
    std::unordered_map<std::string, void*> bindings_;
    std::vector<float> logits_{0.1F, 0.2F, 0.9F, 0.3F};
};

class FakeVisionModule final : public trtmc::ITrtModule {
  public:
    FakeVisionModule() : features_(16, 1.0F) {}
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        return {
            {"image_features", trtmc::Tensor{features_.data(), {4, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {{"pixel_values", {3, 4, 4}, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"image_features", {4, 4}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override { return name == "pixel_values"; }
    bool has_output(const std::string& name) const override { return name == "image_features"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<float> features_;
};

class BlockTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override {
        return {9, 8, 1, 1, 1, 1, 7, 6};
    }
    std::string decode(const std::vector<int32_t>&) const override { return "out"; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }
};

trtmc::LanceConfig config(int32_t prefill_limit) {
    trtmc::LanceConfig value;
    value.vocab_size = 4;
    value.id_eos = 2;
    value.image_token_id = 1;
    value.vision_output_dim = 4;
    value.num_layers = 1;
    value.prefill_max_length = prefill_limit;
    return value;
}

void test_native_cache_contract_and_segmented_mask() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    auto prefill_stats = std::make_shared<NativeTextStats>();
    auto decode_stats = std::make_shared<NativeTextStats>();
    NativeTextModule prefill(prefill_stats, true, stream, 8, 8, true);
    NativeTextModule decode(decode_stats, false, stream, 8, 1, true);
    trtmc::LanceKvCache cache(1, 8, 4, stream, trtmc::DType::kBFloat16);

    cache.bind_cache_inputs(prefill);
    check(prefill.device_ptr("present_k_0") == cache.cache_k(0).data() &&
              prefill.device_ptr("present_v_0") == cache.cache_v(0).data(),
          "native cache: prefill outputs alias user buffers");
    trtmc::TensorMap inputs;
    cache.prepare_prefill_with_bidirectional_block(inputs, 4, 1, 3);
    check(inputs.at("attention_mask").shape == std::vector<int64_t>({4, 4}),
          "native cache: prefill mask uses active width");
    const auto mask = host_values<float>(inputs.at("attention_mask"));
    check(mask[1 * 4 + 2] == 0.0F && mask[0 * 4 + 1] < -1000.0F,
          "native cache: only the vision sub-block is bidirectional");
    check(host_values<int32_t>(inputs.at("cache_write_indices")).front() == 0 &&
              host_values<int32_t>(inputs.at("key_value_lengths")).front() == 4,
          "native cache: prefill scalars identify write range");

    cache.write_prefill_kv({prefill.device_ptr("present_k_0")}, {prefill.device_ptr("present_v_0")},
                           4);
    cache.bind_to(decode);
    inputs.clear();
    cache.prepare_step(inputs);
    check(inputs.count("attention_mask") == 0 &&
              host_values<int32_t>(inputs.at("cache_write_indices")).front() == 4 &&
              host_values<int32_t>(inputs.at("key_value_lengths")).front() == 5,
          "native cache: decode uses fused causal mask and active length");

    NativeTextModule legacy(std::make_shared<NativeTextStats>(), false, stream, 8, 1, false, false);
    bool rejected = false;
    try {
        cache.bind_to(legacy);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "native cache: removed legacy scalar contract fails closed");
    cudaStreamDestroy(stream);
}

void test_text_prefill_chunks_and_overflow() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    auto decode_stats = std::make_shared<NativeTextStats>();
    auto prefill_stats = std::make_shared<NativeTextStats>();
    auto decoder = std::make_unique<NativeTextModule>(decode_stats, false, stream, 8, 1);
    auto prefill = std::make_unique<NativeTextModule>(prefill_stats, true, stream, 8, 2);
    auto cache = std::make_unique<trtmc::LanceKvCache>(1, 8, 4, stream, trtmc::DType::kBFloat16);
    trtmc::LancePreprocessConfig preprocess;
    trtmc::LancePipeline pipeline(std::move(decoder), nullptr, std::move(cache), config(2),
                                  preprocess, stream, nullptr, "", nullptr, std::move(prefill));

    trtmc::GenerateConfig request;
    request.max_new_tokens = 1;
    request.eos_token_id = 99;
    auto result = pipeline.generate_ids({0, 1, 3}, request);
    check(result.token_ids == std::vector<int32_t>({0, 1, 3, 2}),
          "text chunks: generation remains correct");
    check(prefill_stats->token_history == std::vector<std::vector<int32_t>>({{0, 1}, {3}}) &&
              prefill_stats->position_history == std::vector<std::vector<int32_t>>({{0, 1}, {2}}),
          "text chunks: token order and absolute positions are preserved");
    check(prefill_stats->write_index_history == std::vector<int32_t>({0, 2}) &&
              prefill_stats->kv_length_history == std::vector<int32_t>({2, 3}) &&
              decode_stats->calls == 0,
          "text chunks: native KV ranges advance without prompt-linear decode");

    bool overflow = false;
    try {
        (void)pipeline.generate_ids({0, 1, 2, 3, 4, 5, 6, 7}, request);
    } catch (const std::runtime_error&) {
        overflow = true;
    }
    check(overflow && prefill_stats->calls == 2,
          "context overflow is rejected before another enqueue");

    request.max_new_tokens = 2;
    request.eos_token_id = 99;
    (void)pipeline.generate_ids({0, 1, 3}, request);
    check(decode_stats->calls == 1,
          "text decode: the final sampled token does not launch unused decode");
    cudaStreamDestroy(stream);
}

void test_vl_prefill_keeps_vision_span_whole() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    auto decode_stats = std::make_shared<NativeTextStats>();
    auto prefill_stats = std::make_shared<NativeTextStats>();
    auto decoder = std::make_unique<NativeTextModule>(decode_stats, false, stream, 16, 1, true);
    auto prefill = std::make_unique<NativeTextModule>(prefill_stats, true, stream, 16, 6, true);
    auto vision = std::make_unique<FakeVisionModule>();
    auto cache = std::make_unique<trtmc::LanceKvCache>(1, 16, 4, stream, trtmc::DType::kBFloat16);
    trtmc::LancePreprocessConfig preprocess;
    preprocess.preprocessor_type = "simple_chw";
    preprocess.fixed_image_size = 4;
    preprocess.in_channels = 3;
    preprocess.patch_size = 1;
    preprocess.merge_size = 1;
    auto tokenizer = std::make_shared<BlockTokenizer>();
    trtmc::LancePipeline pipeline(std::move(decoder), std::move(vision), std::move(cache),
                                  config(6), preprocess, stream, tokenizer, "", nullptr,
                                  std::move(prefill));

    float pixels[2 * 2 * 3] = {0.5F, 0.5F, 0.5F, 0.4F, 0.4F, 0.4F,
                               0.3F, 0.3F, 0.3F, 0.2F, 0.2F, 0.2F};
    trtmc::GenerateConfig request;
    request.max_new_tokens = 1;
    request.eos_token_id = 99;
    auto result = pipeline.generate("test", pixels, 2, 2, request);
    check(result.token_ids == std::vector<int32_t>({2}), "VL chunks: generation remains correct");
    check(prefill_stats->token_history ==
              std::vector<std::vector<int32_t>>({{9}, {8, 1, 1, 1, 1, 7}, {6}}),
          "VL chunks: bidirectional vision span is never split");
    check(prefill_stats->write_index_history == std::vector<int32_t>({0, 1, 7}) &&
              prefill_stats->kv_length_history == std::vector<int32_t>({1, 7, 8}),
          "VL chunks: native cache ranges follow chunk boundaries");
    check(prefill_stats->embed_selector_history.size() == 3 &&
              prefill_stats->embed_selector_history[1] ==
                  std::vector<float>({0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.0F}),
          "VL chunks: image features stay aligned with placeholders");
    check(prefill_stats->mask_shape_history[1] == std::vector<int64_t>({6, 7}) &&
              std::all_of(prefill_stats->mask_history[1].begin() + 7,
                          prefill_stats->mask_history[1].begin() + 14,
                          [](float value) { return value == 0.0F; }),
          "VL chunks: vision query sees the complete active vision span");
    check(prefill_stats->mrope_history.size() == 3 && prefill_stats->mrope_history[1].size() == 18,
          "VL chunks: mRoPE axes are sliced with each chunk");
    check(decode_stats->calls == 0, "VL chunks: no prompt-linear decode launches");
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_native_cache_contract_and_segmented_mask();
    test_text_prefill_chunks_and_overflow();
    test_vl_prefill_keeps_vision_span_whole();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
