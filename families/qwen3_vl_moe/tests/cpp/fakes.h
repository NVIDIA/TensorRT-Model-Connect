/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/qwen3_vl_moe/runtime/pipeline.h"

#include <map>
#include <stdexcept>
#include <string_view>

namespace test {

inline void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function, std::string_view expected = {}) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception& error) {
        require(expected.empty() ||
                    std::string_view(error.what()).find(expected) != std::string_view::npos,
                "request failed for the wrong reason");
        rejected = true;
    }
    require(rejected, "invalid request was accepted");
}

class Module final : public trtmc::ITrtModule {
  public:
    int calls{0};
    int32_t rank{3};
    std::vector<int64_t> shape{1, 2};
    std::map<std::string, void*> external;
    std::vector<float> logits{0.0F, 1.0F};
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        ++calls;
        const trtmc::Tensor tensor{logits.data(), {1, 2}, trtmc::DType::kFloat32};
        return {{"logits", tensor}, {"image_features", tensor}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return shape; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {1, 2};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string& name, void* pointer) override {
        external[name] = pointer;
    }
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>&) override {
        external[name] = pointer;
    }
    int32_t input_rank(const std::string&) const override { return rank; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

class State final : public trtmc::Qwen3VlMoeInferenceState {
  public:
    int resets{0}, current{0};
    void reset() override {
        ++resets;
        current = 0;
    }
    void bind_to(trtmc::ITrtModule&) override {}
    void prepare_step(trtmc::TensorMap&, int32_t = 1) override {}
    void advance(int32_t n = 1) override { current += n; }
    int32_t position() const override { return current; }
    int32_t max_length() const override { return 5; }
    int32_t num_layers() const override { return 1; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "test"; }
    bool ok() const override { return true; }
};

class Tokenizer final : public trtmc::ITokenizer {
  public:
    std::size_t tokens{2};
    std::vector<int32_t> encode(const std::string&) const override {
        return std::vector<int32_t>(tokens, 0);
    }
    std::string decode(const std::vector<int32_t>& ids) const override {
        return std::string(ids.size(), 'x');
    }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return "x"; }
};

} // namespace test
