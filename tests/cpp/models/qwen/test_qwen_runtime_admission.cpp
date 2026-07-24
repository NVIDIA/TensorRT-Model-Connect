/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen/pipeline.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class AdmissionOnlyModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        ++execution_calls;
        throw std::logic_error("admission-only module must not execute");
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        ++execution_calls;
        throw std::logic_error("admission-only module must not execute");
    }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {
        ++execution_calls;
        throw std::logic_error("admission-only module must not execute");
    }
    void forward_async(const trtmc::TensorMap&) override {
        ++execution_calls;
        throw std::logic_error("admission-only module must not execute");
    }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return false; }
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

    int execution_calls{0};
};

class AdmissionOnlyRuntimeState final : public trtmc::QwenInferenceState {
  public:
    void reset() override { ++reset_calls; }
    void bind_to(trtmc::TrtModule&) override { ++bind_calls; }
    void prepare_step(trtmc::TensorMap&, int32_t) override { ++prepare_calls; }
    void advance(int32_t n_tokens) override {
        position_ += n_tokens;
        ++advance_calls;
    }
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return 8; }
    bool runtime_owned_kv() const override { return true; }
    int32_t prefill_chunk_limit() const override { return 4; }
    std::uint64_t runtime_kv_capacity_tokens() const override { return 8; }
    std::string runtime_memory_receipt_json() const override { return R"({"kv_allocation_id":1})"; }
    int32_t num_layers() const override { return 1; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "admission-test"; }
    bool ok() const override { return true; }

    int reset_calls{0};
    int bind_calls{0};
    int prepare_calls{0};
    int advance_calls{0};

  private:
    int32_t position_{0};
};

void test_qwen_runtime_admission_rejects_before_attention() {
    auto module = std::make_unique<AdmissionOnlyModule>();
    auto* module_observer = module.get();
    auto state = std::make_unique<AdmissionOnlyRuntimeState>();
    auto* state_observer = state.get();

    trtmc::QwenTextGenConfig config;
    config.vocab_size = 4;
    config.id_eos = 2;
    config.has_position_input = false;
    config.max_sequence_length = 6;
    config.runtime_sequence_admission = trtmc::RuntimeSequenceAdmissionContext{
        /*model_context_limit=*/8,
        /*runtime_kv_capacity_tokens=*/8,
        /*request_context_limit=*/6,
        /*kv_bytes_per_token=*/16,
        /*kv_budget_bytes=*/128,
        /*kv_reserved_bytes=*/128,
    };

    trtmc::QwenTextGenerationPipeline pipeline(std::move(module), std::move(state), config,
                                               nullptr);

    auto* qualification = dynamic_cast<trtmc::IRuntimeMemoryQualificationV1*>(&pipeline);
    check(qualification != nullptr,
          "Qwen qualification interface is discoverable across the model DSO");

    bool semantic_rejection = false;
    if (qualification != nullptr) {
        trtmc::RuntimeMemoryQualificationRequestV1 qualification_request;
        qualification_request.input_ids.assign(9, 1);
        qualification_request.max_new_tokens = 0;
        try {
            (void)qualification->qualify_runtime_memory(qualification_request);
        } catch (const trtmc::RuntimeMemoryQualificationAdmissionError& error) {
            semantic_rejection =
                std::string(error.what()).find("semantic model context limit exceeded") !=
                std::string::npos;
        }
    }
    check(semantic_rejection, "Qwen qualification reports a typed semantic context error");

    trtmc::GenerateConfig generation_request;
    generation_request.max_new_tokens = 0;
    bool policy_rejection = false;
    try {
        (void)pipeline.generate_ids(std::vector<int32_t>(7, 1), generation_request);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        policy_rejection =
            message.find("runtime max-sequence policy exceeded") != std::string::npos &&
            message.find("request_context_limit=6") != std::string::npos &&
            message.find("runtime_kv_capacity_tokens=8") != std::string::npos;
    }
    check(policy_rejection, "normal Qwen generation reports the user max-sequence policy");

    check(state_observer->reset_calls == 0,
          "Qwen admission rejects before resetting runtime state");
    check(state_observer->bind_calls == 0,
          "Qwen admission rejects before binding an attention engine");
    check(state_observer->prepare_calls == 0,
          "Qwen admission rejects before preparing an attention invocation");
    check(state_observer->advance_calls == 0, "Qwen admission rejects before advancing KV state");
    check(module_observer->execution_calls == 0,
          "Qwen admission rejects before executing the attention graph");
}

} // namespace

int main() {
    test_qwen_runtime_admission_rejects_before_attention();
    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
