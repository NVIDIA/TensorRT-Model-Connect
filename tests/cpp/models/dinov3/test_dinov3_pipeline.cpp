/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/dinov3/pipeline.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

class FakeDinov3Module final : public trtmc::ITrtModule {
  public:
    explicit FakeDinov3Module(bool include_pooler = true,
                              trtmc::DType io_dtype = trtmc::DType::kFloat32)
        : include_pooler_(include_pooler), io_dtype_(io_dtype) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto input = inputs.find("pixel_values");
        if (input != inputs.end()) {
            input_shape = input->second.shape;
            if (input->second.data != nullptr && input->second.dtype == trtmc::DType::kFloat32) {
                const auto* values = static_cast<const float*>(input->second.data);
                input_values.assign(values, values + input->second.numel());
            } else if (input->second.data != nullptr &&
                       input->second.dtype == trtmc::DType::kFloat16) {
                const auto* values = static_cast<const uint16_t*>(input->second.data);
                input_half_values.assign(values, values + input->second.numel());
            }
        }
        void* hidden_data = io_dtype_ == trtmc::DType::kFloat32
                                ? static_cast<void*>(hidden_.data())
                                : static_cast<void*>(hidden_half_.data());
        void* pooler_data = io_dtype_ == trtmc::DType::kFloat32
                                ? static_cast<void*>(pooler_.data())
                                : static_cast<void*>(pooler_half_.data());
        trtmc::TensorMap outputs{{"last_hidden_state", {hidden_data, {1, 3, 2}, io_dtype_}}};
        if (include_pooler_)
            outputs.emplace("pooler_output", trtmc::Tensor{pooler_data, {1, 2}, io_dtype_});
        return outputs;
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
        return {{"pixel_values", {1, 3, 1, 1}, io_dtype_, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {
            {"last_hidden_state", {1, 3, 2}, io_dtype_, false},
            {"pooler_output", {1, 2}, io_dtype_, false},
        };
    }
    bool has_input(const std::string& name) const override { return name == "pixel_values"; }
    bool has_output(const std::string& name) const override {
        return name == "last_hidden_state" || (include_pooler_ && name == "pooler_output");
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return io_dtype_; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return name == "pooler_output" ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{1, 3, 2};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {1, 3, 1, 1};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    std::vector<int64_t> input_shape;
    std::vector<float> input_values;
    std::vector<uint16_t> input_half_values;

  private:
    bool include_pooler_{true};
    trtmc::DType io_dtype_{trtmc::DType::kFloat32};
    std::vector<float> hidden_{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> pooler_{7.0F, 8.0F};
    std::vector<uint16_t> hidden_half_{0x3C00U, 0xC000U, 0x3800U, 0x0000U, 0x4000U, 0xBC00U};
    std::vector<uint16_t> pooler_half_{0x4200U, 0xC200U};
};

trtmc::Dinov3PreprocessConfig identity_config() {
    trtmc::Dinov3PreprocessConfig config;
    config.input_image_h = 1;
    config.input_image_w = 1;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};
    return config;
}

void test_pipeline_returns_both_named_outputs_with_shapes() {
    auto module = std::make_unique<FakeDinov3Module>();
    auto* module_ptr = module.get();
    trtmc::Dinov3ImageFeaturePipeline pipeline(std::move(module), identity_config(),
                                               "facebook/dinov3-vits16-pretrain-lvd1689m");
    const std::vector<float> image{0.25F, 0.5F, 0.75F};
    const auto result = pipeline.extract_image_features(image.data(), 1, 1);

    check(result.last_hidden_state_shape == std::vector<int64_t>({1, 3, 2}),
          "DINOv3 last_hidden_state shape");
    check(result.last_hidden_state == std::vector<float>({1, 2, 3, 4, 5, 6}),
          "DINOv3 last_hidden_state data");
    check(result.pooler_output_shape == std::vector<int64_t>({1, 2}), "DINOv3 pooler_output shape");
    check(result.pooler_output == std::vector<float>({7, 8}), "DINOv3 pooler_output data");
    check(module_ptr->input_shape == std::vector<int64_t>({1, 3, 1, 1}),
          "DINOv3 engine receives NCHW input");
    check(module_ptr->input_values == image, "DINOv3 engine receives channel-first values");
    check(std::string(pipeline.model_id()) == "facebook/dinov3-vits16-pretrain-lvd1689m",
          "DINOv3 model id");
}

void test_pipeline_requires_pooler_output() {
    trtmc::Dinov3ImageFeaturePipeline pipeline(std::make_unique<FakeDinov3Module>(false),
                                               identity_config());
    const std::vector<float> image(3, 0.0F);
    bool threw = false;
    try {
        (void)pipeline.extract_image_features(image.data(), 1, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "DINOv3 missing pooler_output rejected");
}

void test_pipeline_converts_fp16_engine_io() {
    auto module = std::make_unique<FakeDinov3Module>(true, trtmc::DType::kFloat16);
    auto* module_ptr = module.get();
    trtmc::Dinov3ImageFeaturePipeline pipeline(std::move(module), identity_config());
    const std::vector<float> image{1.0F, 0.5F, 0.0F};
    const auto result = pipeline.extract_image_features(image.data(), 1, 1);
    check(module_ptr->input_half_values == std::vector<uint16_t>({0x3C00U, 0x3800U, 0x0000U}),
          "DINOv3 FP16 input conversion");
    check(result.last_hidden_state == std::vector<float>({1.0F, -2.0F, 0.5F, 0.0F, 2.0F, -1.0F}),
          "DINOv3 FP16 hidden-state conversion");
    check(result.pooler_output == std::vector<float>({3.0F, -3.0F}),
          "DINOv3 FP16 pooler conversion");
}

void test_pipeline_rejects_invalid_module() {
    bool threw = false;
    try {
        trtmc::Dinov3ImageFeaturePipeline pipeline(nullptr);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "DINOv3 null module rejected");
}

} // namespace

int main() {
    test_pipeline_returns_both_named_outputs_with_shapes();
    test_pipeline_requires_pooler_output();
    test_pipeline_converts_fp16_engine_io();
    test_pipeline_rejects_invalid_module();

    if (g_failures != 0) {
        std::cerr << g_failures << " DINOv3 pipeline test(s) failed\n";
        return 1;
    }
    return 0;
}
