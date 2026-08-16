/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/fast_foundation_stereo/stereo_pipeline.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

class FakeTrtModule final : public trtmc::ITrtModule {
  public:
    FakeTrtModule(bool has_disparity, trtmc::DType disparity_dtype,
                  std::vector<int64_t> disparity_shape)
        : has_disparity_(has_disparity), disparity_dtype_(disparity_dtype),
          disparity_shape_(std::move(disparity_shape)) {}

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string& name) const override {
        return name == "disp" && has_disparity_;
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return disparity_dtype_; }
    std::vector<int64_t> tensor_shape(const std::string&) const override {
        return disparity_shape_;
    }
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
    bool has_disparity_;
    trtmc::DType disparity_dtype_;
    std::vector<int64_t> disparity_shape_;
};

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, const char* name) {
    if (std::fabs(actual - expected) > 1.0e-6F) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++failures;
    }
}

void test_preprocess_matches_rgb_chw_replication_contract() {
    constexpr int32_t height = 700;
    constexpr int32_t width = 700;
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3, 0.0F);
    pixels[0] = 0.25F;
    pixels[1] = 0.5F;
    pixels[2] = 1.0F;
    const auto last = pixels.size() - 3;
    pixels[last] = 0.75F;
    pixels[last + 1] = 0.125F;
    pixels[last + 2] = 0.625F;

    std::vector<float> output;
    trtmc::prepare_fast_foundation_stereo_image(pixels.data(), height, width, output);
    check(output.size() == static_cast<std::size_t>(3) * 704 * 704,
          "stereo preprocess output size");
    if (output.size() != static_cast<std::size_t>(3) * 704 * 704)
        return;

    // Top-left padding replicates source pixel (0,0), then HWC becomes CHW.
    check_close(output[0], 0.25F * 255.0F, "stereo red top-left replicate");
    check_close(output[704 * 704], 0.5F * 255.0F, "stereo green top-left replicate");
    check_close(output[2 * 704 * 704], 255.0F, "stereo blue top-left replicate");

    const auto bottom_right = static_cast<std::size_t>(704) * 704 - 1;
    check_close(output[bottom_right], 0.75F * 255.0F, "stereo red bottom-right replicate");
    check_close(output[704 * 704 + bottom_right], 0.125F * 255.0F,
                "stereo green bottom-right replicate");
}

void test_preprocess_rejects_invalid_input() {
    std::vector<float> output;
    bool null_threw = false;
    try {
        trtmc::prepare_fast_foundation_stereo_image(nullptr, 700, 700, output);
    } catch (const std::invalid_argument&) {
        null_threw = true;
    }
    check(null_threw, "stereo preprocess rejects null image");

    float pixel = 0.0F;
    bool shape_threw = false;
    try {
        trtmc::prepare_fast_foundation_stereo_image(&pixel, 699, 700, output);
    } catch (const std::invalid_argument&) {
        shape_threw = true;
    }
    check(shape_threw, "stereo preprocess rejects wrong shape");
}

bool pipeline_construction_throws(bool has_disparity, trtmc::DType disparity_dtype,
                                  std::vector<int64_t> disparity_shape) {
    auto feature =
        std::make_unique<FakeTrtModule>(false, trtmc::DType::kFloat32, std::vector<int64_t>{});
    auto post =
        std::make_unique<FakeTrtModule>(has_disparity, disparity_dtype, std::move(disparity_shape));
    try {
        trtmc::FastFoundationStereoPipeline pipeline(std::move(feature), std::move(post), "test");
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_pipeline_validates_disparity_engine_contract() {
    const std::vector<int64_t> expected_shape{1, 1, 704, 704};
    check(!pipeline_construction_throws(true, trtmc::DType::kFloat32, expected_shape),
          "stereo pipeline accepts exact disparity contract");
    check(pipeline_construction_throws(false, trtmc::DType::kFloat32, expected_shape),
          "stereo pipeline rejects missing disparity output");
    check(pipeline_construction_throws(true, trtmc::DType::kFloat16, expected_shape),
          "stereo pipeline rejects disparity dtype mismatch");
    check(pipeline_construction_throws(true, trtmc::DType::kFloat32, {1, 704, 704}),
          "stereo pipeline rejects disparity shape mismatch");
}

} // namespace

int main() {
    test_preprocess_matches_rgb_chw_replication_contract();
    test_preprocess_rejects_invalid_input();
    test_pipeline_validates_disparity_engine_contract();
    if (failures == 0)
        std::cout << "All Fast Foundation Stereo preprocess tests passed\n";
    return failures == 0 ? 0 : 1;
}
