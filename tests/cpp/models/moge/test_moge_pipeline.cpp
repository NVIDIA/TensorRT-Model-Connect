/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/moge/pipeline.h"

#include <algorithm>
#include <cmath>
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

bool close(float actual, float expected, float tolerance = 2.0e-3F) {
    return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

std::vector<float> affine_points(int32_t height, int32_t width, float focal, float shift) {
    const float aspect = static_cast<float>(width) / height;
    const float diagonal = std::sqrt(1.0F + aspect * aspect);
    const float span_x = aspect / diagonal;
    const float span_y = 1.0F / diagonal;
    std::vector<float> points(static_cast<std::size_t>(height) * width * 3U);
    for (int32_t y = 0; y < height; ++y) {
        const float v = span_y * (2.0F * y + 1.0F - height) / height;
        for (int32_t x = 0; x < width; ++x) {
            const float u = span_x * (2.0F * x + 1.0F - width) / width;
            const float z = 0.8F + 0.03F * x + 0.02F * y;
            const auto offset = (static_cast<std::size_t>(y) * width + x) * 3U;
            points[offset] = u * (z + shift) / focal;
            points[offset + 1U] = v * (z + shift) / focal;
            points[offset + 2U] = z;
        }
    }
    return points;
}

class FakeMogeModule final : public trtmc::ITrtModule {
  public:
    FakeMogeModule(int32_t height, int32_t width, bool invalidate_pixel = false)
        : height_(height), width_(width), points_(affine_points(height, width, 0.8F, 1.25F)),
          mask_(static_cast<std::size_t>(height) * width, 0.9F) {
        if (invalidate_pixel)
            mask_[5] = 0.1F;
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto image = inputs.find("image");
        if (image != inputs.end()) {
            input_shape = image->second.shape;
            const auto* values = static_cast<const float*>(image->second.data);
            input_values.assign(values, values + image->second.numel());
        }
        return {
            {"points", {points_.data(), {1, height_, width_, 3}, trtmc::DType::kFloat32}},
            {"mask", {mask_.data(), {1, height_, width_}, trtmc::DType::kFloat32}},
            {"metric_scale", {scale_.data(), {1}, trtmc::DType::kFloat32}},
        };
    }
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
    bool has_input(const std::string& name) const override { return name == "image"; }
    bool has_output(const std::string& name) const override {
        return name == "points" || name == "mask" || name == "metric_scale";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "image")
            return {1, height_, width_, 3};
        if (name == "points")
            return {1, height_, width_, 3};
        if (name == "mask")
            return {1, height_, width_};
        if (name == "metric_scale")
            return {1};
        throw std::runtime_error("unknown fake tensor");
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {1, height_, width_, 3};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    void invalidate_all() { std::fill(mask_.begin(), mask_.end(), 0.1F); }

    std::vector<int64_t> input_shape;
    std::vector<float> input_values;

  private:
    int32_t height_;
    int32_t width_;
    std::vector<float> points_;
    std::vector<float> mask_;
    std::vector<float> scale_{2.0F};
};

std::vector<float> rgb_image(int32_t height, int32_t width) {
    std::vector<float> image(static_cast<std::size_t>(height) * width * 3U);
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(height) * width; ++pixel) {
        image[pixel * 3U] = 0.1F;
        image[pixel * 3U + 1U] = 0.2F;
        image[pixel * 3U + 2U] = 0.3F;
    }
    return image;
}

void test_pipeline_recovers_metric_geometry_from_hwc_input() {
    constexpr int32_t height = 64;
    constexpr int32_t width = 80;
    auto module = std::make_unique<FakeMogeModule>(height, width);
    auto* module_ptr = module.get();
    trtmc::MogePipeline pipeline(std::move(module), "moge-2-vitl");
    auto image = rgb_image(height, width);

    const auto result = pipeline.estimate_geometry(image.data(), height, width);

    check(result.height == height && result.width == width, "MoGe result dimensions");
    check(result.points.size() == static_cast<std::size_t>(height) * width * 3U,
          "MoGe point-map size");
    check(result.depth.size() == static_cast<std::size_t>(height) * width, "MoGe depth-map size");
    check(result.mask.size() == result.depth.size(), "MoGe mask size");
    check(close(result.intrinsics[0], 0.5122499F), "MoGe recovered normalized fx");
    check(close(result.intrinsics[4], 0.6403124F), "MoGe recovered normalized fy");
    const std::size_t pixel = 7U * width + 11U;
    const float expected_depth = (0.8F + 0.03F * 11 + 0.02F * 7 + 1.25F) * 2.0F;
    check(close(result.depth[pixel], expected_depth), "MoGe applies shift and metric scale");
    check(close(result.points[pixel * 3U + 2U], result.depth[pixel]), "MoGe point z equals depth");
    check(result.mask[pixel] == 1, "MoGe valid mask retained");
    check(module_ptr->input_shape == std::vector<int64_t>({1, height, width, 3}),
          "MoGe engine receives HWC input");
    check(module_ptr->input_values.size() == static_cast<std::size_t>(height) * width * 3U,
          "MoGe engine receives full HWC payload");
    check(module_ptr->input_values.size() >= 3U && module_ptr->input_values[0] == 0.1F &&
              module_ptr->input_values[1] == 0.2F && module_ptr->input_values[2] == 0.3F,
          "MoGe input stays interleaved");
    check(dynamic_cast<trtmc::moge::IGeometryEstimator*>(&pipeline) != nullptr,
          "MoGe exposes its family-owned geometry contract");
    check(std::string(pipeline.model_id()) == "moge-2-vitl", "MoGe model id");
}

void test_invalid_mask_materializes_infinity() {
    constexpr int32_t size = 64;
    auto module = std::make_unique<FakeMogeModule>(size, size, true);
    trtmc::MogePipeline pipeline(std::move(module), "moge-2-vitl");
    auto image = rgb_image(size, size);

    const auto result = pipeline.estimate_geometry(image.data(), size, size);

    check(result.mask[5] == 0, "MoGe invalid pixel mask cleared");
    check(std::isinf(result.depth[5]), "MoGe invalid depth is infinity");
    check(std::isinf(result.points[15]), "MoGe invalid point is infinity");
}

void test_focal_recovery_failure_is_reported() {
    constexpr int32_t size = 64;
    auto module = std::make_unique<FakeMogeModule>(size, size);
    module->invalidate_all();
    trtmc::MogePipeline pipeline(std::move(module), "moge-2-vitl");
    auto image = rgb_image(size, size);

    try {
        (void)pipeline.estimate_geometry(image.data(), size, size);
        check(false, "MoGe rejects geometry without valid focal samples");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("recover camera focal") != std::string::npos,
              "MoGe focal recovery error is explicit");
    }
}

} // namespace

int main() {
    test_pipeline_recovers_metric_geometry_from_hwc_input();
    test_invalid_mask_materializes_infinity();
    test_focal_recovery_failure_is_reported();
    if (g_failures != 0) {
        std::cerr << g_failures << " MoGe pipeline test(s) failed\n";
        return 1;
    }
    return 0;
}
