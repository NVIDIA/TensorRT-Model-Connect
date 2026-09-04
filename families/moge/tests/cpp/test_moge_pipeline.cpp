/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moge/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

std::vector<float> affine_depth(const std::vector<float>& points) {
    std::vector<float> depth(points.size() / 3U);
    for (std::size_t pixel = 0; pixel < depth.size(); ++pixel)
        depth[pixel] = points[pixel * 3U + 2U];
    return depth;
}

std::vector<float> focal_samples(const std::vector<float>& points, int32_t height, int32_t width) {
    constexpr int32_t sample_size = 64;
    std::vector<float> samples(static_cast<std::size_t>(sample_size) * sample_size * 3U);
    for (int32_t out_y = 0; out_y < sample_size; ++out_y) {
        const int32_t y = static_cast<int32_t>(static_cast<int64_t>(out_y) * height / sample_size);
        for (int32_t out_x = 0; out_x < sample_size; ++out_x) {
            const int32_t x =
                static_cast<int32_t>(static_cast<int64_t>(out_x) * width / sample_size);
            const auto source = (static_cast<std::size_t>(y) * width + x) * 3U;
            const auto target = (static_cast<std::size_t>(out_y) * sample_size + out_x) * 3U;
            std::copy_n(points.data() + source, 3, samples.data() + target);
        }
    }
    return samples;
}

class FakeMogeModule final : public trtmc::ITrtModule {
  public:
    FakeMogeModule(int32_t height, int32_t width, bool invalidate_pixel = false,
                   trtmc::DType valid_dtype = trtmc::DType::kFloat16)
        : height_(height), width_(width), min_height_(height), min_width_(width),
          max_height_(height), max_width_(width),
          points_(affine_points(height, width, 0.8F, 1.25F)), depth_(affine_depth(points_)),
          samples_(focal_samples(points_, height, width)),
          valid_(static_cast<std::size_t>(height) * width, uint16_t{0x3C00}),
          valid_dtype_(valid_dtype) {
        if (invalidate_pixel)
            valid_[5] = 0;
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++forward_count_;
        const auto image = inputs.find("image");
        if (image != inputs.end()) {
            input_shape = image->second.shape;
            const auto* values = static_cast<const float*>(image->second.data);
            input_values.assign(values, values + image->second.numel());
        }
        return {
            {"affine_depth", {depth_.data(), {1, height_, width_}, trtmc::DType::kFloat32}},
            {"valid", {valid_.data(), {1, height_, width_}, valid_dtype_}},
            {"focal_samples", {samples_.data(), {1, 64, 64, 3}, trtmc::DType::kFloat32}},
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
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return profile_index_; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return name == "image"; }
    bool has_output(const std::string& name) const override {
        return name == "affine_depth" || name == "valid" || name == "focal_samples" ||
               name == "metric_scale";
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == "valid" ? valid_dtype_ : trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "image")
            return {1, height_, width_, 3};
        if (name == "affine_depth" || name == "valid")
            return {1, height_, width_};
        if (name == "focal_samples")
            return {1, 64, 64, 3};
        if (name == "metric_scale")
            return {1};
        throw std::runtime_error("unknown fake tensor");
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t profile_index,
                                             trtmc::ProfileShapeSelector selector) const override {
        last_queried_profile_index_ = profile_index;
        ++profile_query_count_;
        if (selector == trtmc::ProfileShapeSelector::kMin)
            return {1, min_height_, min_width_, 3};
        if (selector == trtmc::ProfileShapeSelector::kMax)
            return {1, max_height_, max_width_, 3};
        return {1, height_, width_, 3};
    }
    int32_t optimization_profile_count() const override { return profile_index_ + 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 4; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    void invalidate_all() { std::fill(valid_.begin(), valid_.end(), uint16_t{0}); }
    void set_valid(int32_t y, int32_t x, bool value) {
        valid_.at(static_cast<std::size_t>(y) * width_ + x) = value ? uint16_t{0x3C00} : 0;
    }
    void set_valid_neighborhood(int32_t y, int32_t x) {
        for (int32_t neighbor_y = y - 1; neighbor_y <= y + 1; ++neighbor_y) {
            for (int32_t neighbor_x = x - 1; neighbor_x <= x + 1; ++neighbor_x)
                set_valid(neighbor_y, neighbor_x, true);
        }
    }
    void set_focal_sample(int32_t y, int32_t x, float px, float py, float pz) {
        const auto sample = (static_cast<std::size_t>(y) * 64U + x) * 3U;
        samples_.at(sample) = px;
        samples_.at(sample + 1U) = py;
        samples_.at(sample + 2U) = pz;
    }
    uint16_t valid_bits(std::size_t pixel) const { return valid_.at(pixel); }
    int32_t forward_count() const { return forward_count_; }
    int32_t last_queried_profile_index() const { return last_queried_profile_index_; }
    int32_t profile_query_count() const { return profile_query_count_; }
    void set_profile_index(int32_t profile_index) { profile_index_ = profile_index; }
    void set_profile_bounds(int32_t min_height, int32_t min_width, int32_t max_height,
                            int32_t max_width) {
        min_height_ = min_height;
        min_width_ = min_width;
        max_height_ = max_height;
        max_width_ = max_width;
    }

    std::vector<int64_t> input_shape;
    std::vector<float> input_values;

  private:
    int32_t height_;
    int32_t width_;
    int32_t min_height_;
    int32_t min_width_;
    int32_t max_height_;
    int32_t max_width_;
    int32_t forward_count_{0};
    int32_t profile_index_{0};
    mutable int32_t last_queried_profile_index_{-1};
    mutable int32_t profile_query_count_{0};
    std::vector<float> points_;
    std::vector<float> depth_;
    std::vector<float> samples_;
    std::vector<uint16_t> valid_;
    trtmc::DType valid_dtype_;
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
    trtmc::MogePipeline pipeline(std::move(module));
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
    check(module_ptr->valid_bits(0) == 0x3C00, "MoGe valid=true uses FP16 one bits");
}

void test_invalid_mask_materializes_infinity() {
    constexpr int32_t size = 64;
    auto module = std::make_unique<FakeMogeModule>(size, size, true);
    auto* module_ptr = module.get();
    trtmc::MogePipeline pipeline(std::move(module));
    auto image = rgb_image(size, size);

    const auto result = pipeline.estimate_geometry(image.data(), size, size);

    check(result.mask[5] == 0, "MoGe invalid pixel mask cleared");
    check(std::isinf(result.depth[5]), "MoGe invalid depth is infinity");
    check(std::isinf(result.points[15]), "MoGe invalid point is infinity");
    check(module_ptr->valid_bits(5) == 0x0000, "MoGe valid=false uses FP16 zero bits");
    check(module_ptr->valid_bits(0) == 0x3C00, "MoGe retained valid pixel uses FP16 one bits");
}

void test_legacy_int8_valid_contract_is_rejected() {
    constexpr int32_t size = 64;
    auto module = std::make_unique<FakeMogeModule>(size, size, false, trtmc::DType::kInt8);
    trtmc::MogePipeline pipeline(std::move(module));
    auto image = rgb_image(size, size);

    try {
        (void)pipeline.estimate_geometry(image.data(), size, size);
        check(false, "MoGe rejects legacy INT8 valid output");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("output contract mismatch for 'valid'") !=
                  std::string::npos,
              "MoGe legacy INT8 valid rejection is explicit");
    }
}

void test_focal_recovery_failure_is_reported() {
    constexpr int32_t size = 64;
    auto module = std::make_unique<FakeMogeModule>(size, size);
    module->invalidate_all();
    trtmc::MogePipeline pipeline(std::move(module));
    auto image = rgb_image(size, size);

    try {
        (void)pipeline.estimate_geometry(image.data(), size, size);
        check(false, "MoGe rejects geometry without valid focal samples");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("recover camera focal") != std::string::npos,
              "MoGe focal recovery error is explicit");
    }
}

void test_invalid_rgb_values_are_rejected() {
    constexpr int32_t size = 64;
    const std::vector<float> invalid_values = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -0.01F,
        1.01F,
    };
    for (const float value : invalid_values) {
        auto module = std::make_unique<FakeMogeModule>(size, size);
        trtmc::MogePipeline pipeline(std::move(module));
        auto image = rgb_image(size, size);
        image[0] = value;
        try {
            (void)pipeline.estimate_geometry(image.data(), size, size);
            check(false, "MoGe rejects non-finite or out-of-range RGB input");
        } catch (const std::invalid_argument& error) {
            check(std::string(error.what()).find("RGB input values") != std::string::npos,
                  "MoGe RGB rejection is explicit");
        }
    }
}

void test_loaded_engine_profile_bounds_are_enforced_before_forward() {
    constexpr int32_t height = 64;
    constexpr int32_t width = 80;
    auto module = std::make_unique<FakeMogeModule>(height, width);
    module->set_profile_index(2);
    module->set_profile_bounds(height, width, 128, 160);
    auto* module_ptr = module.get();
    trtmc::MogePipeline pipeline(std::move(module));
    check(module_ptr->last_queried_profile_index() == 2 && module_ptr->profile_query_count() == 2,
          "MoGe reads min and max bounds from the active engine profile");

    for (const auto& [input_height, input_width] :
         {std::pair{height - 1, width}, std::pair{height, width - 1}, std::pair{129, width},
          std::pair{height, 161}}) {
        auto image = rgb_image(input_height, input_width);
        try {
            (void)pipeline.estimate_geometry(image.data(), input_height, input_width);
            check(false, "MoGe rejects dimensions outside the loaded engine profile");
        } catch (const std::invalid_argument& error) {
            check(std::string(error.what()).find("outside the bundle profile") != std::string::npos,
                  "MoGe profile-bound rejection is explicit");
        }
    }
    check(module_ptr->forward_count() == 0,
          "MoGe rejects profile-incompatible dimensions before TensorRT forward");
}

void test_invalid_loaded_engine_profile_is_rejected() {
    auto module = std::make_unique<FakeMogeModule>(64, 80);
    module->set_profile_bounds(128, 160, 64, 80);
    try {
        trtmc::MogePipeline pipeline(std::move(module));
        check(false, "MoGe rejects an invalid loaded engine profile");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("invalid TensorRT image profile") != std::string::npos,
              "MoGe invalid engine profile rejection is explicit");
    }
}

void test_focal_sampling_excludes_mapped_image_edges() {
    constexpr int32_t size = 64;
    auto module = std::make_unique<FakeMogeModule>(size, size);
    for (int32_t index = 0; index < size; ++index) {
        module->set_focal_sample(0, index, 1000.0F, -1000.0F, 0.1F);
        module->set_focal_sample(size - 1, index, 1000.0F, -1000.0F, 0.1F);
        module->set_focal_sample(index, 0, 1000.0F, -1000.0F, 0.1F);
        module->set_focal_sample(index, size - 1, 1000.0F, -1000.0F, 0.1F);
    }
    trtmc::MogePipeline pipeline(std::move(module));
    auto image = rgb_image(size, size);

    const auto result = pipeline.estimate_geometry(image.data(), size, size);

    check(close(result.intrinsics[0], 0.5656854F),
          "MoGe focal sampling excludes mapped image-edge points");
    check(close(result.intrinsics[4], 0.5656854F),
          "MoGe image-edge exclusion preserves normalized fy");
}

void test_focal_sampling_excludes_invalid_three_by_three_neighborhood() {
    constexpr int32_t size = 64;
    constexpr int32_t sample_y = 20;
    constexpr int32_t sample_x = 20;
    auto module = std::make_unique<FakeMogeModule>(size, size);
    module->set_focal_sample(sample_y, sample_x, 1000.0F, -1000.0F, 0.1F);
    module->set_valid(sample_y, sample_x + 1, false);
    auto* module_ptr = module.get();
    trtmc::MogePipeline pipeline(std::move(module));
    auto image = rgb_image(size, size);

    const auto result = pipeline.estimate_geometry(image.data(), size, size);

    check(module_ptr->valid_bits(static_cast<std::size_t>(sample_y) * size + sample_x) == 0x3C00,
          "MoGe focal center remains valid when its neighbor is invalid");
    check(close(result.intrinsics[0], 0.5656854F),
          "MoGe focal sampling excludes a center with an invalid neighbor");
}

void test_focal_sampling_retains_complete_interior_neighborhoods() {
    constexpr int32_t size = 64;
    auto module = std::make_unique<FakeMogeModule>(size, size);
    module->invalidate_all();
    for (int32_t y : {16, 32, 48}) {
        for (int32_t x : {16, 32, 48})
            module->set_valid_neighborhood(y, x);
    }
    trtmc::MogePipeline pipeline(std::move(module));
    auto image = rgb_image(size, size);

    const auto result = pipeline.estimate_geometry(image.data(), size, size);

    check(close(result.intrinsics[0], 0.5656854F),
          "MoGe focal sampling retains complete interior neighborhoods");
    check(result.mask[32U * size + 32U] == 1,
          "MoGe retained interior focal center remains valid geometry");
}

} // namespace

int main() {
    test_pipeline_recovers_metric_geometry_from_hwc_input();
    test_invalid_mask_materializes_infinity();
    test_legacy_int8_valid_contract_is_rejected();
    test_focal_recovery_failure_is_reported();
    test_invalid_rgb_values_are_rejected();
    test_loaded_engine_profile_bounds_are_enforced_before_forward();
    test_invalid_loaded_engine_profile_is_rejected();
    test_focal_sampling_excludes_mapped_image_edges();
    test_focal_sampling_excludes_invalid_three_by_three_neighborhood();
    test_focal_sampling_retains_complete_interior_neighborhoods();
    if (g_failures != 0) {
        std::cerr << g_failures << " MoGe pipeline test(s) failed\n";
        return 1;
    }
    return 0;
}
