/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU contract of the OneFormer image_to_semantic_segmentation Task: binding,
// configuration and input validation, the engine's input geometry, label maps
// at the original resolution and engine-output checks, through a fake engine.

#include "families/oneformer/runtime/pipeline.h"

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

using trtmc::internal::ConfigEntry;
using trtmc::internal::ConfigError;
using trtmc::internal::IImageToSemanticSegmentation;
using trtmc::internal::ImageFormat;
using trtmc::internal::ImageToSemanticSegmentationRequest;
using trtmc::internal::ImageView;

constexpr int kClasses = 3;

class FakeEngine final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        input_shape = inputs.at("pixel_values").shape;
        const auto height = (input_shape[2] + 3) / 4, width = (input_shape[3] + 3) / 4;
        // Class 0 on the left third, 1 in the middle, 2 on the right of the stride-4 map.
        scores.assign(static_cast<std::size_t>(kClasses * height * width), 0.0F);
        for (std::int64_t y = 0; y < height; ++y)
            for (std::int64_t x = 0; x < width; ++x) {
                const auto winner = std::min<std::int64_t>(x * 3 / width, kClasses - 1);
                scores[static_cast<std::size_t>((winner * height + y) * width + x)] = 1.0F;
            }
        if (poison)
            scores[0] = std::numeric_limits<float>::quiet_NaN();
        trtmc::TensorMap outputs;
        outputs.emplace("semantic_scores", trtmc::Tensor{scores.data(),
                                                         {1, output_classes, height, width},
                                                         trtmc::DType::kFloat32});
        return outputs;
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
    bool has_input(const std::string& name) const override { return name == "pixel_values"; }
    bool has_output(const std::string& name) const override { return name == "semantic_scores"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        return selector == trtmc::ProfileShapeSelector::kMax
                   ? std::vector<int64_t>{1, 3, 2048, 2048}
                   : std::vector<int64_t>{1, 3, 32, 32};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 4; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};
    bool poison{false};
    std::int64_t output_classes{kClasses};
    std::vector<float> scores;
    std::vector<int64_t> input_shape;
};

void require(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}

template <class Error, class Function>
void rejects(Function function, const std::string& message) {
    try {
        function();
    } catch (const Error&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(message + " (wrong exception: " + error.what() + ")");
    }
    throw std::runtime_error(message + " (accepted)");
}

struct Model {
    FakeEngine* engine{nullptr};
    std::unique_ptr<trtmc::oneformer::SemanticSegmentation> model;
};

Model make() {
    auto engine = std::make_unique<FakeEngine>();
    Model result{engine.get(), nullptr};
    trtmc::oneformer::RuntimeConfig config;
    config.shortest_edge = 1024;
    config.longest_edge = 2048;
    config.normalization.mean = {0.485F, 0.456F, 0.406F};
    config.normalization.std = {0.229F, 0.224F, 0.225F};
    config.class_names = {"road", "sidewalk", "building"};
    result.model =
        std::make_unique<trtmc::oneformer::SemanticSegmentation>(std::move(engine), config);
    return result;
}

std::vector<float> gray(std::uint32_t height, std::uint32_t width) {
    return std::vector<float>(static_cast<std::size_t>(height) * width * 3, 0.5F);
}

ImageToSemanticSegmentationRequest request(const std::vector<float>& pixels, std::uint32_t height,
                                           std::uint32_t width, std::uint32_t channels = 3) {
    return {ImageView{pixels.data(), pixels.size() * sizeof(float), height, width, channels,
                      ImageFormat::Float32}};
}

void test_binds_only_semantic_segmentation() {
    auto made = make();
    require(std::string_view(made.model->task()) == IImageToSemanticSegmentation::kTask,
            "primary task is image_to_semantic_segmentation");
    const auto bindings = made.model->task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == IImageToSemanticSegmentation::kTask &&
                bindings[0].fields.size() == 0,
            "exactly one Task without Config fields is bound");
}

void test_labels_cover_the_original_image() {
    auto made = make();
    const auto pixels = gray(120, 300);
    const auto result = made.model->run(request(pixels, 120, 300), {});
    // 120 x 300 resizes to 1024 x 2560 > 2048, so the longest edge caps it at 819 x 2048.
    require(made.engine->input_shape == std::vector<int64_t>{1, 3, 819, 2048}, "processed size");
    require(result.height == 120 && result.width == 300 && result.labels.size() == 120U * 300U,
            "labels have the original resolution");
    require(result.class_ids == std::vector<std::int32_t>{0, 1, 2}, "class ids");
    require(result.class_names == std::vector<std::string>{"road", "sidewalk", "building"},
            "class names");
    require(result.class_scores.empty() && result.score_height == 0 && result.score_width == 0,
            "no score maps are returned");
    require(!result.ignore_label && !result.background_label && result.vocabulary_id.empty(),
            "no ignore/background label or vocabulary is claimed");
    require(result.labels[0] == 0 && result.labels[150] == 1 && result.labels[299] == 2,
            "left, middle and right thirds keep their classes");
}

void test_uint8_and_float_inputs_agree() {
    auto made = make();
    std::vector<std::uint8_t> bytes(64U * 80U * 3U);
    std::vector<float> floats(bytes.size());
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>((index * 37U) % 256U);
        floats[index] = static_cast<float>(bytes[index]) / 255.0F;
    }
    const auto from_float = made.model->run(request(floats, 64, 80), {});
    const auto from_bytes =
        made.model->run({ImageView{bytes.data(), bytes.size(), 64, 80, 3, ImageFormat::UInt8}}, {});
    require(from_float.labels == from_bytes.labels,
            "uint8 and float32 pixels give the same labels");
}

void test_rejects_invalid_requests() {
    auto made = make();
    const auto pixels = gray(64, 64);
    const ConfigEntry entry{"threshold", 0.5};
    rejects<ConfigError>([&] { made.model->run(request(pixels, 64, 64), {&entry, 1}); },
                         "runtime Config is rejected");
    rejects<std::invalid_argument>([&] { made.model->run(request(pixels, 64, 64, 4), {}); },
                                   "RGBA is rejected");
    rejects<std::invalid_argument>(
        [&] {
            made.model->run({ImageView{pixels.data(), 7, 64, 64, 3, ImageFormat::Float32}}, {});
        },
        "a wrong byte size is rejected");
    rejects<std::invalid_argument>(
        [&] { made.model->run({ImageView{nullptr, 0, 64, 64, 3, ImageFormat::Float32}}, {}); },
        "missing pixels are rejected");
    // 1 x 400 needs a 5 x 2048 processed image, below the 32-pixel engine minimum.
    const auto strip = gray(1, 400);
    rejects<std::invalid_argument>([&] { made.model->run(request(strip, 1, 400), {}); },
                                   "an image outside the engine profile is rejected");
    require(made.engine->calls == 0, "rejected requests never reach the engine");
}

void test_rejects_invalid_engine_outputs() {
    auto made = make();
    const auto pixels = gray(64, 64);
    made.engine->poison = true;
    rejects<std::runtime_error>([&] { made.model->run(request(pixels, 64, 64), {}); },
                                "non-finite scores are rejected");
    made.engine->poison = false;
    made.engine->output_classes = kClasses + 1;
    rejects<std::runtime_error>([&] { made.model->run(request(pixels, 64, 64), {}); },
                                "a class count that differs from the bundle's names is rejected");
}

} // namespace

int main() {
    try {
        test_binds_only_semantic_segmentation();
        test_labels_cover_the_original_image();
        test_uint8_and_float_inputs_agree();
        test_rejects_invalid_requests();
        test_rejects_invalid_engine_outputs();
        std::cout << "oneformer task contract: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "oneformer task contract: " << error.what() << '\n';
        return 1;
    }
}
