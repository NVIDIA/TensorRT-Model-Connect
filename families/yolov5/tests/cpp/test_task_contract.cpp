/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU contract for the YOLOv5 image_to_boxes Task: binding, input validation,
// letterbox undo, suppression and engine-output checks,
// driven through a fake engine so no GPU or TensorRT is needed.

#include "families/yolov5/runtime/pipeline.h"

#include <cmath>
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
using trtmc::internal::IImageToBoxes;
using trtmc::internal::ImageFormat;
using trtmc::internal::ImageToBoxesRequest;

class FakeEngine final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& image = inputs.at("pixel_values");
        input_shape = image.shape;
        trtmc::TensorMap outputs;
        const auto count = static_cast<int64_t>(scores.size());
        if (has_boxes)
            outputs.emplace("boxes", trtmc::Tensor{boxes.data(),
                                                   {1, static_cast<int64_t>(boxes.size() / 4), 4},
                                                   boxes_dtype});
        if (has_scores)
            outputs.emplace("scores", trtmc::Tensor{scores.data(), {1, count}, scores_dtype});
        if (has_classes)
            outputs.emplace("classes", trtmc::Tensor{classes.data(),
                                                     {1, static_cast<int64_t>(classes.size())},
                                                     classes_dtype});
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
    bool has_output(const std::string& name) const override {
        return name == "boxes" || name == "scores" || name == "classes";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 4; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};
    bool has_boxes{true}, has_scores{true}, has_classes{true};
    // Network-input (letterboxed 640x640) XYXY boxes, one row per anchor.
    std::vector<float> boxes{150.0F, 200.0F, 250.0F, 400.0F, 10.0F, 10.0F, 20.0F, 20.0F};
    std::vector<float> scores{0.9F, 0.1F};
    std::vector<int32_t> classes{3, 7};
    std::vector<int64_t> input_shape;
    trtmc::DType boxes_dtype{trtmc::DType::kFloat32};
    trtmc::DType scores_dtype{trtmc::DType::kFloat32};
    trtmc::DType classes_dtype{trtmc::DType::kInt32};
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

bool near(float actual, float expected) {
    return std::fabs(actual - expected) < 1e-4F;
}

constexpr float kScoreThreshold = 0.25F;
constexpr float kIouThreshold = 0.7F;
constexpr std::int32_t kMaxDetections = 300;

std::unique_ptr<trtmc::Yolo11ObjectDetectionPipeline>
make(std::unique_ptr<FakeEngine> engine, std::int32_t max_detections = kMaxDetections) {
    return std::make_unique<trtmc::Yolo11ObjectDetectionPipeline>(
        std::move(engine), trtmc::Yolo11PreprocessConfig{}, kScoreThreshold, kIouThreshold,
        max_detections);
}

struct Image {
    std::vector<float> pixels;
    std::uint32_t height, width;
    ImageToBoxesRequest request() const {
        return {
            {pixels.data(), pixels.size() * sizeof(float), height, width, 3, ImageFormat::Float32}};
    }
};

Image image(std::uint32_t height, std::uint32_t width) {
    return {std::vector<float>(static_cast<std::size_t>(height) * width * 3, 0.5F), height, width};
}

void test_binds_only_image_to_boxes() {
    auto model = make(std::make_unique<FakeEngine>());
    const auto bindings = model->task_bindings();
    require(bindings.size() == 1, "exactly one Task is bound");
    require(bindings[0].key.id == "image_to_boxes" && bindings[0].key.major == 1 &&
                bindings[0].key.minor == 0,
            "the Task is image_to_boxes v1.0");
    require(bindings[0].fields.empty(), "no runtime Config fields are advertised");
    require(bindings[0].implementation == static_cast<IImageToBoxes*>(model.get()),
            "the binding points at the IImageToBoxes subobject");
    require(std::string(model->task()) == "image_to_boxes", "the primary Task matches the binding");
}

void check_letterbox_undo(std::uint32_t height, std::uint32_t width) {
    const auto source = image(height, width);
    trtmc::Yolo11Letterbox letterbox;
    (void)trtmc::preprocess_yolov5_image(source.pixels.data(), static_cast<std::int32_t>(height),
                                         static_cast<std::int32_t>(width),
                                         trtmc::Yolo11PreprocessConfig{}, letterbox);
    const bool wide = width > height;
    require((wide ? letterbox.pad_y : letterbox.pad_x) > 0.0F &&
                (wide ? letterbox.pad_x : letterbox.pad_y) == 0.0F,
            "the fixture is padded on exactly one axis");

    auto engine = std::make_unique<FakeEngine>();
    auto* fake = engine.get();
    auto model = make(std::move(engine));
    const auto result = model->run(source.request(), {});
    require(fake->input_shape == std::vector<int64_t>({1, 3, 640, 640}),
            "the engine receives one planar 640x640 image");
    require(result.image_height == static_cast<std::int32_t>(height) &&
                result.image_width == static_cast<std::int32_t>(width),
            "the result reports the source dimensions");
    require(result.boxes.size() == 1, "the score threshold drops the 0.1 anchor");
    const auto& box = result.boxes[0];
    require(box.score == 0.9F && box.class_id == 3, "score and class are carried unchanged");
    require(near(box.x_min, (150.0F - letterbox.pad_x) / letterbox.scale) &&
                near(box.y_min, (200.0F - letterbox.pad_y) / letterbox.scale) &&
                near(box.x_max, (250.0F - letterbox.pad_x) / letterbox.scale) &&
                near(box.y_max, (400.0F - letterbox.pad_y) / letterbox.scale),
            "padding then scale are undone on both axes");
}

void test_boxes_return_to_original_pixels() {
    check_letterbox_undo(320, 480); // wide: vertical padding
    check_letterbox_undo(480, 320); // tall: horizontal padding
}

void test_boxes_are_clamped_to_the_image() {
    auto engine = std::make_unique<FakeEngine>();
    engine->boxes = {-50.0F, -50.0F, 700.0F, 700.0F};
    engine->scores = {0.5F};
    engine->classes = {0};
    auto model = make(std::move(engine));
    const auto source = image(640, 320);
    const auto result = model->run(source.request(), {});
    require(result.boxes.size() == 1, "one box is kept");
    const auto& box = result.boxes[0];
    require(box.x_min == 0.0F && box.y_min == 0.0F && box.x_max == 320.0F && box.y_max == 640.0F,
            "x is clamped to the width and y to the height");
}

void test_a_score_at_the_threshold_is_kept() {
    auto engine = std::make_unique<FakeEngine>();
    engine->scores = {0.25F, 0.2499F};
    auto model = make(std::move(engine));
    const auto source = image(640, 640);
    const auto result = model->run(source.request(), {});
    require(result.boxes.size() == 1 && result.boxes[0].score == 0.25F,
            "the threshold is inclusive, as before the migration");
}

void test_suppression_is_per_class_and_ordered() {
    auto engine = std::make_unique<FakeEngine>();
    engine->boxes = {10, 10, 30, 30, 11, 11, 31, 31, 10, 10, 30, 30, 200, 200, 220, 220};
    engine->scores = {0.8F, 0.9F, 0.85F, 0.3F};
    engine->classes = {0, 0, 1, 0};
    auto model = make(std::move(engine));
    const auto source = image(640, 640);
    const auto result = model->run(source.request(), {});
    require(result.boxes.size() == 3, "the overlapping same-class box is suppressed");
    require(result.boxes[0].score == 0.9F && result.boxes[0].class_id == 0,
            "the best box of class 0 is first");
    require(result.boxes[1].score == 0.85F && result.boxes[1].class_id == 1,
            "an overlapping box of another class is kept");
    require(result.boxes[2].score == 0.3F, "a distant box of the same class is kept");
}

void test_max_detections_caps_the_result() {
    auto engine = std::make_unique<FakeEngine>();
    engine->boxes = {0, 0, 10, 10, 100, 100, 110, 110, 200, 200, 210, 210};
    engine->scores = {0.5F, 0.7F, 0.6F};
    engine->classes = {0, 0, 0};
    auto model = make(std::move(engine), 2);
    const auto source = image(640, 640);
    const auto result = model->run(source.request(), {});
    require(result.boxes.size() == 2 && result.boxes[0].score == 0.7F &&
                result.boxes[1].score == 0.6F,
            "max_detections keeps the highest scores");
}

void test_an_empty_detection_is_valid() {
    auto engine = std::make_unique<FakeEngine>();
    engine->scores = {0.1F, std::numeric_limits<float>::quiet_NaN()};
    auto model = make(std::move(engine));
    const auto source = image(64, 48);
    const auto result = model->run(source.request(), {});
    require(result.boxes.empty() && result.image_height == 64 && result.image_width == 48,
            "nothing above the threshold (NaN included) is an empty result, not an error");
}

void test_invalid_requests_fail_before_the_engine() {
    auto engine = std::make_unique<FakeEngine>();
    auto* fake = engine.get();
    auto model = make(std::move(engine));
    const auto source = image(32, 32);
    const auto valid = source.request();

    const ConfigEntry unknown{"score_threshold", 0.5};
    rejects<ConfigError>([&] { model->run(valid, {&unknown, 1}); }, "unknown Config is rejected");

    auto bad = valid;
    bad.image.byte_size -= sizeof(float);
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "short buffer");
    bad = valid;
    bad.image.byte_size += sizeof(float);
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "long buffer");
    bad = valid;
    bad.image.channels = 4;
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "four channels");
    bad = valid;
    bad.image.format = ImageFormat::UInt8;
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "uint8 pixels");
    bad = valid;
    bad.image.data = nullptr;
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "null pixels");
    bad = valid;
    bad.image.height = 0;
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "zero height");
    bad = valid;
    bad.image.width = 0;
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "zero width");
    bad = valid;
    bad.image.height = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) + 1U;
    rejects<std::invalid_argument>([&] { model->run(bad, {}); }, "height beyond int32");
    require(fake->calls == 0, "no invalid request reaches the engine");
}

template <class Mutate>
void rejects_engine_output(Mutate mutate, const std::string& message) {
    auto engine = std::make_unique<FakeEngine>();
    mutate(*engine);
    auto model = make(std::move(engine));
    const auto source = image(640, 640);
    rejects<std::runtime_error>([&] { model->run(source.request(), {}); }, message);
}

void test_unusable_engine_outputs_are_errors() {
    rejects_engine_output([](FakeEngine& e) { e.has_boxes = false; }, "missing boxes");
    rejects_engine_output([](FakeEngine& e) { e.has_scores = false; }, "missing scores");
    rejects_engine_output([](FakeEngine& e) { e.has_classes = false; }, "missing classes");
    rejects_engine_output([](FakeEngine& e) { e.boxes_dtype = trtmc::DType::kFloat16; },
                          "fp16 boxes");
    rejects_engine_output([](FakeEngine& e) { e.scores_dtype = trtmc::DType::kFloat16; },
                          "fp16 scores");
    rejects_engine_output([](FakeEngine& e) { e.scores.pop_back(); }, "length mismatch");
    rejects_engine_output([](FakeEngine& e) { e.classes.pop_back(); }, "short classes");
    rejects_engine_output([](FakeEngine& e) { e.classes_dtype = trtmc::DType::kFloat32; },
                          "float classes");
}

} // namespace

int main() {
    try {
        test_binds_only_image_to_boxes();
        test_boxes_return_to_original_pixels();
        test_boxes_are_clamped_to_the_image();
        test_a_score_at_the_threshold_is_kept();
        test_suppression_is_per_class_and_ordered();
        test_max_detections_caps_the_result();
        test_an_empty_detection_is_valid();
        test_invalid_requests_fail_before_the_engine();
        test_unusable_engine_outputs_are_errors();
    } catch (const std::exception& error) {
        std::cerr << "yolov5 task contract failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "yolov5 task contract: 9 cases passed\n";
    return 0;
}
