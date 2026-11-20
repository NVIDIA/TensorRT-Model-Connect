/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU contract for the Table Transformer image_to_boxes Task: binding, input validation,
// coordinate scaling, threshold filtering and engine-output checks,
// driven through a fake engine so no GPU or TensorRT is needed.

#include "families/table_transformer/runtime/pipeline.h"

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
        if (has_boxes)
            outputs.emplace("pred_boxes", trtmc::Tensor{null_boxes ? nullptr : boxes.data(),
                                                        boxes_shape, boxes_dtype});
        if (has_logits)
            outputs.emplace("logits", trtmc::Tensor{null_logits ? nullptr : logits.data(),
                                                    logits_shape, logits_dtype});
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
        return name == "pred_boxes" || name == "logits";
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
    bool has_boxes{true}, has_logits{true};
    bool null_boxes{false}, null_logits{false};
    // 2 queries, normalized [cx, cy, w, h]
    std::vector<float> boxes{0.5F, 0.5F, 0.2F, 0.4F, 0.1F, 0.2F, 0.1F, 0.1F};
    std::vector<int64_t> boxes_shape{1, 2, 4};
    // 2 queries, 3 classes (classes 0, 1 are labels, class 2 is background)
    // query 0: class 1 has strong logit -> score > 0.5
    // query 1: class 2 (background) has strong logit -> label score < 0.5
    std::vector<float> logits{0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 5.0F};
    std::vector<int64_t> logits_shape{1, 2, 3};
    std::vector<int64_t> input_shape;
    trtmc::DType boxes_dtype{trtmc::DType::kFloat32};
    trtmc::DType logits_dtype{trtmc::DType::kFloat32};
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
    return std::abs(actual - expected) < 1e-2F;
}

std::unique_ptr<trtmc::TableTransformerObjectDetectionPipeline>
make(std::unique_ptr<FakeEngine> engine) {
    trtmc::TableTransformerPreprocessConfig config;
    config.input_image_h = 1333;
    config.input_image_w = 1333;
    config.shortest_edge = 800;
    config.longest_edge = 1333;
    config.dynamic_image_shapes = true;
    config.image_mean = {0.485F, 0.456F, 0.406F};
    config.image_std = {0.229F, 0.224F, 0.225F};
    return std::make_unique<trtmc::TableTransformerObjectDetectionPipeline>(std::move(engine),
                                                                            config);
}

struct SourceImage {
    std::vector<float> pixels;
    std::uint32_t height{0}, width{0};

    ImageToBoxesRequest request() const {
        return {trtmc::internal::ImageView{pixels.data(), pixels.size() * sizeof(float), height,
                                           width, 3, ImageFormat::Float32}};
    }
};

SourceImage image(std::uint32_t height, std::uint32_t width) {
    return {std::vector<float>(static_cast<std::size_t>(height) * width * 3, 0.5F), height, width};
}

void test_binds_only_image_to_boxes() {
    auto model = make(std::make_unique<FakeEngine>());
    require(std::string_view(model->task()) == IImageToBoxes::kTask,
            "primary task is image_to_boxes");
    const auto bindings = model->task_bindings();
    require(bindings.size() == 1, "exactly one task is exposed");
    require(bindings[0].key.id == IImageToBoxes::kTask, "task id is image_to_boxes");
    require(bindings[0].key.major == 1 && bindings[0].key.minor == 0, "task version is 1.0");
    require(bindings[0].implementation != nullptr, "implementation pointer is non-null");
    require(bindings[0].fields.empty(), "Table Transformer exposes no runtime Config fields");
}

void test_boxes_scale_to_original_image_dimensions() {
    auto engine = std::make_unique<FakeEngine>();
    // query 0: cx=0.5, cy=0.5, w=0.2, h=0.4
    // On 800 width, 600 height:
    // x0 = (0.5 - 0.1) * 800 = 320
    // y0 = (0.5 - 0.2) * 600 = 180
    // x1 = (0.5 + 0.1) * 800 = 480
    // y1 = (0.5 + 0.2) * 600 = 420
    engine->boxes = {0.5F, 0.5F, 0.2F, 0.4F};
    engine->boxes_shape = {1, 1, 4};
    engine->logits = {0.0F, 10.0F, 0.0F};
    engine->logits_shape = {1, 1, 3};
    auto model = make(std::move(engine));
    const auto source = image(600, 800);
    const auto result = model->run(source.request(), {});
    require(result.boxes.size() == 1, "one box detected");
    const auto& box = result.boxes[0];
    require(near(box.x_min, 320.0F) && near(box.y_min, 180.0F) && near(box.x_max, 480.0F) &&
                near(box.y_max, 420.0F),
            "box coordinates scale to the original image dimensions");
    require(box.class_id == 1, "predicted class is label 1");
    require(box.score > 0.99F, "score reflects softmax output");
}

void test_scores_above_threshold_are_kept_and_sorted_descending() {
    auto engine = std::make_unique<FakeEngine>();
    // 3 queries:
    // query 0: class 0 logit 5.0 -> score ~0.84
    // query 1: class 1 logit 8.0 -> score ~0.99
    // query 2: background logit 10.0 -> label score < 0.01 (dropped)
    engine->boxes = {0.2F, 0.2F, 0.1F, 0.1F, 0.5F, 0.5F, 0.2F, 0.2F, 0.8F, 0.8F, 0.1F, 0.1F};
    engine->boxes_shape = {1, 3, 4};
    engine->logits = {5.0F, 0.0F, 3.0F, 0.0F, 8.0F, 1.0F, 0.0F, 0.0F, 10.0F};
    engine->logits_shape = {1, 3, 3};
    auto model = make(std::move(engine));
    const auto source = image(400, 400);
    const auto result = model->run(source.request(), {});
    require(result.boxes.size() == 2, "only 2 boxes pass confidence threshold of 0.5");
    require(result.boxes[0].score > result.boxes[1].score, "boxes are sorted by score descending");
    require(result.boxes[0].class_id == 1, "highest scoring box is class 1");
    require(result.boxes[1].class_id == 0, "second box is class 0");
}

void test_an_empty_detection_is_valid() {
    auto engine = std::make_unique<FakeEngine>();
    // All queries predict background
    engine->boxes = {0.5F, 0.5F, 0.2F, 0.2F};
    engine->boxes_shape = {1, 1, 4};
    engine->logits = {0.0F, 0.0F, 10.0F};
    engine->logits_shape = {1, 1, 3};
    auto model = make(std::move(engine));
    const auto source = image(64, 48);
    const auto result = model->run(source.request(), {});
    require(result.boxes.empty() && result.image_height == 64 && result.image_width == 48,
            "no detection above threshold produces an empty result with image size");
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
    const auto source = image(400, 400);
    rejects<std::runtime_error>([&] { model->run(source.request(), {}); }, message);
}

void test_unusable_engine_outputs_are_errors() {
    rejects_engine_output([](FakeEngine& e) { e.has_boxes = false; }, "missing boxes");
    rejects_engine_output([](FakeEngine& e) { e.has_logits = false; }, "missing logits");
    rejects_engine_output([](FakeEngine& e) { e.boxes_shape = {1, 2}; }, "rank 2 boxes");
    rejects_engine_output([](FakeEngine& e) { e.logits_shape = {1, 2}; }, "rank 2 logits");
    rejects_engine_output([](FakeEngine& e) { e.boxes_shape = {1, 1, 4}; }, "truncated boxes");
    rejects_engine_output([](FakeEngine& e) { e.boxes_shape = {1, 0, 4}; }, "empty boxes");
    rejects_engine_output([](FakeEngine& e) { e.null_boxes = true; }, "null boxes");
    rejects_engine_output([](FakeEngine& e) { e.null_logits = true; }, "null logits");
    rejects_engine_output([](FakeEngine& e) { e.boxes_dtype = trtmc::DType::kFloat16; },
                          "FP16 boxes");
    rejects_engine_output([](FakeEngine& e) { e.logits_dtype = trtmc::DType::kFloat16; },
                          "FP16 logits");
    rejects_engine_output([](FakeEngine& e) { e.boxes_shape = {1, 2, 5}; }, "five-value boxes");
    rejects_engine_output([](FakeEngine& e) { e.boxes_shape = {1, 3, 4}; }, "extra box query");
    rejects_engine_output([](FakeEngine& e) { e.boxes_shape = {2, 2, 4}; }, "batched boxes");
    rejects_engine_output([](FakeEngine& e) { e.logits_shape = {2, 2, 3}; }, "batched logits");
    rejects_engine_output([](FakeEngine& e) { e.boxes_shape = {1, 2, 4, 1}; }, "rank 4 boxes");
    rejects_engine_output([](FakeEngine& e) { e.logits_shape = {1, 2, 3, 1}; }, "rank 4 logits");
    rejects_engine_output([](FakeEngine& e) { e.logits_shape = {1, 2, 1}; }, "no label class");
    rejects_engine_output([](FakeEngine& e) { e.logits_shape = {1, -1, 3}; }, "negative queries");
    const int64_t beyond_int32 = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    rejects_engine_output([=](FakeEngine& e) { e.logits_shape = {1, beyond_int32, 3}; },
                          "queries beyond int32");
    rejects_engine_output([=](FakeEngine& e) { e.logits_shape = {1, 2, beyond_int32}; },
                          "classes beyond int32");
}

void test_nonfinite_logits_are_errors() {
    for (const float value :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()}) {
        for (std::size_t index = 0; index < 6; ++index) {
            rejects_engine_output([=](FakeEngine& e) { e.logits[index] = value; },
                                  "non-finite label or background logit");
        }
    }
}

} // namespace

int main() {
    try {
        test_binds_only_image_to_boxes();
        test_boxes_scale_to_original_image_dimensions();
        test_scores_above_threshold_are_kept_and_sorted_descending();
        test_an_empty_detection_is_valid();
        test_invalid_requests_fail_before_the_engine();
        test_unusable_engine_outputs_are_errors();
        test_nonfinite_logits_are_errors();
    } catch (const std::exception& error) {
        std::cerr << "table_transformer task contract failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "table_transformer task contract: 7 cases passed\n";
    return 0;
}
