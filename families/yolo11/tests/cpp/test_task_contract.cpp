/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolo11/runtime/pipeline.h"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& image = inputs.at("pixel_values");
        input_shape = image.shape;
        const auto* data = static_cast<const float*>(image.data);
        input_pixels.assign(data, data + image.numel());

        trtmc::TensorMap outputs;
        if (has_boxes) {
            outputs.emplace("boxes", trtmc::Tensor{boxes.data(),
                                                   {1, static_cast<int64_t>(boxes.size() / 4), 4},
                                                   boxes_dtype});
        }
        if (has_scores) {
            outputs.emplace("scores", trtmc::Tensor{scores.data(),
                                                    {1, static_cast<int64_t>(scores.size())},
                                                    scores_dtype});
        }
        if (has_classes) {
            outputs.emplace("classes", trtmc::Tensor{classes.data(),
                                                     {1, static_cast<int64_t>(classes.size())},
                                                     classes_dtype});
        }
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
    trtmc::DType tensor_dtype(const std::string& name) const override {
        if (name == "boxes")
            return boxes_dtype;
        if (name == "scores")
            return scores_dtype;
        return classes_dtype;
    }
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
    bool has_boxes{true};
    bool has_scores{true};
    bool has_classes{true};
    std::vector<float> boxes{10.0F, 10.0F, 30.0F, 30.0F, 12.0F, 12.0F, 32.0F, 32.0F};
    std::vector<float> scores{0.8F, 0.1F};
    std::vector<int32_t> classes{0, 0};
    std::vector<float> input_pixels;
    std::vector<int64_t> input_shape;
    trtmc::DType boxes_dtype{trtmc::DType::kFloat32};
    trtmc::DType scores_dtype{trtmc::DType::kFloat32};
    trtmc::DType classes_dtype{trtmc::DType::kInt32};
};

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

template <class Error, class Function>
void rejects(Function function, const char* message) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

trtmc::Yolo11PreprocessConfig preprocessing() {
    return {640, 640, 0.447F};
}

trtmc::internal::ImageToBoxesRequest
request(const std::vector<float>& pixels, std::uint32_t height = 640, std::uint32_t width = 640) {
    return {{pixels.data(), pixels.size() * sizeof(float), height, width, 3,
             trtmc::internal::ImageFormat::Float32}};
}

void test_binding_and_complete_owned_boxes() {
    trtmc::internal::DetectedBoxesResult result;
    const std::vector<float> pixels(640 * 640 * 3, 0.5F);
    {
        auto module = std::make_unique<RecordingModule>();
        auto* recording = module.get();
        trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                                   300);
        const auto bindings = model.task_bindings();
        require(bindings.size() == 1 && bindings[0].key.id == "image_to_boxes" &&
                    bindings[0].key.major == 1 && bindings[0].key.minor == 0,
                "family must publish exactly image_to_boxes v1.0");
        require(bindings[0].fields.empty(), "family has no runtime config options");
        require(bindings[0].implementation == static_cast<trtmc::internal::IImageToBoxes*>(&model),
                "bind must preserve the interface subobject address");
        require(std::string(model.task()) == "image_to_boxes",
                "bundle primary task must match the binding");

        result = static_cast<trtmc::internal::IImageToBoxes*>(bindings[0].implementation)
                     ->run(request(pixels), {});
        require(result.image_height == 640 && result.image_width == 640,
                "result preserves input dimensions");
        require(result.boxes.size() == 1, "filtering drops anchor below score threshold");
        require(result.boxes[0].score == 0.8F && result.boxes[0].class_id == 0,
                "kept box preserves score and class");
        require(recording->input_shape == std::vector<int64_t>({1, 3, 640, 640}),
                "preprocessing preserves the NCHW input layout");

        recording->scores.assign(2, 0.0F);
        model.run(request(pixels), {});
    }
    require(result.boxes.size() == 1 && result.boxes[0].score == 0.8F,
            "result must own all detections across another call and model destruction");
}

void test_suppression_and_nms() {
    auto module = std::make_unique<RecordingModule>();
    // Anchor 0: score 0.9, class 0, box [10, 10, 30, 30]
    // Anchor 1: score 0.8, class 0, box [11, 11, 31, 31] (high overlap with anchor 0, same class)
    // Anchor 2: score 0.85, class 1, box [10, 10, 30, 30] (high overlap with anchor 0, different
    // class)
    module->boxes = {10.0F, 10.0F, 30.0F, 30.0F, 11.0F, 11.0F,
                     31.0F, 31.0F, 10.0F, 10.0F, 30.0F, 30.0F};
    module->scores = {0.9F, 0.8F, 0.85F};
    module->classes = {0, 0, 1};

    trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                               300);
    const std::vector<float> pixels(640 * 640 * 3, 0.5F);
    const auto result = model.run(request(pixels), {});

    require(result.boxes.size() == 2, "nms suppresses overlapping box of same class only");
    require(result.boxes[0].score == 0.9F && result.boxes[0].class_id == 0,
            "highest score box retained");
    require(result.boxes[1].score == 0.85F && result.boxes[1].class_id == 1,
            "different class box retained despite overlap");
}

void test_invalid_inputs_and_configs() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                               300);
    const std::vector<float> pixels(640 * 640 * 3, 0.5F);
    const auto input = request(pixels);

    const trtmc::internal::ConfigEntry unsupported{"score_threshold", 0.5};
    rejects<trtmc::internal::ConfigError>([&] { model.run(input, {&unsupported, 1}); },
                                          "unsupported config must fail");

    auto invalid = input;
    invalid.image.byte_size -= sizeof(float);
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject truncated image");

    invalid = input;
    invalid.image.channels = 4;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject non-RGB image");

    invalid = input;
    invalid.image.format = trtmc::internal::ImageFormat::UInt8;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject non-float32 image");

    invalid = input;
    invalid.image.height = 0;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject zero height");

    invalid = input;
    invalid.image.width = 0;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject zero width");

    invalid = input;
    invalid.image.data = nullptr;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject null image data");

    require(recording->calls == 0, "invalid inputs and configs must fail before engine forward");
}

void test_invalid_engine_outputs() {
    const std::vector<float> pixels(640 * 640 * 3, 0.5F);
    const auto input = request(pixels);

    {
        auto module = std::make_unique<RecordingModule>();
        module->has_boxes = false;
        trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                                   300);
        rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject missing boxes tensor");
    }
    {
        auto module = std::make_unique<RecordingModule>();
        module->has_scores = false;
        trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                                   300);
        rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject missing scores tensor");
    }
    {
        auto module = std::make_unique<RecordingModule>();
        module->has_classes = false;
        trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                                   300);
        rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject missing classes tensor");
    }
    {
        auto module = std::make_unique<RecordingModule>();
        module->boxes_dtype = trtmc::DType::kFloat16;
        trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                                   300);
        rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject wrong boxes dtype");
    }
    {
        auto module = std::make_unique<RecordingModule>();
        module->scores.pop_back();
        trtmc::Yolo11ObjectDetectionPipeline model(std::move(module), preprocessing(), 0.25F, 0.7F,
                                                   300);
        rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject tensor length mismatch");
    }
}

} // namespace

int main() {
    try {
        test_binding_and_complete_owned_boxes();
        test_suppression_and_nms();
        test_invalid_inputs_and_configs();
        test_invalid_engine_outputs();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
