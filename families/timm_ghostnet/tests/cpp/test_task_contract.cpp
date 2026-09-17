/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_ghostnet/runtime/pipeline.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& image = inputs.at("pixel_values");
        shape = image.shape;
        const auto* data = static_cast<const float*>(image.data);
        pixels.assign(data, data + image.numel());
        return {{"logits", {logits.data(), {1, static_cast<int64_t>(logits.size())}, dtype}}};
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
    bool has_output(const std::string& name) const override { return name == "logits"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return dtype; }
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
    std::vector<float> logits{-2.0F, 4.0F, 0.5F, 3.0F, -1.0F};
    std::vector<float> pixels;
    std::vector<int64_t> shape;
    trtmc::DType dtype{trtmc::DType::kFloat32};
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

trtmc::TimmGhostNetPreprocessConfig preprocessing() {
    return {2, 2, {0.25F, 0.5F, 0.75F}, {0.5F, 0.25F, 0.125F}, 1.0F, "bilinear"};
}

trtmc::internal::ImageToClassScoresRequest request(const std::vector<float>& pixels) {
    return {{pixels.data(), pixels.size() * sizeof(float), 2, 2, 3,
             trtmc::internal::ImageFormat::Float32}};
}

void test_binding_and_complete_owned_logits() {
    trtmc::internal::LabelScoresResult result;
    {
        auto module = std::make_unique<RecordingModule>();
        auto* recording = module.get();
        trtmc::TimmGhostNetImageClassificationPipeline model(std::move(module), preprocessing(), 5,
                                                             "", {});
        const auto bindings = model.task_bindings();
        require(bindings.size() == 1 && bindings[0].key.id == "image_to_class_scores" &&
                    bindings[0].key.major == 1 && bindings[0].key.minor == 0,
                "family must publish exactly its implemented semantic task");
        require(bindings[0].fields.empty(), "family has no runtime config options");
        require(bindings[0].implementation ==
                    static_cast<trtmc::internal::IImageToClassScores*>(&model),
                "bind must preserve the interface subobject address");
        require(std::string(model.task()) == "image_to_class_scores",
                "bundle primary task must match the binding");
        const std::vector<float> pixels(12, 0.75F);
        result = static_cast<trtmc::internal::IImageToClassScores*>(bindings[0].implementation)
                     ->run(request(pixels), {});
        require(result.scores == recording->logits, "return all raw logits in class order");
        require(result.kind == trtmc::internal::ScoreKind::Logit, "scores are logits");
        require(result.labels.empty() && result.vocabulary_id.empty(),
                "unknown vocabulary must not acquire invented identity or labels");
        require(recording->shape == std::vector<int64_t>({1, 3, 2, 2}),
                "preprocessing must preserve the engine NCHW input layout");
        require(recording->pixels == std::vector<float>({1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}),
                "preprocessing must preserve checkpoint normalization");
        recording->logits.assign(5, 0.0F);
        model.run(request(pixels), {});
    }
    require(result.scores == std::vector<float>({-2, 4, 0.5F, 3, -1}),
            "result must own all logits across another call and model destruction");
}

void test_metadata_and_invalid_inputs() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    const std::vector<std::string> labels{"first", "second", "third", "fourth", "fifth"};
    trtmc::TimmGhostNetImageClassificationPipeline model(std::move(module), preprocessing(), 5,
                                                         "test:five-classes", labels);
    const std::vector<float> pixels(12, 0.75F);
    const auto input = request(pixels);
    const auto result = model.run(input, {});
    require(result.vocabulary_id == "test:five-classes" && result.labels == labels,
            "return checkpoint-provided identity without reordering");

    const trtmc::internal::ConfigEntry unsupported{"top_k", std::int64_t{1}};
    rejects<trtmc::internal::ConfigError>([&] { model.run(input, {&unsupported, 1}); },
                                          "unsupported config must fail, not truncate logits");
    auto invalid = input;
    invalid.image.byte_size -= sizeof(float);
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject truncated image");
    invalid = input;
    invalid.image.channels = 4;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject non-RGB image");
    invalid = input;
    invalid.image.format = trtmc::internal::ImageFormat::UInt8;
    rejects<std::invalid_argument>([&] { model.run(invalid, {}); }, "reject unsupported format");
    require(recording->calls == 1, "invalid input and config must fail before engine execution");
    recording->logits.pop_back();
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject incomplete class output");
    recording->logits.push_back(0.0F);
    recording->dtype = trtmc::DType::kFloat16;
    rejects<std::runtime_error>([&] { model.run(input, {}); }, "reject wrong output dtype");
}

} // namespace

int main() {
    try {
        test_binding_and_complete_owned_logits();
        test_metadata_and_invalid_inputs();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
