/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/pointnet/runtime/pipeline.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& points = inputs.at("point");
        input_shape = points.shape;
        const auto* data = static_cast<const float*>(points.data);
        input_values.assign(data, data + points.numel());
        return {{"pred", {logits.data(), {1, 3, 4}, dtype}}};
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
    bool has_input(const std::string& name) const override { return name == "point"; }
    bool has_output(const std::string& name) const override { return name == "pred"; }
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
    int32_t input_rank(const std::string&) const override { return 3; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};
    std::vector<float> logits{0.1F, 0.9F, 0.2F, 0.3F, 0.8F, 0.1F,
                              0.1F, 0.2F, 0.1F, 0.1F, 0.1F, 0.7F};
    std::vector<float> input_values;
    std::vector<int64_t> input_shape;
    trtmc::DType dtype{trtmc::DType::kFloat32};
};

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

trtmc::internal::PointsToSemanticSegmentationRequest request() {
    static const std::vector<float> points{0, 1, 2, 3, 4, 5, 6, 7, 8};
    return {trtmc::Span<const float>{points.data(), points.size()}, 3, 3};
}

void test_binding_and_segmentation() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::PointNetPipeline model(std::move(module), 3, 4, 3);
    const auto bindings = model.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "points_to_semantic_segmentation" &&
                bindings[0].key.major == 1 && bindings[0].key.minor == 0,
            "family must publish exactly its implemented semantic task");
    require(bindings[0].fields.empty(), "family has no runtime config options");
    require(std::string(model.task()) == "points_to_semantic_segmentation",
            "bundle primary task must match the binding");
    const auto result = model.run(request(), {});
    require(recording->input_shape == std::vector<int64_t>({1, 3, 3}),
            "points must be transposed to the engine [1, C, N] layout");
    require(recording->input_values == std::vector<float>({0, 3, 6, 1, 4, 7, 2, 5, 8}),
            "transpose must preserve XYZ channel-major order");
    require(result.labels == std::vector<int32_t>({1, 0, 3}),
            "labels are argmax per point over the class axis");
    require(result.num_points == 3 && result.num_classes == 4,
            "result metadata must match the request and bundle");
}

void test_rejects_invalid_inputs() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::PointNetPipeline model(std::move(module), 3, 4, 3);
    auto invalid = request();
    invalid.num_points = 4;
    try {
        model.run(invalid, {});
        throw std::runtime_error("expected point-count rejection");
    } catch (const std::invalid_argument&) {
    }
    invalid = request();
    invalid.input_dim = 4;
    try {
        model.run(invalid, {});
        throw std::runtime_error("expected input-dimension rejection");
    } catch (const std::invalid_argument&) {
    }
    require(recording->calls == 0, "invalid inputs must fail before engine execution");
}

} // namespace

int main() {
    try {
        test_binding_and_segmentation();
        test_rejects_invalid_inputs();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
