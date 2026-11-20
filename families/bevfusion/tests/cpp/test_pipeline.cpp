/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/bevfusion/runtime/pipeline.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        if (inputs.size() != 2 || inputs.at("camera_bev").data != camera ||
            inputs.at("lidar_bev").data != lidar)
            throw std::runtime_error("input operands must be forwarded without repacking");
        return {{"map_probabilities", {values.data(), output_shape, output_dtype}}};
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
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {{"camera_bev", {1, 80, 128, 128}}, {"lidar_bev", {1, 256, 128, 128}}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"map_probabilities", {1, 6, 200, 200}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override {
        return name == "camera_bev" || name == "lidar_bev";
    }
    bool has_output(const std::string& name) const override { return name == "map_probabilities"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "camera_bev")
            return {1, 80, 128, 128};
        if (name == "lidar_bev")
            return {1, 256, 128, 128};
        return {1, 6, 200, 200};
    }
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
    const void* camera{nullptr};
    const void* lidar{nullptr};
    std::vector<float> values = std::vector<float>(6 * 200 * 200, 0.25F);
    std::vector<int64_t> output_shape{1, 6, 200, 200};
    trtmc::DType output_dtype{trtmc::DType::kFloat32};
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Exception, class Function>
void rejects(Function function) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("invalid operands were accepted");
}

void test_forward() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::bevfusion::Pipeline pipeline(std::move(module));
    std::vector<float> camera(80 * 128 * 128), lidar(256 * 128 * 128);
    recording->camera = camera.data();
    recording->lidar = lidar.data();
    trtmc::Tensor a{camera.data(), {1, 80, 128, 128}, trtmc::DType::kFloat32};
    trtmc::Tensor b{lidar.data(), {1, 256, 128, 128}, trtmc::DType::kFloat32};
    auto result = pipeline.forward(a, b);
    require(recording->calls == 1 && result.size() == 6 * 200 * 200,
            "execute exactly one complete BEV forward");
    recording->values.assign(result.size(), 0.75F);
    const auto next = pipeline.forward(a, b);
    require(result.front() == 0.25F && next.front() == 0.75F,
            "results own their probability buffers across calls");
    auto invalid = a;
    invalid.shape = {1, 128, 128, 80};
    rejects<std::invalid_argument>([&] { pipeline.forward(invalid, b); });
    invalid = a;
    invalid.dtype = trtmc::DType::kFloat16;
    rejects<std::invalid_argument>([&] { pipeline.forward(invalid, b); });
    invalid = a;
    invalid.data = nullptr;
    rejects<std::invalid_argument>([&] { pipeline.forward(invalid, b); });
    require(recording->calls == 2, "reject invalid operands before engine execution");
    recording->output_shape = {1, 200, 200, 6};
    rejects<std::runtime_error>([&] { pipeline.forward(a, b); });
    recording->output_shape = {1, 6, 200, 200};
    recording->output_dtype = trtmc::DType::kFloat16;
    rejects<std::runtime_error>([&] { pipeline.forward(a, b); });
    recording->output_dtype = trtmc::DType::kFloat32;
    trtmc::internal::SpatialFeaturesToClassScoresRequest request{
        {{"lidar_bev", {lidar.data(), lidar.size()}, 256, 128, 128},
         {"camera_bev", {camera.data(), camera.size()}, 80, 128, 128}}};
    const auto scores = pipeline.run(request, {});
    require(scores.classes == 6 && scores.height == 200 && scores.width == 200 &&
                scores.class_names.front() == "drivable_area" &&
                scores.class_names.back() == "divider" &&
                scores.kind == trtmc::internal::ScoreKind::Probability,
            "Task returns independent class probabilities with the source class order");
    const auto bindings = pipeline.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "spatial_features_to_class_scores" &&
                bindings[0].fields.empty(),
            "publish exactly the implemented stateless Task");
    const auto calls = recording->calls;
    request.inputs[0].name = "unknown";
    rejects<std::invalid_argument>([&] { pipeline.run(request, {}); });
    require(recording->calls == calls, "reject unknown feature names before engine execution");
}

} // namespace

int main() {
    try {
        test_forward();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
