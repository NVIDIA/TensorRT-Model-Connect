/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/uni3d/runtime/pipeline.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kMinPoints = 2;
constexpr std::int32_t kMaxPoints = 4;
constexpr std::int32_t kInputDim = 6;
constexpr std::int32_t kEmbedDim = 3;

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& points = inputs.at("points");
        input_shape = points.shape;
        input_data = points.data;
        return {{"embedding", {embedding.data(), {1, kEmbedDim}, trtmc::DType::kFloat32}}};
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
    bool has_input(const std::string& name) const override { return name == "points"; }
    bool has_output(const std::string& name) const override { return name == "embedding"; }
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
    int32_t input_rank(const std::string&) const override { return 3; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};
    std::vector<float> embedding{3.0F, 0.0F, 4.0F};
    std::vector<int64_t> input_shape;
    const void* input_data{nullptr};
};

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

const std::vector<float>& cloud() {
    static const std::vector<float> points{0.1F, 0.2F, 0.3F, 0.4F, 0.4F,  0.4F, -0.1F, 0.0F, 0.5F,
                                           0.4F, 0.4F, 0.4F, 0.6F, -0.2F, 0.1F, 0.4F,  0.4F, 0.4F};
    return points;
}

trtmc::internal::PointsToEmbeddingRequest request() {
    return {trtmc::Span<const float>{cloud().data(), cloud().size()}, 3, kInputDim};
}

std::unique_ptr<trtmc::Uni3DPipeline> pipeline(RecordingModule** recording) {
    auto module = std::make_unique<RecordingModule>();
    *recording = module.get();
    return std::make_unique<trtmc::Uni3DPipeline>(std::move(module), kMinPoints, kMaxPoints,
                                                  kInputDim, kEmbedDim);
}

void test_binding_and_embedding() {
    RecordingModule* recording = nullptr;
    auto model = pipeline(&recording);
    const auto bindings = model->task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "points_to_embedding" &&
                bindings[0].key.major == 1 && bindings[0].key.minor == 0,
            "family must publish exactly its implemented semantic task");
    require(bindings[0].fields.empty(), "family has no runtime config options");
    require(std::string(model->task()) == "points_to_embedding",
            "bundle primary task must match the binding");

    const auto result = model->run(request(), {});
    require(recording->input_shape == std::vector<int64_t>({1, 3, kInputDim}),
            "points must reach the engine as [1, N, 6]");
    require(recording->input_data == cloud().data(),
            "the row-major request buffer is passed to the engine unchanged");
    require(result.values == std::vector<float>({0.6F, 0.0F, 0.8F}),
            "the embedding is l2-normalized");
    require(!result.embedding_space.empty() && result.pooling == "class_token" &&
                result.normalization == "l2",
            "embedding_space, pooling and normalization must be set for the embedding storage");
}

template <typename Mutate>
void expect_rejected(const char* what, Mutate mutate) {
    RecordingModule* recording = nullptr;
    auto model = pipeline(&recording);
    auto invalid = request();
    mutate(invalid);
    try {
        model->run(invalid, {});
    } catch (const std::invalid_argument&) {
        require(recording->calls == 0, "invalid inputs must fail before engine execution");
        return;
    }
    throw std::runtime_error(std::string("expected rejection: ") + what);
}

void test_rejects_invalid_inputs() {
    using Request = trtmc::internal::PointsToEmbeddingRequest;
    expect_rejected("xyz-only input", [](Request& r) { r.input_dim = 3; });
    expect_rejected("too few points", [](Request& r) {
        r.num_points = 1;
        r.points = trtmc::Span<const float>{cloud().data(), 6};
    });
    expect_rejected("too many points", [](Request& r) { r.num_points = 5; });
    expect_rejected("empty cloud", [](Request& r) {
        r.num_points = 0;
        r.points = trtmc::Span<const float>{nullptr, 0};
    });
    expect_rejected("span size mismatch", [](Request& r) { r.num_points = 2; });
    expect_rejected("null points", [](Request& r) {
        r.points = trtmc::Span<const float>{nullptr, cloud().size()};
    });
}

void test_rejects_a_zero_embedding() {
    RecordingModule* recording = nullptr;
    auto model = pipeline(&recording);
    recording->embedding = {0.0F, 0.0F, 0.0F};
    try {
        model->run(request(), {});
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("expected a zero embedding to be rejected");
}

} // namespace

int main() {
    try {
        test_binding_and_embedding();
        test_rejects_invalid_inputs();
        test_rejects_a_zero_embedding();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
