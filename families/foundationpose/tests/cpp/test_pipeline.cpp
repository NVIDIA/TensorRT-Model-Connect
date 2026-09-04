/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/foundationpose/runtime/pipeline.h"

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

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    explicit FakeModule(bool scorer) : scorer_(scorer) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++forward_calls;
        const auto batch = static_cast<int32_t>(inputs.at("input1").shape.at(0));
        batches.push_back(batch);
        if (scorer_) {
            values_.resize(batch);
            for (int32_t index = 0; index < batch; ++index)
                values_[index] = static_cast<float>(index) - 1.0F;
            return {{"output1", {values_.data(), {1, batch}, trtmc::DType::kFloat32}}};
        }
        translation_.assign(static_cast<std::size_t>(batch) * 3, 0.0F);
        rotation_.assign(static_cast<std::size_t>(batch) * 3, 0.0F);
        for (int32_t index = 0; index < batch; ++index) {
            translation_[static_cast<std::size_t>(index) * 3] = 0.1F;
            rotation_[static_cast<std::size_t>(index) * 3 + 2] = 0.2F;
        }
        return {
            {"output1", {translation_.data(), {batch, 3}, trtmc::DType::kFloat32}},
            {"output2", {rotation_.data(), {batch, 3}, trtmc::DType::kFloat32}},
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
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "input1" || name == "input2";
    }
    bool has_output(const std::string& name) const override {
        return name == "output1" || (!scorer_ && name == "output2");
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
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override { ++reset_calls; }
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    bool scorer_{false};
    int forward_calls{0};
    int reset_calls{0};
    std::vector<int32_t> batches;
    std::vector<float> translation_;
    std::vector<float> rotation_;
    std::vector<float> values_;
};

std::vector<float> identity_poses(int32_t count) {
    std::vector<float> poses(static_cast<std::size_t>(count) * 16, 0.0F);
    for (int32_t index = 0; index < count; ++index) {
        auto* pose = poses.data() + static_cast<std::size_t>(index) * 16;
        pose[0] = pose[5] = pose[10] = pose[15] = 1.0F;
    }
    return poses;
}

trtmc::PoseCropProvider crops(int32_t expected_count, int* calls = nullptr) {
    return [=](const std::vector<float>& poses, trtmc::PoseCropStage, int32_t) mutable {
        if (calls != nullptr)
            ++*calls;
        const auto values = static_cast<std::size_t>(expected_count) * 2 * 2 * 6;
        trtmc::PoseCropBatch result;
        result.rendered_features.assign(values, poses.at(3));
        result.observed_features.assign(values, 0.25F);
        result.num_hypotheses = expected_count;
        result.height = 2;
        result.width = 2;
        result.channels = 6;
        return result;
    };
}

void test_refinement_scoring_and_chunking() {
    auto refiner = std::make_unique<FakeModule>(false);
    auto scorer = std::make_unique<FakeModule>(true);
    auto* refiner_ptr = refiner.get();
    auto* scorer_ptr = scorer.get();
    trtmc::FoundationPosePipeline pipeline(std::move(refiner), std::move(scorer), 2, 2, 6, 2, 5,
                                           10);
    int crop_calls = 0;
    trtmc::PoseEstimationRequest request;
    request.candidate_poses = identity_poses(5);
    request.num_hypotheses = 5;
    request.mesh_diameter = 2.0F;
    request.refinement_iterations = 2;
    request.crop_provider = crops(5, &crop_calls);
    const auto result = pipeline.estimate_pose_hypotheses(request);

    check(result.num_hypotheses == 5 && result.refined_poses.size() == 80,
          "FoundationPose result shape");
    check(result.scores == std::vector<float>({-1.0F, 0.0F, 1.0F, 2.0F, 3.0F}),
          "FoundationPose score logits");
    check(result.best_index == 4, "FoundationPose best hypothesis selection");
    check(result.all_poses_rigid, "FoundationPose decoded transforms remain rigid");
    check(std::abs(result.refined_poses[3] - 0.2F) < 1.0e-6F,
          "FoundationPose iterative metric translation decoding");
    check(refiner_ptr->batches == std::vector<int32_t>({2, 2, 1, 2, 2, 1}),
          "FoundationPose chunks refiner batches at the profile limit");
    check(scorer_ptr->batches == std::vector<int32_t>({5}),
          "FoundationPose scores all hypotheses jointly");
    check(crop_calls == 3, "FoundationPose asks for rerendered crops each iteration and scoring");
}

void test_tracking_and_reset() {
    auto refiner = std::make_unique<FakeModule>(false);
    auto scorer = std::make_unique<FakeModule>(true);
    auto* refiner_ptr = refiner.get();
    auto* scorer_ptr = scorer.get();
    trtmc::FoundationPosePipeline pipeline(std::move(refiner), std::move(scorer), 2, 2, 6, 2, 5,
                                           10);
    trtmc::PoseEstimationRequest request;
    request.candidate_poses = identity_poses(1);
    request.num_hypotheses = 1;
    request.mesh_diameter = 1.0F;
    request.score_hypotheses = false;
    request.crop_provider = crops(1);
    const auto first = pipeline.estimate_pose_hypotheses(request);
    request.candidate_poses.clear();
    request.use_tracked_pose = true;
    const auto tracked = pipeline.estimate_pose_hypotheses(request);
    check(tracked.refined_poses[3] > first.refined_poses[3],
          "FoundationPose tracking refines the selected prior pose");
    pipeline.reset_pose_tracking();
    bool rejected = false;
    try {
        (void)pipeline.estimate_pose_hypotheses(request);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "FoundationPose reset clears tracking state");
    check(refiner_ptr->reset_calls == 1 && scorer_ptr->reset_calls == 1,
          "FoundationPose reset reuses both engine contexts");
}

void test_invalid_contracts() {
    trtmc::FoundationPosePipeline pipeline(std::make_unique<FakeModule>(false),
                                           std::make_unique<FakeModule>(true), 2, 2, 6, 2, 5, 10);
    auto valid = trtmc::PoseEstimationRequest{};
    valid.candidate_poses = identity_poses(1);
    valid.num_hypotheses = 1;
    valid.mesh_diameter = 1.0F;
    valid.score_hypotheses = false;
    valid.crop_provider = crops(1);
    auto rejects = [&](const trtmc::PoseEstimationRequest& request) {
        try {
            (void)pipeline.estimate_pose_hypotheses(request);
            return false;
        } catch (const std::invalid_argument&) {
            return true;
        }
    };
    auto value = valid;
    value.candidate_poses[15] = 0.0F;
    check(rejects(value), "FoundationPose rejects non-rigid candidates");
    value = valid;
    value.mesh_diameter = std::numeric_limits<float>::quiet_NaN();
    check(rejects(value), "FoundationPose rejects invalid mesh scale");
    value = valid;
    value.num_hypotheses = 6;
    value.candidate_poses = identity_poses(6);
    value.crop_provider = crops(6);
    check(rejects(value), "FoundationPose rejects batches above the bundle limit");
    value = valid;
    value.crop_provider = [](const std::vector<float>&, trtmc::PoseCropStage, int32_t) {
        return trtmc::PoseCropBatch{{0.0F}, {0.0F}, 1, 2, 2, 6};
    };
    check(rejects(value), "FoundationPose rejects malformed crop tensors");
    value = valid;
    value.crop_provider = [](const std::vector<float>&, trtmc::PoseCropStage, int32_t) {
        trtmc::PoseCropBatch result;
        result.rendered_features.assign(24, 0.0F);
        result.observed_features.assign(24, 0.0F);
        result.observed_features[3] = std::numeric_limits<float>::infinity();
        result.num_hypotheses = 1;
        result.height = result.width = 2;
        result.channels = 6;
        return result;
    };
    check(rejects(value), "FoundationPose rejects non-finite crop features");
}

} // namespace

int main() {
    test_refinement_scoring_and_chunking();
    test_tracking_and_reset();
    test_invalid_contracts();
    if (failures != 0) {
        std::cerr << failures << " FoundationPose pipeline test(s) failed\n";
        return 1;
    }
    return 0;
}
