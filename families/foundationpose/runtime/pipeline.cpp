/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/foundationpose/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

constexpr float kRotationNormalizer = 0.3490658503988659F;
constexpr float kRigidTolerance = 2.0e-3F;

void validate_module(const ITrtModule* module, const char* label,
                     const char* second_output = nullptr) {
    if (module == nullptr || !module->ok())
        throw std::runtime_error(std::string("FoundationPose: invalid ") + label + " module");
    if (!module->has_input("input1") || !module->has_input("input2") ||
        !module->has_output("output1") ||
        (second_output != nullptr && !module->has_output(second_output)))
        throw std::runtime_error(std::string("FoundationPose: ") + label +
                                 " engine tensor contract mismatch");
}

bool finite_values(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

void require_argument(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

void require_runtime(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

float determinant(const float* pose) {
    return pose[0] * (pose[5] * pose[10] - pose[6] * pose[9]) -
           pose[1] * (pose[4] * pose[10] - pose[6] * pose[8]) +
           pose[2] * (pose[4] * pose[9] - pose[5] * pose[8]);
}

bool valid_homogeneous_row(const float* pose) {
    return std::abs(pose[12]) <= kRigidTolerance && std::abs(pose[13]) <= kRigidTolerance &&
           std::abs(pose[14]) <= kRigidTolerance && std::abs(pose[15] - 1.0F) <= kRigidTolerance;
}

bool orthonormal_rotation(const float* pose) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            float dot = 0.0F;
            for (int axis = 0; axis < 3; ++axis)
                dot += pose[row * 4 + axis] * pose[col * 4 + axis];
            const float expected = row == col ? 1.0F : 0.0F;
            if (std::abs(dot - expected) > kRigidTolerance)
                return false;
        }
    }
    return true;
}

bool rigid_pose(const float* pose) {
    return std::all_of(pose, pose + 16, [](float value) { return std::isfinite(value); }) &&
           valid_homogeneous_row(pose) && orthonormal_rotation(pose) &&
           std::abs(determinant(pose) - 1.0F) <= 3.0F * kRigidTolerance;
}

void apply_delta(float* pose, const float* translation, const float* rotation,
                 float mesh_diameter) {
    const float x = std::tanh(rotation[0]) * kRotationNormalizer;
    const float y = std::tanh(rotation[1]) * kRotationNormalizer;
    const float z = std::tanh(rotation[2]) * kRotationNormalizer;
    const float theta = std::sqrt(x * x + y * y + z * z);
    float a = 1.0F;
    float b = 0.5F;
    if (theta > 1.0e-7F) {
        a = std::sin(theta) / theta;
        b = (1.0F - std::cos(theta)) / (theta * theta);
    }
    // Transpose of the conventional Rodrigues exponential, matching the
    // FoundationPose decoder's so3_exp_map(...).permute(0, 2, 1).
    const float r[9] = {
        1.0F - b * (y * y + z * z), b * x * y + a * z,          b * x * z - a * y,
        b * x * y - a * z,          1.0F - b * (x * x + z * z), b * y * z + a * x,
        b * x * z + a * y,          b * y * z - a * x,          1.0F - b * (x * x + y * y),
    };
    float old[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            old[row * 3 + col] = pose[row * 4 + col];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            float value = 0.0F;
            for (int axis = 0; axis < 3; ++axis)
                value += r[row * 3 + axis] * old[axis * 3 + col];
            pose[row * 4 + col] = value;
        }
    const float scale = mesh_diameter * 0.5F;
    pose[3] += translation[0] * scale;
    pose[7] += translation[1] * scale;
    pose[11] += translation[2] * scale;
}

const Tensor& require_output(const TensorMap& outputs, const char* name, std::size_t values) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.data == nullptr || it->second.dtype != DType::kFloat32 ||
        it->second.numel() != values)
        throw std::runtime_error(std::string("FoundationPose engine returned invalid ") + name);
    return it->second;
}

} // namespace

FoundationPosePipeline::FoundationPosePipeline(std::unique_ptr<ITrtModule> refiner,
                                               std::unique_ptr<ITrtModule> scorer,
                                               int32_t crop_height, int32_t crop_width,
                                               int32_t channels, int32_t max_refiner_batch,
                                               int32_t max_hypotheses,
                                               int32_t max_refinement_iterations)
    : refiner_(std::move(refiner)), scorer_(std::move(scorer)), crop_height_(crop_height),
      crop_width_(crop_width), channels_(channels), max_refiner_batch_(max_refiner_batch),
      max_hypotheses_(max_hypotheses), max_refinement_iterations_(max_refinement_iterations) {
    validate_module(refiner_.get(), "refiner", "output2");
    validate_module(scorer_.get(), "scorer");
    if (crop_height_ <= 0 || crop_width_ <= 0 || channels_ != 6 || max_refiner_batch_ <= 0 ||
        max_hypotheses_ < max_refiner_batch_ || max_refinement_iterations_ <= 0)
        throw std::runtime_error("FoundationPose: invalid bundle contract dimensions");
}

PoseCropBatch FoundationPosePipeline::request_crops(const PoseEstimationRequest& request,
                                                    const std::vector<float>& poses,
                                                    PoseCropStage stage, int32_t iteration) const {
    if (!request.crop_provider)
        throw std::invalid_argument("FoundationPose crop provider is required");
    auto crops = request.crop_provider(poses, stage, iteration);
    const auto count =
        static_cast<std::size_t>(request.num_hypotheses) * crop_height_ * crop_width_ * channels_;
    if (crops.num_hypotheses != request.num_hypotheses || crops.height != crop_height_ ||
        crops.width != crop_width_ || crops.channels != channels_ ||
        crops.rendered_features.size() != count || crops.observed_features.size() != count)
        throw std::invalid_argument("FoundationPose crop provider returned an invalid shape");
    if (!finite_values(crops.rendered_features) || !finite_values(crops.observed_features))
        throw std::invalid_argument("FoundationPose crop features must be finite");
    return crops;
}

PoseEstimationResult
FoundationPosePipeline::estimate_pose_hypotheses(const PoseEstimationRequest& request) {
    auto poses = initial_poses(request);
    validate_request(request, poses);
    PoseEstimationResult result;
    result.num_hypotheses = request.num_hypotheses;
    result.refinement_ms = refine_poses(request, poses);
    if (request.score_hypotheses)
        score_poses(request, poses, result);
    else
        result.best_index = 0;
    result.refined_poses = std::move(poses);
    validate_result(result);
    result.all_poses_rigid = true;
    if (request.update_tracking_state) {
        const auto begin = result.refined_poses.begin() + result.best_index * 16;
        tracked_pose_.assign(begin, begin + 16);
    }
    return result;
}

std::vector<float>
FoundationPosePipeline::initial_poses(const PoseEstimationRequest& request) const {
    if (request.use_tracked_pose) {
        require_argument(request.num_hypotheses == 1,
                         "FoundationPose tracked refinement requires one hypothesis");
        require_argument(request.candidate_poses.empty(),
                         "FoundationPose tracked refinement does not accept candidates");
        require_argument(!tracked_pose_.empty(), "FoundationPose tracking state is empty");
        return tracked_pose_;
    }
    return request.candidate_poses;
}

void FoundationPosePipeline::validate_request(const PoseEstimationRequest& request,
                                              const std::vector<float>& poses) const {
    require_argument(request.num_hypotheses > 0,
                     "FoundationPose hypothesis count must be positive");
    require_argument(request.num_hypotheses <= max_hypotheses_,
                     "FoundationPose hypothesis count exceeds the bundle limit");
    require_argument(poses.size() == static_cast<std::size_t>(request.num_hypotheses) * 16,
                     "FoundationPose candidate pose shape is invalid");
    require_argument(std::isfinite(request.mesh_diameter),
                     "FoundationPose mesh diameter must be finite");
    require_argument(request.mesh_diameter > 0.0F, "FoundationPose mesh diameter must be positive");
    require_argument(request.refinement_iterations > 0,
                     "FoundationPose refinement iterations must be positive");
    require_argument(request.refinement_iterations <= max_refinement_iterations_,
                     "FoundationPose refinement iterations exceed the bundle limit");
    for (int32_t index = 0; index < request.num_hypotheses; ++index)
        require_argument(rigid_pose(poses.data() + static_cast<std::size_t>(index) * 16),
                         "FoundationPose candidates must be rigid transforms");
    require_argument(request.score_hypotheses || request.num_hypotheses == 1,
                     "FoundationPose multi-hypothesis requests require scoring");
}

double FoundationPosePipeline::refine_poses(const PoseEstimationRequest& request,
                                            std::vector<float>& poses) {
    double elapsed_ms = 0.0;
    for (int32_t iteration = 0; iteration < request.refinement_iterations; ++iteration) {
        auto crops = request_crops(request, poses, PoseCropStage::kRefinement, iteration);
        const auto begin = std::chrono::steady_clock::now();
        const auto values_per_crop =
            static_cast<std::size_t>(crop_height_) * crop_width_ * channels_;
        for (int32_t offset = 0; offset < request.num_hypotheses; offset += max_refiner_batch_) {
            const int32_t batch = std::min(max_refiner_batch_, request.num_hypotheses - offset);
            const auto value_offset = static_cast<std::size_t>(offset) * values_per_crop;
            Tensor rendered{crops.rendered_features.data() + value_offset,
                            {batch, crop_height_, crop_width_, channels_},
                            DType::kFloat32};
            Tensor observed{crops.observed_features.data() + value_offset,
                            {batch, crop_height_, crop_width_, channels_},
                            DType::kFloat32};
            const auto outputs = refiner_->forward({{"input1", rendered}, {"input2", observed}});
            const auto& translation = require_output(outputs, "output1", batch * 3U);
            const auto& rotation = require_output(outputs, "output2", batch * 3U);
            const auto* translations = static_cast<const float*>(translation.data);
            const auto* rotations = static_cast<const float*>(rotation.data);
            for (int32_t local = 0; local < batch; ++local)
                apply_delta(poses.data() + static_cast<std::size_t>(offset + local) * 16,
                            translations + local * 3, rotations + local * 3, request.mesh_diameter);
        }
        elapsed_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
    }
    return elapsed_ms;
}

void FoundationPosePipeline::score_poses(const PoseEstimationRequest& request,
                                         const std::vector<float>& poses,
                                         PoseEstimationResult& result) {
    auto crops =
        request_crops(request, poses, PoseCropStage::kScoring, request.refinement_iterations);
    Tensor rendered{crops.rendered_features.data(),
                    {request.num_hypotheses, crop_height_, crop_width_, channels_},
                    DType::kFloat32};
    Tensor observed{crops.observed_features.data(),
                    {request.num_hypotheses, crop_height_, crop_width_, channels_},
                    DType::kFloat32};
    const auto begin = std::chrono::steady_clock::now();
    const auto outputs = scorer_->forward({{"input1", rendered}, {"input2", observed}});
    result.scoring_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    const auto& scores = require_output(outputs, "output1", request.num_hypotheses);
    const auto* score_values = static_cast<const float*>(scores.data);
    result.scores.assign(score_values, score_values + request.num_hypotheses);
    require_runtime(finite_values(result.scores),
                    "FoundationPose scorer returned non-finite logits");
    result.best_index = static_cast<int32_t>(std::distance(
        result.scores.begin(), std::max_element(result.scores.begin(), result.scores.end())));
}

void FoundationPosePipeline::validate_result(const PoseEstimationResult& result) const {
    for (int32_t index = 0; index < result.num_hypotheses; ++index)
        require_runtime(
            rigid_pose(result.refined_poses.data() + static_cast<std::size_t>(index) * 16),
            "FoundationPose decoder produced a non-rigid transform");
}

void FoundationPosePipeline::reset_pose_tracking() {
    tracked_pose_.clear();
    refiner_->reset_execution_context();
    scorer_->reset_execution_context();
}

} // namespace trtmc
