/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::openpi {

struct RobotImage {
    std::string name;
    std::vector<float> pixels; // RGB HWC float32 in [0, 1].
    int32_t height{0};
    int32_t width{0};
    int32_t channels{3};
    bool valid{true};
};

struct ActionRequest {
    std::string prompt;
    std::vector<RobotImage> cameras;
    std::vector<float> state;
    std::vector<float> initial_noise;
    int32_t seed{-1};
    int32_t denoise_steps{-1};
};

struct PolicyTimings {
    double preprocess_ms{0.0};
    double prefill_ms{0.0};
    double denoise_ms{0.0};
    double postprocess_ms{0.0};
};

struct ActionResult {
    std::vector<float> actions; // Row-major [horizon, action_dim].
    int32_t horizon{0};
    int32_t action_dim{0};
    PolicyTimings timings;
};

class IOpenPIActionPipeline {
  public:
    virtual ~IOpenPIActionPipeline() = default;
    virtual ActionResult predict_actions(const ActionRequest& request) = 0;
};

enum class DiagnosticTensorType : uint8_t {
    kBool,
    kInt32,
    kBFloat16,
    kFloat32,
};

enum class DiagnosticStage : uint8_t {
    kPreprocess,
    kVision,
    kPrefix,
    kFlow,
    kPostprocess,
};

enum class DiagnosticRole : uint8_t {
    kInput,
    kIntermediate,
    kOutput,
};

struct DiagnosticTensor {
    std::string name;
    DiagnosticStage stage{DiagnosticStage::kPreprocess};
    DiagnosticRole role{DiagnosticRole::kIntermediate};
    DiagnosticTensorType dtype{DiagnosticTensorType::kFloat32};
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

struct ActionDiagnosticResult {
    ActionResult result;
    std::vector<DiagnosticTensor> tensors;
};

// OpenPI qualification capture is model-owned and absent from production
// robot-policy APIs.
class IOpenPIDiagnosticPipeline {
  public:
    virtual ~IOpenPIDiagnosticPipeline() = default;
    virtual ActionDiagnosticResult
    predict_actions_with_diagnostics(const ActionRequest& request) = 0;
};

} // namespace trtmc::openpi
