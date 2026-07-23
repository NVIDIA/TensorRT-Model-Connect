/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/openpi/api.h"
#include "runtime/models/openpi/config.h"
#include "runtime/models/openpi/paligemma_bpe.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::openpi {

struct PreparedOpenPIInputs {
    std::vector<float> pixel_values;        // [3, 3, 224, 224], NCHW
    std::vector<float> preprocessed_images; // [1, 3, 224, 224, 3], NHWC
    std::vector<int32_t> token_ids;         // [1, 200]
    std::vector<uint8_t> token_mask;        // [1, 200], bool storage
    std::vector<uint8_t> image_mask;        // [1, 3], bool storage
    std::vector<float> normalized_state;    // [1, 32]
    std::vector<uint8_t> prefix_mask;       // [1, 968], TensorRT bool storage
    std::vector<int32_t> prefix_positions;  // [1, 968]
    std::vector<int32_t> suffix_positions;  // [1, H]
    std::vector<float> initial_noise;       // [1, H, 32]
    EulerSchedule schedule;
};

// CPU-only request preparation seam used by the runtime and model-owned tests.
PreparedOpenPIInputs prepare_openpi_inputs(const OpenPIConfig& config,
                                           const OpenPINormalization& normalization,
                                           const PaligemmaBpeTokenizer& tokenizer,
                                           const ActionRequest& request,
                                           bool retain_diagnostics = false);

// Validate both fixed-shape plans before any GPU workspace is allocated.
void validate_openpi_engine_contracts(const ITrtModule& prefill, const ITrtModule& action_step,
                                      const OpenPIConfig& config);

class OpenPIPipeline final : public IPipeline,
                             public IOpenPIActionPipeline,
                             public IOpenPIDiagnosticPipeline {
  public:
    OpenPIPipeline(std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> action_step,
                   OpenPIConfig config, OpenPINormalization normalization,
                   PaligemmaBpeTokenizer tokenizer, std::string model_id);
    ~OpenPIPipeline() override;

    OpenPIPipeline(const OpenPIPipeline&) = delete;
    OpenPIPipeline& operator=(const OpenPIPipeline&) = delete;

    ActionResult predict_actions(const ActionRequest& request) override;
    ActionDiagnosticResult predict_actions_with_diagnostics(const ActionRequest& request) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "OpenPIPipeline"; }

  private:
    struct DeviceWorkspace;

    void ensure_workspace();
    void upload_request(const PreparedOpenPIInputs& inputs);
    std::vector<float> execute_device_resident_flow(const PreparedOpenPIInputs& inputs,
                                                    double& prefill_ms, double& denoise_ms);
    ActionDiagnosticResult execute_diagnostic_flow(const PreparedOpenPIInputs& inputs,
                                                   double preprocess_ms);

    // The action module borrows the prefill module's stream. Destruction is
    // therefore explicit in ~OpenPIPipeline: action first, prefill second.
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> action_step_;
    OpenPIConfig config_;
    OpenPINormalization normalization_;
    PaligemmaBpeTokenizer tokenizer_;
    std::string model_id_;
    std::unique_ptr<DeviceWorkspace> workspace_;
};

} // namespace trtmc::openpi
