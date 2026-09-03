/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/boltz2/engine_contract.h"
#include "runtime/models/boltz2/feature_bundle.h"
#include "runtime/models/boltz2/prepared_request.h"
#include "runtime/models/boltz2/random_samples.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc::boltz2 {

struct EngineSet {
    std::unique_ptr<ITrtModule> input;
    std::unique_ptr<ITrtModule> trunk_init;
    std::unique_ptr<ITrtModule> msa;
    std::array<std::unique_ptr<ITrtModule>, kPairformerSegments> pairformer;
    std::unique_ptr<ITrtModule> conditioning;
    std::unique_ptr<ITrtModule> score_input;
    std::array<std::unique_ptr<ITrtModule>, kTokenSegments> score_token;
    std::unique_ptr<ITrtModule> score_output;
    std::unique_ptr<ITrtModule> confidence;
};

struct BundleArtifacts {
    FeatureBundle features;
    std::string request;
    std::string structure_metadata_json;
    RandomSamples random_samples;
};

class Boltz2Pipeline final : public IPipeline {
  public:
    Boltz2Pipeline(EngineSet engines, BundleArtifacts artifacts, std::string model_id,
                   std::string preprocessor_python);

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Boltz2StructurePredictionPipeline"; }
    StructurePredictionResult predict_structure(const std::string& input,
                                                const StructurePredictionConfig& cfg = {}) override;
    std::string prepare_structure_input(const std::string& input,
                                        const std::string& input_path) override;

  private:
    void validateAndBindEngines();
    void validateStreams();
    void configureProfile();
    void activateRequest(PreparedRequest request);
    void uploadFeatures();
    void allocateRuntimeTensors();
    void bindTrunkEngines();
    ITrtModule* bindPairformerEngines();
    void bindDiffusionEngines(ITrtModule& trunk_output);
    void bindConfidenceEngine(ITrtModule& trunk_output);
    void runTrunk();
    void runConditioning();
    std::vector<float> runDiffusionScore(const std::vector<float>& model_input, float time_value);
    std::vector<float> sampleCoordinates(int32_t seed, int32_t sampling_steps);
    StructureConfidence runConfidence(const std::vector<float>& coordinates);
    std::string writeStructure(const std::vector<float>& coordinates, StructureFormat format,
                               const StructureConfidence& confidence) const;
    std::string resultMetadata(const StructurePredictionConfig& cfg,
                               const StructureConfidence& confidence) const;

    const FeatureTensor& feature(std::string_view name) const;
    void bindFeature(ITrtModule& module, std::string_view name);

    EngineSet engines_;
    BundleArtifacts artifacts_;
    std::string model_id_;
    std::string preprocessor_python_;
    int token_count_{0};
    int atom_count_{0};
    int active_token_count_{0};
    int active_atom_count_{0};
    cudaStream_t stream_{nullptr};
    std::unordered_map<std::string, DeviceTensor> device_features_;
    DeviceTensor zero_s_;
    DeviceTensor zero_z_;
    DeviceTensor r_noisy_;
    DeviceTensor time_;
    DeviceTensor x_pred_;
};

} // namespace trtmc::boltz2
