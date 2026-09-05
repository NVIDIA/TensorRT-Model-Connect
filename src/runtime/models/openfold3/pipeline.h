/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/openfold3/engine_contract.h"
#include "runtime/models/openfold3/feature_bundle.h"
#include "runtime/models/openfold3/random_samples.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

namespace trtmc::openfold3 {

struct EngineSet {
    std::unique_ptr<ITrtModule> input;
    std::unique_ptr<ITrtModule> trunk_cycle;
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
    std::string precision;
};

class OpenFold3Pipeline final : public IPipeline {
  public:
    OpenFold3Pipeline(EngineSet engines, BundleArtifacts artifacts, std::string model_id);

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "OpenFold3StructurePredictionPipeline"; }
    StructurePredictionResult predict_structure(const std::string& input,
                                                const StructurePredictionConfig& cfg = {}) override;

  private:
    const FeatureTensor& feature(std::string_view name) const;
    void bindFeature(ITrtModule& module, std::string_view name);
    void validateAndBind();
    void validateEngineSet();
    void validateProfile();
    void validateRandomSamples() const;
    void validateRandomPadding() const;
    void uploadFeaturesAndAllocate();
    void bindInputAndTrunk();
    void bindPairformer();
    void bindDiffusion();
    void bindConfidence();
    void runTrunk();
    std::vector<float> runDiffusionStep(const std::vector<float>& noisy, float time);
    std::vector<float> sampleCoordinates();
    StructureConfidence runConfidence(const std::vector<float>& coordinates);
    std::string writeMmcif(const std::vector<float>& coordinates,
                           const StructureConfidence& confidence) const;
    std::string resultMetadata(const StructurePredictionConfig& cfg,
                               const StructureConfidence& confidence) const;

    EngineSet engines_;
    BundleArtifacts artifacts_;
    std::string model_id_;
    int token_count_{0};
    int atom_count_{0};
    int padded_atom_count_{0};
    cudaStream_t stream_{nullptr};
    std::unordered_map<std::string, DeviceTensor> device_features_;
    DeviceTensor zero_s_;
    DeviceTensor zero_z_;
    DeviceTensor noisy_;
    DeviceTensor time_;
    DeviceTensor confidence_positions_;
    float last_gpde_{0.0F};
    std::vector<float> last_pde_;
    std::vector<float> last_pae_;
};

} // namespace trtmc::openfold3
