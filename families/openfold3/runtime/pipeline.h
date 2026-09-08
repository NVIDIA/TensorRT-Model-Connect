/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/openfold3/runtime/engine_contract.h"
#include "families/openfold3/runtime/feature_bundle.h"
#include "families/openfold3/runtime/random_samples.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

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

struct StructureConfidence {
    float ptm{0.0F};
    float iptm{0.0F};
    float average_plddt{0.0F};
    std::vector<float> plddt;
};

class OpenFold3Pipeline final : public IStructurePrediction {
  public:
    OpenFold3Pipeline(EngineSet engines, BundleArtifacts artifacts);

    StructurePredictionResult predict_structure(const std::string& input) override;

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
    std::string resultMetadata(const StructureConfidence& confidence) const;

    EngineSet engines_;
    BundleArtifacts artifacts_;
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
