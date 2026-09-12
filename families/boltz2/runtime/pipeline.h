/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/boltz2/runtime/engine_contract.h"
#include "families/boltz2/runtime/feature_bundle.h"
#include "families/boltz2/runtime/prepared_request.h"
#include "families/boltz2/runtime/random_samples.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc::boltz2 {

struct EngineSet {
    std::unique_ptr<ITrtModule> input;
    std::unique_ptr<ITrtModule> trunk_init;
    std::unique_ptr<ITrtModule> template_engine;
    std::unique_ptr<ITrtModule> msa;
    std::array<std::unique_ptr<ITrtModule>, kPairformerSegments> pairformer;
    std::unique_ptr<ITrtModule> conditioning;
    std::unique_ptr<ITrtModule> score_input;
    std::array<std::unique_ptr<ITrtModule>, kTokenSegments> score_token;
    std::unique_ptr<ITrtModule> score_output;
    std::unique_ptr<ITrtModule> confidence;
    std::array<std::unique_ptr<ITrtModule>, 2> affinity;
};

struct BundleArtifacts {
    FeatureBundle features;
    std::string request;
    std::string structure_metadata_json;
    RandomSamples random_samples;
};

struct AffinityPrediction {
    float value{0.0F};
    float probability{0.0F};
    std::array<float, 2> member_values{};
    std::array<float, 2> member_probabilities{};
};

class Boltz2Pipeline final : public IStructurePrediction {
  public:
    Boltz2Pipeline(EngineSet engines, BundleArtifacts artifacts);

    StructurePredictionResult predict_structure(const StructurePredictionRequest& request) override;

  private:
    void validateAndBindEngines();
    void validateStreams();
    void configureProfile();
    void activateRequest(PreparedRequest request);
    void uploadFeatures();
    void allocateRuntimeTensors();
    void bindTrunkEngines();
    void bindTemplatePath();
    ITrtModule* bindPairformerEngines();
    void bindDiffusionEngines(ITrtModule& trunk_output);
    void bindConfidenceEngine(ITrtModule& trunk_output);
    void bindAffinityEngines(ITrtModule& trunk_output);
    void bindInputEmbedding(bool affinity);
    void runTrunk(int recycling_steps, bool affinity);
    void runConditioning();
    std::vector<float> runDiffusionScore(const std::vector<float>& model_input, float time_value);
    std::vector<float> sampleCoordinates(int32_t seed, int32_t sampling_steps, int sample_index);
    StructureConfidence runConfidence(const std::vector<float>& coordinates);
    AffinityPrediction runAffinity(const std::vector<float>& coordinates);
    AffinityPrediction predictAffinity(int32_t seed, int32_t sampling_steps);
    std::string writeStructure(const std::vector<float>& coordinates, StructureFormat format,
                               const StructureConfidence& confidence) const;
    std::string resultMetadata(const StructurePredictionConfig& cfg,
                               const StructureConfidence& confidence,
                               const std::optional<AffinityPrediction>& affinity) const;

    const FeatureTensor& feature(std::string_view name) const;
    void bindFeature(ITrtModule& module, std::string_view name);

    EngineSet engines_;
    BundleArtifacts artifacts_;
    int token_count_{0};
    int atom_count_{0};
    int active_token_count_{0};
    int active_atom_count_{0};
    bool use_templates_{false};
    bool has_affinity_{false};
    std::vector<int32_t> confidence_chain_ids_;
    std::vector<std::vector<float>> confidence_chain_pairs_;
    cudaStream_t stream_{nullptr};
    std::unordered_map<std::string, DeviceTensor> device_features_;
    DeviceTensor zero_s_;
    DeviceTensor zero_z_;
    DeviceTensor r_noisy_;
    DeviceTensor time_;
    DeviceTensor x_pred_;
};

} // namespace trtmc::boltz2
