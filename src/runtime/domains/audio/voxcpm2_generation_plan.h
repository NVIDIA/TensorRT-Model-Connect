#pragma once

#include "runtime/domains/audio/voxcpm2_component_contract.h"
#include "runtime/domains/audio/voxcpm2_config.h"

#include <array>
#include <cstddef>
#include <sstream>
#include <string>

namespace trtmc::runtime::builders::audio {

enum class VoxCPM2StageKind {
    kLocEnc,
    kTslm,
    kRalm,
    kLocDit,
    kAudioVae,
};

struct VoxCPM2GenerationStage {
    VoxCPM2StageKind kind;
    const char* name;
    const char* engine_section;
    const char* input_artifact;
    const char* output_artifact;
    VoxCPM2TensorContract input_tensor;
    VoxCPM2TensorContract output_tensor;
};

inline constexpr std::array<VoxCPM2StageKind, 5> kVoxCPM2StageKinds{{
    VoxCPM2StageKind::kLocEnc,
    VoxCPM2StageKind::kTslm,
    VoxCPM2StageKind::kRalm,
    VoxCPM2StageKind::kLocDit,
    VoxCPM2StageKind::kAudioVae,
}};

inline constexpr std::array<VoxCPM2GenerationStage, 5> kVoxCPM2GenerationStages{{
    {kVoxCPM2StageKinds[0], kVoxCPM2ComponentSpecs[0].name,
     kVoxCPM2ComponentSpecs[0].engine_section, kVoxCPM2ComponentSpecs[0].input_artifact,
     kVoxCPM2ComponentSpecs[0].output_artifact, kVoxCPM2ComponentSpecs[0].input_tensor,
     kVoxCPM2ComponentSpecs[0].output_tensor},
    {kVoxCPM2StageKinds[1], kVoxCPM2ComponentSpecs[1].name,
     kVoxCPM2ComponentSpecs[1].engine_section, kVoxCPM2ComponentSpecs[1].input_artifact,
     kVoxCPM2ComponentSpecs[1].output_artifact, kVoxCPM2ComponentSpecs[1].input_tensor,
     kVoxCPM2ComponentSpecs[1].output_tensor},
    {kVoxCPM2StageKinds[2], kVoxCPM2ComponentSpecs[2].name,
     kVoxCPM2ComponentSpecs[2].engine_section, kVoxCPM2ComponentSpecs[2].input_artifact,
     kVoxCPM2ComponentSpecs[2].output_artifact, kVoxCPM2ComponentSpecs[2].input_tensor,
     kVoxCPM2ComponentSpecs[2].output_tensor},
    {kVoxCPM2StageKinds[3], kVoxCPM2ComponentSpecs[3].name,
     kVoxCPM2ComponentSpecs[3].engine_section, kVoxCPM2ComponentSpecs[3].input_artifact,
     kVoxCPM2ComponentSpecs[3].output_artifact, kVoxCPM2ComponentSpecs[3].input_tensor,
     kVoxCPM2ComponentSpecs[3].output_tensor},
    {kVoxCPM2StageKinds[4], kVoxCPM2ComponentSpecs[4].name,
     kVoxCPM2ComponentSpecs[4].engine_section, kVoxCPM2ComponentSpecs[4].input_artifact,
     kVoxCPM2ComponentSpecs[4].output_artifact, kVoxCPM2ComponentSpecs[4].input_tensor,
     kVoxCPM2ComponentSpecs[4].output_tensor},
}};

struct VoxCPM2GenerationPlan {
    ::trtmc::VoxCPM2Config config;
    std::array<VoxCPM2GenerationStage, 5> stages{kVoxCPM2GenerationStages};
    const char* output_wav_artifact{"trt_output.wav"};
};

inline VoxCPM2GenerationPlan make_voxcpm2_generation_plan(const ::trtmc::VoxCPM2Config& cfg) {
    return VoxCPM2GenerationPlan{cfg, kVoxCPM2GenerationStages, "trt_output.wav"};
}

inline bool voxcpm2_generation_plan_matches_component_contract() {
    if (kVoxCPM2GenerationStages.size() != kVoxCPM2ComponentSpecs.size())
        return false;
    for (std::size_t i = 0; i < kVoxCPM2GenerationStages.size(); ++i) {
        if (std::string(kVoxCPM2GenerationStages[i].name) != kVoxCPM2ComponentSpecs[i].name)
            return false;
        if (std::string(kVoxCPM2GenerationStages[i].engine_section) !=
            kVoxCPM2ComponentSpecs[i].engine_section)
            return false;
        if (std::string(kVoxCPM2GenerationStages[i].input_artifact) !=
            kVoxCPM2ComponentSpecs[i].input_artifact)
            return false;
        if (std::string(kVoxCPM2GenerationStages[i].output_artifact) !=
            kVoxCPM2ComponentSpecs[i].output_artifact)
            return false;
        if (std::string(kVoxCPM2GenerationStages[i].input_tensor.name) !=
            kVoxCPM2ComponentSpecs[i].input_artifact)
            return false;
        if (std::string(kVoxCPM2GenerationStages[i].output_tensor.name) !=
            kVoxCPM2ComponentSpecs[i].output_artifact)
            return false;
        if (kVoxCPM2GenerationStages[i].input_tensor.rank !=
            kVoxCPM2ComponentSpecs[i].input_tensor.rank)
            return false;
        if (kVoxCPM2GenerationStages[i].output_tensor.rank !=
            kVoxCPM2ComponentSpecs[i].output_tensor.rank)
            return false;
    }
    return true;
}

inline std::string describe_voxcpm2_generation_plan(const VoxCPM2GenerationPlan& plan) {
    std::ostringstream os;
    os << "VoxCPM2 generation plan: ";
    for (std::size_t i = 0; i < plan.stages.size(); ++i) {
        if (i > 0)
            os << " -> ";
        os << plan.stages[i].name << "(" << plan.stages[i].input_artifact << "=>"
           << plan.stages[i].output_artifact << ", section=" << plan.stages[i].engine_section
           << ", input=" << voxcpm2_dtype_contract_name(plan.stages[i].input_tensor.dtype_contract)
           << "[" << plan.stages[i].input_tensor.symbolic_shape << "]"
           << ", output="
           << voxcpm2_dtype_contract_name(plan.stages[i].output_tensor.dtype_contract) << "["
           << plan.stages[i].output_tensor.symbolic_shape << "]"
           << ")";
    }
    os << "; output_wav_artifact=" << plan.output_wav_artifact
       << "; sample_rate=" << plan.config.sample_rate << "; cfg_value=" << plan.config.cfg_value
       << "; inference_timesteps=" << plan.config.inference_timesteps;
    return os.str();
}

} // namespace trtmc::runtime::builders::audio
