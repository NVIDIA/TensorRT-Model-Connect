#pragma once

#include "trtmc/runtime/tensor.h"

#include <array>
#include <cstddef>

namespace trtmc::runtime::builders::audio {

enum class VoxCPM2TensorDTypeContract {
    kInt8,
    kInt32,
    kFloat32,
    kFloat32OrBFloat16,
};

struct VoxCPM2TensorContract {
    const char* name;
    VoxCPM2TensorDTypeContract dtype_contract;
    std::size_t rank;
    const char* symbolic_shape;
};

struct VoxCPM2ComponentSpec {
    const char* name;
    const char* engine_section;
    const char* input_artifact;
    const char* output_artifact;
    VoxCPM2TensorContract input_tensor;
    VoxCPM2TensorContract output_tensor;
};

inline constexpr VoxCPM2TensorContract kVoxCPM2TextUtf8Tensor{
    "text_utf8", VoxCPM2TensorDTypeContract::kInt8, 1, "utf8_bytes"};
inline constexpr VoxCPM2TensorContract kVoxCPM2TextTokensTensor{
    "text_tokens", VoxCPM2TensorDTypeContract::kInt32, 1, "text_tokens"};
inline constexpr VoxCPM2TensorContract kVoxCPM2TextMaskTensor{
    "text_mask", VoxCPM2TensorDTypeContract::kFloat32OrBFloat16, 1, "text_tokens"};
inline constexpr VoxCPM2TensorContract kVoxCPM2AudioMaskTensor{
    "audio_mask", VoxCPM2TensorDTypeContract::kFloat32OrBFloat16, 1, "text_tokens"};
inline constexpr VoxCPM2TensorContract kVoxCPM2AudioFeatsTensor{
    "audio_feats", VoxCPM2TensorDTypeContract::kFloat32OrBFloat16, 3,
    "text_steps,patch_size,feat_dim"};
inline constexpr VoxCPM2TensorContract kVoxCPM2LocalTextFeaturesTensor{
    "local_text_features", VoxCPM2TensorDTypeContract::kFloat32OrBFloat16, 2,
    "text_steps,feat_dim"};
inline constexpr VoxCPM2TensorContract kVoxCPM2SemanticLmStatesTensor{
    "semantic_lm_states", VoxCPM2TensorDTypeContract::kFloat32OrBFloat16, 2,
    "lm_steps,lm_hidden_size"};
inline constexpr VoxCPM2TensorContract kVoxCPM2ResidualHiddenTensor{
    "residual_hidden", VoxCPM2TensorDTypeContract::kFloat32OrBFloat16, 2,
    "lm_steps,residual_hidden_size"};
inline constexpr VoxCPM2TensorContract kVoxCPM2AudioVaeLatentsTensor{
    "audio_vae_latents", VoxCPM2TensorDTypeContract::kFloat32OrBFloat16, 2,
    "audio_frames,audio_vae_latent_dim"};
inline constexpr VoxCPM2TensorContract kVoxCPM2WaveformF32Tensor{
    "waveform_f32", VoxCPM2TensorDTypeContract::kFloat32, 1, "audio_samples"};

inline constexpr std::array<VoxCPM2ComponentSpec, 5> kVoxCPM2ComponentSpecs{{
    {"locenc", "locenc_engine_plan", "audio_feats", "local_text_features",
     kVoxCPM2AudioFeatsTensor, kVoxCPM2LocalTextFeaturesTensor},
    {"tslm", "tslm_engine_plan", "local_text_features", "semantic_lm_states",
     kVoxCPM2LocalTextFeaturesTensor, kVoxCPM2SemanticLmStatesTensor},
    {"ralm", "ralm_engine_plan", "semantic_lm_states", "residual_hidden",
     kVoxCPM2SemanticLmStatesTensor, kVoxCPM2ResidualHiddenTensor},
    {"locdit", "locdit_engine_plan", "residual_hidden", "audio_vae_latents",
     kVoxCPM2ResidualHiddenTensor, kVoxCPM2AudioVaeLatentsTensor},
    {"audiovae", "audiovae_engine_plan", "audio_vae_latents", "waveform_f32",
     kVoxCPM2AudioVaeLatentsTensor, kVoxCPM2WaveformF32Tensor},
}};

inline const char* voxcpm2_dtype_contract_name(VoxCPM2TensorDTypeContract dtype_contract) {
    switch (dtype_contract) {
    case VoxCPM2TensorDTypeContract::kInt8:
        return "int8";
    case VoxCPM2TensorDTypeContract::kInt32:
        return "int32";
    case VoxCPM2TensorDTypeContract::kFloat32:
        return "float32";
    case VoxCPM2TensorDTypeContract::kFloat32OrBFloat16:
        return "float32|bfloat16";
    }
    return "unknown";
}

inline bool voxcpm2_dtype_matches(VoxCPM2TensorDTypeContract dtype_contract, ::trtmc::DType dtype) {
    switch (dtype_contract) {
    case VoxCPM2TensorDTypeContract::kInt8:
        return dtype == ::trtmc::DType::kInt8;
    case VoxCPM2TensorDTypeContract::kInt32:
        return dtype == ::trtmc::DType::kInt32;
    case VoxCPM2TensorDTypeContract::kFloat32:
        return dtype == ::trtmc::DType::kFloat32;
    case VoxCPM2TensorDTypeContract::kFloat32OrBFloat16:
        return dtype == ::trtmc::DType::kFloat32 || dtype == ::trtmc::DType::kBFloat16;
    }
    return false;
}

} // namespace trtmc::runtime::builders::audio
