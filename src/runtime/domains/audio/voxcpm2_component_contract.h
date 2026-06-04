#pragma once

#include <array>

namespace trtmc::runtime::builders::audio {

struct VoxCPM2ComponentSpec {
    const char* name;
    const char* engine_section;
    const char* input_artifact;
    const char* output_artifact;
};

inline constexpr std::array<VoxCPM2ComponentSpec, 5> kVoxCPM2ComponentSpecs{{
    {"locenc", "locenc_engine_plan", "text_utf8", "local_text_features"},
    {"tslm", "tslm_engine_plan", "local_text_features", "semantic_lm_states"},
    {"ralm", "ralm_engine_plan", "semantic_lm_states", "acoustic_residual_states"},
    {"locdit", "locdit_engine_plan", "acoustic_residual_states", "audio_vae_latents"},
    {"audiovae", "audiovae_engine_plan", "audio_vae_latents", "waveform_f32"},
}};

} // namespace trtmc::runtime::builders::audio
