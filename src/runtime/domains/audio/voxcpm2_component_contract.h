#pragma once

#include <array>

namespace trtmc::runtime::builders::audio {

struct VoxCPM2ComponentSpec {
    const char* name;
    const char* engine_section;
};

inline constexpr std::array<VoxCPM2ComponentSpec, 5> kVoxCPM2ComponentSpecs{{
    {"locenc", "locenc_engine_plan"},
    {"tslm", "tslm_engine_plan"},
    {"ralm", "ralm_engine_plan"},
    {"locdit", "locdit_engine_plan"},
    {"audiovae", "audiovae_engine_plan"},
}};

} // namespace trtmc::runtime::builders::audio
