#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

constexpr int32_t kMagpieCodecInputLimit = 2016;
constexpr int32_t kMagpieCodecSamplesPerFrame = 1024;

struct MagpieCodecPlan {
    int32_t max_codec_frames{0};
    int32_t padded_frames{0};
    int32_t input_len{0};
    std::size_t input_elems{0};
    std::size_t input_bytes{0};
    std::size_t len_bytes{sizeof(int32_t)};
    std::size_t output_elems{0};
    std::size_t output_bytes{0};
    std::size_t valid_samples{0};
};

inline MagpieCodecPlan make_magpie_codec_plan(int32_t num_frames, int32_t num_cb,
                                              int32_t max_codec_frames) {
    MagpieCodecPlan plan;
    plan.max_codec_frames = std::max(max_codec_frames, 0);
    plan.padded_frames = std::min(std::max(num_frames, 0), plan.max_codec_frames);
    plan.input_len = plan.padded_frames;
    plan.input_elems = static_cast<std::size_t>(std::max(num_cb, 0)) *
                       static_cast<std::size_t>(plan.max_codec_frames);
    plan.input_bytes = plan.input_elems * sizeof(int32_t);
    plan.output_elems =
        static_cast<std::size_t>(plan.max_codec_frames) * kMagpieCodecSamplesPerFrame;
    plan.output_bytes = plan.output_elems * sizeof(float);
    plan.valid_samples = static_cast<std::size_t>(plan.padded_frames) * kMagpieCodecSamplesPerFrame;
    return plan;
}

inline std::vector<int32_t> build_magpie_codec_input(const std::vector<int32_t>& codes,
                                                     int32_t num_cb, const MagpieCodecPlan& plan) {
    std::vector<int32_t> codec_input(plan.input_elems, 0);
    for (int32_t f = 0; f < plan.padded_frames; ++f) {
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            const auto src_idx = static_cast<std::size_t>(f) * num_cb + cb;
            if (src_idx >= codes.size()) {
                continue;
            }

            int32_t code = codes[src_idx];
            if (code >= kMagpieCodecInputLimit) {
                code = 0;
            }
            codec_input[static_cast<std::size_t>(cb) * plan.max_codec_frames + f] = code;
        }
    }
    return codec_input;
}

} // namespace trtmc
