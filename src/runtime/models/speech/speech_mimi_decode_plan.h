#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

struct MimiDecodeLayout {
    int32_t dec_codebooks{0};
    int32_t dec_frames{0};
    int32_t total_output_elems{0};
    std::size_t input_elems{0};
    std::size_t input_bytes{0};
    std::size_t output_bytes{0};
};

inline MimiDecodeLayout build_mimi_decode_layout(int32_t dec_codebooks, int32_t dec_frames,
                                                 const std::vector<int32_t>& output_dims) {
    MimiDecodeLayout layout;
    layout.dec_codebooks = dec_codebooks;
    layout.dec_frames = dec_frames;
    layout.total_output_elems = 1;
    for (int32_t dim : output_dims) {
        layout.total_output_elems *= dim;
    }
    layout.input_elems = static_cast<std::size_t>(layout.dec_codebooks) * layout.dec_frames;
    layout.input_bytes = layout.input_elems * sizeof(float);
    layout.output_bytes = static_cast<std::size_t>(layout.total_output_elems) * sizeof(float);
    return layout;
}

inline std::vector<float> build_mimi_decoder_input(const std::vector<int32_t>& codec_tokens,
                                                   int32_t num_frames, int32_t actual_codebooks,
                                                   int32_t dec_frames, int32_t dec_codebooks) {
    const auto input_elems = static_cast<std::size_t>(dec_codebooks) * dec_frames;
    std::vector<float> input_tokens(input_elems, 0.0F);
    const int32_t frames_to_copy = std::min(num_frames, dec_frames);
    const int32_t cbs_to_copy = std::min(actual_codebooks, dec_codebooks);
    for (int32_t frame = 0; frame < frames_to_copy; ++frame) {
        for (int32_t cb = 0; cb < cbs_to_copy; ++cb) {
            const auto src_idx = static_cast<std::size_t>(frame) * actual_codebooks + cb;
            const auto dst_idx = static_cast<std::size_t>(cb) * dec_frames + frame;
            if (src_idx < codec_tokens.size()) {
                input_tokens[dst_idx] = static_cast<float>(codec_tokens[src_idx]);
            }
        }
    }
    return input_tokens;
}

inline void waveform_stats(const std::vector<float>& waveform, int32_t total_output_elems,
                           float& rms, float& mx) {
    rms = 0.0F;
    mx = 0.0F;
    for (float sample : waveform) {
        rms += sample * sample;
        mx = std::max(mx, std::abs(sample));
    }
    rms = std::sqrt(rms / std::max(1, total_output_elems));
}

} // namespace trtmc
