#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {
namespace canary {

struct MelResult {
    std::vector<float> data; // [n_mels, n_frames] row-major
    int32_t n_mels{0};
    int32_t n_frames{0};      // total frames (audio is chunk-padded, so this is the full length)
    int32_t valid_frames{0};  // frames covering the real (pre-chunk-padding) audio
};

MelResult extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                                  int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft,
                                  int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate);

} // namespace canary
} // namespace trtmc
