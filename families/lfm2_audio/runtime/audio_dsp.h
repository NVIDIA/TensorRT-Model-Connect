/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::lfm2_audio {
struct MelFeatures {
    // Time-major [frames, 128], including the reference's masked final frame.
    std::vector<float> values;
    std::int32_t frames{0};
};
std::vector<float> resample(const float* audio, std::size_t count, std::int32_t source_rate);
MelFeatures extract_mel(const float* audio, std::size_t count, std::int32_t source_rate,
                        const std::vector<float>& filters);
// Time-major [frames, 1282]: 641 log-magnitudes followed by 641 phases.
std::vector<float> inverse_spectrogram(const std::vector<float>& spectra);

// Streaming overlap-add. append returns only samples that future spectral
// frames cannot change; finish emits the final cropped tail exactly once.
class StreamingIstft {
  public:
    StreamingIstft();
    std::vector<float> append(const float* spectra, std::size_t frames);
    std::vector<float> finish();
    void reset();

  private:
    void flush(std::size_t end, std::vector<float>& output);
    std::array<float, 1280> signal_{}, envelope_{}, window_{};
    std::size_t frames_{0}, finalized_{0};
    bool finished_{false};
};
} // namespace trtmc::lfm2_audio
