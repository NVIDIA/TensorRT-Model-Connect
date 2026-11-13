/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/audio.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <vector>

namespace trtmc {

struct GemmaAudioConfig {
    int sample_rate{16000}, frame_length{320}, hop_length{160}, fft_length{512};
    int mel_bins{128}, max_frames{0}, hidden_size{0};
    int audio_token{0}, begin_token{0}, end_token{0};
    double input_scale{1}, preemphasis{0}, mel_floor{0.001};
    bool preemphasis_htk{true};
    bool waveform{false}; // Unified input: mel_bins is the raw samples-per-token width.
    std::vector<double> mean, stddev;
    std::vector<float> window;
    std::vector<double> mel_filters;
};

struct GemmaAudioFeatures {
    std::vector<float> mel;
    std::vector<int32_t> mask;
    int soft_tokens{0};
};

GemmaAudioFeatures gemma4_audio_features(const internal::AudioView&, const GemmaAudioConfig&);

class GemmaAudioEncoder {
  public:
    GemmaAudioEncoder(std::unique_ptr<ITrtModule>, GemmaAudioConfig);
    const GemmaAudioConfig& config() const { return config_; }
    std::vector<float> encode(const GemmaAudioFeatures&);

  private:
    std::unique_ptr<ITrtModule> module_;
    GemmaAudioConfig config_;
};

} // namespace trtmc
