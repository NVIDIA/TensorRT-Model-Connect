/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::cli::io {

struct LoadedImage {
    std::vector<float> pixels;
    std::int32_t height{0};
    std::int32_t width{0};

    bool empty() const { return pixels.empty(); }
};

struct LoadedAudio {
    std::vector<float> samples; // Interleaved, without implicit downmixing.
    std::int32_t sample_rate{0};
    std::int32_t channels{0};
};

AudioResult read_wav(const std::string& path);
void write_wav(const AudioResult& audio, const std::string& path);
LoadedAudio read_wav_interleaved(const std::string& path);
void write_wav_interleaved(Span<const float> samples, std::int32_t sample_rate,
                           std::int32_t channels, const std::string& path);

LoadedImage read_image(const std::string& path);
void save_png(const std::string& path, const std::vector<float>& pixels, int width, int height,
              int channels = 3);
void save_png(const std::string& path, Span<const float> pixels, int width, int height,
              int channels = 3);
void save_png(const ImageResult& image, const std::string& path);

} // namespace trtmc::cli::io
