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

AudioResult read_wav(const std::string& path);
void write_wav(const AudioResult& audio, const std::string& path);

LoadedImage read_image(const std::string& path);
void save_png(const std::string& path, const std::vector<float>& pixels, int width, int height);
void save_png(const ImageResult& image, const std::string& path);

} // namespace trtmc::cli::io
