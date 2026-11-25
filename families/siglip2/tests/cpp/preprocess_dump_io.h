/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// The file siglip2_preprocess_dump writes: int32 grid_height, int32 grid_width, float32
// pixel_values, float32 taps.
#pragma once

#include "families/siglip2/runtime/naflex_preprocess.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace trtmc {

inline void write_siglip2_preprocess_dump(const std::string& path, const Siglip2Patches& patches) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&patches.grid_height), sizeof(std::int32_t));
    output.write(reinterpret_cast<const char*>(&patches.grid_width), sizeof(std::int32_t));
    output.write(reinterpret_cast<const char*>(patches.pixel_values.data()),
                 static_cast<std::streamsize>(patches.pixel_values.size() * sizeof(float)));
    output.write(reinterpret_cast<const char*>(patches.position_weights.data()),
                 static_cast<std::streamsize>(patches.position_weights.size() * sizeof(float)));
    // A short write may only fail when the last buffer is flushed: close before checking.
    output.close();
    if (!output)
        throw std::runtime_error("could not write the output file");
}

} // namespace trtmc
