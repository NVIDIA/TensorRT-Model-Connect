/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_engine_contract.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace trtmc::sam2 {

struct DecodedSam2Jpeg {
    std::int32_t height{kOriginalImageHeight};
    std::int32_t width{kOriginalImageWidth};
    // Tightly packed HWC RGB uint8 pixels. The decoder accepts only the
    // delivered 1088x1280, three-component JPEG contract.
    std::vector<std::uint8_t> rgb_hwc;
};

// Decode caller-owned compressed bytes using the source-compatible libjpeg
// settings. Taking the vector by value prevents a caller mutation from racing
// the decoder; lvalue callers receive an owned copy and rvalue callers transfer
// ownership without another copy.
[[nodiscard]] DecodedSam2Jpeg decodeSam2JpegBytes(std::vector<std::uint8_t> encoded_jpeg);

// Snapshot a non-symlink regular file into owned memory, then decode it through
// the same byte API. Directories, devices, pipes, empty files, oversized files,
// short reads, and files whose size changes during the snapshot are rejected.
[[nodiscard]] DecodedSam2Jpeg decodeSam2JpegFile(const std::filesystem::path& path);

} // namespace trtmc::sam2
