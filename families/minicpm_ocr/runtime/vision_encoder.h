/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minicpm_ocr/runtime/image_processor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::minicpm_ocr {

// Runs the vision engine once per image piece (overview or slice); attention
// never crosses pieces in the reference.
class VisionEncoder {
  public:
    VisionEncoder(std::unique_ptr<ITrtModule> engine, int32_t hidden_size, DType dtype,
                  int32_t patch_values, int32_t max_piece_patches, int32_t merge_size);

    // Appends one row per visual token, piece by piece, as [rows, hidden] in the
    // decoder dtype. Returns the number of appended rows.
    std::size_t encode(const ProcessedImage& image, std::vector<uint8_t>& features);

  private:
    std::unique_ptr<ITrtModule> engine_;
    int32_t hidden_size_;
    DType dtype_;
    int32_t patch_values_;
    int32_t merge_size_;
};

} // namespace trtmc::minicpm_ocr
