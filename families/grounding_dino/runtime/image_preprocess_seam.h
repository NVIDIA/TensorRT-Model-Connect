/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc {

struct GroundingDinoPreprocessConfig {
    std::int32_t input_image_h{800};
    std::int32_t input_image_w{800};
    std::array<float, 3> image_mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> image_std{0.229F, 0.224F, 0.225F};
};

// Resize straight to the engine's square input, then normalise.
//
// Deliberately NOT the checkpoint's aspect-ratio-preserving resize: shifted
// window attention needs a static partition, so the engine is built for one
// fixed size and the aspect ratio is not preserved. The bundle records that
// size, and boxes are mapped back to the original image on the way out.
//
// The CLI delivers pixels already in [0, 1] (apps/cli/io.cpp divides by 255),
// so there is no second division here.
std::vector<float> preprocess_grounding_dino_image(const float* pixels, std::int32_t image_height,
                                                   std::int32_t image_width,
                                                   const GroundingDinoPreprocessConfig& config);

} // namespace trtmc
