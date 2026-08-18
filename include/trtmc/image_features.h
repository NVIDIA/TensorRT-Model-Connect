/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct ImageFeaturesResult {
    std::vector<float> last_hidden_state;
    std::vector<int64_t> last_hidden_state_shape;
    std::vector<float> pooler_output;
    std::vector<int64_t> pooler_output_shape;
};

// Optional capability implemented only by image-feature model pipelines.
// Keeping it separate from IPipeline preserves the optimized-runtime ABI.
class IImageFeatureExtractor {
  public:
    virtual ~IImageFeatureExtractor();
    virtual ImageFeaturesResult extract_image_features(const float* pixels, int32_t height,
                                                       int32_t width) = 0;
};

} // namespace trtmc
