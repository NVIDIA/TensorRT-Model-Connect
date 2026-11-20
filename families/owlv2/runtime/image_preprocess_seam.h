/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct Owlv2PreprocessConfig {
    std::int32_t input_image_h{960};
    std::int32_t input_image_w{960};
    std::vector<float> image_mean{0.48145466F, 0.4578275F, 0.40821073F};
    std::vector<float> image_std{0.26862954F, 0.26130258F, 0.27577711F};
};

// The side of the square the source is padded to: the longer edge, with the
// padding on the bottom and the right. Exposed so the unit test can pin the
// rule without running an engine.
std::int32_t compute_owlv2_pad_side(std::int32_t image_height, std::int32_t image_width);

// The anti-aliasing sigma for a pad_side -> target resize, or 0 when the
// resize is an upsample. The reference skips the filter entirely at 0, so this
// returning 0 means "no blur", not "a blur with no width".
float compute_owlv2_antialias_sigma(std::int32_t pad_side, std::int32_t target);

std::vector<float> preprocess_owlv2_image(const float* image_pixels, std::int32_t image_height,
                                          std::int32_t image_width,
                                          const Owlv2PreprocessConfig& config);

} // namespace trtmc
