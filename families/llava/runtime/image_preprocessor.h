/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/llava/runtime/decoded_image.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {
struct LlavaPreprocessConfig {
    int32_t fixed_image_size{336};
    int32_t num_image_pad_tokens{576};
    int32_t image_token_id{32000};
    int32_t vision_output_dim{4096};
    std::string vl_prompt_template{"USER: {image_pads}\n{prompt} ASSISTANT:"};
    std::string image_token_str{"<image>"};
};
struct LlavaPreprocessedImage {
    std::vector<float> pixel_values;
    std::vector<int32_t> image_grid_hws;
    std::vector<int32_t> hd_crop_valid;
    std::vector<int32_t> hd_token_rows;
    int32_t channels{3};
    int32_t height{336};
    int32_t width{336};
    bool ok{false};
};
LlavaPreprocessedImage llava_preprocess_decoded_image(const runtime::adapters::io::DecodedImage&,
                                                      const LlavaPreprocessConfig&);
std::string llava_format_prompt(const std::string&, const LlavaPreprocessConfig&, int32_t);
LlavaPreprocessConfig llava_parse_preprocess_config(const std::string&);
} // namespace trtmc
