/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/qwen_vl/decoded_image.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

// VL preprocessing config parsed from bundle's config.json + preprocessor_config.json.
struct QwenVlPreprocessConfig {
    int32_t fixed_image_size{448};
    int32_t patch_size{14};
    int32_t merge_size{2};
    int32_t temporal_patch_size{2};
    int32_t in_channels{3};
    float image_mean[3]{0.48145466F, 0.4578275F, 0.40821073F};
    float image_std[3]{0.26862954F, 0.26130258F, 0.27577711F};
    int32_t num_image_pad_tokens{256};
    int32_t image_token_id{-1};
    int32_t vision_output_dim{0};
    std::string vl_prompt_template;
    std::string image_token_str;
    // Preprocessing strategy: "merge_group_chw", "simple_chw",
    // "center_crop_chw", "aspect_preserve_chw", or "pad_center_chw"
    std::string preprocessor_type{"merge_group_chw"};
    // Interpolation mode for resize: "bicubic", "bilinear", or "nearest"
    std::string interpolation{"bicubic"};
};

// Preprocessed pixel values ready for the vision TRT engine.
struct QwenVlPreprocessedImage {
    std::vector<float> pixel_values;     // Layout depends on preprocessor_type
    std::vector<int32_t> image_grid_hws; // [height, width] patch grid for patchified inputs
    int32_t channels{0};                 // C*T for merge_group_chw, C for simple_chw
    int32_t height{0};
    int32_t width{0};
    bool ok{false};
};

// Load and preprocess a single image for the vision encoder.
// Dispatches to the appropriate strategy based on config.preprocessor_type:
//   "merge_group_chw":    [C*T, H, W] with merge-group patch permutation
//   "simple_chw":          [C, H, W] standard resize + normalize
//   "center_crop_chw":     [C, H, W] center-crop to square, then resize + normalize
//   "aspect_preserve_chw": [C, H, W] aspect-ratio-preserving resize + zero-pad
//   "pad_center_chw":      [C, H, W] aspect-ratio-preserving resize + center-pad with mean color
//   "patchify_chw": [num_patches, C, patch_size, patch_size]
//
// NOTE: Only single-image input is supported. Multi-image batching
// is not yet implemented; callers must process one image at a time.
runtime::adapters::io::DecodedImage qwen_vl_decode_image_rgb(const std::string& image_path);

QwenVlPreprocessedImage
qwen_vl_preprocess_decoded_image(const runtime::adapters::io::DecodedImage& image,
                                 const QwenVlPreprocessConfig& config);

QwenVlPreprocessedImage qwen_vl_load_and_preprocess_image(const std::string& image_path,
                                                          const QwenVlPreprocessConfig& config);

// Format a VL prompt with image pad tokens from the template.
// Replaces {image_pads} and {prompt} in vl_prompt_template.
std::string qwen_vl_format_prompt(const std::string& user_prompt,
                                  const QwenVlPreprocessConfig& config);

// Parse QwenVlPreprocessConfig from config.json text.
QwenVlPreprocessConfig qwen_vl_parse_preprocess_config(const std::string& config_text,
                                                       const std::string& preprocessor_config_text);

} // namespace trtmc
