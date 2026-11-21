/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/internvl/runtime/decoded_image.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

// VL preprocessing config parsed from bundle's config.json + preprocessor_config.json.
struct InternVlPreprocessConfig {
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
    // "dynamic_tiles_chw": tile counts of the processor's crop_to_patches.
    int32_t min_image_tiles{1};
    int32_t max_image_tiles{1};
    bool use_thumbnail{false};
};

// Preprocessed pixel values ready for the vision TRT engine.
struct InternVlPreprocessedImage {
    std::vector<float> pixel_values;     // Layout depends on preprocessor_type
    std::vector<int32_t> image_grid_hws; // [height, width] patch grid for patchified inputs
    int32_t channels{0};                 // C*T for merge_group_chw, C for simple_chw
    int32_t height{0};
    int32_t width{0};
    int32_t num_tiles{1}; // images of [channels, height, width] in pixel_values
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
InternVlPreprocessedImage
internvl_preprocess_decoded_image(const runtime::adapters::io::DecodedImage& image,
                                  const InternVlPreprocessConfig& config);

// Format a VL prompt with image pad tokens from the template.
// Replaces {image_pads} and {prompt} in vl_prompt_template.
std::string internvl_format_prompt(const std::string& user_prompt,
                                   const InternVlPreprocessConfig& config);

// As above, with num_image_pad_tokens pads for each of num_tiles tiles.
std::string internvl_format_prompt(const std::string& user_prompt,
                                   const InternVlPreprocessConfig& config, int32_t num_tiles);

// The (columns, rows) tile grid GotOcr2's get_optimal_tiled_canvas selects:
// the closest aspect ratio, preferring more tiles on a tie until they cover
// twice the image area.
std::pair<int32_t, int32_t> internvl_tile_grid(int32_t height, int32_t width, int32_t tile_size,
                                               int32_t min_tiles, int32_t max_tiles);

// Parse InternVlPreprocessConfig from config.json text.
InternVlPreprocessConfig internvl_parse_preprocess_config(const std::string& config_text);

} // namespace trtmc
