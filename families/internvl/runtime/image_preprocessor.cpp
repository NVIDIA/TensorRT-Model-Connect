/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/internvl/runtime/image_preprocessor.h"

#include "families/internvl/runtime/internvl_resample.h"
#include "image_transform_helper.h"
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

// ---------------------------------------------------------------------------
// Interpolation filter resolution
// ---------------------------------------------------------------------------

static stbir_filter resolve_stbir_filter(const std::string& interpolation) {
    if (interpolation == "bilinear")
        return STBIR_FILTER_TRIANGLE;
    if (interpolation == "nearest")
        return STBIR_FILTER_POINT_SAMPLE;
    // "bicubic" or anything else -> Catmull-Rom (matches PIL BICUBIC)
    return STBIR_FILTER_CATMULLROM;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct LoadedImage {
    std::vector<float> img_chw; // [C, H, W] normalized
    int target_size{0};
    int channels{0};
    bool ok{false};
};

// Resize raw uint8 RGB buffer to target_size x target_size using the given filter.
static std::vector<unsigned char> resize_raw(const unsigned char* raw, int width, int height,
                                             int target_size, stbir_filter filter) {
    std::vector<unsigned char> resized(static_cast<std::size_t>(target_size) * target_size * 3);

    void* result =
        stbir_resize(raw, width, height, width * 3, resized.data(), target_size, target_size,
                     target_size * 3, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, filter);

    if (result == nullptr) {
        return {};
    }
    return resized;
}

// Convert resized uint8 HWC buffer to float32 CHW, normalizing per channel.
static bool normalize_to_chw(const std::vector<unsigned char>& resized, int target_size,
                             const InternVlPreprocessConfig& config, std::vector<float>& out_chw) {
    ImageNormalizationParams params;
    params.width = target_size;
    params.height = target_size;
    params.channels = config.in_channels;
    params.image_mean[0] = config.image_mean[0];
    params.image_mean[1] = config.image_mean[1];
    params.image_mean[2] = config.image_mean[2];
    params.image_std[0] = config.image_std[0];
    params.image_std[1] = config.image_std[1];
    params.image_std[2] = config.image_std[2];
    return normalize_hwc_u8_to_chw(resized, params, out_chw);
}

// ---------------------------------------------------------------------------
// Load strategies
// ---------------------------------------------------------------------------

static LoadedImage load_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                         const InternVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess image: decoded image missing" << std::endl;
        return loaded;
    }

    const int target_size = config.fixed_image_size;

    // 2. Resize to fixed_image_size x fixed_image_size
    auto resized = resize_raw(image.pixels.data(), image.width, image.height, target_size,
                              resolve_stbir_filter(config.interpolation));

    if (resized.empty()) {
        std::cerr << "[trtmc] Failed to resize image" << std::endl;
        return loaded;
    }

    // 3. Normalize to [C, H, W]
    if (!normalize_to_chw(resized, target_size, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize image" << std::endl;
        return loaded;
    }
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Center-crop to square, then resize + normalize.
static LoadedImage load_crop_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                              const InternVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess cropped image: decoded image missing"
                  << std::endl;
        return loaded;
    }

    // Center-crop to square
    const int crop_size = std::min(image.width, image.height);
    const int x_off = (image.width - crop_size) / 2;
    const int y_off = (image.height - crop_size) / 2;

    std::vector<unsigned char> cropped(static_cast<std::size_t>(crop_size) * crop_size * 3);
    for (int y = 0; y < crop_size; ++y) {
        const unsigned char* src_row =
            image.pixels.data() + (static_cast<std::size_t>(y + y_off) * image.width + x_off) * 3;
        unsigned char* dst_row = cropped.data() + static_cast<std::size_t>(y) * crop_size * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(crop_size) * 3);
    }

    const int target_size = config.fixed_image_size;
    auto resized = resize_raw(cropped.data(), crop_size, crop_size, target_size,
                              resolve_stbir_filter(config.interpolation));
    if (resized.empty()) {
        std::cerr << "[trtmc] Failed to resize cropped image" << std::endl;
        return loaded;
    }

    if (!normalize_to_chw(resized, target_size, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize cropped image" << std::endl;
        return loaded;
    }
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Aspect-ratio-preserving resize + zero-pad to square, then normalize.
static LoadedImage
load_aspect_preserve_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                      const InternVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess aspect-preserve image: decoded image missing"
                  << std::endl;
        return loaded;
    }

    const int target_size = config.fixed_image_size;
    const stbir_filter filter = resolve_stbir_filter(config.interpolation);

    // Compute scaled dimensions that fit inside target_size x target_size
    const float scale =
        static_cast<float>(target_size) / static_cast<float>(std::max(image.width, image.height));
    const int new_w = std::max(1, static_cast<int>(image.width * scale));
    const int new_h = std::max(1, static_cast<int>(image.height * scale));

    // Resize preserving aspect ratio
    std::vector<unsigned char> resized_small(static_cast<std::size_t>(new_w) * new_h * 3);

    void* resize_result = stbir_resize(
        image.pixels.data(), image.width, image.height, image.width * 3, resized_small.data(),
        new_w, new_h, new_w * 3, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, filter);

    if (resize_result == nullptr) {
        std::cerr << "[trtmc] Failed to resize image (aspect-preserve)" << std::endl;
        return loaded;
    }

    // Zero-pad to target_size x target_size (top-left aligned)
    std::vector<unsigned char> padded(static_cast<std::size_t>(target_size) * target_size * 3, 0);
    for (int y = 0; y < new_h; ++y) {
        const unsigned char* src_row =
            resized_small.data() + static_cast<std::size_t>(y) * new_w * 3;
        unsigned char* dst_row = padded.data() + static_cast<std::size_t>(y) * target_size * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(new_w) * 3);
    }

    if (!normalize_to_chw(padded, target_size, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize aspect-preserve image" << std::endl;
        return loaded;
    }
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Aspect-ratio-preserving resize + center-pad with mean color, then normalize.
// Matches PIL ImageOps.pad(image, (size, size), color=mean*255).
static LoadedImage
load_pad_center_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                 const InternVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess pad-center image: decoded image missing"
                  << std::endl;
        return loaded;
    }

    const int target_size = config.fixed_image_size;
    const stbir_filter filter = resolve_stbir_filter(config.interpolation);

    // Compute scaled dimensions that fit inside target_size x target_size
    // This matches PIL ImageOps.pad behavior: scale to fit, then center.
    const float scale_w = static_cast<float>(target_size) / static_cast<float>(image.width);
    const float scale_h = static_cast<float>(target_size) / static_cast<float>(image.height);
    const float scale = std::min(scale_w, scale_h);
    const int new_w = std::max(1, static_cast<int>(image.width * scale));
    const int new_h = std::max(1, static_cast<int>(image.height * scale));

    // Resize preserving aspect ratio
    std::vector<unsigned char> resized_small(static_cast<std::size_t>(new_w) * new_h * 3);

    void* resize_result = stbir_resize(
        image.pixels.data(), image.width, image.height, image.width * 3, resized_small.data(),
        new_w, new_h, new_w * 3, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, filter);

    if (resize_result == nullptr) {
        std::cerr << "[trtmc] Failed to resize image (pad-center)" << std::endl;
        return loaded;
    }

    // Fill pad with mean color (mean * 255), matching ImageOps.pad color arg
    const unsigned char pad_r = static_cast<unsigned char>(config.image_mean[0] * 255.0F);
    const unsigned char pad_g = static_cast<unsigned char>(config.image_mean[1] * 255.0F);
    const unsigned char pad_b = static_cast<unsigned char>(config.image_mean[2] * 255.0F);

    std::vector<unsigned char> padded(static_cast<std::size_t>(target_size) * target_size * 3);
    for (std::size_t i = 0; i < padded.size(); i += 3) {
        padded[i + 0] = pad_r;
        padded[i + 1] = pad_g;
        padded[i + 2] = pad_b;
    }

    // Center the resized image in the padded canvas
    const int x_off = (target_size - new_w) / 2;
    const int y_off = (target_size - new_h) / 2;
    for (int y = 0; y < new_h; ++y) {
        const unsigned char* src_row =
            resized_small.data() + static_cast<std::size_t>(y) * new_w * 3;
        unsigned char* dst_row =
            padded.data() + (static_cast<std::size_t>(y + y_off) * target_size + x_off) * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(new_w) * 3);
    }

    if (!normalize_to_chw(padded, target_size, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize pad-center image" << std::endl;
        return loaded;
    }
    loaded.target_size = target_size;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// ---------------------------------------------------------------------------
// Strategy: merge_group_chw
// ---------------------------------------------------------------------------

static InternVlPreprocessedImage
preprocess_merge_group_chw(const LoadedImage& loaded, const InternVlPreprocessConfig& config) {
    InternVlPreprocessedImage result;
    ImageTransformParams params;
    params.layout = ImageTransformLayout::kMergeGroupChw;
    params.target_size = loaded.target_size;
    params.channels = loaded.channels;
    params.patch_size = config.patch_size;
    params.merge_size = config.merge_size;
    params.temporal_patch_size = config.temporal_patch_size;

    result.height = loaded.target_size;
    result.width = loaded.target_size;
    result.ok = transform_chw_layout(loaded.img_chw, params, result.pixel_values, result.channels);
    return result;
}

// ---------------------------------------------------------------------------
// Strategy: simple_chw
// ---------------------------------------------------------------------------

static InternVlPreprocessedImage preprocess_simple_chw(const LoadedImage& loaded,
                                                       const InternVlPreprocessConfig& config) {
    InternVlPreprocessedImage result;
    ImageTransformParams params;
    params.layout = ImageTransformLayout::kSimpleChw;
    params.target_size = loaded.target_size;
    params.channels = loaded.channels;

    result.height = loaded.target_size;
    result.width = loaded.target_size;
    result.ok = transform_chw_layout(loaded.img_chw, params, result.pixel_values, result.channels);

    (void)config;
    return result;
}

// ---------------------------------------------------------------------------
// Strategy: patchify_chw
// ---------------------------------------------------------------------------

static InternVlPreprocessedImage preprocess_patchify_chw(const LoadedImage& loaded,
                                                         const InternVlPreprocessConfig& config) {
    InternVlPreprocessedImage result;
    const int patch = config.patch_size;
    const int channels = loaded.channels;
    const int height = loaded.target_size;
    const int width = loaded.target_size;
    if (patch <= 0 || height % patch != 0 || width % patch != 0 || channels <= 0) {
        std::cerr << "[trtmc] Invalid patchify shape" << std::endl;
        return result;
    }

    const int grid_h = height / patch;
    const int grid_w = width / patch;
    const int num_patches = grid_h * grid_w;
    result.pixel_values.resize(static_cast<std::size_t>(num_patches) * channels * patch * patch);

    for (int gh = 0; gh < grid_h; ++gh) {
        for (int gw = 0; gw < grid_w; ++gw) {
            const int patch_idx = gh * grid_w + gw;
            for (int c = 0; c < channels; ++c) {
                for (int ph = 0; ph < patch; ++ph) {
                    for (int pw = 0; pw < patch; ++pw) {
                        const std::size_t dst =
                            (((static_cast<std::size_t>(patch_idx) * channels + c) * patch + ph) *
                                 patch +
                             pw);
                        const std::size_t src =
                            (static_cast<std::size_t>(c) * height + gh * patch + ph) * width +
                            gw * patch + pw;
                        result.pixel_values[dst] = loaded.img_chw[src];
                    }
                }
            }
        }
    }

    result.image_grid_hws = {grid_h, grid_w};
    result.channels = channels;
    result.height = height;
    result.width = width;
    result.ok = true;
    return result;
}

// ---------------------------------------------------------------------------
// Dynamic tiling (GotOcr2 / InternVL crop_to_patches)
// The grid selection follows get_optimal_tiled_canvas in Hugging Face
// Transformers' GotOcr2 image processor (Apache-2.0).
// ---------------------------------------------------------------------------

std::pair<int32_t, int32_t> internvl_tile_grid(int32_t height, int32_t width, int32_t tile_size,
                                               int32_t min_tiles, int32_t max_tiles) {
    std::vector<std::pair<int32_t, int32_t>> grids;
    for (int32_t columns = 1; columns <= max_tiles; ++columns)
        for (int32_t rows = 1; rows <= max_tiles; ++rows)
            if (columns * rows <= max_tiles && columns * rows >= min_tiles)
                grids.emplace_back(columns, rows);
    std::stable_sort(grids.begin(), grids.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first * lhs.second < rhs.first * rhs.second;
    });
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    const double area = static_cast<double>(width) * static_cast<double>(height);
    double best_difference = std::numeric_limits<double>::infinity();
    std::pair<int32_t, int32_t> best{1, 1};
    for (const auto& grid : grids) {
        const double difference =
            std::abs(aspect - static_cast<double>(grid.first) / static_cast<double>(grid.second));
        if (difference < best_difference) {
            best_difference = difference;
            best = grid;
        } else if (difference == best_difference &&
                   area > 0.5 * tile_size * tile_size * grid.first * grid.second) {
            best = grid;
        }
    }
    return best;
}

// The processor resizes the image to the tile grid, splits it row by row,
// and appends a whole-image thumbnail when there is more than one tile.
static InternVlPreprocessedImage
preprocess_dynamic_tiles(const runtime::adapters::io::DecodedImage& image,
                         const InternVlPreprocessConfig& config) {
    InternVlPreprocessedImage result;
    if (image.empty() || image.channels != 3)
        return result;
    const int32_t tile = config.fixed_image_size;
    const auto [columns, rows] = internvl_tile_grid(image.height, image.width, tile,
                                                    config.min_image_tiles, config.max_image_tiles);
    const auto canvas = internvl::torch_aa_bicubic_resize_rgb(
        image.pixels, image.height, image.width, rows * tile, columns * tile);
    std::vector<std::vector<unsigned char>> tiles;
    for (int32_t index = 0; index < columns * rows; ++index) {
        const int32_t column = index % columns;
        const int32_t row = index / columns;
        std::vector<unsigned char> crop(static_cast<std::size_t>(tile) * tile * 3);
        for (int32_t y = 0; y < tile; ++y) {
            const auto* source = canvas.data() + ((static_cast<std::size_t>(row) * tile + y) *
                                                      static_cast<std::size_t>(columns) * tile +
                                                  static_cast<std::size_t>(column) * tile) *
                                                     3;
            std::memcpy(crop.data() + static_cast<std::size_t>(y) * tile * 3, source,
                        static_cast<std::size_t>(tile) * 3);
        }
        tiles.push_back(std::move(crop));
    }
    if (config.use_thumbnail && tiles.size() != 1)
        tiles.push_back(internvl::torch_aa_bicubic_resize_rgb(image.pixels, image.height,
                                                              image.width, tile, tile));
    for (const auto& pixels : tiles) {
        std::vector<float> chw;
        if (!normalize_to_chw(pixels, tile, config, chw))
            return InternVlPreprocessedImage{};
        result.pixel_values.insert(result.pixel_values.end(), chw.begin(), chw.end());
    }
    result.num_tiles = static_cast<int32_t>(tiles.size());
    result.channels = config.in_channels;
    result.height = tile;
    result.width = tile;
    result.ok = true;
    return result;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

using LoadImageFn = LoadedImage (*)(const runtime::adapters::io::DecodedImage& image,
                                    const InternVlPreprocessConfig& config);

using PreprocessImageFn = InternVlPreprocessedImage (*)(const LoadedImage& loaded,
                                                        const InternVlPreprocessConfig& config);

struct PreprocessDispatch {
    LoadImageFn load_fn;
    PreprocessImageFn preprocess_fn;
    bool warn_unknown_type{false};
};

static PreprocessDispatch resolve_preprocess_dispatch(const std::string& preprocessor_type) {
    if (preprocessor_type == "center_crop_chw")
        return {load_crop_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "aspect_preserve_chw")
        return {load_aspect_preserve_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "pad_center_chw")
        return {load_pad_center_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "simple_chw")
        return {load_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "patchify_chw")
        return {load_resize_normalize, preprocess_patchify_chw, false};

    const bool warn_unknown = (preprocessor_type != "merge_group_chw");
    return {load_resize_normalize, preprocess_merge_group_chw, warn_unknown};
}

static InternVlPreprocessedImage
run_preprocess_dispatch(const runtime::adapters::io::DecodedImage& image,
                        const InternVlPreprocessConfig& config,
                        const PreprocessDispatch& dispatch) {
    LoadedImage loaded = dispatch.load_fn(image, config);
    if (!loaded.ok) {
        return InternVlPreprocessedImage{};
    }
    return dispatch.preprocess_fn(loaded, config);
}

InternVlPreprocessedImage
internvl_preprocess_decoded_image(const runtime::adapters::io::DecodedImage& image,
                                  const InternVlPreprocessConfig& config) {
    if (config.preprocessor_type == "dynamic_tiles_chw")
        return preprocess_dynamic_tiles(image, config);
    const auto dispatch = resolve_preprocess_dispatch(config.preprocessor_type);
    if (dispatch.warn_unknown_type) {
        std::cerr << "[trtmc] WARNING: Unknown preprocessor_type \"" << config.preprocessor_type
                  << "\", falling back to merge_group_chw" << std::endl;
    }

    return run_preprocess_dispatch(image, config, dispatch);
}

std::string internvl_format_prompt(const std::string& user_prompt,
                                   const InternVlPreprocessConfig& config) {
    return internvl_format_prompt(user_prompt, config, 1);
}

std::string internvl_format_prompt(const std::string& user_prompt,
                                   const InternVlPreprocessConfig& config, int32_t num_tiles) {
    // Build image_pads string: image_token_str num_image_pad_tokens times per tile
    const int32_t pad_count = config.num_image_pad_tokens * std::max(num_tiles, 1);
    std::string image_pads;
    image_pads.reserve(static_cast<std::size_t>(pad_count) * config.image_token_str.size());
    for (int32_t i = 0; i < pad_count; ++i) {
        image_pads += config.image_token_str;
    }

    // Replace {image_pads} and {prompt} in the template
    std::string result = config.vl_prompt_template;

    const std::string pads_placeholder = "{image_pads}";
    const std::size_t pads_pos = result.find(pads_placeholder);
    if (pads_pos != std::string::npos) {
        result.replace(pads_pos, pads_placeholder.size(), image_pads);
    }

    const std::string prompt_placeholder = "{prompt}";
    const std::size_t prompt_pos = result.find(prompt_placeholder);
    if (prompt_pos != std::string::npos) {
        result.replace(prompt_pos, prompt_placeholder.size(), user_prompt);
    }

    return result;
}

nlohmann::json parse_preprocess_document(const std::string& text) {
    auto document = nlohmann::json::parse(text);
    if (!document.is_object())
        throw std::runtime_error("runtime.json must be a JSON object");
    return document;
}

const nlohmann::json& require_preprocess_member(const nlohmann::json& document, const char* key) {
    const auto found = document.find(key);
    if (found == document.end())
        throw std::runtime_error(std::string("runtime.json is missing '") + key + "'");
    return *found;
}

int32_t require_preprocess_int(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_number_integer() && !value.is_number_unsigned())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be an integer");
    return value.get<int32_t>();
}

float require_preprocess_number(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_number())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be numeric");
    return value.get<float>();
}

std::string require_preprocess_string(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_string())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be a string");
    return value.get<std::string>();
}

bool require_preprocess_bool(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_boolean())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be a boolean");
    return value.get<bool>();
}

void require_preprocess_triplet(const nlohmann::json& document, const char* key,
                                float (&target)[3]) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_array() || value.size() != 3)
        throw std::runtime_error(std::string("runtime.json '") + key + "' must have 3 numbers");
    for (std::size_t index = 0; index < 3; ++index) {
        if (!value[index].is_number())
            throw std::runtime_error(std::string("runtime.json '") + key + "' must have 3 numbers");
        target[index] = value[index].get<float>();
    }
}

InternVlPreprocessConfig internvl_parse_preprocess_config(const std::string& config_text) {
    const auto document = parse_preprocess_document(config_text);
    InternVlPreprocessConfig cfg;
    cfg.preprocessor_type = require_preprocess_string(document, "preprocessor_type");
    cfg.image_token_id = require_preprocess_int(document, "image_token_id");
    cfg.fixed_image_size = require_preprocess_int(document, "fixed_image_size");
    cfg.patch_size = require_preprocess_int(document, "patch_size");
    cfg.merge_size = require_preprocess_int(document, "merge_size");
    cfg.temporal_patch_size = require_preprocess_int(document, "temporal_patch_size");
    cfg.num_image_pad_tokens = require_preprocess_int(document, "num_image_pad_tokens");
    cfg.vision_output_dim = require_preprocess_int(document, "vision_output_dim");
    cfg.vl_prompt_template = require_preprocess_string(document, "vl_prompt_template");
    cfg.image_token_str = require_preprocess_string(document, "image_token_str");
    cfg.interpolation = require_preprocess_string(document, "interpolation");
    require_preprocess_triplet(document, "image_mean", cfg.image_mean);
    require_preprocess_triplet(document, "image_std", cfg.image_std);
    cfg.min_image_tiles = document.value("min_image_tiles", cfg.min_image_tiles);
    cfg.max_image_tiles = document.value("max_image_tiles", cfg.max_image_tiles);
    cfg.use_thumbnail = document.value("use_thumbnail", cfg.use_thumbnail);
    if (cfg.min_image_tiles <= 0 || cfg.max_image_tiles < cfg.min_image_tiles)
        throw std::runtime_error("runtime.json has invalid image tile limits");
    if (cfg.fixed_image_size <= 0 || cfg.patch_size <= 0 || cfg.merge_size <= 0 ||
        cfg.temporal_patch_size <= 0 || cfg.vision_output_dim <= 0)
        throw std::runtime_error("runtime.json has invalid preprocessing geometry");
    if (cfg.interpolation != "nearest" && cfg.interpolation != "bilinear" &&
        cfg.interpolation != "bicubic")
        throw std::runtime_error("runtime.json has unsupported interpolation");
    return cfg;
}

} // namespace trtmc
