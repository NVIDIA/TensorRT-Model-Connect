/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lance/image_preprocessor.h"

#include "image_transform_helper.h"
#include "stb_image.h"
#include "stb_image_resize2.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

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
                             const LancePreprocessConfig& config, std::vector<float>& out_chw) {
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
                                         const LancePreprocessConfig& config) {
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
                                              const LancePreprocessConfig& config) {
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
                                      const LancePreprocessConfig& config) {
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
                                 const LancePreprocessConfig& config) {
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

static LancePreprocessedImage preprocess_merge_group_chw(const LoadedImage& loaded,
                                                         const LancePreprocessConfig& config) {
    LancePreprocessedImage result;
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

static LancePreprocessedImage preprocess_simple_chw(const LoadedImage& loaded,
                                                    const LancePreprocessConfig& config) {
    LancePreprocessedImage result;
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

static LancePreprocessedImage preprocess_patchify_chw(const LoadedImage& loaded,
                                                      const LancePreprocessConfig& config) {
    LancePreprocessedImage result;
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
// Dispatcher
// ---------------------------------------------------------------------------

using LoadImageFn = LoadedImage (*)(const runtime::adapters::io::DecodedImage& image,
                                    const LancePreprocessConfig& config);

using PreprocessImageFn = LancePreprocessedImage (*)(const LoadedImage& loaded,
                                                     const LancePreprocessConfig& config);

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

static LancePreprocessedImage
run_preprocess_dispatch(const runtime::adapters::io::DecodedImage& image,
                        const LancePreprocessConfig& config, const PreprocessDispatch& dispatch) {
    LoadedImage loaded = dispatch.load_fn(image, config);
    if (!loaded.ok) {
        return LancePreprocessedImage{};
    }
    return dispatch.preprocess_fn(loaded, config);
}

runtime::adapters::io::DecodedImage lance_decode_image_rgb(const std::string& image_path) {
    runtime::adapters::io::DecodedImage image;
    if (image_path.empty()) {
        return image;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr) {
        std::cerr << "[trtmc] Failed to load image: " << image_path << " (" << stbi_failure_reason()
                  << ")" << std::endl;
        return image;
    }

    image.width = width;
    image.height = height;
    image.channels = 3;
    image.pixels.assign(raw, raw + static_cast<std::size_t>(width) * height * 3);
    stbi_image_free(raw);
    return image;
}

LancePreprocessedImage
lance_preprocess_decoded_image(const runtime::adapters::io::DecodedImage& image,
                               const LancePreprocessConfig& config) {
    const auto dispatch = resolve_preprocess_dispatch(config.preprocessor_type);
    if (dispatch.warn_unknown_type) {
        std::cerr << "[trtmc] WARNING: Unknown preprocessor_type \"" << config.preprocessor_type
                  << "\", falling back to merge_group_chw" << std::endl;
    }

    return run_preprocess_dispatch(image, config, dispatch);
}

LancePreprocessedImage lance_load_and_preprocess_image(const std::string& image_path,
                                                       const LancePreprocessConfig& config) {
    return lance_preprocess_decoded_image(lance_decode_image_rgb(image_path), config);
}

LanceMropePositions lance_build_mrope_positions(const std::vector<int32_t>& input_ids,
                                                int32_t image_token_id, int32_t num_image_features,
                                                int32_t grid_height, int32_t grid_width) {
    LanceMropePositions result;
    result.token_positions.resize(input_ids.size());

    const auto fill_text_positions = [&result, &input_ids]() {
        for (std::size_t i = 0; i < input_ids.size(); ++i) {
            const int32_t position = static_cast<int32_t>(i);
            result.token_positions[i] = {position, position, position};
        }
        result.next_position = static_cast<int32_t>(input_ids.size());
    };

    const auto image_begin = std::find(input_ids.begin(), input_ids.end(), image_token_id);
    const bool valid_grid =
        grid_height > 0 && grid_width > 0 && grid_height * grid_width == num_image_features;
    if (image_begin == input_ids.end() || !valid_grid) {
        fill_text_positions();
        return result;
    }

    const std::size_t image_offset =
        static_cast<std::size_t>(std::distance(input_ids.begin(), image_begin));
    const std::size_t image_end = image_offset + static_cast<std::size_t>(num_image_features);
    if (image_end > input_ids.size()) {
        fill_text_positions();
        return result;
    }

    for (std::size_t i = 0; i < image_offset; ++i) {
        const int32_t position = static_cast<int32_t>(i);
        result.token_positions[i] = {position, position, position};
    }

    const int32_t vision_base = static_cast<int32_t>(image_offset);
    for (int32_t i = 0; i < num_image_features; ++i) {
        result.token_positions[image_offset + static_cast<std::size_t>(i)] = {
            vision_base, vision_base + i / grid_width, vision_base + i % grid_width};
    }

    int32_t text_position = vision_base + std::max(grid_height, grid_width);
    for (std::size_t i = image_end; i < input_ids.size(); ++i) {
        result.token_positions[i] = {text_position, text_position, text_position};
        ++text_position;
    }
    result.next_position = text_position;
    return result;
}

std::string lance_format_prompt(const std::string& user_prompt,
                                const LancePreprocessConfig& config) {
    // Build image_pads string: repeat image_token_str num_image_pad_tokens times
    std::string image_pads;
    image_pads.reserve(static_cast<std::size_t>(config.num_image_pad_tokens) *
                       config.image_token_str.size());
    for (int32_t i = 0; i < config.num_image_pad_tokens; ++i) {
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

static void assign_image_norm_triplet(const std::vector<float>& values, float (&target)[3]) {
    if (values.size() < 3) {
        return;
    }
    target[0] = values[0];
    target[1] = values[1];
    target[2] = values[2];
}

static void unescape_newline_sequences(std::string& text) {
    std::size_t pos = 0;
    while ((pos = text.find("\\n", pos)) != std::string::npos) {
        text.replace(pos, 2, "\n");
        ++pos;
    }
}

static void parse_base_vl_config(const std::string& config_text, LancePreprocessConfig& cfg) {
    cfg.preprocessor_type =
        extract_json_string(config_text, "preprocessor_type", "merge_group_chw");
    cfg.image_token_id = extract_json_int(config_text, "image_token_id", -1);
    cfg.fixed_image_size = extract_json_int(config_text, "fixed_image_size", 448);
    cfg.patch_size = extract_json_int(config_text, "patch_size", cfg.patch_size);
    cfg.merge_size = extract_json_int(config_text, "merge_size", cfg.merge_size);
    cfg.temporal_patch_size =
        extract_json_int(config_text, "temporal_patch_size", cfg.temporal_patch_size);
    cfg.num_image_pad_tokens = extract_json_int(config_text, "num_image_pad_tokens", 256);
    cfg.vision_output_dim = extract_json_int(config_text, "vision_output_dim", 0);
    cfg.vl_prompt_template = extract_json_string(config_text, "vl_prompt_template", "");
    cfg.image_token_str = extract_json_string(config_text, "image_token_str", "");
    cfg.interpolation = extract_json_string(config_text, "interpolation", "bicubic");
}

static void maybe_apply_resample_fallback(const std::string& config_text,
                                          const std::string& preprocessor_config_text,
                                          LancePreprocessConfig& cfg) {
    if (!extract_json_string(config_text, "interpolation", "").empty()) {
        return;
    }

    const int resample = extract_json_int(preprocessor_config_text, "resample", -1);
    if (resample == 0)
        cfg.interpolation = "nearest";
    else if (resample == 2)
        cfg.interpolation = "bilinear";
    else if (resample == 3)
        cfg.interpolation = "bicubic";
}

static void apply_preprocessor_config_overrides(const std::string& config_text,
                                                const std::string& preprocessor_config_text,
                                                LancePreprocessConfig& cfg) {
    if (preprocessor_config_text.empty()) {
        return;
    }

    cfg.patch_size = extract_json_int(preprocessor_config_text, "patch_size", 14);
    cfg.merge_size = extract_json_int(preprocessor_config_text, "merge_size", 2);
    cfg.temporal_patch_size = extract_json_int(preprocessor_config_text, "temporal_patch_size", 2);

    auto mean_vals = extract_json_float_array(preprocessor_config_text, "image_mean", 3);
    assign_image_norm_triplet(mean_vals, cfg.image_mean);

    auto std_vals = extract_json_float_array(preprocessor_config_text, "image_std", 3);
    assign_image_norm_triplet(std_vals, cfg.image_std);

    maybe_apply_resample_fallback(config_text, preprocessor_config_text, cfg);
}

static void apply_config_image_norm_overrides(const std::string& config_text,
                                              LancePreprocessConfig& cfg) {
    auto mean_vals = extract_json_float_array(config_text, "image_mean", 3);
    assign_image_norm_triplet(mean_vals, cfg.image_mean);

    auto std_vals = extract_json_float_array(config_text, "image_std", 3);
    assign_image_norm_triplet(std_vals, cfg.image_std);
}

LancePreprocessConfig lance_parse_preprocess_config(const std::string& config_text,
                                                    const std::string& preprocessor_config_text) {
    LancePreprocessConfig cfg;
    parse_base_vl_config(config_text, cfg);
    unescape_newline_sequences(cfg.vl_prompt_template);
    apply_preprocessor_config_overrides(config_text, preprocessor_config_text, cfg);
    apply_config_image_norm_overrides(config_text, cfg);

    return cfg;
}

} // namespace trtmc
