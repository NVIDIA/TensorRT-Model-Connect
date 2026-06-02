// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-IMG-CPP-01
// Architecture:   ARCH-IMG-001
// Unit Design:    UD-IMG-01
// Intent:         All 4 preprocessing strategies, config parsing, prompt formatting
// Preconditions:  Test image data available (via TempDirGuard)
// Postconditions: Each strategy produces correctly shaped output
// =============================================================================

// =============================================================================
// Test suite: VL image preprocessing (stb_image-based)
// =============================================================================
//
// Tests load_and_preprocess_image(), format_vl_prompt(), and
// parse_vl_preprocess_config() from image_preprocessor.h.
//
// These tests are CPU-only, no GPU/TRT required. Image loading tests use
// a small in-memory PPM image written to a temp file.
// =============================================================================

#include "runtime/domains/multimodal/image_preprocessor.h"
#include "test_helpers.h"
#include "trtmc/runtime/domains/multimodal/image_transform_helper.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void check_near(float actual, float expected, float tol, const char* test_name)
{
    check(std::abs(actual - expected) <= tol, test_name);
}

// Pure helper test: HWC uint8 -> CHW float normalization in-memory.
static void test_helper_normalize_hwc_to_chw()
{
    const std::vector<unsigned char> image_hwc = {
        // Pixel 0 (x=0): R, G, B
        0, 64, 255,
        // Pixel 1 (x=1): R, G, B
        255, 128, 0
    };

    trtmc::ImageNormalizationParams params;
    params.width = 2;
    params.height = 1;
    params.channels = 3;
    params.image_mean[0] = 0.0F;
    params.image_mean[1] = 0.0F;
    params.image_mean[2] = 0.0F;
    params.image_std[0] = 1.0F;
    params.image_std[1] = 1.0F;
    params.image_std[2] = 1.0F;

    std::vector<float> out_chw;
    const bool ok = trtmc::normalize_hwc_u8_to_chw(image_hwc, params, out_chw);
    check(ok, "helper normalize: returns true");
    check(out_chw.size() == 6, "helper normalize: output size is C*H*W");

    check_near(out_chw[0], 0.0F / 255.0F, 1e-6F, "helper normalize: R(0,0)");
    check_near(out_chw[1], 255.0F / 255.0F, 1e-6F, "helper normalize: R(0,1)");
    check_near(out_chw[2], 64.0F / 255.0F, 1e-6F, "helper normalize: G(0,0)");
    check_near(out_chw[3], 128.0F / 255.0F, 1e-6F, "helper normalize: G(0,1)");
    check_near(out_chw[4], 255.0F / 255.0F, 1e-6F, "helper normalize: B(0,0)");
    check_near(out_chw[5], 0.0F / 255.0F, 1e-6F, "helper normalize: B(0,1)");
}

// Pure helper test: std <= 1e-8 uses inv_std=1 branch (no division by near-zero).
static void test_helper_normalize_std_floor_branch()
{
    const std::vector<unsigned char> image_hwc = {128, 10, 20};

    trtmc::ImageNormalizationParams params;
    params.width = 1;
    params.height = 1;
    params.channels = 3;
    params.image_mean[0] = 0.5F;
    params.image_mean[1] = 0.0F;
    params.image_mean[2] = 0.0F;
    params.image_std[0] = 0.0F;   // hits fallback branch
    params.image_std[1] = 0.5F;
    params.image_std[2] = 1.0F;

    std::vector<float> out_chw;
    const bool ok = trtmc::normalize_hwc_u8_to_chw(image_hwc, params, out_chw);
    check(ok, "helper normalize std floor: returns true");
    check(out_chw.size() == 3, "helper normalize std floor: output size is 3");

    const float expected_c0 = 128.0F / 255.0F - 0.5F;
    check_near(out_chw[0], expected_c0, 1e-6F, "helper normalize std floor: C0 fallback");
}

// Pure helper test: simple CHW layout branch keeps data unchanged.
static void test_helper_transform_simple_chw_branch()
{
    const std::vector<float> input_chw = {
        // Channel 0 (2x2)
        1.0F, 2.0F, 3.0F, 4.0F,
        // Channel 1 (2x2)
        5.0F, 6.0F, 7.0F, 8.0F
    };

    trtmc::ImageTransformParams params;
    params.layout = trtmc::ImageTransformLayout::kSimpleChw;
    params.target_size = 2;
    params.channels = 2;

    std::vector<float> out_values;
    int32_t out_channels = 0;
    const bool ok = trtmc::transform_chw_layout(input_chw, params, out_values, out_channels);

    check(ok, "helper simple transform: returns true");
    check(out_channels == 2, "helper simple transform: out_channels=2");
    check(out_values == input_chw, "helper simple transform: values unchanged");
}

// Pure helper test: qwen merge-group branch reorders patch positions and duplicates T channels.
static void test_helper_transform_qwen_merge_group_branch()
{
    std::vector<float> input_chw(16);
    for (int i = 0; i < 16; ++i)
    {
        input_chw[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }

    trtmc::ImageTransformParams params;
    params.layout = trtmc::ImageTransformLayout::kQwenMergeGroup;
    params.target_size = 4;
    params.channels = 1;
    params.patch_size = 1;
    params.merge_size = 2;
    params.temporal_patch_size = 2;

    std::vector<float> out_values;
    int32_t out_channels = 0;
    const bool ok = trtmc::transform_chw_layout(input_chw, params, out_values, out_channels);
    check(ok, "helper qwen transform: returns true");
    check(out_channels == 2, "helper qwen transform: out_channels=C*T=2");
    check(out_values.size() == 32, "helper qwen transform: output size=2*4*4");

    const std::vector<float> expected_channel = {
        0.0F, 1.0F, 4.0F, 5.0F,
        2.0F, 3.0F, 6.0F, 7.0F,
        8.0F, 9.0F, 12.0F, 13.0F,
        10.0F, 11.0F, 14.0F, 15.0F
    };

    bool first_channel_ok = true;
    for (std::size_t i = 0; i < expected_channel.size(); ++i)
    {
        if (std::abs(out_values[i] - expected_channel[i]) > 1e-6F)
        {
            first_channel_ok = false;
            break;
        }
    }
    check(first_channel_ok, "helper qwen transform: merge-group reorder matches expected");

    bool second_channel_ok = true;
    const std::size_t offset = 16;
    for (std::size_t i = 0; i < expected_channel.size(); ++i)
    {
        if (std::abs(out_values[offset + i] - expected_channel[i]) > 1e-6F)
        {
            second_channel_ok = false;
            break;
        }
    }
    check(second_channel_ok, "helper qwen transform: temporal channel duplicated");
}

// Write a tiny 4x4 PPM image (binary format) to a file.
static std::string write_test_ppm(const std::string& dir)
{
    const std::string path = dir + "/test_image.ppm";
    std::ofstream out(path, std::ios::binary);
    out << "P6\n4 4\n255\n";
    // 4x4 pixels, each RGB
    for (int i = 0; i < 16; ++i)
    {
        unsigned char r = static_cast<unsigned char>(i * 16);
        unsigned char g = static_cast<unsigned char>(128);
        unsigned char b = static_cast<unsigned char>(255 - i * 16);
        out.put(static_cast<char>(r));
        out.put(static_cast<char>(g));
        out.put(static_cast<char>(b));
    }
    out.close();
    return path;
}

// Test: qwen_merge_group strategy — default, produces [C*T, H, W] with permutation.
static void test_qwen_merge_group_strategy()
{
    trtmc_test::TempDirGuard tmp;
    const std::string& dir = tmp.path();

    // Write test image
    const std::string image_path = write_test_ppm(dir);

    // Configure for a small fixed size
    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 8;  // small for testing
    config.temporal_patch_size = 2;
    config.in_channels = 3;
    config.preprocessor_type = "qwen_merge_group";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtmc::load_and_preprocess_image(image_path, config);

    check(result.ok, "qwen_merge_group: image loaded successfully");
    check(result.channels == 6, "qwen_merge_group: channels = T*C = 2*3 = 6");
    check(result.height == 8, "qwen_merge_group: height = fixed_image_size = 8");
    check(result.width == 8, "qwen_merge_group: width = fixed_image_size = 8");

    const std::size_t expected_size = 6 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "qwen_merge_group: pixel_values size = channels * H * W");

    // Check normalization range: (pixel/255 - 0.5) / 0.5 is in [-1, 1]
    bool in_range = true;
    for (float v : result.pixel_values)
    {
        if (v < -1.1F || v > 1.1F)
        {
            in_range = false;
            break;
        }
    }
    check(in_range, "qwen_merge_group: all normalized values in [-1.1, 1.1]");
}

// Test: simple_chw strategy — produces [C, H, W] without permutation.
static void test_simple_chw_strategy()
{
    trtmc_test::TempDirGuard tmp;
    const std::string& dir = tmp.path();

    // Write test image
    const std::string image_path = write_test_ppm(dir);

    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.in_channels = 3;
    config.preprocessor_type = "simple_chw";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtmc::load_and_preprocess_image(image_path, config);

    check(result.ok, "simple_chw: image loaded successfully");
    check(result.channels == 3, "simple_chw: channels = C = 3 (no temporal)");
    check(result.height == 8, "simple_chw: height = fixed_image_size = 8");
    check(result.width == 8, "simple_chw: width = fixed_image_size = 8");

    const std::size_t expected_size = 3 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "simple_chw: pixel_values size = C * H * W");

    // Check normalization range: (pixel/255 - 0.5) / 0.5 is in [-1, 1]
    bool in_range = true;
    for (float v : result.pixel_values)
    {
        if (v < -1.1F || v > 1.1F)
        {
            in_range = false;
            break;
        }
    }
    check(in_range, "simple_chw: all normalized values in [-1.1, 1.1]");
}

// Test: locateanything_patchify strategy produces [patches, C, pH, pW] plus grid metadata.
static void test_locateanything_patchify_strategy()
{
    trtmc_test::TempDirGuard tmp;
    const std::string& dir = tmp.path();
    const std::string image_path = write_test_ppm(dir);

    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 4;
    config.patch_size = 2;
    config.in_channels = 3;
    config.preprocessor_type = "locateanything_patchify";
    config.interpolation = "nearest";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtmc::load_and_preprocess_image(image_path, config);

    check(result.ok, "locateanything_patchify: image loaded successfully");
    check(result.channels == 3, "locateanything_patchify: channels = C = 3");
    check(result.height == 4, "locateanything_patchify: height = fixed_image_size = 4");
    check(result.width == 4, "locateanything_patchify: width = fixed_image_size = 4");
    check(result.image_grid_hws.size() == 2, "locateanything_patchify: grid has two entries");
    check(result.image_grid_hws[0] == 2, "locateanything_patchify: grid height = 2");
    check(result.image_grid_hws[1] == 2, "locateanything_patchify: grid width = 2");

    const std::size_t expected_size = 4 * 3 * 2 * 2;
    check(result.pixel_values.size() == expected_size,
          "locateanything_patchify: pixel_values size = patches * C * pH * pW");
    check_near(result.pixel_values[0], -1.0F, 1e-5F,
               "locateanything_patchify: first patch first red pixel normalized");
}

// Test: load non-existent image returns ok=false.
static void test_load_missing_image()
{
    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 8;

    auto result = trtmc::load_and_preprocess_image("/nonexistent/image.jpg", config);
    check(!result.ok, "missing image returns ok=false");
}

// Test: format_vl_prompt replaces {image_pads} and {prompt}.
static void test_format_vl_prompt()
{
    trtmc::VLPreprocessConfig config;
    config.num_image_pad_tokens = 3;
    config.image_token_str = "<|pad|>";
    config.vl_prompt_template = "USER: {image_pads}\n{prompt}\nASST:";

    const std::string result = trtmc::format_vl_prompt("Describe this", config);

    // Should contain 3 copies of <|pad|>
    check(result.find("<|pad|><|pad|><|pad|>") != std::string::npos,
          "3 image pad tokens present");

    // Should contain the user prompt
    check(result.find("Describe this") != std::string::npos,
          "user prompt present");

    // Should have the template structure
    check(result.find("USER: ") == 0, "starts with USER: ");
    check(result.find("ASST:") != std::string::npos, "ends with ASST:");
}

// Test: parse_vl_preprocess_config extracts fields correctly.
static void test_parse_vl_config()
{
    const std::string config_json = R"({
        "image_token_id": 151655,
        "fixed_image_size": 448,
        "num_image_pad_tokens": 256,
        "vision_output_dim": 2048,
        "vl_prompt_template": "test {image_pads} {prompt}",
        "image_token_str": "<|image_pad|>",
        "preprocessor_type": "qwen_merge_group"
    })";

    const std::string preproc_json = R"({
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711]
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, preproc_json);

    check(cfg.image_token_id == 151655, "image_token_id = 151655");
    check(cfg.fixed_image_size == 448, "fixed_image_size = 448");
    check(cfg.num_image_pad_tokens == 256, "num_image_pad_tokens = 256");
    check(cfg.vision_output_dim == 2048, "vision_output_dim = 2048");
    check(cfg.patch_size == 14, "patch_size = 14");
    check(cfg.merge_size == 2, "merge_size = 2");
    check(cfg.temporal_patch_size == 2, "temporal_patch_size = 2");
    check(cfg.image_token_str == "<|image_pad|>", "image_token_str parsed");
    check(cfg.preprocessor_type == "qwen_merge_group", "preprocessor_type parsed");

    // Check float array parsing
    check(std::abs(cfg.image_mean[0] - 0.48145466F) < 1e-5F, "image_mean[0]");
    check(std::abs(cfg.image_mean[1] - 0.4578275F) < 1e-5F, "image_mean[1]");
    check(std::abs(cfg.image_std[2] - 0.27577711F) < 1e-5F, "image_std[2]");
}

// Test: preprocessor_type defaults to "qwen_merge_group" when absent.
static void test_parse_vl_config_default_preprocessor_type()
{
    const std::string config_json = R"({
        "image_token_id": 100
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");
    check(cfg.preprocessor_type == "qwen_merge_group",
          "preprocessor_type defaults to qwen_merge_group");
}

// Test: preprocessor_type = "simple_chw" round-trips through parse.
static void test_parse_vl_config_simple_chw()
{
    const std::string config_json = R"({
        "preprocessor_type": "simple_chw"
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");
    check(cfg.preprocessor_type == "simple_chw",
          "preprocessor_type simple_chw parsed correctly");
}

// Write a non-square 6x4 PPM image (binary format) to a file.
static std::string write_test_ppm_nonsquare(const std::string& dir)
{
    const std::string path = dir + "/test_nonsquare.ppm";
    std::ofstream out(path, std::ios::binary);
    out << "P6\n6 4\n255\n";
    // 6x4 pixels, each RGB
    for (int i = 0; i < 24; ++i)
    {
        unsigned char r = static_cast<unsigned char>(i * 10);
        unsigned char g = static_cast<unsigned char>(128);
        unsigned char b = static_cast<unsigned char>(255 - i * 10);
        out.put(static_cast<char>(r));
        out.put(static_cast<char>(g));
        out.put(static_cast<char>(b));
    }
    out.close();
    return path;
}

// Test Gap 1: unknown preprocessor_type falls back to qwen_merge_group with ok=true.
static void test_unknown_preprocessor_type_fallback()
{
    trtmc_test::TempDirGuard tmp;
    const std::string& dir = tmp.path();
    const std::string image_path = write_test_ppm(dir);

    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.temporal_patch_size = 2;
    config.in_channels = 3;
    config.preprocessor_type = "bogus";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtmc::load_and_preprocess_image(image_path, config);

    check(result.ok, "unknown type fallback: ok=true");
    // Should produce qwen_merge_group output (C*T channels)
    check(result.channels == 6, "unknown type fallback: channels = C*T = 6");
    check(result.height == 8, "unknown type fallback: height = 8");
    check(result.width == 8, "unknown type fallback: width = 8");
}

// Test Gap 2: center_crop_chw strategy with non-square image.
static void test_center_crop_chw_strategy()
{
    trtmc_test::TempDirGuard tmp;
    const std::string& dir = tmp.path();
    const std::string image_path = write_test_ppm_nonsquare(dir);

    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.in_channels = 3;
    config.preprocessor_type = "center_crop_chw";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtmc::load_and_preprocess_image(image_path, config);

    check(result.ok, "center_crop_chw: ok=true");
    check(result.channels == 3, "center_crop_chw: channels = 3");
    check(result.height == 8, "center_crop_chw: height = 8");
    check(result.width == 8, "center_crop_chw: width = 8");

    const std::size_t expected_size = 3 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "center_crop_chw: pixel_values size = C * H * W");

    bool in_range = true;
    for (float v : result.pixel_values)
    {
        if (v < -1.1F || v > 1.1F)
        {
            in_range = false;
            break;
        }
    }
    check(in_range, "center_crop_chw: all normalized values in [-1.1, 1.1]");
}

// Test Gap 3: aspect_preserve_chw strategy with non-square image.
static void test_aspect_preserve_chw_strategy()
{
    trtmc_test::TempDirGuard tmp;
    const std::string& dir = tmp.path();
    const std::string image_path = write_test_ppm_nonsquare(dir);

    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.in_channels = 3;
    config.preprocessor_type = "aspect_preserve_chw";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtmc::load_and_preprocess_image(image_path, config);

    check(result.ok, "aspect_preserve_chw: ok=true");
    check(result.channels == 3, "aspect_preserve_chw: channels = 3");
    check(result.height == 8, "aspect_preserve_chw: height = 8");
    check(result.width == 8, "aspect_preserve_chw: width = 8");

    const std::size_t expected_size = 3 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "aspect_preserve_chw: pixel_values size = C * H * W");

    // Padded region (bottom rows) should have normalized-zero values:
    // (0/255 - 0.5) / 0.5 = -1.0
    // The 6x4 image scaled to fit 8x8 -> new_w=8, new_h=5 (6/6*8=8, 4/6*8=5.33->5)
    // So rows 5-7 should be padded zeros -> normalized to -1.0
    // Check last row of first channel
    const float expected_pad = (0.0F / 255.0F - 0.5F) / 0.5F;  // -1.0
    bool pad_ok = true;
    for (int x = 0; x < 8; ++x)
    {
        // Channel 0, row 7, col x
        const std::size_t idx = static_cast<std::size_t>(0) * 64 + 7 * 8 + x;
        if (std::abs(result.pixel_values[idx] - expected_pad) > 0.01F)
        {
            pad_ok = false;
            break;
        }
    }
    check(pad_ok, "aspect_preserve_chw: padded rows have correct normalized zero value");
}

// Test: pad_center_chw strategy — aspect-ratio-preserving resize + center-pad with mean color.
static void test_pad_center_chw_strategy()
{
    trtmc_test::TempDirGuard tmp;
    const std::string& dir = tmp.path();
    const std::string image_path = write_test_ppm_nonsquare(dir);

    trtmc::VLPreprocessConfig config;
    config.fixed_image_size = 8;
    config.in_channels = 3;
    config.preprocessor_type = "pad_center_chw";
    config.image_mean[0] = 0.5F;
    config.image_mean[1] = 0.5F;
    config.image_mean[2] = 0.5F;
    config.image_std[0] = 0.5F;
    config.image_std[1] = 0.5F;
    config.image_std[2] = 0.5F;

    auto result = trtmc::load_and_preprocess_image(image_path, config);

    check(result.ok, "pad_center_chw: ok=true");
    check(result.channels == 3, "pad_center_chw: channels = 3");
    check(result.height == 8, "pad_center_chw: height = 8");
    check(result.width == 8, "pad_center_chw: width = 8");

    const std::size_t expected_size = 3 * 8 * 8;
    check(result.pixel_values.size() == expected_size,
          "pad_center_chw: pixel_values size = C * H * W");

    // Padded region should have normalized-mean-color values:
    // (128/255 - 0.5) / 0.5 = (0.502 - 0.5) / 0.5 ≈ 0.004 (close to 0)
    // The 6x4 image scaled to fit 8x8 -> new_w=8, new_h=5
    // Center-padded: y_off = (8-5)/2 = 1, so row 0 and rows 6-7 are padded
    // Pad color = mean * 255 = 0.5 * 255 = 127 (truncated from 127.5)
    // Normalized: (127/255 - 0.5) / 0.5 = (0.498 - 0.5) / 0.5 = -0.004
    const float pad_pixel = static_cast<float>(static_cast<unsigned char>(0.5F * 255.0F));
    const float expected_pad = (pad_pixel / 255.0F - 0.5F) / 0.5F;
    bool pad_ok = true;
    // Check row 0 of first channel (should be padded)
    for (int x = 0; x < 8; ++x)
    {
        const std::size_t idx = static_cast<std::size_t>(0) * 64 + 0 * 8 + x;
        if (std::abs(result.pixel_values[idx] - expected_pad) > 0.05F)
        {
            pad_ok = false;
            break;
        }
    }
    check(pad_ok, "pad_center_chw: center-padded rows have mean-color value");
}

// Test Gap 4: interpolation defaults to "bicubic".
static void test_parse_interpolation_default()
{
    const std::string config_json = R"({
        "preprocessor_type": "simple_chw"
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");
    check(cfg.interpolation == "bicubic",
          "interpolation defaults to bicubic");
}

// Test Gap 4: interpolation = "bilinear" round-trips.
static void test_parse_interpolation_bilinear()
{
    const std::string config_json = R"({
        "interpolation": "bilinear"
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");
    check(cfg.interpolation == "bilinear",
          "interpolation bilinear parsed from config.json");
}

// Test Gap 4: resample int from preprocessor_config.json maps correctly.
static void test_parse_resample_from_preprocessor()
{
    // config.json does NOT set interpolation -> fallback to resample int
    const std::string config_json = R"({
        "preprocessor_type": "simple_chw"
    })";

    const std::string preproc_json = R"({
        "resample": 2
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, preproc_json);
    check(cfg.interpolation == "bilinear",
          "resample=2 maps to bilinear");

    // Test resample=3 -> bicubic
    const std::string preproc_json3 = R"({
        "resample": 3
    })";
    auto cfg3 = trtmc::parse_vl_preprocess_config(config_json, preproc_json3);
    check(cfg3.interpolation == "bicubic",
          "resample=3 maps to bicubic");

    // Test resample=0 -> nearest
    const std::string preproc_json0 = R"({
        "resample": 0
    })";
    auto cfg0 = trtmc::parse_vl_preprocess_config(config_json, preproc_json0);
    check(cfg0.interpolation == "nearest",
          "resample=0 maps to nearest");

    // Test: explicit config.json interpolation overrides resample
    const std::string config_explicit = R"({
        "interpolation": "nearest"
    })";
    auto cfg_override = trtmc::parse_vl_preprocess_config(config_explicit, preproc_json);
    check(cfg_override.interpolation == "nearest",
          "explicit interpolation overrides resample");
}

// Test: parse_vl_preprocess_config with complete JSON (all fields populated).
static void test_parse_complete_json()
{
    const std::string config_json = R"({
        "image_token_id": 200,
        "fixed_image_size": 336,
        "num_image_pad_tokens": 128,
        "vision_output_dim": 4096,
        "vl_prompt_template": "system {image_pads}\nuser: {prompt}\nassistant:",
        "image_token_str": "<img>",
        "preprocessor_type": "center_crop_chw",
        "interpolation": "bilinear"
    })";

    const std::string preproc_json = R"({
        "patch_size": 16,
        "merge_size": 4,
        "temporal_patch_size": 1,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25]
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, preproc_json);

    check(cfg.image_token_id == 200, "complete: image_token_id=200");
    check(cfg.fixed_image_size == 336, "complete: fixed_image_size=336");
    check(cfg.num_image_pad_tokens == 128, "complete: num_image_pad_tokens=128");
    check(cfg.vision_output_dim == 4096, "complete: vision_output_dim=4096");
    check(cfg.image_token_str == "<img>", "complete: image_token_str=<img>");
    check(cfg.preprocessor_type == "center_crop_chw", "complete: preprocessor_type=center_crop_chw");
    check(cfg.interpolation == "bilinear", "complete: interpolation=bilinear");
    check(cfg.patch_size == 16, "complete: patch_size=16");
    check(cfg.merge_size == 4, "complete: merge_size=4");
    check(cfg.temporal_patch_size == 1, "complete: temporal_patch_size=1");
    check(std::abs(cfg.image_mean[0] - 0.5F) < 1e-5F, "complete: image_mean[0]=0.5");
    check(std::abs(cfg.image_mean[1] - 0.5F) < 1e-5F, "complete: image_mean[1]=0.5");
    check(std::abs(cfg.image_mean[2] - 0.5F) < 1e-5F, "complete: image_mean[2]=0.5");
    check(std::abs(cfg.image_std[0] - 0.25F) < 1e-5F, "complete: image_std[0]=0.25");
    check(std::abs(cfg.image_std[1] - 0.25F) < 1e-5F, "complete: image_std[1]=0.25");
    check(std::abs(cfg.image_std[2] - 0.25F) < 1e-5F, "complete: image_std[2]=0.25");

    // Verify template substitution includes the newline unescaping
    check(cfg.vl_prompt_template.find("user:") != std::string::npos,
          "complete: template contains user:");
}

// Test: parse_vl_preprocess_config with missing fields uses defaults.
static void test_parse_missing_fields_defaults()
{
    // Config JSON with only one field
    const std::string config_json = R"({
        "image_token_id": 42
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");

    check(cfg.image_token_id == 42, "defaults: image_token_id=42");
    check(cfg.fixed_image_size == 448, "defaults: fixed_image_size=448");
    check(cfg.num_image_pad_tokens == 256, "defaults: num_image_pad_tokens=256");
    check(cfg.vision_output_dim == 0, "defaults: vision_output_dim=0");
    check(cfg.preprocessor_type == "qwen_merge_group", "defaults: preprocessor_type=qwen_merge_group");
    check(cfg.interpolation == "bicubic", "defaults: interpolation=bicubic");
    check(cfg.vl_prompt_template.empty(), "defaults: vl_prompt_template empty");
    check(cfg.image_token_str.empty(), "defaults: image_token_str empty");
    // Default image_mean/std from struct init
    check(std::abs(cfg.image_mean[0] - 0.48145466F) < 1e-5F, "defaults: image_mean[0] is default");
    check(std::abs(cfg.image_std[0] - 0.26862954F) < 1e-5F, "defaults: image_std[0] is default");
}

// Test: parse_vl_preprocess_config with empty JSON string.
static void test_parse_empty_json()
{
    auto cfg = trtmc::parse_vl_preprocess_config("", "");

    // All fields should be at their struct default values
    check(cfg.image_token_id == -1, "empty: image_token_id=-1");
    check(cfg.fixed_image_size == 448, "empty: fixed_image_size=448");
    check(cfg.num_image_pad_tokens == 256, "empty: num_image_pad_tokens=256");
    check(cfg.preprocessor_type == "qwen_merge_group", "empty: preprocessor_type=qwen_merge_group");
    check(cfg.interpolation == "bicubic", "empty: interpolation=bicubic");
    check(cfg.patch_size == 14, "empty: patch_size=14 (struct default)");
    check(cfg.merge_size == 2, "empty: merge_size=2 (struct default)");
}

// Test: parse_vl_preprocess_config with empty config but populated preprocessor.
static void test_parse_only_preprocessor_config()
{
    const std::string preproc_json = R"({
        "patch_size": 32,
        "merge_size": 1,
        "temporal_patch_size": 4,
        "image_mean": [0.1, 0.2, 0.3],
        "image_std": [0.4, 0.5, 0.6]
    })";

    auto cfg = trtmc::parse_vl_preprocess_config("{}", preproc_json);

    check(cfg.patch_size == 32, "preproc_only: patch_size=32");
    check(cfg.merge_size == 1, "preproc_only: merge_size=1");
    check(cfg.temporal_patch_size == 4, "preproc_only: temporal_patch_size=4");
    check(std::abs(cfg.image_mean[0] - 0.1F) < 1e-5F, "preproc_only: image_mean[0]=0.1");
    check(std::abs(cfg.image_mean[1] - 0.2F) < 1e-5F, "preproc_only: image_mean[1]=0.2");
    check(std::abs(cfg.image_mean[2] - 0.3F) < 1e-5F, "preproc_only: image_mean[2]=0.3");
    check(std::abs(cfg.image_std[0] - 0.4F) < 1e-5F, "preproc_only: image_std[0]=0.4");
    check(std::abs(cfg.image_std[1] - 0.5F) < 1e-5F, "preproc_only: image_std[1]=0.5");
    check(std::abs(cfg.image_std[2] - 0.6F) < 1e-5F, "preproc_only: image_std[2]=0.6");
}

// Test: format_vl_prompt with empty template returns empty string.
static void test_format_vl_prompt_empty_template()
{
    trtmc::VLPreprocessConfig config;
    config.num_image_pad_tokens = 5;
    config.image_token_str = "<pad>";
    config.vl_prompt_template = "";

    const std::string result = trtmc::format_vl_prompt("Hello", config);
    check(result.empty(), "empty_template: returns empty string");
}

// Test: format_vl_prompt with template missing placeholders.
static void test_format_vl_prompt_no_placeholders()
{
    trtmc::VLPreprocessConfig config;
    config.num_image_pad_tokens = 2;
    config.image_token_str = "<tok>";
    config.vl_prompt_template = "Fixed template with no substitution.";

    const std::string result = trtmc::format_vl_prompt("User input", config);
    check(result == "Fixed template with no substitution.",
          "no_placeholders: template returned unchanged");
}

// Test: preprocessor_type = "aspect_preserve_chw" round-trips through parse.
static void test_parse_vl_config_aspect_preserve()
{
    const std::string config_json = R"({
        "preprocessor_type": "aspect_preserve_chw"
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");
    check(cfg.preprocessor_type == "aspect_preserve_chw",
          "preprocessor_type aspect_preserve_chw parsed correctly");
}

// Test: preprocessor_type = "center_crop_chw" round-trips through parse.
static void test_parse_vl_config_center_crop()
{
    const std::string config_json = R"({
        "preprocessor_type": "center_crop_chw"
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");
    check(cfg.preprocessor_type == "center_crop_chw",
          "preprocessor_type center_crop_chw parsed correctly");
}

// Test: vl_prompt_template with escaped newlines (\n) gets unescaped.
static void test_parse_vl_prompt_template_newline_unescape()
{
    const std::string config_json = R"({
        "vl_prompt_template": "Line1\\nLine2\\nLine3"
    })";

    auto cfg = trtmc::parse_vl_preprocess_config(config_json, "");
    // The \\n sequences should be unescaped to real newlines
    check(cfg.vl_prompt_template.find('\n') != std::string::npos,
          "newline_unescape: template contains real newlines");
}

int main()
{
    test_helper_normalize_hwc_to_chw();
    test_helper_normalize_std_floor_branch();
    test_helper_transform_simple_chw_branch();
    test_helper_transform_qwen_merge_group_branch();

    test_qwen_merge_group_strategy();
    test_simple_chw_strategy();
    test_locateanything_patchify_strategy();
    test_load_missing_image();
    test_format_vl_prompt();
    test_parse_vl_config();
    test_parse_vl_config_default_preprocessor_type();
    test_parse_vl_config_simple_chw();
    test_unknown_preprocessor_type_fallback();
    test_center_crop_chw_strategy();
    test_aspect_preserve_chw_strategy();
    test_pad_center_chw_strategy();
    test_parse_interpolation_default();
    test_parse_interpolation_bilinear();
    test_parse_resample_from_preprocessor();
    // New tests
    test_parse_complete_json();
    test_parse_missing_fields_defaults();
    test_parse_empty_json();
    test_parse_only_preprocessor_config();
    test_format_vl_prompt_empty_template();
    test_format_vl_prompt_no_placeholders();
    test_parse_vl_config_aspect_preserve();
    test_parse_vl_config_center_crop();
    test_parse_vl_prompt_template_newline_unescape();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All image preprocessor tests passed.\n";
    return 0;
}
