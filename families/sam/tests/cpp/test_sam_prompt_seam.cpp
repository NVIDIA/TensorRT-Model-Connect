/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-04
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-SEG-01
// Intent:         SAM prompt seam: point sparse prompt building, embedding encoding, multimask
// selection, image preprocess and mask postprocess Preconditions:  Point coordinates and embedding
// data available Postconditions: Sparse prompts padded correctly, missing data returns zeros,
// multimask selection correct, separable resampling bit-identical to the per-sample definition
// =============================================================================

#include "families/sam/runtime/sam_output_selection.h"
#include "families/sam/runtime/sam_postprocess_seam.h"
#include "families/sam/runtime/sam_preprocess.h"
#include "families/sam/runtime/sam_prompt_seam.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++g_failures;
    }
}

bool bit_identical(const std::vector<float>& actual, const std::vector<float>& expected) {
    return actual.size() == expected.size() &&
           std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
}

std::vector<float> seeded_values(std::size_t count, unsigned seed, float low, float high) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(low, high);
    std::vector<float> values(count);
    for (auto& value : values)
        value = uniform(rng);
    return values;
}

// The per-sample mask resize: sample_mask_bilinear at every destination pixel.
std::vector<float> reference_resize_mask(const std::vector<float>& mask, int32_t src_w,
                                         int32_t src_h, int32_t dst_w, int32_t dst_h) {
    if (src_w == dst_w && src_h == dst_h)
        return mask;
    std::vector<float> resized(static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h));
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    for (int32_t y = 0; y < dst_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
        for (int32_t x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
            resized[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
                    static_cast<std::size_t>(x)] =
                trtmc::sample_mask_bilinear(mask, src_w, src_h, src_x, src_y);
        }
    }
    return resized;
}

// The per-mask restore: upsample to image_size, crop the rescaled region, resize to the original.
std::vector<float> reference_restore_mask(const std::vector<float>& mask, int32_t mask_w,
                                          int32_t mask_h, int32_t image_size, int32_t rescaled_w,
                                          int32_t rescaled_h, int32_t original_w,
                                          int32_t original_h) {
    const auto upsampled = reference_resize_mask(mask, mask_w, mask_h, image_size, image_size);
    std::vector<float> cropped;
    for (int32_t y = 0; y < rescaled_h; ++y) {
        const auto row = upsampled.begin() + static_cast<std::ptrdiff_t>(y) * image_size;
        cropped.insert(cropped.end(), row, row + rescaled_w);
    }
    return reference_resize_mask(cropped, rescaled_w, rescaled_h, original_w, original_h);
}

// The per-sample image preprocess: bilinear sample of the HWC image, then (value - mean) / std.
std::vector<float> reference_preprocess(const std::vector<float>& image, int32_t height,
                                        int32_t width, int32_t image_size, int32_t rescaled_w,
                                        int32_t rescaled_h, const std::array<float, 3>& mean,
                                        const std::array<float, 3>& stdv) {
    std::vector<float> pixels(3 * static_cast<std::size_t>(image_size) * image_size, 0.0F);
    const auto sample = [&](int32_t channel, float x, float y) {
        const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(width - 1));
        const float clamped_y = std::clamp(y, 0.0F, static_cast<float>(height - 1));
        const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
        const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
        const int32_t x1 = std::min(x0 + 1, width - 1);
        const int32_t y1 = std::min(y0 + 1, height - 1);
        const float wx = clamped_x - static_cast<float>(x0);
        const float wy = clamped_y - static_cast<float>(y0);
        const auto at = [&](int32_t yy, int32_t xx) {
            return image[static_cast<std::size_t>((yy * width + xx) * 3 + channel)];
        };
        const float top = at(y0, x0) * (1.0F - wx) + at(y0, x1) * wx;
        const float bottom = at(y1, x0) * (1.0F - wx) + at(y1, x1) * wx;
        return top * (1.0F - wy) + bottom * wy;
    };
    for (int32_t y = 0; y < rescaled_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5F) * static_cast<float>(height) /
                                static_cast<float>(rescaled_h) -
                            0.5F;
        for (int32_t x = 0; x < rescaled_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5F) * static_cast<float>(width) /
                                    static_cast<float>(rescaled_w) -
                                0.5F;
            for (int32_t c = 0; c < 3; ++c) {
                const float value = sample(c, src_x, src_y);
                pixels[static_cast<std::size_t>(c) * image_size * image_size +
                       static_cast<std::size_t>(y) * image_size + x] =
                    (value - mean[static_cast<std::size_t>(c)]) / stdv[static_cast<std::size_t>(c)];
            }
        }
    }
    return pixels;
}

void check_close(float actual, float expected, const char* test_name) {
    constexpr float kTolerance = 1e-5F;
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << test_name << " actual=" << actual << " expected=" << expected
                  << '\n';
        ++g_failures;
    }
}

void test_build_sam_point_sparse_prompt_happy_path() {
    const std::vector<float> shared_image_pe = {1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> point_embed_fg = {0.1F, 0.2F, 0.3F, 0.4F};
    const std::vector<float> point_embed_bg = {9.0F, 9.0F, 9.0F, 9.0F};
    const std::vector<float> not_a_point_embed = {9.0F, 8.0F, 7.0F, 6.0F};

    const auto sparse =
        trtmc::build_sam_point_sparse_prompt(0.5F, 0.5F, true, 9, 9, 10, 4, shared_image_pe,
                                             point_embed_fg, point_embed_bg, not_a_point_embed);

    check(sparse.size() == 8, "happy path: sparse size");
    if (sparse.size() != 8) {
        return;
    }

    check_close(sparse[0], 0.1F, "happy path: point value[0]");
    check_close(sparse[1], 0.2F, "happy path: point value[1]");
    check_close(sparse[2], 1.3F, "happy path: point value[2]");
    check_close(sparse[3], 1.4F, "happy path: point value[3]");
    check_close(sparse[4], 9.0F, "happy path: pad value[0]");
    check_close(sparse[5], 8.0F, "happy path: pad value[1]");
    check_close(sparse[6], 7.0F, "happy path: pad value[2]");
    check_close(sparse[7], 6.0F, "happy path: pad value[3]");
}

void test_fractional_prompt_matches_reference_pixel_quantization() {
    check_close(trtmc::quantize_sam_fractional_point(0.50390625F, 640), 322.0F / 640.0F,
                "prompt quantization: x uses the reference integer pixel");
    check_close(trtmc::quantize_sam_fractional_point(0.6461318F, 349), 225.0F / 349.0F,
                "prompt quantization: y uses the reference integer pixel");
    check_close(trtmc::quantize_sam_fractional_point(0.25F, 0), 0.25F,
                "prompt quantization: invalid image size preserves the fraction");
}

void test_encode_sam_point_embedding_missing_data_returns_zeroes() {
    const std::vector<float> shared_image_pe = {1.0F, 2.0F, 3.0F};
    const std::vector<float> point_embed_fg = {0.1F, 0.2F, 0.3F, 0.4F};
    const std::vector<float> point_embed_bg = {5.0F, 6.0F, 7.0F};

    const auto embedding = trtmc::encode_sam_point_embedding(
        2.0F, 3.0F, false, 10, 4, shared_image_pe, point_embed_fg, point_embed_bg);

    check(embedding.size() == 4, "missing data: embedding size");
    if (embedding.size() != 4) {
        return;
    }

    check_close(embedding[0], 0.0F, "missing data: value[0]");
    check_close(embedding[1], 0.0F, "missing data: value[1]");
    check_close(embedding[2], 0.0F, "missing data: value[2]");
    check_close(embedding[3], 0.0F, "missing data: value[3]");
}

void test_build_sam_point_sparse_prompt_short_padding_embedding() {
    const std::vector<float> shared_image_pe = {1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> point_embed_fg = {8.0F, 8.0F, 8.0F, 8.0F};
    const std::vector<float> point_embed_bg = {0.5F, 0.5F, 0.5F, 0.5F};
    const std::vector<float> not_a_point_embed = {1.0F, 2.0F, 3.0F};

    const auto sparse =
        trtmc::build_sam_point_sparse_prompt(0.5F, 0.5F, false, 9, 9, 10, 4, shared_image_pe,
                                             point_embed_fg, point_embed_bg, not_a_point_embed);

    check(sparse.size() == 8, "short padding: sparse size");
    if (sparse.size() != 8) {
        return;
    }

    check_close(sparse[0], 0.5F, "short padding: point value[0]");
    check_close(sparse[1], 0.5F, "short padding: point value[1]");
    check_close(sparse[2], 1.5F, "short padding: point value[2]");
    check_close(sparse[3], 1.5F, "short padding: point value[3]");
    check_close(sparse[4], 0.0F, "short padding: pad value[0]");
    check_close(sparse[5], 0.0F, "short padding: pad value[1]");
    check_close(sparse[6], 0.0F, "short padding: pad value[2]");
    check_close(sparse[7], 0.0F, "short padding: pad value[3]");
}

void test_select_sam_multimask_outputs_keeps_trailing_masks() {
    trtmc::SamResult result;
    result.mask_height = 1;
    result.mask_width = 2;
    result.num_masks = 4;
    result.masks = {
        10.0F, 11.0F, 20.0F, 21.0F, 30.0F, 31.0F, 40.0F, 41.0F,
    };
    result.iou_scores = {0.1F, 0.2F, 0.3F, 0.4F};

    const auto trimmed = trtmc::select_sam_multimask_outputs(std::move(result), 3);

    check(trimmed.num_masks == 3, "multimask: keeps requested count");
    check(trimmed.masks == std::vector<float>({20.0F, 21.0F, 30.0F, 31.0F, 40.0F, 41.0F}),
          "multimask: keeps trailing mask values");
    check(trimmed.iou_scores == std::vector<float>({0.2F, 0.3F, 0.4F}),
          "multimask: keeps trailing mask scores");
}

void test_select_sam_multimask_outputs_keeps_original_for_invalid_requests() {
    trtmc::SamResult result;
    result.mask_height = 1;
    result.mask_width = 2;
    result.num_masks = 2;
    result.masks = {1.0F, 2.0F, 3.0F, 4.0F};
    result.iou_scores = {0.1F, 0.2F};

    const auto keep_all = trtmc::select_sam_multimask_outputs(result, 2);
    check(keep_all.num_masks == 2, "multimask invalid: keeps original when count already fits");
    check(keep_all.masks == result.masks, "multimask invalid: preserves masks");

    const auto invalid_shape = trtmc::select_sam_multimask_outputs(
        trtmc::SamResult{result.masks, result.iou_scores, 2, 0, 2}, 1);
    check(invalid_shape.num_masks == 2,
          "multimask invalid: keeps original when mask shape invalid");
}

void test_sample_and_resize_sam_mask_helpers_handle_clamp_identity_and_invalid_inputs() {
    const std::vector<float> mask = {
        1.0F,
        2.0F,
        3.0F,
        4.0F,
    };

    check_close(trtmc::sample_mask_bilinear(mask, 2, 2, -5.0F, -5.0F), 1.0F,
                "sample bilinear: clamps to top-left");
    check_close(trtmc::sample_mask_bilinear(mask, 2, 2, 5.0F, 5.0F), 4.0F,
                "sample bilinear: clamps to bottom-right");
    check_close(trtmc::sample_mask_bilinear(mask, 2, 2, 0.5F, 0.5F), 2.5F,
                "sample bilinear: averages center point");
    check_close(trtmc::sample_mask_bilinear({}, 2, 2, 0.0F, 0.0F), 0.0F,
                "sample bilinear: returns zero for empty mask");

    check(trtmc::resize_mask_bilinear(mask, 2, 2, 2, 2) == mask,
          "resize bilinear: identity resize preserves mask");
    check(trtmc::resize_mask_bilinear(mask, 2, 2, 0, 2).empty(),
          "resize bilinear: invalid destination returns empty mask");
}

void test_postprocess_sam_result_crops_padding_region() {
    trtmc::SamResult result;
    result.num_masks = 1;
    result.mask_width = 4;
    result.mask_height = 4;
    result.masks = {
        0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F,
        0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F,
    };

    const auto processed = trtmc::postprocess_sam_result(std::move(result),
                                                         /*image_size=*/4,
                                                         /*rescaled_w=*/2,
                                                         /*rescaled_h=*/4,
                                                         /*original_w=*/2,
                                                         /*original_h=*/4);

    check(processed.mask_width == 2, "postprocess: width updated to original");
    check(processed.mask_height == 4, "postprocess: height updated to original");
    check(processed.masks == std::vector<float>(8, 0.0F),
          "postprocess: right-padding region is cropped away");
}

void test_postprocess_sam_result_preserves_invalid_requests_and_incomplete_payload() {
    trtmc::SamResult invalid_request;
    invalid_request.num_masks = 1;
    invalid_request.mask_width = 2;
    invalid_request.mask_height = 2;
    invalid_request.masks = {1.0F, 2.0F, 3.0F, 4.0F};

    const auto unchanged_invalid = trtmc::postprocess_sam_result(invalid_request,
                                                                 /*image_size=*/0,
                                                                 /*rescaled_w=*/2,
                                                                 /*rescaled_h=*/2,
                                                                 /*original_w=*/2,
                                                                 /*original_h=*/2);
    check(unchanged_invalid.masks == invalid_request.masks,
          "postprocess invalid: returns original result");

    trtmc::SamResult incomplete_payload = invalid_request;
    incomplete_payload.masks = {1.0F, 2.0F, 3.0F};
    const auto unchanged_incomplete = trtmc::postprocess_sam_result(incomplete_payload,
                                                                    /*image_size=*/2,
                                                                    /*rescaled_w=*/2,
                                                                    /*rescaled_h=*/2,
                                                                    /*original_w=*/2,
                                                                    /*original_h=*/2);
    check(unchanged_incomplete.masks == incomplete_payload.masks,
          "postprocess incomplete: returns original result");

    const auto unchanged_oversized = trtmc::postprocess_sam_result(invalid_request,
                                                                   /*image_size=*/4,
                                                                   /*rescaled_w=*/5,
                                                                   /*rescaled_h=*/2,
                                                                   /*original_w=*/5,
                                                                   /*original_h=*/2);
    check(unchanged_oversized.masks == invalid_request.masks &&
              unchanged_oversized.mask_width == invalid_request.mask_width,
          "postprocess oversized: a rescaled width above image_size returns original result");

    const auto unchanged_too_tall = trtmc::postprocess_sam_result(invalid_request,
                                                                  /*image_size=*/4,
                                                                  /*rescaled_w=*/2,
                                                                  /*rescaled_h=*/5,
                                                                  /*original_w=*/2,
                                                                  /*original_h=*/5);
    check(unchanged_too_tall.masks == invalid_request.masks &&
              unchanged_too_tall.mask_height == invalid_request.mask_height,
          "postprocess oversized: a rescaled height above image_size returns original result");
}

void test_resize_mask_bilinear_matches_per_sample_reference() {
    // up, down, mixed, one axis unchanged, and degenerate one-sample axes
    const int32_t sizes[][4] = {{256, 256, 1024, 1024}, {1024, 611, 640, 382}, {37, 53, 101, 29},
                                {64, 48, 64, 97},       {5, 1, 3, 7},          {1, 1, 4, 4}};
    unsigned seed = 1;
    for (const auto& size : sizes) {
        auto mask = seeded_values(static_cast<std::size_t>(size[0]) * size[1], seed++, -20.F, 20.F);
        mask[0] = -0.0F;
        const auto actual = trtmc::resize_mask_bilinear(mask, size[0], size[1], size[2], size[3]);
        const auto expected = reference_resize_mask(mask, size[0], size[1], size[2], size[3]);
        check(bit_identical(actual, expected), "resize bilinear: bit-identical to per-sample");
    }
    check(trtmc::resize_mask_bilinear({}, 2, 2, 3, 3) == std::vector<float>(9, 0.0F),
          "resize bilinear: empty mask resizes to zeros");
}

void test_postprocess_sam_result_matches_per_sample_restore() {
    // {mask_w, mask_h, image_size, rescaled_w, rescaled_h, original_w, original_h}
    const int32_t cases[][7] = {
        {256, 256, 1024, 1024, 611, 640, 382},    // landscape (the family's test image geometry)
        {256, 256, 1024, 611, 1024, 382, 640},    // portrait
        {256, 256, 1024, 1024, 1024, 1024, 1024}, // no crop, no restore resize
        {256, 256, 1024, 1024, 768, 1024, 768},   // crop only
        {256, 256, 1024, 1024, 1, 2000, 1},       // restore resizes the width only
        {256, 256, 1024, 7, 1024, 7, 1500},       // restore resizes the height only
        {16, 16, 16, 16, 9, 30, 17},              // the upsample is a copy
        {16, 8, 16, 16, 8, 30, 17},               // the mask width alone is image_size
        {8, 16, 16, 16, 16, 40, 16},              // the mask height alone is image_size
        {8, 8, 32, 32, 3, 100, 9},
    };
    unsigned seed = 100;
    for (const auto& c : cases) {
        trtmc::SamResult result;
        result.num_masks = 3;
        result.mask_width = c[0];
        result.mask_height = c[1];
        const auto mask_area = static_cast<std::size_t>(c[0]) * c[1];
        result.masks = seeded_values(3 * mask_area, seed++, -20.F, 20.F);
        result.masks[1] = -0.0F;
        result.iou_scores = {0.5F, 0.6F, 0.7F};

        std::vector<float> expected;
        for (int32_t m = 0; m < 3; ++m) {
            const std::vector<float> mask(result.masks.begin() + m * mask_area,
                                          result.masks.begin() + (m + 1) * mask_area);
            const auto restored =
                reference_restore_mask(mask, c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
            expected.insert(expected.end(), restored.begin(), restored.end());
        }
        const auto processed =
            trtmc::postprocess_sam_result(std::move(result), c[2], c[3], c[4], c[5], c[6]);
        check(processed.mask_width == c[5] && processed.mask_height == c[6],
              "postprocess per-sample: original size");
        check(bit_identical(processed.masks, expected),
              "postprocess per-sample: bit-identical masks");
    }
}

// Every request is written into ONE reused buffer that starts as NaN, so a
// value the writer fails to overwrite (padding left by a larger earlier
// request) differs from the reference's fresh zeros.
void check_write_sam_pixel_values_matches_per_sample_reference(
    const trtmc::SamConfig& config, const std::vector<std::array<int32_t, 2>>& sizes,
    const std::array<float, 3>& mean, const std::array<float, 3>& stdv, unsigned seed,
    const char* test_name) {
    std::vector<float> pixel_values(3 * static_cast<std::size_t>(config.image_size) *
                                        static_cast<std::size_t>(config.image_size),
                                    std::numeric_limits<float>::quiet_NaN());
    for (const auto& size : sizes) {
        const auto image =
            seeded_values(static_cast<std::size_t>(size[0]) * size[1] * 3, seed++, 0.F, 1.F);
        const auto plan = trtmc::plan_sam_preprocess(image.data(), size[0], size[1], config);
        trtmc::write_sam_pixel_values(pixel_values.data(), plan, image.data(), config);
        const auto expected =
            reference_preprocess(image, size[0], size[1], config.image_size, plan.rescaled_width,
                                 plan.rescaled_height, mean, stdv);
        check(plan.original_height == size[0] && plan.original_width == size[1] &&
                  bit_identical(pixel_values, expected),
              test_name);
    }
}

void test_write_sam_pixel_values_matches_per_sample_reference() {
    check_write_sam_pixel_values_matches_per_sample_reference(
        trtmc::SamConfig{},
        {{382, 640},
         {640, 382},
         {1024, 1024},
         {768, 1024},
         {1080, 1920},
         {60, 100},
         {333, 777},
         {1, 5}},
        {0.485F, 0.456F, 0.406F}, {0.229F, 0.224F, 0.225F}, 200,
        "preprocess per-sample: bit-identical pixel values");

    // The config's own normalization and image_size, not the defaults.
    trtmc::SamConfig custom;
    custom.image_size = 32;
    custom.image_mean = {0.4F, 0.5F, 0.6F};
    custom.image_std = {0.2F, 0.25F, 0.3F};
    check_write_sam_pixel_values_matches_per_sample_reference(
        custom, {{19, 32}, {32, 19}, {13, 13}, {1, 7}, {9, 1}, {40, 33}, {33, 40}, {5, 3}},
        {0.4F, 0.5F, 0.6F}, {0.2F, 0.25F, 0.3F}, 300,
        "preprocess per-sample: custom config bit-identical");
}

void test_plan_sam_preprocess_rejects_invalid_requests() {
    trtmc::SamConfig config;
    const std::vector<float> image(4 * 4 * 3, 0.5F);
    check(trtmc::plan_sam_preprocess(nullptr, 4, 4, config).rescaled_width == 0,
          "plan invalid: no pixels");
    check(trtmc::plan_sam_preprocess(image.data(), 0, 4, config).rescaled_width == 0,
          "plan invalid: empty height");
    check(trtmc::plan_sam_preprocess(image.data(), 4, -1, config).rescaled_width == 0,
          "plan invalid: negative width");
    config.image_size = 0;
    check(trtmc::plan_sam_preprocess(image.data(), 4, 4, config).rescaled_width == 0,
          "plan invalid: no image_size");
}

} // namespace

int main() {
    test_build_sam_point_sparse_prompt_happy_path();
    test_fractional_prompt_matches_reference_pixel_quantization();
    test_encode_sam_point_embedding_missing_data_returns_zeroes();
    test_build_sam_point_sparse_prompt_short_padding_embedding();
    test_select_sam_multimask_outputs_keeps_trailing_masks();
    test_select_sam_multimask_outputs_keeps_original_for_invalid_requests();
    test_sample_and_resize_sam_mask_helpers_handle_clamp_identity_and_invalid_inputs();
    test_postprocess_sam_result_crops_padding_region();
    test_postprocess_sam_result_preserves_invalid_requests_and_incomplete_payload();
    test_resize_mask_bilinear_matches_per_sample_reference();
    test_postprocess_sam_result_matches_per_sample_restore();
    test_write_sam_pixel_values_matches_per_sample_reference();
    test_plan_sam_preprocess_rejects_invalid_requests();

    if (g_failures != 0) {
        std::cerr << g_failures << " SAM prompt seam test(s) failed" << '\n';
        return 1;
    }

    std::cout << "SAM prompt seam tests passed" << '\n';
    return 0;
}
