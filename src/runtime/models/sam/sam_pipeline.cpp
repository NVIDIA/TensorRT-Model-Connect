/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam/sam_pipeline.h"

#include "runtime/models/sam/sam_output_selection.h"
#include "runtime/models/sam/sam_postprocess_seam.h"
#include "runtime/models/sam/sam_prompt_seam.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct SamPreprocessPlan {
    std::vector<float> pixel_values;
    int32_t rescaled_width{0};
    int32_t rescaled_height{0};
    int32_t original_width{0};
    int32_t original_height{0};
};

std::array<float, 3> sam_channel_values(const std::vector<float>& values,
                                        std::array<float, 3> defaults) {
    const auto count = std::min(values.size(), defaults.size());
    for (std::size_t i = 0; i < count; ++i)
        defaults[i] = values[i];
    return defaults;
}

float sample_hwc_channel(const float* pixels, int32_t height, int32_t width, int32_t channel,
                         float x, float y) {
    const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(width - 1));
    const float clamped_y = std::clamp(y, 0.0F, static_cast<float>(height - 1));
    const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
    const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
    const int32_t x1 = std::min(x0 + 1, width - 1);
    const int32_t y1 = std::min(y0 + 1, height - 1);
    const float wx = clamped_x - static_cast<float>(x0);
    const float wy = clamped_y - static_cast<float>(y0);

    const auto idx = [width](int32_t yy, int32_t xx, int32_t cc) {
        return static_cast<std::size_t>((yy * width + xx) * 3 + cc);
    };

    const float v00 = pixels[idx(y0, x0, channel)];
    const float v01 = pixels[idx(y0, x1, channel)];
    const float v10 = pixels[idx(y1, x0, channel)];
    const float v11 = pixels[idx(y1, x1, channel)];
    const float top = v00 * (1.0F - wx) + v01 * wx;
    const float bottom = v10 * (1.0F - wx) + v11 * wx;
    return top * (1.0F - wy) + bottom * wy;
}

void fill_sam_preprocessed_pixels(SamPreprocessPlan& plan, const float* image_pixels,
                                  int32_t height, int32_t width, const SamConfig& config,
                                  const std::array<float, 3>& mean,
                                  const std::array<float, 3>& stdv) {
    for (int32_t y = 0; y < plan.rescaled_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5F) * static_cast<float>(height) /
                                static_cast<float>(plan.rescaled_height) -
                            0.5F;
        for (int32_t x = 0; x < plan.rescaled_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5F) * static_cast<float>(width) /
                                    static_cast<float>(plan.rescaled_width) -
                                0.5F;
            for (int32_t c = 0; c < 3; ++c) {
                float value = sample_hwc_channel(image_pixels, height, width, c, src_x, src_y);
                value =
                    (value - mean[static_cast<std::size_t>(c)]) / stdv[static_cast<std::size_t>(c)];
                plan.pixel_values[static_cast<std::size_t>(c) *
                                      static_cast<std::size_t>(config.image_size) *
                                      static_cast<std::size_t>(config.image_size) +
                                  static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(config.image_size) +
                                  static_cast<std::size_t>(x)] = value;
            }
        }
    }
}

SamPreprocessPlan build_sam_preprocess_plan(const float* image_pixels, int32_t height,
                                            int32_t width, const SamConfig& config) {
    SamPreprocessPlan plan;
    if (image_pixels == nullptr || height <= 0 || width <= 0 || config.image_size <= 0)
        return plan;

    const int32_t image_size = config.image_size;
    const int32_t longest_side = std::max(width, height);
    const float scale = static_cast<float>(image_size) / static_cast<float>(longest_side);
    plan.rescaled_width =
        std::max(1, static_cast<int32_t>(std::round(static_cast<float>(width) * scale)));
    plan.rescaled_height =
        std::max(1, static_cast<int32_t>(std::round(static_cast<float>(height) * scale)));
    plan.original_width = width;
    plan.original_height = height;
    plan.pixel_values.assign(static_cast<std::size_t>(3) * static_cast<std::size_t>(image_size) *
                                 static_cast<std::size_t>(image_size),
                             0.0F);

    const auto mean = sam_channel_values(config.image_mean, {0.485F, 0.456F, 0.406F});
    const auto stdv = sam_channel_values(config.image_std, {0.229F, 0.224F, 0.225F});
    fill_sam_preprocessed_pixels(plan, image_pixels, height, width, config, mean, stdv);

    return plan;
}

bool parse_mask_shape(const std::vector<int64_t>& shape, int32_t& num_masks, int32_t& height,
                      int32_t& width) {
    if (shape.size() == 4) {
        num_masks = static_cast<int32_t>(shape[1]);
        height = static_cast<int32_t>(shape[2]);
        width = static_cast<int32_t>(shape[3]);
    } else if (shape.size() == 3) {
        num_masks = static_cast<int32_t>(shape[0]);
        height = static_cast<int32_t>(shape[1]);
        width = static_cast<int32_t>(shape[2]);
    } else {
        return false;
    }
    return num_masks > 0 && height > 0 && width > 0;
}

SamResult parse_sam_outputs(const TensorMap& outputs) {
    SamResult result;
    for (const auto& [name, tensor] : outputs) {
        const auto* data = static_cast<const float*>(tensor.data);
        if (data == nullptr)
            continue;
        if (name == "masks" || name.find("mask") != std::string::npos) {
            int32_t num_masks = 0;
            int32_t height = 0;
            int32_t width = 0;
            if (!parse_mask_shape(tensor.shape, num_masks, height, width))
                continue;
            const auto value_count = static_cast<std::size_t>(num_masks) *
                                     static_cast<std::size_t>(height) *
                                     static_cast<std::size_t>(width);
            result.masks.assign(data, data + value_count);
            result.num_masks = num_masks;
            result.mask_height = height;
            result.mask_width = width;
        } else if (name == "iou_scores" || name.find("iou") != std::string::npos ||
                   name.find("score") != std::string::npos) {
            const auto value_count = static_cast<std::size_t>(tensor.numel());
            result.iou_scores.assign(data, data + value_count);
        }
    }
    return result;
}

PromptedSegmentationResult to_public_result(SamResult result) {
    PromptedSegmentationResult out;
    out.masks = std::move(result.masks);
    out.iou_scores = std::move(result.iou_scores);
    out.num_masks = result.num_masks;
    out.height = result.mask_height;
    out.width = result.mask_width;
    return out;
}

} // namespace

SamPipeline::SamPipeline(std::unique_ptr<TrtModule> image_encoder,
                         std::unique_ptr<TrtModule> mask_decoder, SamConfig config,
                         std::string model_id_str)
    : image_encoder_(std::move(image_encoder)), mask_decoder_(std::move(mask_decoder)),
      config_(std::move(config)), model_id_(std::move(model_id_str)) {
    if (!image_encoder_ || !image_encoder_->ok())
        throw std::runtime_error("SamPipeline: invalid image_encoder");
    if (!mask_decoder_ || !mask_decoder_->ok())
        throw std::runtime_error("SamPipeline: invalid mask_decoder");
}

SegmentResult SamPipeline::segment(const float* pixels, int32_t height, int32_t width) {
    auto prompted = segment_prompted(pixels, height, width);
    SegmentResult result;
    result.height = prompted.height;
    result.width = prompted.width;
    const auto mask_area = static_cast<std::size_t>(std::max(0, prompted.height)) *
                           static_cast<std::size_t>(std::max(0, prompted.width));
    if (mask_area == 0 || prompted.masks.size() < mask_area)
        return result;
    result.mask.resize(mask_area);
    for (std::size_t i = 0; i < mask_area; ++i)
        result.mask[i] = prompted.masks[i] > 0.0F ? 1 : 0;
    return result;
}

PromptedSegmentationResult SamPipeline::segment_prompted(const float* image_pixels,
                                                         int32_t image_height, int32_t image_width,
                                                         float point_x, float point_y,
                                                         bool is_foreground) {
    auto plan = build_sam_preprocess_plan(image_pixels, image_height, image_width, config_);
    if (plan.pixel_values.empty())
        return {};

    Tensor img_t;
    img_t.data = plan.pixel_values.data();
    img_t.shape = {1, 3, config_.image_size, config_.image_size};
    img_t.dtype = DType::kFloat32;

    auto enc_out = image_encoder_->forward({{"pixel_values", img_t}});

    const float reference_point_x = quantize_sam_fractional_point(point_x, plan.original_width);
    const float reference_point_y = quantize_sam_fractional_point(point_y, plan.original_height);
    auto sparse_prompt = build_sam_point_sparse_prompt(
        reference_point_x, reference_point_y, is_foreground, plan.rescaled_width,
        plan.rescaled_height, config_.image_size, config_.decoder_hidden_size,
        config_.shared_image_pe, config_.point_embed_fg, config_.point_embed_bg,
        config_.not_a_point_embed);
    Tensor sparse_t;
    sparse_t.data = sparse_prompt.data();
    sparse_t.shape = {2, config_.decoder_hidden_size};
    sparse_t.dtype = DType::kFloat32;

    TensorMap decoder_inputs;
    for (auto& [name, tensor] : enc_out)
        decoder_inputs[name] = tensor;
    decoder_inputs["sparse_prompt_embeddings"] = sparse_t;

    auto dec_out = mask_decoder_->forward(decoder_inputs);

    auto sam = parse_sam_outputs(dec_out);
    sam = select_sam_multimask_outputs(std::move(sam), config_.num_multimask_outputs);
    sam = postprocess_sam_result(std::move(sam), config_.image_size, plan.rescaled_width,
                                 plan.rescaled_height, plan.original_width, plan.original_height);
    return to_public_result(std::move(sam));
}

} // namespace trtmc
