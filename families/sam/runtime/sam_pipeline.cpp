/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam/runtime/sam_pipeline.h"

#include "families/sam/runtime/sam_output_selection.h"
#include "families/sam/runtime/sam_postprocess_seam.h"
#include "families/sam/runtime/sam_preprocess.h"
#include "families/sam/runtime/sam_prompt_seam.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

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

SamPipeline::SamPipeline(std::unique_ptr<ITrtModule> image_encoder,
                         std::unique_ptr<ITrtModule> mask_decoder, SamConfig config,
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
