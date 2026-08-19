/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam3_video_session.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace trtmc {

namespace {

std::size_t checked_image_element_count(int32_t height, int32_t width) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("SAM3 video frame dimensions must be positive");
    const auto h = static_cast<std::size_t>(height);
    const auto w = static_cast<std::size_t>(width);
    if (h > std::numeric_limits<std::size_t>::max() / w ||
        h * w > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::overflow_error("SAM3 video frame dimensions overflow the pixel buffer size");
    }
    return h * w * 3U;
}

std::size_t checked_mask_element_count(std::size_t objects, int32_t height, int32_t width) {
    const auto area = checked_image_element_count(height, width) / 3U;
    if (objects != 0 && area > std::numeric_limits<std::size_t>::max() / objects)
        throw std::overflow_error("SAM3 video result dimensions overflow the mask buffer size");
    return objects * area;
}

void validate_ids(const std::vector<int32_t>& ids, const char* field) {
    std::unordered_set<int32_t> unique;
    unique.reserve(ids.size());
    for (const auto id : ids) {
        if (id < 0 || !unique.insert(id).second)
            throw std::runtime_error(std::string("SAM3 video result has invalid ") + field);
    }
}

bool result_geometry_matches(const Sam3VideoFrame& frame, const Sam3VideoFrameResult& result) {
    return result.frame_idx == frame.frame_idx && result.height == frame.height &&
           result.width == frame.width;
}

bool result_buffers_align(const Sam3VideoFrameResult& result, std::size_t objects) {
    return result.masks.size() ==
               checked_mask_element_count(objects, result.height, result.width) &&
           result.detection_scores.size() == objects && result.tracker_scores.size() == objects &&
           result.boxes.size() == objects * 4U;
}

bool finite_values(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool result_scores_are_finite(const Sam3VideoFrameResult& result) {
    return finite_values(result.detection_scores) && finite_values(result.tracker_scores) &&
           finite_values(result.boxes);
}

bool masks_are_binary_and_finite(const std::vector<float>& masks) {
    return std::all_of(masks.begin(), masks.end(), [](float value) {
        return std::isfinite(value) && (value == 0.0F || value == 1.0F);
    });
}

void validate_result(const Sam3VideoFrame& frame, const Sam3VideoFrameResult& result,
                     bool validate_binary_masks) {
    if (!result_geometry_matches(frame, result)) {
        throw std::runtime_error("SAM3 video frame result geometry does not match its input");
    }
    const auto objects = result.object_ids.size();
    if (!result_buffers_align(result, objects)) {
        throw std::runtime_error("SAM3 video result buffers do not align with object IDs");
    }
    validate_ids(result.object_ids, "object IDs");
    validate_ids(result.removed_object_ids, "removed object IDs");
    validate_ids(result.suppressed_object_ids, "suppressed object IDs");
    if (!result_scores_are_finite(result)) {
        throw std::runtime_error("SAM3 video result contains non-finite values");
    }
    if (validate_binary_masks && !masks_are_binary_and_finite(result.masks))
        throw std::runtime_error("SAM3 video result masks must be finite and binary");
}

std::vector<Sam3VideoFrame> make_borrowed_tail(const Sam3VideoFrameView* frames,
                                               std::size_t num_frames, int32_t max_video_frames) {
    if (frames == nullptr || num_frames == 0)
        throw std::invalid_argument("SAM3 continuation requires frame zero");
    if (num_frames > static_cast<std::size_t>(max_video_frames))
        throw std::length_error("SAM3 video session frame limit exceeded");
    std::vector<Sam3VideoFrame> tail;
    tail.reserve(num_frames - 1U);
    for (std::size_t index = 0; index < num_frames; ++index) {
        const auto count = checked_image_element_count(frames[index].height, frames[index].width);
        if (frames[index].pixels == nullptr)
            throw std::invalid_argument("SAM3 video frame pixels must not be null");
        if (index == 0)
            continue;
        Sam3VideoFrame frame;
        frame.frame_idx = static_cast<int32_t>(index);
        frame.height = frames[index].height;
        frame.width = frames[index].width;
        frame.borrowed_pixels = frames[index].pixels;
        frame.borrowed_pixel_count = count;
        tail.push_back(std::move(frame));
    }
    return tail;
}

} // namespace

Sam3VideoSegmentationSession::Sam3VideoSegmentationSession(std::string text_prompt,
                                                           Sam3VideoFrameProcessor processor,
                                                           int32_t max_video_frames)
    : Sam3VideoSegmentationSession(std::move(text_prompt), std::move(processor), max_video_frames,
                                   MaskValidation::kFull) {}

Sam3VideoSegmentationSession::Sam3VideoSegmentationSession(std::string text_prompt,
                                                           Sam3VideoFrameProcessor processor,
                                                           int32_t max_video_frames,
                                                           MaskValidation mask_validation)
    : processor_(std::move(processor)), max_video_frames_(max_video_frames),
      mask_validation_(mask_validation) {
    if (text_prompt.empty() || !processor_ || max_video_frames_ <= 0)
        throw std::invalid_argument("SAM3 video session configuration is invalid");
}

Sam3VideoFrameResult Sam3VideoSegmentationSession::accept_prompt_frame(const float* image_pixels,
                                                                       int32_t image_height,
                                                                       int32_t image_width) {
    if (poisoned_)
        throw std::runtime_error("SAM3 video session is poisoned after a processing failure");
    if (prompt_processed_)
        throw std::runtime_error("SAM3 prompt frame is one-shot");
    const auto count = checked_image_element_count(image_height, image_width);
    if (image_pixels == nullptr)
        throw std::invalid_argument("SAM3 video frame pixels must not be null");
    Sam3VideoFrame frame;
    frame.frame_idx = 0;
    frame.height = image_height;
    frame.width = image_width;
    frame.owned_pixels.assign(image_pixels, image_pixels + count);
    try {
        auto result = processor_.accept_prompt(frame);
        validate_result(frame, result, mask_validation_ == MaskValidation::kFull);
        prompt_processed_ = true;
        return result;
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

std::vector<Sam3VideoFrameResult> Sam3VideoSegmentationSession::propagate_borrowed_continuation(
    Sam3VideoFrameResult prompt_result, const Sam3VideoFrameView* frames, std::size_t num_frames) {
    if (poisoned_)
        throw std::runtime_error("SAM3 video session is poisoned after a processing failure");
    if (!prompt_processed_ || completed_)
        throw std::runtime_error("SAM3 continuation requires one unconsumed prompt result");
    auto tail = make_borrowed_tail(frames, num_frames, max_video_frames_);
    Sam3VideoFrame prompt;
    prompt.frame_idx = 0;
    prompt.height = frames[0].height;
    prompt.width = frames[0].width;
    prompt.borrowed_pixels = frames[0].pixels;
    prompt.borrowed_pixel_count = checked_image_element_count(prompt.height, prompt.width);
    validate_result(prompt, prompt_result, mask_validation_ == MaskValidation::kFull);
    try {
        auto results = processor_.continue_borrowed(std::move(prompt_result), tail,
                                                    static_cast<int32_t>(num_frames));
        if (results.size() != num_frames)
            throw std::runtime_error("SAM3 continuation returned the wrong frame count");
        validate_result(prompt, results.front(), mask_validation_ == MaskValidation::kFull);
        for (std::size_t index = 0; index < tail.size(); ++index)
            validate_result(tail[index], results[index + 1U],
                            mask_validation_ == MaskValidation::kFull);
        completed_ = true;
        return results;
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

} // namespace trtmc
