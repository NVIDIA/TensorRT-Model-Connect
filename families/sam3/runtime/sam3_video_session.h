/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace trtmc {

class Sam3Pipeline;

// Full per-frame SAM3 result. Object-indexed vectors share object_ids order;
// masks are binary float32 [objects, height, width] and boxes are absolute xyxy.
struct Sam3VideoFrameResult {
    int32_t frame_idx{-1};
    int32_t height{0};
    int32_t width{0};
    std::vector<int32_t> object_ids;
    std::vector<float> masks;
    std::vector<float> detection_scores;
    std::vector<float> tracker_scores;
    std::vector<float> boxes;
    std::vector<int32_t> removed_object_ids;
    std::vector<int32_t> suppressed_object_ids;
};

struct Sam3VideoFrameView {
    const float* pixels{nullptr};
    int32_t height{0};
    int32_t width{0};
};

// Internal synchronous frame view. Frame zero owns a copy; continuation frames
// borrow storage for the duration of propagate_borrowed_continuation().
struct Sam3VideoFrame {
    int32_t frame_idx{-1};
    int32_t height{0};
    int32_t width{0};
    std::vector<float> owned_pixels;
    const float* borrowed_pixels{nullptr};
    std::size_t borrowed_pixel_count{0};

    const float* pixel_data() const {
        return borrowed_pixels != nullptr ? borrowed_pixels : owned_pixels.data();
    }
    std::size_t pixel_count() const {
        return borrowed_pixels != nullptr ? borrowed_pixel_count : owned_pixels.size();
    }
};

struct Sam3VideoFrameProcessor {
    using AcceptPrompt = std::function<Sam3VideoFrameResult(const Sam3VideoFrame&)>;
    using ContinueBorrowed = std::function<std::vector<Sam3VideoFrameResult>(
        Sam3VideoFrameResult, const std::vector<Sam3VideoFrame>&, int32_t)>;

    AcceptPrompt accept_prompt;
    ContinueBorrowed continue_borrowed;

    explicit operator bool() const {
        return static_cast<bool>(accept_prompt) && static_cast<bool>(continue_borrowed);
    }
};

// Exact customer schedule: execute and return frame zero first, then process a
// complete borrowed tail in strict temporal order. A session is one-shot.
class Sam3VideoSegmentationSession final {
  public:
    Sam3VideoSegmentationSession(std::string text_prompt, Sam3VideoFrameProcessor processor,
                                 int32_t max_video_frames = 1024);

    Sam3VideoFrameResult accept_prompt_frame(const float* image_pixels, int32_t image_height,
                                             int32_t image_width);
    std::vector<Sam3VideoFrameResult>
    propagate_borrowed_continuation(Sam3VideoFrameResult prompt_result,
                                    const Sam3VideoFrameView* frames, std::size_t num_frames);

  private:
    friend class Sam3Pipeline;

    enum class MaskValidation { kFull, kBinaryByConstruction };
    Sam3VideoSegmentationSession(std::string text_prompt, Sam3VideoFrameProcessor processor,
                                 int32_t max_video_frames, MaskValidation mask_validation);

    Sam3VideoFrameProcessor processor_;
    int32_t max_video_frames_{1024};
    MaskValidation mask_validation_{MaskValidation::kFull};
    bool prompt_processed_{false};
    bool completed_{false};
    bool poisoned_{false};
};

} // namespace trtmc
