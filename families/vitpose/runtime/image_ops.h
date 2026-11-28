/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::vitpose {

// The person crop is sized in units of this many pixels, in the processor's box encoding and in
// its warp and keypoint transforms alike.
constexpr double kPixelStd = 200.0;

// One person box in the reference processor's COCO order: the top-left corner, width and height,
// in original-image pixels.
struct PersonBox {
    double x{0.0};
    double y{0.0};
    double width{0.0};
    double height{0.0};
};

// The crop around a box as the reference keeps it: a float32 centre and a float32 scale (the box
// side after the aspect-ratio fit and the padding, in units of kPixelStd pixels).
struct CropGeometry {
    std::array<float, 2> center{};
    std::array<float, 2> scale{};
};

// The checkpoint's preprocessor_config.json and the processor's fixed box padding.
struct PreprocessConfig {
    std::int32_t height{0}; // model input rows (size.height)
    std::int32_t width{0};  // model input columns (size.width)
    double padding_factor{0.0};
    double rescale_factor{0.0};
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
};

// box_to_center_and_scale of the reference processor: double arithmetic on the box, float32
// results. The box must have a positive, finite width and height.
CropGeometry crop_geometry(const PersonBox& box, const PreprocessConfig& config);

// The model's view of one box: the unbiased (UDP) warp of the reference processor, sampled the
// way scipy.ndimage.affine_transform(order=1, mode="constant") samples 8-bit pixels (bilinear,
// zero outside the image, rounded half up). Input and output are contiguous HWC RGB.
std::vector<std::uint8_t> warp_crop(const std::uint8_t* rgb, std::int32_t height,
                                    std::int32_t width, const CropGeometry& geometry,
                                    const PreprocessConfig& config);

// True when every 8-bit value normalizes to a finite float32 the way normalize_crop() computes
// it.
bool normalization_is_finite(const PreprocessConfig& config);

// The fast reference processor's fused rescale and normalization: (value - mean / rescale) /
// (std / rescale) in float32. Returns CHW float32 pixel values.
std::vector<float> normalize_crop(const std::vector<std::uint8_t>& crop,
                                  const PreprocessConfig& config);

struct Keypoint {
    float x{0.0F}; // original-image pixels, pixel centres at integer coordinates
    float y{0.0F};
    float score{0.0F};   // the heatmap's maximum
    bool located{false}; // false when the maximum is not positive: the reference marks it -1
};

// The reference's keypoints_from_heatmaps for one person: the first maximum of every heatmap,
// refined by DARK (Gaussian modulation with `kernel` taps, then one Newton step on the log
// heatmap) and mapped back to the original image through the crop. `heatmaps` is
// [keypoints, height, width] float32; every value must be finite.
std::vector<Keypoint> decode_heatmaps(const float* heatmaps, std::int32_t keypoints,
                                      std::int32_t height, std::int32_t width,
                                      const CropGeometry& geometry, std::int32_t kernel);

} // namespace trtmc::vitpose
