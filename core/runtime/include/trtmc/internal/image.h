/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/task.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace trtmc::internal {

enum class ImageFormat : std::uint32_t { UInt8, Float32 };

// Borrowed, contiguous host HWC. UInt8 values are [0,255]; Float32 values are
// [0,1]. Color channels are RGB/RGBA, or one grayscale channel. A family
// declares which formats/channel counts it accepts and owns preprocessing.
struct ImageView {
    const void* data{nullptr};
    std::size_t byte_size{0};
    std::uint32_t height{0};
    std::uint32_t width{0};
    std::uint32_t channels{3};
    ImageFormat format{ImageFormat::Float32};
};

// An aligned host mask: zero preserves source content, one selects generation.
// Intermediate values express a soft mask only when the family accepts them.
struct ImageMaskView {
    Span<const float> values;
    std::uint32_t height{0};
    std::uint32_t width{0};
};

// Reuse existing owned float pixels. Image outputs require exactly one frame.
// A non-output distributed participant may return the existing explicit
// zero-frame/zero-shape/empty-RGB completion instead of decoded media.
using ImageResult = trtmc::ImageResult;

inline bool is_worker_completion(const ImageResult& result) noexcept {
    return result.num_frames == 0 && result.height == 0 && result.width == 0 &&
           result.channels == 3 && result.pixels.empty();
}

struct TextToImageRequest {
    std::string_view prompt;
    // Borrowed float32 in the loaded family's latent layout. Empty selects
    // family initialization; unsupported nonempty replay must be rejected.
    Span<const float> initial_latents{};
};
struct ImagesTextToImageEditRequest {
    Span<const ImageView> images;
    std::string_view prompt;
    Span<const float> initial_latents{};
};
struct MaskedImageTextToImageRequest {
    ImageView source;
    ImageMaskView mask;
    std::string_view prompt;
};
struct BatchTextToImageItem {
    TextToImageRequest input;
    ConfigView config;
};
struct BatchTextToImageRequest {
    Span<const BatchTextToImageItem> items;
};

class ITextToImage {
  public:
    using TaskInterface = ITextToImage;
    static constexpr std::string_view kTask = "text_to_image";
    virtual ~ITextToImage() = default;
    virtual ImageResult run(const TextToImageRequest&, ConfigView) = 0;
};

class IImagesTextToImageEdit {
  public:
    using TaskInterface = IImagesTextToImageEdit;
    static constexpr std::string_view kTask = "images_text_to_image_edit";
    virtual ~IImagesTextToImageEdit() = default;
    virtual ImageResult run(const ImagesTextToImageEditRequest&, ConfigView) = 0;
};

class IMaskedImageTextToImage {
  public:
    using TaskInterface = IMaskedImageTextToImage;
    static constexpr std::string_view kTask = "masked_image_text_to_image";
    virtual ~IMaskedImageTextToImage() = default;
    virtual ImageResult run(const MaskedImageTextToImageRequest&, ConfigView) = 0;
};

class IBatchTextToImage {
  public:
    using TaskInterface = IBatchTextToImage;
    static constexpr std::string_view kTask = "batch_text_to_image";
    virtual ~IBatchTextToImage() = default;
    // One native batch invocation, preserving item order. Config preflight for
    // every item must succeed before execution starts; no implicit serial run.
    virtual std::vector<ImageResult> run_batch(const BatchTextToImageRequest&) = 0;
};

} // namespace trtmc::internal
