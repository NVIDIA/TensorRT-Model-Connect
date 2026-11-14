/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::oneformer {

struct ImageSize {
    std::int32_t height{0};
    std::int32_t width{0};
};

struct Normalization {
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
    double rescale_factor{1.0 / 255.0};
};

// The processor's target size: shortest edge to `shortest`, unless the longest
// edge would exceed `longest` (Transformers `get_size_with_aspect_ratio`).
ImageSize processed_size(ImageSize source, std::int32_t shortest, std::int32_t longest);

// ATen's antialiased bilinear resize of 8-bit interleaved channels (separable,
// int16 weights, horizontal pass first, 8-bit intermediate). torchvision 0.27
// and later run this kernel in `resize(BILINEAR, antialias=True)` for uint8 CPU
// tensors on AVX2 / AVX-512 x86 and on aarch64; on other x86 CPUs (and on
// aarch64 before 0.27) they resize in float32 and round instead.
std::vector<std::uint8_t> resize_uint8_antialias(const std::uint8_t* pixels, ImageSize source,
                                                 ImageSize target, std::int32_t channels);

// The fused rescale + normalize of the fast processor, as planar float32 CHW.
std::vector<float> normalize_planar(const std::vector<std::uint8_t>& pixels, ImageSize size,
                                    const Normalization& normalization);

// ATen's float `resize(BILINEAR, antialias=True)` of [classes, H, W] score maps to
// `target`, then the per-pixel argmax (first maximum on ties).
std::vector<std::int32_t> resize_argmax(const float* scores, std::int32_t classes, ImageSize source,
                                        ImageSize target);

} // namespace trtmc::oneformer
