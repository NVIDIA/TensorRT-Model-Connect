/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/d3/runtime/media.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace trtmc::d3 {
uint16_t bfloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
TensorMap VisualInputs::tensors() {
    return {
        {"patches", {patches.data(), {count, 1536}, DType::kBFloat16}},
        {"rotary_angles",
         {angles.data(), {count, static_cast<int64_t>(angles.size()) / count}, DType::kFloat32}},
        {"position_indices", {indices.data(), {4, count}, DType::kInt32}},
        {"position_weights", {factors.data(), {4, count}, DType::kFloat32}}};
}
VisualInputs prepare_image(const ImageResult& image, int32_t height, int32_t width,
                           int32_t positions, int32_t head_dim) {
    if (height < 32 || width < 32 || height % 32 || width % 32 || image.height != height ||
        image.width != width || image.channels != 3 || image.num_frames != 1 ||
        image.pixels.size() != static_cast<std::size_t>(height) * width * 3)
        throw std::invalid_argument("D3 requires RGB images at the exact bundle image dimensions");
    for (const float p : image.pixels)
        if (!std::isfinite(p) || p < 0 || p > 255 || std::floor(p) != p)
            throw std::invalid_argument("D3 image pixels must represent RGB8 values in [0,255]");
    const int32_t gh = height / 16, gw = width / 16;
    const int32_t side = static_cast<int32_t>(std::sqrt(positions));
    if (side * side != positions || head_dim <= 0 || head_dim % 4)
        throw std::invalid_argument("invalid D3 vision geometry");
    VisualInputs out;
    out.count = static_cast<int64_t>(gh) * gw;
    out.patches.resize(out.count * 1536);
    out.angles.resize(out.count * head_dim / 2);
    out.indices.resize(4 * out.count);
    out.factors.resize(4 * out.count);
    std::size_t token = 0;
    for (int32_t by = 0; by < gh; by += 2)
        for (int32_t bx = 0; bx < gw; bx += 2)
            for (int32_t dy = 0; dy < 2; ++dy)
                for (int32_t dx = 0; dx < 2; ++dx, ++token) {
                    const int32_t y = by + dy, x = bx + dx;
                    std::size_t offset = token * 1536;
                    for (int32_t c = 0; c < 3; ++c)
                        for (int32_t t = 0; t < 2; ++t)
                            for (int32_t py = 0; py < 16; ++py)
                                for (int32_t px = 0; px < 16; ++px) {
                                    const float p =
                                        image.pixels[((y * 16 + py) * width + x * 16 + px) * 3 + c];
                                    out.patches[offset++] = bfloat16((p - 127.5F) / 127.5F);
                                }
                    for (int32_t i = 0; i < head_dim / 4; ++i) {
                        const float inv =
                            (1.0F / std::pow(10000.0F, static_cast<float>(2 * i) / (head_dim / 2)));
                        out.angles[token * head_dim / 2 + i] = y * inv;
                        out.angles[token * head_dim / 2 + head_dim / 4 + i] = x * inv;
                    }
                    const float fy =
                        static_cast<float>(static_cast<double>(side - 1) * y / (gh - 1));
                    const float fx =
                        static_cast<float>(static_cast<double>(side - 1) * x / (gw - 1));
                    const auto iy = static_cast<int32_t>(fy), ix = static_cast<int32_t>(fx);
                    const float ry = fy - iy, rx = fx - ix;
                    for (int32_t k = 0; k < 4; ++k) {
                        out.indices[k * out.count + token] =
                            std::min(iy + k / 2, side - 1) * side + std::min(ix + k % 2, side - 1);
                        out.factors[k * out.count + token] =
                            ((k / 2 ? ry : 1 - ry) * (k % 2 ? rx : 1 - rx));
                    }
                }
    return out;
}
std::vector<float> rope_angles(const std::vector<int32_t>& ids, int32_t image_token, int32_t height,
                               int32_t width, std::size_t image_count) {
    std::vector<float> result(ids.size() * 32);
    int32_t logical = 0;
    std::size_t images = 0;
    for (std::size_t i = 0; i < ids.size();) {
        const bool visual = ids[i] == image_token;
        const int32_t count = visual ? height * width : 1;
        if (visual && (images++ >= image_count || i + count > ids.size()))
            throw std::invalid_argument("D3 image token span does not match images");
        for (int32_t j = 0; j < count; ++j, ++i) {
            if (visual && ids[i] != image_token)
                throw std::invalid_argument("D3 image tokens must form a contiguous span");
            const int32_t coordinates[3] = {logical, logical + (visual ? j / width : 0),
                                            logical + (visual ? j % width : 0)};
            for (int32_t f = 0; f < 32; ++f) {
                const int32_t axis = f % 3 == 1 && f < 33 ? 1 : f % 3 == 2 && f < 30 ? 2 : 0;
                result[i * 32 + f] = coordinates[axis] *
                                     (1.0F / std::pow(10000000.0F, static_cast<float>(2 * f) / 64));
            }
        }
        logical += visual ? std::max(height, width) : 1;
    }
    if (images != image_count)
        throw std::invalid_argument("D3 images have no matching prompt span");
    return result;
}
} // namespace trtmc::d3
