/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace trtmc {

inline float quantize_sam_fractional_point(float fraction, int32_t original_size) {
    if (original_size <= 0)
        return fraction;
    return std::floor(fraction * static_cast<float>(original_size)) /
           static_cast<float>(original_size);
}

inline std::vector<float> encode_sam_point_embedding(float x, float y, bool is_foreground,
                                                     int32_t image_size,
                                                     int32_t decoder_hidden_size,
                                                     const std::vector<float>& shared_image_pe,
                                                     const std::vector<float>& point_embed_fg,
                                                     const std::vector<float>& point_embed_bg) {
    const int32_t dim = decoder_hidden_size;
    const int32_t num_pos_feats = dim / 2;

    const float nx = (x + 0.5F) / static_cast<float>(image_size);
    const float ny = (y + 0.5F) / static_cast<float>(image_size);

    const float cx = 2.0F * nx - 1.0F;
    const float cy = 2.0F * ny - 1.0F;

    std::vector<float> sparse(static_cast<std::size_t>(dim), 0.0F);
    if (static_cast<int32_t>(shared_image_pe.size()) >= 2 * num_pos_feats) {
        for (int32_t i = 0; i < num_pos_feats; ++i) {
            const float b = cx * shared_image_pe[static_cast<std::size_t>(i)] +
                            cy * shared_image_pe[static_cast<std::size_t>(num_pos_feats + i)];
            const float angle = 2.0F * 3.14159265358979F * b;
            sparse[static_cast<std::size_t>(i)] = std::sin(angle);
            sparse[static_cast<std::size_t>(num_pos_feats + i)] = std::cos(angle);
        }
    }

    const auto& point_embed = is_foreground ? point_embed_fg : point_embed_bg;
    if (static_cast<int32_t>(point_embed.size()) >= dim) {
        for (int32_t i = 0; i < dim; ++i) {
            sparse[static_cast<std::size_t>(i)] += point_embed[static_cast<std::size_t>(i)];
        }
    }

    return sparse;
}

inline std::vector<float> build_sam_point_sparse_prompt(
    float point_x, float point_y, bool is_foreground, int32_t rescaled_w, int32_t rescaled_h,
    int32_t image_size, int32_t decoder_hidden_size, const std::vector<float>& shared_image_pe,
    const std::vector<float>& point_embed_fg, const std::vector<float>& point_embed_bg,
    const std::vector<float>& not_a_point_embed) {
    const int32_t dim = decoder_hidden_size;
    const float px = point_x * static_cast<float>(rescaled_w);
    const float py = point_y * static_cast<float>(rescaled_h);
    const auto point_emb =
        encode_sam_point_embedding(px, py, is_foreground, image_size, decoder_hidden_size,
                                   shared_image_pe, point_embed_fg, point_embed_bg);

    std::vector<float> pad_emb(static_cast<std::size_t>(dim), 0.0F);
    if (static_cast<int32_t>(not_a_point_embed.size()) >= dim) {
        for (int32_t i = 0; i < dim; ++i) {
            pad_emb[static_cast<std::size_t>(i)] = not_a_point_embed[static_cast<std::size_t>(i)];
        }
    }

    std::vector<float> sparse(static_cast<std::size_t>(2) * static_cast<std::size_t>(dim));
    std::memcpy(sparse.data(), point_emb.data(), static_cast<std::size_t>(dim) * sizeof(float));
    std::memcpy(sparse.data() + static_cast<std::size_t>(dim), pad_emb.data(),
                static_cast<std::size_t>(dim) * sizeof(float));
    return sparse;
}

} // namespace trtmc
