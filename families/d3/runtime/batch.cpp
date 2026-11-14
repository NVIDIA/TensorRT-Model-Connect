/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/d3/runtime/batch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
namespace trtmc::d3 {
PackedRows pack_rows(const std::vector<BatchRow>& rows, std::size_t begin, std::size_t end) {
    if (begin >= end || end > rows.size())
        throw std::invalid_argument("invalid D3 batch row range");
    PackedRows out;
    out.batch = static_cast<int64_t>(end - begin);
    out.sequence = 0;
    for (auto i = begin; i < end; ++i) {
        if (rows[i].ids.empty() || rows[i].angles.size() != rows[i].ids.size() * 32)
            throw std::invalid_argument("invalid D3 row token/rotary geometry");
        out.lengths.push_back(static_cast<int64_t>(rows[i].ids.size()));
        out.sequence = std::max(out.sequence, out.lengths.back());
    }
    out.ids.assign(out.batch * out.sequence, 0);
    out.angles.assign(out.batch * out.sequence * 32, 0);
    out.mask.assign(out.batch * out.sequence, -std::numeric_limits<float>::infinity());
    for (auto i = begin; i < end; ++i) {
        const auto offset = (i - begin) * out.sequence;
        std::copy(rows[i].ids.begin(), rows[i].ids.end(), out.ids.begin() + offset);
        std::copy(rows[i].angles.begin(), rows[i].angles.end(), out.angles.begin() + offset * 32);
        std::fill_n(out.mask.begin() + offset, rows[i].ids.size(), 0);
    }
    return out;
}
VisualInputs pack_visual(const std::vector<VisualInputs>& images, std::size_t begin,
                         std::size_t end) {
    if (begin >= end || end > images.size())
        throw std::invalid_argument("invalid D3 visual batch range");
    const auto patches = images[begin].count;
    if (patches < 1)
        throw std::invalid_argument("empty D3 image patches");
    VisualInputs out;
    out.count = patches * static_cast<int64_t>(end - begin);
    out.indices.resize(4 * out.count);
    out.factors.resize(4 * out.count);
    for (auto i = begin; i < end; ++i) {
        const auto& image = images[i];
        if (image.count != patches ||
            image.indices.size() != static_cast<std::size_t>(4 * patches) ||
            image.factors.size() != image.indices.size())
            throw std::invalid_argument("D3 batch images must have the same patch grid");
        out.patches.insert(out.patches.end(), image.patches.begin(), image.patches.end());
        out.angles.insert(out.angles.end(), image.angles.begin(), image.angles.end());
        for (int row = 0; row < 4; ++row) {
            const auto source = row * patches;
            const auto target = row * out.count + (i - begin) * patches;
            std::copy_n(image.indices.begin() + source, patches, out.indices.begin() + target);
            std::copy_n(image.factors.begin() + source, patches, out.factors.begin() + target);
        }
    }
    return out;
}
} // namespace trtmc::d3
