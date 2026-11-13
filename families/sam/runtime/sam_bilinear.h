/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Separable bilinear resampling for SAM's host-side image preprocess and mask
// postprocess.
//
// The per-sample definition (sample_mask_bilinear in sam_postprocess_seam.h)
// clamps a source coordinate to [0, len - 1], takes the floor tap and the
// clamped next tap, interpolates each of the two source rows horizontally,
// then blends the two rows vertically:
//
//   top    = v00 * (1 - wx) + v01 * wx
//   bottom = v10 * (1 - wx) + v11 * wx
//   value  = top * (1 - wy) + bottom * wy
//
// The taps and weights depend only on the destination column (or row), and
// `top`/`bottom` only on the source row and destination column. So each axis
// is computed once per resize, each source row is interpolated horizontally
// once per resize, and the vertical blend runs as a contiguous loop the
// compiler vectorizes. Every value is the same expression on the same
// operands as the per-sample definition, so the result is identical bit for
// bit.

#include "families/sam/runtime/sam_parallel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

// One axis of a bilinear resize: for each destination index, the two source
// taps and the interpolation weight of the upper tap.
struct SamBilinearAxis {
    std::vector<int32_t> lo;
    std::vector<int32_t> hi;
    std::vector<float> weight;         // coordinate - lo
    std::vector<float> inverse_weight; // 1 - weight

    [[nodiscard]] std::size_t size() const { return lo.size(); }
};

// The first `count` destination indices of an axis with `src_len` source
// samples. `source_coordinate(i)` returns destination index i's unclamped
// source coordinate, spelled exactly as the caller's per-sample loop spells it.
template <typename SourceCoordinate>
SamBilinearAxis make_sam_bilinear_axis(int32_t src_len, int32_t count,
                                       SourceCoordinate&& source_coordinate) {
    SamBilinearAxis axis;
    const auto n = static_cast<std::size_t>(std::max(count, 0));
    axis.lo.resize(n);
    axis.hi.resize(n);
    axis.weight.resize(n);
    axis.inverse_weight.resize(n);
    const float max_coordinate = static_cast<float>(src_len - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const float clamped =
            std::clamp(source_coordinate(static_cast<int32_t>(i)), 0.0F, max_coordinate);
        const auto lo = static_cast<int32_t>(std::floor(clamped));
        const float weight = clamped - static_cast<float>(lo);
        axis.lo[i] = lo;
        axis.hi[i] = std::min(lo + 1, src_len - 1);
        axis.weight[i] = weight;
        axis.inverse_weight[i] = 1.0F - weight;
    }
    return axis;
}

// Two horizontally interpolated source rows, reused across destination rows:
// a bilinear upsample reads each source row for several destination rows.
// `fill(source_row, dst)` writes one source row's horizontal pass into dst.
class SamBilinearRowCache {
  public:
    explicit SamBilinearRowCache(std::size_t row_values)
        : storage_(2 * row_values), slots_{storage_.data(), storage_.data() + row_values} {}
    SamBilinearRowCache(const SamBilinearRowCache&) = delete;
    SamBilinearRowCache& operator=(const SamBilinearRowCache&) = delete;

    // Source row `row`, computed unless cached; `keep` (the other row the
    // current destination row needs) is never evicted.
    template <typename Fill>
    const float* row(int32_t row, int32_t keep, Fill&& fill) {
        for (int slot = 0; slot < 2; ++slot) {
            if (rows_[slot] == row) {
                return slots_[slot];
            }
        }
        const int slot = rows_[0] == keep ? 1 : 0;
        fill(row, slots_[slot]);
        rows_[slot] = row;
        return slots_[slot];
    }

  private:
    std::vector<float> storage_;
    float* slots_[2];
    int32_t rows_[2]{-1, -1};
};

// The horizontal pass of one single-channel source row: the per-sample
// definition's `top` (or `bottom`) for every destination column. `dst` never
// overlaps `src_row`; saying so (__restrict) keeps the gather loop vectorized
// wherever this is inlined.
inline void sam_bilinear_row(const float* __restrict src_row, const SamBilinearAxis& x_axis,
                             float* __restrict dst) {
    const int32_t* lo = x_axis.lo.data();
    const int32_t* hi = x_axis.hi.data();
    const float* weight = x_axis.weight.data();
    const float* inverse_weight = x_axis.inverse_weight.data();
    const std::size_t width = x_axis.size();
    for (std::size_t x = 0; x < width; ++x) {
        dst[x] = src_row[lo[x]] * inverse_weight[x] + src_row[hi[x]] * weight[x];
    }
}

// Destination rows [y_begin, y_end) of a single-channel plane resize
// (`src_w` floats per source row), written at `dst` with `dst_stride` floats
// between rows (`dst` addresses destination row 0).
inline void sam_bilinear_resize_rows(const float* src, int32_t src_w, const SamBilinearAxis& x_axis,
                                     const SamBilinearAxis& y_axis, std::size_t y_begin,
                                     std::size_t y_end, float* dst, std::size_t dst_stride) {
    const std::size_t width = x_axis.size();
    SamBilinearRowCache cache(width);
    const auto fill = [&](int32_t row, float* out) {
        sam_bilinear_row(src + static_cast<std::size_t>(row) * static_cast<std::size_t>(src_w),
                         x_axis, out);
    };
    for (std::size_t y = y_begin; y < y_end; ++y) {
        const float* top = cache.row(y_axis.lo[y], y_axis.hi[y], fill);
        const float* bottom = cache.row(y_axis.hi[y], y_axis.lo[y], fill);
        const float weight = y_axis.weight[y];
        const float inverse_weight = y_axis.inverse_weight[y];
        float* out = dst + y * dst_stride;
        for (std::size_t x = 0; x < width; ++x) {
            out[x] = top[x] * inverse_weight + bottom[x] * weight;
        }
    }
}

// Resizes a single-channel plane to y_axis.size() rows of x_axis.size()
// values, written at `dst` with `dst_stride` floats between rows, on up to
// `workers` threads of at least `min_values` values each (sam_parallel.h).
inline void sam_bilinear_resize_plane(const float* src, int32_t src_w,
                                      const SamBilinearAxis& x_axis, const SamBilinearAxis& y_axis,
                                      float* dst, std::size_t dst_stride,
                                      int32_t workers = sam_host_worker_count(),
                                      int64_t min_values = kSamMinValuesPerWorker) {
    sam_parallel_for_rows(static_cast<int64_t>(y_axis.size()),
                          sam_min_rows_per_worker(static_cast<int64_t>(x_axis.size()), min_values),
                          workers, [&](int64_t begin, int64_t end) {
                              sam_bilinear_resize_rows(
                                  src, src_w, x_axis, y_axis, static_cast<std::size_t>(begin),
                                  static_cast<std::size_t>(end), dst, dst_stride);
                          });
}

} // namespace trtmc
