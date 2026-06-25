#pragma once

#include "runtime/models/sam/sam_types.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace trtmc {

inline SamResult select_sam_multimask_outputs(SamResult result, int32_t num_multimask_outputs) {
    if (num_multimask_outputs <= 0 || result.num_masks <= num_multimask_outputs ||
        result.mask_height <= 0 || result.mask_width <= 0) {
        return result;
    }

    const int32_t keep_count = std::min(num_multimask_outputs, result.num_masks);
    const int32_t start_mask = std::max(0, result.num_masks - keep_count);
    const std::size_t mask_area = static_cast<std::size_t>(result.mask_height) * result.mask_width;
    const std::size_t start_offset = static_cast<std::size_t>(start_mask) * mask_area;
    const std::size_t keep_values = static_cast<std::size_t>(keep_count) * mask_area;

    if (result.masks.size() >= start_offset + keep_values) {
        result.masks = std::vector<float>(
            result.masks.begin() + static_cast<std::ptrdiff_t>(start_offset),
            result.masks.begin() + static_cast<std::ptrdiff_t>(start_offset + keep_values));
    }
    if (result.iou_scores.size() >= static_cast<std::size_t>(start_mask + keep_count)) {
        result.iou_scores = std::vector<float>(result.iou_scores.begin() + start_mask,
                                               result.iou_scores.begin() + start_mask + keep_count);
    }
    result.num_masks = keep_count;
    return result;
}

} // namespace trtmc
