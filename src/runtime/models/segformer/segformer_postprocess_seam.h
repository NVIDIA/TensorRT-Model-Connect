#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace trtmc {

struct SegformerLogitsShape {
    int32_t num_classes{0};
    int32_t output_h{0};
    int32_t output_w{0};
};

enum class SegformerPostprocessStatus {
    kOk = 0,
    kInvalidShape = 1,
    kLogitsSizeMismatch = 2,
};

inline bool get_segformer_logits_expected_size(const SegformerLogitsShape& shape,
                                               std::size_t& expected_size) {
    expected_size = 0;
    if (shape.num_classes <= 0 || shape.output_h <= 0 || shape.output_w <= 0) {
        return false;
    }

    const std::size_t classes = static_cast<std::size_t>(shape.num_classes);
    const std::size_t output_h = static_cast<std::size_t>(shape.output_h);
    const std::size_t output_w = static_cast<std::size_t>(shape.output_w);
    constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();

    if (classes > (kMaxSize / output_h)) {
        return false;
    }
    const std::size_t classes_by_h = classes * output_h;
    if (classes_by_h > (kMaxSize / output_w)) {
        return false;
    }

    expected_size = classes_by_h * output_w;
    return true;
}

inline SegformerPostprocessStatus
compute_segformer_class_map_from_logits(const std::vector<float>& logits,
                                        const SegformerLogitsShape& shape,
                                        std::vector<int32_t>& class_map) {
    std::size_t expected_logits_size = 0;
    if (!get_segformer_logits_expected_size(shape, expected_logits_size)) {
        class_map.clear();
        return SegformerPostprocessStatus::kInvalidShape;
    }

    if (logits.size() != expected_logits_size) {
        class_map.clear();
        return SegformerPostprocessStatus::kLogitsSizeMismatch;
    }

    const int32_t num_classes = shape.num_classes;
    const int32_t output_h = shape.output_h;
    const int32_t output_w = shape.output_w;
    const std::size_t plane_size =
        static_cast<std::size_t>(output_h) * static_cast<std::size_t>(output_w);

    class_map.resize(plane_size);
    for (int32_t y = 0; y < output_h; ++y) {
        for (int32_t x = 0; x < output_w; ++x) {
            const std::size_t pixel_index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(output_w) +
                static_cast<std::size_t>(x);

            int32_t best_class = 0;
            float best_val = -1e30F;
            for (int32_t c = 0; c < num_classes; ++c) {
                const float val = logits[static_cast<std::size_t>(c) * plane_size + pixel_index];
                if (val > best_val) {
                    best_val = val;
                    best_class = c;
                }
            }

            class_map[pixel_index] = best_class;
        }
    }

    return SegformerPostprocessStatus::kOk;
}

} // namespace trtmc
