/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_bbox_postprocess.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::uint16_t bfloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

struct Outputs {
    static constexpr std::array<int, 3> kSizes{128, 64, 32};

    Outputs() {
        for (std::size_t level = 0; level < kSizes.size(); ++level) {
            const auto area = static_cast<std::size_t>(kSizes[level] * kSizes[level]);
            cls[level].assign(2U * area, bfloat16(-20.0F));
            reg[level].assign(4U * area, bfloat16(0.0F));
        }
    }

    void candidate(int stride, int y, int x, int label, float logit,
                   const std::array<float, 4>& ltrb) {
        const std::size_t level = stride == 8 ? 0U : stride == 16 ? 1U : 2U;
        const auto size = static_cast<std::size_t>(kSizes[level]);
        const auto area = size * size;
        const auto anchor = static_cast<std::size_t>(y) * size + static_cast<std::size_t>(x);
        cls[level][static_cast<std::size_t>(label) * area + anchor] = bfloat16(logit);
        for (std::size_t coordinate = 0; coordinate < ltrb.size(); ++coordinate)
            reg[level][coordinate * area + anchor] = bfloat16(ltrb[coordinate]);
    }

    trtmc::Sam2BBoxRawOutputs views() const {
        return {view(cls[0], 2, 128), view(cls[1], 2, 64), view(cls[2], 2, 32),
                view(reg[0], 4, 128), view(reg[1], 4, 64), view(reg[2], 4, 32)};
    }

    std::array<std::vector<std::uint16_t>, 3> cls;
    std::array<std::vector<std::uint16_t>, 3> reg;

  private:
    static trtmc::Sam2BBoxTensorView view(const std::vector<std::uint16_t>& data,
                                          std::int64_t channels, std::int64_t size) {
        return {data.data(), {1, channels, size, size}, data.size()};
    }
};

template <typename Mutate>
void check_rejected(Mutate mutate, const char* message) {
    Outputs outputs;
    auto views = outputs.views();
    mutate(outputs, views);
    try {
        (void)trtmc::decode_sam2_bbox_outputs(views);
    } catch (const trtmc::Sam2BBoxPostprocessError&) {
        return;
    }
    check(false, message);
}

void test_captured_bfloat16_box() {
    Outputs outputs;
    outputs.candidate(16, 42, 37, 1, -0.33984375F, {1.703125F, 2.40625F, 3.15625F, 1.6953125F});
    const auto detections = trtmc::decode_sam2_bbox_outputs(outputs.views());
    const auto& detection = trtmc::require_exactly_one_sam2_bbox_detection(detections);

    check(detection.label == 1, "captured label drifted");
    check(detection.score == 0.416015625F, "captured BF16 score drifted");
    check(detection.model_xyxy_1024 == std::array<float, 4>{572.0F, 640.0F, 652.0F, 708.0F},
          "captured model box drifted");
    check(detection.original_xyxy == std::array<float, 4>{607.75F, 800.0F, 692.75F, 885.0F},
          "captured original-space box drifted");
}

void test_selection_and_exact_one_policy() {
    Outputs outputs;
    outputs.candidate(8, 0, 0, 0, 5.0F, {0.5F, 0.5F, 1.0F, 0.75F});
    outputs.candidate(8, 0, 1, 1, 4.0F, {0.5F, 0.5F, 1.0F, 0.75F});
    outputs.candidate(8, 0, 2, 1, 3.0F, {1.5625F, 0.5F, -0.0625F, 0.75F});
    const auto detections = trtmc::decode_sam2_bbox_outputs(outputs.views());
    check(detections.size() == 2 && detections[0].label == 0 &&
              detections[0].flattened_anchor_index == 0 && detections[1].label == 1 &&
              detections[1].flattened_anchor_index == 1,
          "class-agnostic NMS survivor drifted");

    bool rejected = false;
    try {
        (void)trtmc::require_exactly_one_sam2_bbox_detection(detections);
    } catch (const trtmc::Sam2BBoxPostprocessError&) {
        rejected = true;
    }
    check(rejected, "multiple detections must not acquire an implicit top-one policy");
}

void test_rejects_invalid_tensor_contract() {
    check_rejected([](auto&, auto& views) { views.bbox_cls_stride_8.shape[0] = 2; },
                   "bbox output shape drift must fail closed");
    check_rejected([](auto&, auto& views) { --views.bbox_cls_stride_8.element_count; },
                   "truncated bbox output must fail closed");
    check_rejected([](auto&, auto& views) { views.bbox_cls_stride_8.data = nullptr; },
                   "null bbox output must fail closed");
    check_rejected([](auto& outputs, auto&) { outputs.cls[0][0] = 0x7FC0U; },
                   "non-finite bbox output must fail closed");
    check_rejected([](auto& outputs, auto&) { outputs.reg[2][0] = 0x7F7FU; },
                   "finite bbox distance overflow must fail closed");
}

} // namespace

int main() {
    static_assert(trtmc::kSam2BBoxPreNmsTopK == 100);
    static_assert(trtmc::kSam2BBoxNmsIouThreshold == 0.2F);
    test_captured_bfloat16_box();
    test_selection_and_exact_one_policy();
    test_rejects_invalid_tensor_contract();
    return 0;
}
