/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SAM2-HOI-CPP-01
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SAM2-HOI-01
// Intent:         SAM2+HOI CPU postprocess ordering, thresholds, NMS, and pair formation
// Preconditions:  Fixed-shape detector outputs and deterministic synthetic boxes
// Postconditions: Reference operation order and strict boundary behavior are preserved
// =============================================================================

#include "runtime/models/sam2_hoi/hoi_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using trtmc::sam2_hoi::HoiInteractionPair;
using trtmc::sam2_hoi::HoiPostprocessResult;
using trtmc::sam2_hoi::HoiPostprocessStatus;

constexpr float kBoxScale = 1024.0F;

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, const char* label, float tolerance = 1.0e-5F) {
    check(std::abs(actual - expected) <= tolerance, label);
}

struct Fixture {
    std::vector<float> class_probabilities =
        std::vector<float>(trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiClassCount, 0.0F);
    std::vector<float> boxes = std::vector<float>(trtmc::sam2_hoi::kHoiQueryCount * 4, 0.0F);
    std::vector<float> embeddings = std::vector<float>(
        trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiEmbeddingSize, 0.0F);
};

void set_box(Fixture& fixture, std::size_t query_index, const std::array<float, 4>& box_xyxy) {
    const std::size_t offset = query_index * 4;
    fixture.boxes[offset] = (box_xyxy[0] + box_xyxy[2]) * 0.5F / kBoxScale;
    fixture.boxes[offset + 1] = (box_xyxy[1] + box_xyxy[3]) * 0.5F / kBoxScale;
    fixture.boxes[offset + 2] = (box_xyxy[2] - box_xyxy[0]) / kBoxScale;
    fixture.boxes[offset + 3] = (box_xyxy[3] - box_xyxy[1]) / kBoxScale;
}

void set_candidate(Fixture& fixture, std::size_t query_index, int32_t label, float score,
                   const std::array<float, 4>& box_xyxy) {
    fixture.class_probabilities[query_index * trtmc::sam2_hoi::kHoiClassCount +
                                static_cast<std::size_t>(label)] = score;
    set_box(fixture, query_index, box_xyxy);
}

void set_embedding_tag(Fixture& fixture, std::size_t query_index, float tag) {
    const std::size_t begin = query_index * trtmc::sam2_hoi::kHoiEmbeddingSize;
    std::fill(fixture.embeddings.begin() + static_cast<std::ptrdiff_t>(begin),
              fixture.embeddings.begin() +
                  static_cast<std::ptrdiff_t>(begin + trtmc::sam2_hoi::kHoiEmbeddingSize),
              tag);
}

void check_pair(const HoiInteractionPair& pair, int32_t source, int32_t target, const char* label) {
    check(pair.source_detection_index == source && pair.target_detection_index == target, label);
}

void test_reference_sequence_uses_pre_dedup_hands_and_callback_embeddings() {
    Fixture fixture;
    set_candidate(fixture, 0, 0, 0.90F, {0.0F, 0.0F, 100.0F, 100.0F});
    set_candidate(fixture, 1, 1, 0.85F, {50.0F, 0.0F, 150.0F, 100.0F});
    set_candidate(fixture, 2, 2, 0.80F, {149.0F, 0.0F, 249.0F, 100.0F});
    set_candidate(fixture, 3, 2, 0.70F, {400.0F, 400.0F, 450.0F, 450.0F});
    set_candidate(fixture, 4, 3, 0.75F, {149.0F, 50.0F, 249.0F, 150.0F});
    set_embedding_tag(fixture, 0, 10.0F);
    set_embedding_tag(fixture, 2, 30.0F);
    set_embedding_tag(fixture, 4, 50.0F);

    bool callback_called = false;
    HoiPostprocessResult result;
    const auto status = trtmc::sam2_hoi::postprocess_hoi(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings,
        [&](const std::vector<float>& pair_embeddings, std::size_t pair_count) {
            callback_called = true;
            check(pair_count == 2, "reference: callback receives two candidate pairs");
            check(pair_embeddings.size() == 2 * 2 * trtmc::sam2_hoi::kHoiEmbeddingSize,
                  "reference: callback pair-feature shape");
            if (pair_embeddings.size() == 2 * 2 * trtmc::sam2_hoi::kHoiEmbeddingSize) {
                check_close(pair_embeddings[0], 10.0F, "reference: first pair source embedding");
                check_close(pair_embeddings[trtmc::sam2_hoi::kHoiEmbeddingSize], 30.0F,
                            "reference: first pair target embedding");
                check_close(pair_embeddings[2 * trtmc::sam2_hoi::kHoiEmbeddingSize], 30.0F,
                            "reference: second pair source embedding");
                check_close(pair_embeddings[3 * trtmc::sam2_hoi::kHoiEmbeddingSize], 50.0F,
                            "reference: second pair target embedding");
            }
            return std::vector<float>{0.5F, 0.5001F};
        },
        result);

    check(status == HoiPostprocessStatus::kOk, "reference: status ok");
    check(callback_called, "reference: callback invoked");
    check(result.detections.size() == 3, "reference: three filtered detections");
    if (result.detections.size() == 3) {
        check(result.detections[0].label == 0 && result.detections[0].query_index == 0,
              "reference: surviving hand is first");
        check(result.detections[1].label == 2 && result.detections[1].query_index == 2,
              "reference: object overlapping only removed hand survives");
        check(result.detections[2].label == 3 && result.detections[2].query_index == 4,
              "reference: second object survives");
        check_close(result.detections[0].box_xyxy[0], 0.0F, "reference: scaled box x1");
        check_close(result.detections[0].box_xyxy[2], 100.0F, "reference: scaled box x2");
    }
    check(result.interaction_pairs.size() == 1,
          "reference: strict interaction threshold keeps one pair");
    if (result.interaction_pairs.size() == 1) {
        check_pair(result.interaction_pairs[0], 1, 2,
                   "reference: retained label-2 to label-3 pair");
    }
    check(result.interaction_probabilities.size() == 1,
          "reference: one retained interaction probability");
    if (result.interaction_probabilities.size() == 1) {
        check_close(result.interaction_probabilities[0], 0.5001F,
                    "reference: retained interaction probability value");
    }
}

void test_nms_threshold_boundaries_and_cartesian_pair_order() {
    Fixture fixture;
    set_candidate(fixture, 0, 0, 0.95F, {49.0F, 0.0F, 51.0F, 100.0F});
    set_candidate(fixture, 1, 2, 0.90F, {0.0F, 0.0F, 100.0F, 100.0F});
    set_candidate(fixture, 2, 2, 0.80F, {0.0F, 0.0F, 50.0F, 100.0F});
    set_candidate(fixture, 3, 3, 0.70F, {0.0F, 0.0F, 75.0F, 100.0F});
    set_candidate(fixture, 4, 3, 0.35F, {50.0F, 0.0F, 60.0F, 100.0F});

    HoiPostprocessResult result;
    const auto status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings,
        {0.60F, 0.70F, 0.80F, 0.90F}, result);

    check(status == HoiPostprocessStatus::kOk, "boundaries: status ok");
    check(result.detections.size() == 4, "boundaries: IoU equal to 0.5 and 0.75 is retained");
    if (result.detections.size() == 4) {
        check(result.detections[0].label == 0 && result.detections[1].label == 2 &&
                  result.detections[2].label == 2 && result.detections[3].label == 3,
              "boundaries: score order is preserved");
    }
    check(result.interaction_pairs.size() == 4, "boundaries: all four Cartesian pairs survive");
    if (result.interaction_pairs.size() == 4) {
        check_pair(result.interaction_pairs[0], 0, 1, "boundaries: hand to first label-2 pair");
        check_pair(result.interaction_pairs[1], 0, 2, "boundaries: hand to second label-2 pair");
        check_pair(result.interaction_pairs[2], 1, 3, "boundaries: first label-2 to label-3 pair");
        check_pair(result.interaction_pairs[3], 2, 3, "boundaries: second label-2 to label-3 pair");
    }
    check(std::none_of(result.detections.begin(), result.detections.end(),
                       [](const auto& detection) { return detection.query_index == 4; }),
          "boundaries: score exactly 0.35 is rejected");
}

void test_combined_nms_intersection_and_hand_nms_order() {
    Fixture fixture;
    set_candidate(fixture, 0, 0, 0.90F, {0.0F, 0.0F, 100.0F, 100.0F});
    set_candidate(fixture, 1, 0, 0.80F, {20.0F, 0.0F, 120.0F, 100.0F});
    set_candidate(fixture, 2, 1, 0.85F, {10.0F, 0.0F, 110.0F, 100.0F});
    set_candidate(fixture, 3, 1, 0.70F, {60.0F, 0.0F, 160.0F, 100.0F});
    set_candidate(fixture, 5, 0, 0.60F, {100.0F, 0.0F, 200.0F, 100.0F});

    HoiPostprocessResult result;
    const auto status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings, {}, result);

    check(status == HoiPostprocessStatus::kOk, "combined NMS: status ok");
    check(result.detections.size() == 2, "combined NMS: two hands survive");
    if (result.detections.size() == 2) {
        check(result.detections[0].query_index == 0,
              "combined NMS: highest-score hand remains first");
        check(result.detections[1].query_index == 3,
              "combined NMS: hand at exact 0.25 IoU survives");
    }
    check(result.interaction_pairs.empty(), "combined NMS: no object pairs");
}

void test_bfloat16_box_arithmetic_and_batched_nms_coordinate_rounding() {
    Fixture box_fixture;
    box_fixture.class_probabilities[0] = 0.90F;
    box_fixture.boxes[0] = 0.5277F;
    box_fixture.boxes[1] = 0.8289F;
    box_fixture.boxes[2] = 0.0997F;
    box_fixture.boxes[3] = 0.0588F;

    HoiPostprocessResult box_result;
    auto status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        box_fixture.class_probabilities, box_fixture.boxes, box_fixture.embeddings, {}, box_result);
    check(status == HoiPostprocessStatus::kOk, "BF16 boxes: status ok");
    check(box_result.detections.size() == 1, "BF16 boxes: one hand survives");
    if (box_result.detections.size() == 1) {
        const auto& box = box_result.detections[0].box_xyxy;
        check_close(box[0], 488.0F, "BF16 boxes: x1 rounds after every source operation");
        check_close(box[1], 816.0F, "BF16 boxes: y1 rounds after every source operation");
        check_close(box[2], 592.0F, "BF16 boxes: x2 rounds after every source operation");
        check_close(box[3], 880.0F, "BF16 boxes: y2 rounds after every source operation");
    }

    Fixture nms_fixture;
    set_candidate(nms_fixture, 0, 0, 0.90F, {100.0F, 300.0F, 120.0F, 320.0F});
    set_candidate(nms_fixture, 1, 2, 0.80F, {31.0F, 276.0F, 312.0F, 512.0F});
    set_candidate(nms_fixture, 2, 2, 0.70F, {108.0F, 256.0F, 390.0F, 496.0F});

    HoiPostprocessResult nms_result;
    status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        nms_fixture.class_probabilities, nms_fixture.boxes, nms_fixture.embeddings, {0.60F, 0.70F},
        nms_result);
    check(status == HoiPostprocessStatus::kOk, "BF16 batched NMS: status ok");
    check(nms_result.detections.size() == 3,
          "BF16 batched NMS: coordinate rounding retains IoU exactly 0.5");
    check(nms_result.interaction_pairs.size() == 2,
          "BF16 batched NMS: both source hand-object pairs survive");
}

void test_stable_flattened_topk_excludes_rank_301() {
    Fixture fixture;
    for (std::size_t query_index = 0; query_index < 300; ++query_index) {
        set_candidate(fixture, query_index, 0, 0.40F, {0.0F, 0.0F, 10.0F, 10.0F});
    }
    set_candidate(fixture, 300, 2, 0.40F, {9.0F, 0.0F, 19.0F, 10.0F});

    HoiPostprocessResult result;
    const auto status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings, {}, result);

    check(status == HoiPostprocessStatus::kOk, "top-k: status ok");
    check(result.detections.size() == 1, "top-k: duplicate hands collapse to one");
    if (result.detections.size() == 1) {
        check(result.detections[0].query_index == 0,
              "top-k: stable tie keeps the lowest flattened index");
    }
    check(result.interaction_pairs.empty(), "top-k: rank-301 object is excluded before NMS");
}

void test_stable_class_tie_prefers_lower_flattened_index() {
    Fixture fixture;
    set_candidate(fixture, 0, 0, 0.90F, {0.0F, 0.0F, 100.0F, 100.0F});
    fixture.class_probabilities[1] = 0.90F;
    set_candidate(fixture, 1, 2, 0.80F, {99.0F, 0.0F, 199.0F, 100.0F});
    set_candidate(fixture, 2, 2, 0.35F, {98.0F, 0.0F, 198.0F, 100.0F});

    HoiPostprocessResult result;
    const auto status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings, {0.60F}, result);

    check(status == HoiPostprocessStatus::kOk, "stable class tie: status ok");
    check(result.detections.size() == 2, "stable class tie: hand and object survive");
    if (result.detections.size() == 2) {
        check(result.detections[0].label == 0,
              "stable class tie: lower flattened class index wins global NMS");
        check(result.detections[1].query_index == 1,
              "stable class tie: score-threshold object is retained");
    }
    check(result.interaction_pairs.size() == 1, "stable class tie: one hand-object pair is formed");
}

void test_no_hand_returns_all_empty_without_callback() {
    Fixture fixture;
    set_candidate(fixture, 0, 2, 0.90F, {0.0F, 0.0F, 100.0F, 100.0F});
    set_candidate(fixture, 1, 3, 0.80F, {50.0F, 0.0F, 150.0F, 100.0F});

    bool callback_called = false;
    HoiPostprocessResult result;
    const auto status = trtmc::sam2_hoi::postprocess_hoi(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings,
        [&](const std::vector<float>&, std::size_t) {
            callback_called = true;
            return std::vector<float>{};
        },
        result);

    check(status == HoiPostprocessStatus::kOk, "no hand: status ok");
    check(!callback_called, "no hand: callback is not invoked");
    check(result.detections.empty() && result.interaction_pairs.empty() &&
              result.interaction_probabilities.empty(),
          "no hand: all result fields are empty");
}

void test_invalid_inputs_and_interaction_contract_clear_output() {
    Fixture fixture;
    set_candidate(fixture, 0, 0, 0.90F, {0.0F, 0.0F, 100.0F, 100.0F});
    set_candidate(fixture, 1, 2, 0.80F, {99.0F, 0.0F, 199.0F, 100.0F});

    HoiPostprocessResult result;
    result.detections.push_back({});
    const trtmc::sam2_hoi::HoiInteractionProbabilityCallback missing_callback;
    auto status = trtmc::sam2_hoi::postprocess_hoi(fixture.class_probabilities, fixture.boxes,
                                                   fixture.embeddings, missing_callback, result);
    check(status == HoiPostprocessStatus::kMissingInteractionEvaluator,
          "invalid: missing callback rejected");
    check(result.detections.empty(), "invalid: missing callback clears output");

    status = trtmc::sam2_hoi::postprocess_hoi(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings,
        [](const std::vector<float>&, std::size_t) { return std::vector<float>{}; }, result);
    check(status == HoiPostprocessStatus::kInteractionProbabilityCountMismatch,
          "invalid: callback probability count rejected");
    check(result.detections.empty(), "invalid: callback mismatch leaves output empty");

    std::vector<float> short_probabilities = fixture.class_probabilities;
    short_probabilities.pop_back();
    status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        short_probabilities, fixture.boxes, fixture.embeddings, {0.60F}, result);
    check(status == HoiPostprocessStatus::kInputSizeMismatch,
          "invalid: class-probability shape rejected");
    check(result.detections.empty(), "invalid: input mismatch leaves output empty");

    fixture.class_probabilities[0] = std::numeric_limits<float>::quiet_NaN();
    status = trtmc::sam2_hoi::postprocess_hoi_with_probabilities(
        fixture.class_probabilities, fixture.boxes, fixture.embeddings, {0.60F}, result);
    check(status == HoiPostprocessStatus::kNonFiniteValue,
          "invalid: non-finite detector value rejected");
}

} // namespace

int main() {
    test_reference_sequence_uses_pre_dedup_hands_and_callback_embeddings();
    test_nms_threshold_boundaries_and_cartesian_pair_order();
    test_combined_nms_intersection_and_hand_nms_order();
    test_bfloat16_box_arithmetic_and_batched_nms_coordinate_rounding();
    test_stable_flattened_topk_excludes_rank_301();
    test_stable_class_tie_prefers_lower_flattened_index();
    test_no_hand_returns_all_empty_without_callback();
    test_invalid_inputs_and_interaction_contract_clear_output();

    if (failures != 0) {
        std::cerr << failures << " SAM2+HOI postprocess test(s) failed" << '\n';
        return 1;
    }
    std::cout << "SAM2+HOI postprocess tests passed" << '\n';
    return 0;
}
