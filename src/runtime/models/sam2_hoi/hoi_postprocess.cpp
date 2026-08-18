/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/hoi_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <utility>

namespace trtmc::sam2_hoi {
namespace {

constexpr float kBoxScale = 1024.0F;
constexpr float kScoreThreshold = 0.35F;
constexpr float kClassAwareNmsThreshold = 0.5F;
constexpr float kGlobalNmsThreshold = 0.75F;
constexpr float kHandNmsThreshold = 0.25F;
constexpr float kObjectHandOverlapThreshold = 1.0e-6F;
constexpr float kInteractionThreshold = 0.5F;

struct Candidate {
    HoiDetection detection;
};

struct PreparedResult {
    HoiPostprocessResult result;
    std::vector<HoiInteractionPair> candidate_pairs;
    std::vector<float> pair_embeddings;
};

bool all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

float round_to_bfloat16(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding_bias = 0x00007FFFU + ((bits >> 16U) & 1U);
    bits = (bits + rounding_bias) & 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float bfloat16_add(float left, float right) {
    return round_to_bfloat16(round_to_bfloat16(left) + round_to_bfloat16(right));
}

float bfloat16_subtract(float left, float right) {
    return round_to_bfloat16(round_to_bfloat16(left) - round_to_bfloat16(right));
}

float bfloat16_multiply(float left, float right) {
    return round_to_bfloat16(round_to_bfloat16(left) * round_to_bfloat16(right));
}

HoiPostprocessStatus validate_inputs(const std::vector<float>& class_probabilities,
                                     const std::vector<float>& normalized_boxes_cxcywh,
                                     const std::vector<float>& query_embeddings) {
    if (class_probabilities.size() != kHoiQueryCount * kHoiClassCount ||
        normalized_boxes_cxcywh.size() != kHoiQueryCount * 4 ||
        query_embeddings.size() != kHoiQueryCount * kHoiEmbeddingSize) {
        return HoiPostprocessStatus::kInputSizeMismatch;
    }
    if (!all_finite(class_probabilities) || !all_finite(normalized_boxes_cxcywh) ||
        !all_finite(query_embeddings)) {
        return HoiPostprocessStatus::kNonFiniteValue;
    }
    return HoiPostprocessStatus::kOk;
}

std::array<float, 4> cxcywh_to_scaled_xyxy(const std::vector<float>& boxes,
                                           std::size_t query_index) {
    const std::size_t offset = query_index * 4;
    const float cx = round_to_bfloat16(boxes[offset]);
    const float cy = round_to_bfloat16(boxes[offset + 1]);
    const float half_width = bfloat16_multiply(boxes[offset + 2], 0.5F);
    const float half_height = bfloat16_multiply(boxes[offset + 3], 0.5F);
    return {
        bfloat16_multiply(kBoxScale, bfloat16_subtract(cx, half_width)),
        bfloat16_multiply(kBoxScale, bfloat16_subtract(cy, half_height)),
        bfloat16_multiply(kBoxScale, bfloat16_add(cx, half_width)),
        bfloat16_multiply(kBoxScale, bfloat16_add(cy, half_height)),
    };
}

float box_iou(const std::array<float, 4>& left, const std::array<float, 4>& right) {
    const float intersection_width =
        std::max(0.0F, std::min(left[2], right[2]) - std::max(left[0], right[0]));
    const float intersection_height =
        std::max(0.0F, std::min(left[3], right[3]) - std::max(left[1], right[1]));
    const float intersection = intersection_width * intersection_height;
    const float left_area = std::max(0.0F, left[2] - left[0]) * std::max(0.0F, left[3] - left[1]);
    const float right_area =
        std::max(0.0F, right[2] - right[0]) * std::max(0.0F, right[3] - right[1]);
    const float union_area = left_area + right_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<std::array<float, 4>>
make_class_aware_nms_boxes(const std::vector<Candidate>& candidates) {
    std::vector<std::array<float, 4>> boxes;
    if (candidates.empty())
        return boxes;

    float max_coordinate = candidates.front().detection.box_xyxy.front();
    for (const auto& candidate : candidates) {
        for (const float coordinate : candidate.detection.box_xyxy)
            max_coordinate = std::max(max_coordinate, coordinate);
    }

    const float offset_scale = bfloat16_add(max_coordinate, 1.0F);
    boxes.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const float offset =
            bfloat16_multiply(static_cast<float>(candidate.detection.label), offset_scale);
        auto box = candidate.detection.box_xyxy;
        for (float& coordinate : box)
            coordinate = bfloat16_add(coordinate, offset);
        boxes.push_back(box);
    }
    return boxes;
}

const std::array<float, 4>& nms_box(const std::vector<Candidate>& candidates,
                                    const std::vector<std::array<float, 4>>& class_aware_boxes,
                                    std::size_t index, bool class_aware) {
    if (class_aware)
        return class_aware_boxes[index];
    return candidates[index].detection.box_xyxy;
}

bool is_suppressed_by_kept_candidate(const std::vector<Candidate>& candidates,
                                     const std::vector<std::array<float, 4>>& class_aware_boxes,
                                     const std::vector<std::size_t>& keep,
                                     std::size_t candidate_index, float iou_threshold,
                                     bool class_aware) {
    const auto& candidate_box =
        nms_box(candidates, class_aware_boxes, candidate_index, class_aware);
    for (const std::size_t kept_index : keep) {
        const auto& kept_box = nms_box(candidates, class_aware_boxes, kept_index, class_aware);
        if (box_iou(kept_box, candidate_box) > iou_threshold)
            return true;
    }
    return false;
}

std::vector<std::size_t> nms_keep_indices(const std::vector<Candidate>& candidates,
                                          float iou_threshold, bool class_aware) {
    const auto boxes_for_nms =
        class_aware ? make_class_aware_nms_boxes(candidates) : std::vector<std::array<float, 4>>{};

    std::vector<std::size_t> keep;
    keep.reserve(candidates.size());
    for (std::size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
        if (!is_suppressed_by_kept_candidate(candidates, boxes_for_nms, keep, candidate_index,
                                             iou_threshold, class_aware)) {
            keep.push_back(candidate_index);
        }
    }
    return keep;
}

std::vector<Candidate>
make_thresholded_topk_candidates(const std::vector<float>& class_probabilities,
                                 const std::vector<float>& normalized_boxes_cxcywh) {
    std::vector<std::size_t> flattened_indices(class_probabilities.size());
    std::iota(flattened_indices.begin(), flattened_indices.end(), std::size_t{0});
    std::stable_sort(flattened_indices.begin(), flattened_indices.end(),
                     [&class_probabilities](std::size_t left, std::size_t right) {
                         return class_probabilities[left] > class_probabilities[right];
                     });
    flattened_indices.resize(kHoiTopK);

    std::vector<Candidate> candidates;
    candidates.reserve(kHoiTopK);
    for (const std::size_t flattened_index : flattened_indices) {
        const float score = round_to_bfloat16(class_probabilities[flattened_index]);
        if (!(score > kScoreThreshold)) {
            continue;
        }
        const std::size_t query_index = flattened_index / kHoiClassCount;
        Candidate candidate;
        candidate.detection.box_xyxy = cxcywh_to_scaled_xyxy(normalized_boxes_cxcywh, query_index);
        candidate.detection.score = score;
        candidate.detection.label = static_cast<int32_t>(flattened_index % kHoiClassCount);
        candidate.detection.query_index = static_cast<int32_t>(query_index);
        candidates.push_back(candidate);
    }
    return candidates;
}

std::vector<Candidate> apply_detector_nms(const std::vector<Candidate>& candidates) {
    const std::vector<std::size_t> class_aware_keep =
        nms_keep_indices(candidates, kClassAwareNmsThreshold, true);
    const std::vector<std::size_t> global_keep =
        nms_keep_indices(candidates, kGlobalNmsThreshold, false);

    std::vector<bool> kept_by_global(candidates.size(), false);
    for (const std::size_t index : global_keep) {
        kept_by_global[index] = true;
    }

    std::vector<Candidate> kept;
    kept.reserve(class_aware_keep.size());
    for (const std::size_t index : class_aware_keep) {
        if (kept_by_global[index]) {
            kept.push_back(candidates[index]);
        }
    }
    return kept;
}

std::vector<Candidate> apply_hand_and_object_filter(const std::vector<Candidate>& candidates,
                                                    bool& had_hand) {
    std::vector<std::size_t> hand_indices;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (candidates[index].detection.label <= 1) {
            hand_indices.push_back(index);
        }
    }
    had_hand = !hand_indices.empty();
    if (!had_hand) {
        return {};
    }

    std::vector<Candidate> hands_before_deduplication;
    hands_before_deduplication.reserve(hand_indices.size());
    for (const std::size_t index : hand_indices) {
        hands_before_deduplication.push_back(candidates[index]);
    }
    const std::vector<std::size_t> kept_hand_local_indices =
        nms_keep_indices(hands_before_deduplication, kHandNmsThreshold, false);

    std::vector<bool> keep_hand(candidates.size(), false);
    for (const std::size_t local_index : kept_hand_local_indices) {
        keep_hand[hand_indices[local_index]] = true;
    }

    std::vector<Candidate> kept;
    kept.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const Candidate& candidate = candidates[index];
        if (candidate.detection.label <= 1) {
            if (keep_hand[index]) {
                kept.push_back(candidate);
            }
            continue;
        }

        const bool overlaps_pre_deduplication_hand =
            std::any_of(hands_before_deduplication.begin(), hands_before_deduplication.end(),
                        [&candidate](const Candidate& hand) {
                            return box_iou(candidate.detection.box_xyxy, hand.detection.box_xyxy) >=
                                   kObjectHandOverlapThreshold;
                        });
        if (overlaps_pre_deduplication_hand) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

void append_cartesian_pairs(const std::vector<int32_t>& sources,
                            const std::vector<int32_t>& targets,
                            std::vector<HoiInteractionPair>& pairs) {
    for (const int32_t source : sources) {
        for (const int32_t target : targets) {
            pairs.push_back({source, target});
        }
    }
}

PreparedResult prepare_result(const std::vector<float>& class_probabilities,
                              const std::vector<float>& normalized_boxes_cxcywh,
                              const std::vector<float>& query_embeddings,
                              bool build_pair_embeddings) {
    PreparedResult prepared;
    const std::vector<Candidate> thresholded_candidates =
        make_thresholded_topk_candidates(class_probabilities, normalized_boxes_cxcywh);
    const std::vector<Candidate> detector_nms_candidates =
        apply_detector_nms(thresholded_candidates);
    bool had_hand = false;
    const std::vector<Candidate> filtered_candidates =
        apply_hand_and_object_filter(detector_nms_candidates, had_hand);
    if (!had_hand) {
        return prepared;
    }

    prepared.result.detections.reserve(filtered_candidates.size());
    std::vector<int32_t> hand_indices;
    std::vector<int32_t> first_object_indices;
    std::vector<int32_t> second_object_indices;
    for (std::size_t index = 0; index < filtered_candidates.size(); ++index) {
        const HoiDetection& detection = filtered_candidates[index].detection;
        prepared.result.detections.push_back(detection);
        const int32_t output_index = static_cast<int32_t>(index);
        if (detection.label <= 1) {
            hand_indices.push_back(output_index);
        } else if (detection.label == 2) {
            first_object_indices.push_back(output_index);
        } else if (detection.label == 3) {
            second_object_indices.push_back(output_index);
        }
    }

    append_cartesian_pairs(hand_indices, first_object_indices, prepared.candidate_pairs);
    append_cartesian_pairs(first_object_indices, second_object_indices, prepared.candidate_pairs);

    if (!build_pair_embeddings) {
        return prepared;
    }
    prepared.pair_embeddings.reserve(prepared.candidate_pairs.size() * 2 * kHoiEmbeddingSize);
    for (const HoiInteractionPair& pair : prepared.candidate_pairs) {
        const auto append_detection_embedding = [&](int32_t detection_index) {
            const std::size_t query_index = static_cast<std::size_t>(
                prepared.result.detections[static_cast<std::size_t>(detection_index)].query_index);
            const auto begin = query_embeddings.begin() +
                               static_cast<std::ptrdiff_t>(query_index * kHoiEmbeddingSize);
            prepared.pair_embeddings.insert(prepared.pair_embeddings.end(), begin,
                                            begin + static_cast<std::ptrdiff_t>(kHoiEmbeddingSize));
        };
        append_detection_embedding(pair.source_detection_index);
        append_detection_embedding(pair.target_detection_index);
    }
    return prepared;
}

HoiPostprocessStatus finish_result(PreparedResult&& prepared,
                                   const std::vector<float>& interaction_probabilities,
                                   HoiPostprocessResult& result) {
    if (interaction_probabilities.size() != prepared.candidate_pairs.size()) {
        return HoiPostprocessStatus::kInteractionProbabilityCountMismatch;
    }
    if (!all_finite(interaction_probabilities)) {
        return HoiPostprocessStatus::kNonFiniteValue;
    }

    for (std::size_t index = 0; index < prepared.candidate_pairs.size(); ++index) {
        const float probability = interaction_probabilities[index];
        if (probability > kInteractionThreshold) {
            prepared.result.interaction_pairs.push_back(prepared.candidate_pairs[index]);
            prepared.result.interaction_probabilities.push_back(probability);
        }
    }
    result = std::move(prepared.result);
    return HoiPostprocessStatus::kOk;
}

} // namespace

HoiPostprocessStatus
postprocess_hoi(const std::vector<float>& class_probabilities,
                const std::vector<float>& normalized_boxes_cxcywh,
                const std::vector<float>& query_embeddings,
                const HoiInteractionProbabilityCallback& interaction_probability_callback,
                HoiPostprocessResult& result) {
    result = {};
    const HoiPostprocessStatus input_status =
        validate_inputs(class_probabilities, normalized_boxes_cxcywh, query_embeddings);
    if (input_status != HoiPostprocessStatus::kOk) {
        return input_status;
    }

    PreparedResult prepared =
        prepare_result(class_probabilities, normalized_boxes_cxcywh, query_embeddings, true);
    if (prepared.candidate_pairs.empty()) {
        result = std::move(prepared.result);
        return HoiPostprocessStatus::kOk;
    }
    if (!interaction_probability_callback) {
        return HoiPostprocessStatus::kMissingInteractionEvaluator;
    }
    const std::vector<float> probabilities =
        interaction_probability_callback(prepared.pair_embeddings, prepared.candidate_pairs.size());
    return finish_result(std::move(prepared), probabilities, result);
}

HoiPostprocessStatus postprocess_hoi_with_probabilities(
    const std::vector<float>& class_probabilities,
    const std::vector<float>& normalized_boxes_cxcywh, const std::vector<float>& query_embeddings,
    const std::vector<float>& interaction_probabilities, HoiPostprocessResult& result) {
    result = {};
    const HoiPostprocessStatus input_status =
        validate_inputs(class_probabilities, normalized_boxes_cxcywh, query_embeddings);
    if (input_status != HoiPostprocessStatus::kOk) {
        return input_status;
    }

    PreparedResult prepared =
        prepare_result(class_probabilities, normalized_boxes_cxcywh, query_embeddings, false);
    return finish_result(std::move(prepared), interaction_probabilities, result);
}

} // namespace trtmc::sam2_hoi
