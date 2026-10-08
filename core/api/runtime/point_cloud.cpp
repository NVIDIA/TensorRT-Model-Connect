/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/point_cloud.h"

#include "api_internal.h"
#include "trtmc/point_cloud.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::api {

namespace {

internal::PointsToSemanticSegmentationRequest
convert(const trtmc_points_to_semantic_segmentation_request_v1& input) {
    require(input.points != nullptr || input.num_points == 0,
            "point-cloud input points pointer is null");
    require(input.input_dim > 0, "point-cloud input_dim must be positive");
    const auto count = checked_size(input.num_points, input.input_dim);
    return {Span<const float>{input.points, count}, static_cast<std::uint32_t>(input.num_points),
            input.input_dim};
}

struct PointCloudSegmentationStorage final : ResultStorage {
    explicit PointCloudSegmentationStorage(internal::PointsToSemanticSegmentationResult value)
        : result(std::move(value)) {
        if (result.labels.size() != result.num_points)
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family point labels do not match the point count"};
        if (!result.class_names.empty() && result.class_names.size() != result.class_ids.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family point class names and class IDs differ"};
        if (!result.class_scores.empty() &&
            result.class_scores.size() !=
                static_cast<std::size_t>(result.num_points) * result.num_classes)
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family point class score count is invalid"};
        switch (result.score_kind) {
        case internal::ScoreKind::Logit:
        case internal::ScoreKind::Probability:
        case internal::ScoreKind::Unbounded:
            break;
        default:
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an unknown score kind"};
        }
        class_names.reserve(result.class_names.size());
        for (const auto& name : result.class_names)
            class_names.push_back(borrowed_string(name));
        vocabulary_id = borrowed_string(result.vocabulary_id);
    }

    internal::PointsToSemanticSegmentationResult result;
    std::vector<trtmc_string_view> class_names;
    trtmc_string_view vocabulary_id;
};

template <class Interface, class Request>
trtmc_status TRTMC_CALL run(trtmc_model* model, const Request* input,
                            const trtmc_config_view_v1* config, trtmc_result** out,
                            trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input != nullptr && out != nullptr, "point-cloud input or result output is null");
        const auto request = convert(*input);
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        validate_task_config(model_owner(model), internal::contract_key<Interface>(),
                             options.view());
        auto result = family.run(request, options.view());
        *out = make_result<PointCloudSegmentationStorage>(std::move(result));
    });
}

void fill_point_cloud_view(const PointCloudSegmentationStorage& storage,
                           trtmc_points_to_semantic_segmentation_view_v1* output) noexcept {
    *output = {
        storage.result.labels.data(),
        storage.result.labels.size(),
        storage.result.num_classes,
        {storage.result.class_ids.data(), storage.result.class_ids.size()},
        {storage.class_names.data(), storage.class_names.size()},
        storage.vocabulary_id,
        storage.result.ignore_label.has_value(),
        storage.result.ignore_label.value_or(0),
        storage.result.background_label.has_value(),
        storage.result.background_label.value_or(0),
        storage.result.class_scores.data(),
        storage.result.class_scores.size(),
        static_cast<std::uint32_t>(storage.result.score_kind),
    };
}

trtmc_status TRTMC_CALL result_view(const trtmc_result* result,
                                    trtmc_points_to_semantic_segmentation_view_v1* output,
                                    trtmc_error** error) noexcept {
    if (output)
        *output = {};
    return guarded(error, [&] {
        require(output != nullptr, "point-cloud result output is required");
        fill_point_cloud_view(require_result<PointCloudSegmentationStorage>(result), output);
    });
}

const trtmc_points_to_semantic_segmentation_api_v1 points_to_semantic_segmentation_api = {
    {1, 0, sizeof(trtmc_points_to_semantic_segmentation_api_v1)},
    run<internal::IPointsToSemanticSegmentation, trtmc_points_to_semantic_segmentation_request_v1>,
    result_view};
static_assert(offsetof(trtmc_points_to_semantic_segmentation_api_v1, header) == 0);

const TaskBinding bindings[] = {
    {internal::IPointsToSemanticSegmentation::kTask, 1, 0,
     &points_to_semantic_segmentation_api.header},
};

} // namespace

Span<const TaskBinding> point_cloud_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
