/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/core.hpp"
#include "trtmc/point_cloud.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace trtmc {

struct PointsToSemanticSegmentationRequest {
    Span<const float> points; // Row-major [num_points, input_dim] float32.
    std::uint32_t num_points{0};
    std::uint32_t input_dim{3};
};

using PointsToSemanticSegmentationResult =
    detail::ViewResult<trtmc_points_to_semantic_segmentation_view_v1>;

namespace detail {

struct PointsToSemanticSegmentationWireRequest {
    trtmc_points_to_semantic_segmentation_request_v1 wire;
};

inline auto point_cloud_request(const PointsToSemanticSegmentationRequest& input) {
    return PointsToSemanticSegmentationWireRequest{
        {input.points.data(), static_cast<std::uint64_t>(input.points.size()), input.input_dim}};
}

template <class Result, class Table, class Request>
Result point_cloud_call(const std::shared_ptr<ModelState>& state, const Table* table,
                        const Request& input, const Config& config) {
    auto request = point_cloud_request(input);
    auto entries = config.c_entries();
    auto options = entries.view();
    trtmc_result* raw = nullptr;
    trtmc_error* error = nullptr;
    const auto status = table->run(state->handle, &request.wire, &options, &raw, &error);
    ResultOwner owner(state, raw);
    check(state->api, status, error);
    return Result(std::move(owner), table->result_view);
}

} // namespace detail

class PointsToSemanticSegmentation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_POINTS_TO_SEMANTIC_SEGMENTATION;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    PointsToSemanticSegmentationResult run(const PointsToSemanticSegmentationRequest& input,
                                           const Config& config = {}) const {
        return detail::point_cloud_call<PointsToSemanticSegmentationResult>(state_, api_, input,
                                                                            config);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_points_to_semantic_segmentation_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible point-cloud Task table");
    }

  private:
    friend class Model;
    PointsToSemanticSegmentation(std::shared_ptr<detail::ModelState> state,
                                 const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_points_to_semantic_segmentation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_points_to_semantic_segmentation_api_v1* api_;
};

} // namespace trtmc
