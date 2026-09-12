/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/perception.h"

#include "api_internal.h"
#include "trtmc/perception.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace trtmc::api {
namespace {

void valid_output(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
std::size_t image_area(std::uint32_t height, std::uint32_t width) {
    require(height && width, "spatial dimensions must be positive");
    checked_size(height, width);
    return static_cast<std::size_t>(height) * width;
}
std::size_t output_area(std::uint32_t height, std::uint32_t width) {
    try {
        return image_area(height, width);
    } catch (const ApiFailure&) {
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "invalid output spatial dimensions"};
    }
}
std::vector<trtmc_string_view> string_views(const std::vector<std::string>& strings) {
    std::vector<trtmc_string_view> views;
    views.reserve(strings.size());
    for (const auto& item : strings)
        views.push_back(borrowed_string(item));
    return views;
}
internal::PixelPoint point(trtmc_pixel_point_v1 p) {
    require(std::isfinite(p.x) && std::isfinite(p.y), "pixel coordinates must be finite");
    return {p.x, p.y};
}
internal::PixelBox box(trtmc_pixel_box_v1 b) {
    require(std::isfinite(b.x_min) && std::isfinite(b.y_min) && std::isfinite(b.x_max) &&
                std::isfinite(b.y_max) && b.x_min <= b.x_max && b.y_min <= b.y_max,
            "box must contain ordered finite XYXY coordinates");
    return {b.x_min, b.y_min, b.x_max, b.y_max};
}
trtmc_pixel_box_v1 box_view(const internal::PixelBox& b) {
    return {b.x_min, b.y_min, b.x_max, b.y_max};
}
trtmc_pixel_point_v1 point_view(const internal::PixelPoint& p) {
    return {p.x, p.y};
}
std::vector<internal::PointPrompt> points(const trtmc_point_prompt_v1* data, std::uint64_t count,
                                          bool required) {
    const auto input = checked_span(data, count);
    require(!required || !input.empty(), "point prompts are required");
    std::vector<internal::PointPrompt> result;
    result.reserve(input.size());
    for (const auto& value : input) {
        require(value.foreground <= 1, "point polarity must be zero or one");
        result.push_back({point(value.point), value.foreground != 0});
    }
    return result;
}
internal::PoseMatricesView poses(const trtmc_pose_matrices_v1& input, bool optional = false) {
    if (optional && input.count == 0) {
        require(input.value_count == 0, "empty poses have nonempty values");
        return {};
    }
    require(input.count > 0, "pose matrices are required");
    checked_size(input.count, 16);
    require(input.value_count == input.count * 16, "pose matrix count must match N,4,4");
    return {checked_span(input.values, input.value_count), input.count};
}

struct SemanticStorage final : ResultStorage {
    explicit SemanticStorage(internal::SemanticSegmentationResult result)
        : value(std::move(result)), names(string_views(value.class_names)) {
        const auto area = output_area(value.height, value.width);
        valid_output(value.labels.size() == area && !value.class_ids.empty() &&
                         !value.vocabulary_id.empty(),
                     "semantic labels or vocabulary are incomplete");
        valid_output(names.empty() || names.size() == value.class_ids.size(),
                     "semantic class names do not match class IDs");
        if (!value.class_scores.empty()) {
            const auto score_area = output_area(value.score_height, value.score_width);
            valid_output(value.class_ids.size() <= static_cast<std::size_t>(-1) / score_area &&
                             value.class_scores.size() == value.class_ids.size() * score_area,
                         "semantic score map shape mismatch");
            valid_output(value.score_kind == internal::ScoreKind::Logit ||
                             value.score_kind == internal::ScoreKind::Probability,
                         "semantic score maps must be logits or probabilities");
        } else
            valid_output(value.score_height == 0 && value.score_width == 0,
                         "empty score maps must have zero dimensions");
        view = {value.labels.data(),
                value.labels.size(),
                value.height,
                value.width,
                {value.class_ids.data(), value.class_ids.size()},
                {names.data(), names.size()},
                borrowed_string(value.vocabulary_id),
                value.ignore_label.has_value(),
                value.ignore_label.value_or(0),
                value.background_label.has_value(),
                value.background_label.value_or(0),
                value.class_scores.data(),
                value.class_scores.size(),
                value.score_height,
                value.score_width,
                static_cast<std::uint32_t>(value.score_kind)};
    }
    internal::SemanticSegmentationResult value;
    std::vector<trtmc_string_view> names;
    trtmc_semantic_segmentation_view_v1 view{};
};
struct MasksStorage final : ResultStorage {
    explicit MasksStorage(internal::MasksResult result) : value(std::move(result)) {
        const auto area = output_area(value.height, value.width);
        valid_output(value.count <= static_cast<std::uint64_t>(-1) / area &&
                         value.masks.size() == value.count * area,
                     "mask shape must be N,H,W");
        valid_output(value.kind == internal::MaskKind::Logits ||
                         value.kind == internal::MaskKind::Probability ||
                         value.kind == internal::MaskKind::Binary,
                     "unknown mask interpretation");
        const auto optional_count = [&](std::size_t count) {
            valid_output(count == 0 || count == value.count, "mask metadata count mismatch");
        };
        optional_count(value.predicted_iou.size());
        optional_count(value.confidence.size());
        optional_count(value.stability.size());
        optional_count(value.boxes.size());
        optional_count(value.object_ids.size());
        optional_count(value.proposals.size());
        for (const auto& b : value.boxes)
            boxes.push_back(box_view(b));
        seeds.resize(value.proposals.size());
        proposals.reserve(value.proposals.size());
        for (std::size_t i = 0; i < value.proposals.size(); ++i) {
            for (const auto& p : value.proposals[i].seed_points)
                seeds[i].push_back(point_view(p));
            proposals.push_back({value.proposals[i].area, box_view(value.proposals[i].crop_box),
                                 seeds[i].data(), seeds[i].size()});
        }
        if (!value.low_res_logits.empty()) {
            const auto low_area = output_area(value.low_res_height, value.low_res_width);
            valid_output(value.count <= static_cast<std::uint64_t>(-1) / low_area &&
                             value.low_res_logits.size() == value.count * low_area,
                         "low-resolution prior logits shape mismatch");
        } else
            valid_output(value.low_res_height == 0 && value.low_res_width == 0,
                         "empty prior logits require zero dimensions");
        view = {value.masks.data(),
                value.masks.size(),
                value.count,
                value.height,
                value.width,
                static_cast<std::uint32_t>(value.kind),
                value.predicted_iou.data(),
                value.predicted_iou.size(),
                value.confidence.data(),
                value.confidence.size(),
                value.stability.data(),
                value.stability.size(),
                boxes.data(),
                boxes.size(),
                {value.object_ids.data(), value.object_ids.size()},
                proposals.data(),
                proposals.size(),
                value.low_res_logits.data(),
                value.low_res_logits.size(),
                value.low_res_height,
                value.low_res_width};
    }
    internal::MasksResult value;
    std::vector<trtmc_pixel_box_v1> boxes;
    std::vector<std::vector<trtmc_pixel_point_v1>> seeds;
    std::vector<trtmc_mask_proposal_v1> proposals;
    trtmc_masks_view_v1 view{};
};
struct DisparityStorage final : ResultStorage {
    explicit DisparityStorage(internal::DisparityResult result)
        : value(std::move(result)), view{matrix_result_view(value.disparity)} {}
    internal::DisparityResult value;
    trtmc_disparity_view_v1 view;
};
struct GeometryStorage final : ResultStorage {
    explicit GeometryStorage(internal::MetricGeometryResult result) : value(std::move(result)) {
        const auto area = output_area(value.height, value.width);
        valid_output(area <= static_cast<std::size_t>(-1) / 3 && value.points.size() == area * 3 &&
                         value.depth.size() == area && value.valid.size() == area,
                     "metric geometry shape mismatch");
        view.points = value.points.data();
        view.point_value_count = value.points.size();
        view.depth = value.depth.data();
        view.valid = value.valid.data();
        view.pixel_count = area;
        view.height = value.height;
        view.width = value.width;
        std::copy(value.normalized_intrinsics.begin(), value.normalized_intrinsics.end(),
                  view.normalized_intrinsics);
    }
    internal::MetricGeometryResult value;
    trtmc_metric_geometry_view_v1 view{};
};
struct BoxesStorage final : ResultStorage {
    explicit BoxesStorage(internal::GroundedBoxesResult result) : value(std::move(result)) {
        for (const auto& b : value.boxes)
            boxes.push_back({box_view(b.box), borrowed_string(b.label), b.confidence.has_value(),
                             b.confidence.value_or(0)});
        view = {boxes.data(), boxes.size(), borrowed_string(value.raw_response),
                value.parse_complete};
    }
    internal::GroundedBoxesResult value;
    std::vector<trtmc_grounded_box_v1> boxes;
    trtmc_grounded_boxes_view_v1 view{};
};
struct DetectedBoxesStorage final : ResultStorage {
    explicit DetectedBoxesStorage(internal::DetectedBoxesResult result) {
        valid_output(result.image_height > 0 && result.image_width > 0,
                     "detection image dimensions must be positive");
        boxes.reserve(result.boxes.size());
        for (const auto& item : result.boxes) {
            valid_output(std::isfinite(item.x_min) && std::isfinite(item.y_min) &&
                             std::isfinite(item.x_max) && std::isfinite(item.y_max) &&
                             item.x_min <= item.x_max && item.y_min <= item.y_max &&
                             std::isfinite(item.score),
                         "detections require finite ordered XYXY coordinates and scores");
            boxes.push_back(
                {{item.x_min, item.y_min, item.x_max, item.y_max}, item.score, item.class_id});
        }
        view = {boxes.data(), boxes.size(), static_cast<std::uint32_t>(result.image_height),
                static_cast<std::uint32_t>(result.image_width)};
    }
    std::vector<trtmc_detected_box_v1> boxes;
    trtmc_detected_boxes_view_v1 view{};
};
struct BoxesBatchStorage final : ResultStorage {
    explicit BoxesBatchStorage(std::vector<internal::GroundedBoxesResult> results) {
        for (auto& result : results)
            items.push_back(std::make_unique<BoxesStorage>(std::move(result)));
    }
    std::vector<std::unique_ptr<BoxesStorage>> items;
};
trtmc_status TRTMC_CALL boxes_batch_run(trtmc_model* model,
                                        const trtmc_batch_image_text_to_boxes_request_v1* input,
                                        trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input && out, "boxes batch request or result is null");
        const auto supplied = checked_span(input->items, input->count);
        require(!supplied.empty(), "boxes batch requires requests");
        std::vector<internal::BatchImageTextToBoxesItem> items;
        std::vector<ConvertedConfig> configs;
        items.reserve(supplied.size());
        configs.reserve(supplied.size());
        for (const auto& item : supplied) {
            configs.emplace_back(&item.config);

            items.push_back({{image_input(item.input.image), string_view(item.input.text)},
                             configs.back().view()});
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IBatchImageTextToBoxes>(
            model, internal::IBatchImageTextToBoxes::kTask);
        validate_batch_configs(model_owner(model),
                               internal::contract_key<internal::IBatchImageTextToBoxes>(), configs);
        auto results = family.run_batch({{items.data(), items.size()}});
        valid_output(results.size() == items.size(), "family changed boxes batch item count");
        *out = make_result<BoxesBatchStorage>(std::move(results));
    });
}
trtmc_status TRTMC_CALL boxes_batch_count(const trtmc_result* input, uint64_t* out,
                                          trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out, "boxes batch count output is null");
        *out = require_result<BoxesBatchStorage>(input).items.size();
    });
}
trtmc_status TRTMC_CALL boxes_batch_item(const trtmc_result* input, uint64_t index,
                                         trtmc_grounded_boxes_view_v1* out,
                                         trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "boxes batch item output is null");
        const auto& items = require_result<BoxesBatchStorage>(input).items;
        require(index < items.size(), "boxes batch item index is out of range");
        *out = items[static_cast<size_t>(index)]->view;
    });
}
const trtmc_batch_image_text_to_boxes_api_v1 boxes_batch_api = {
    {1, 0, sizeof(trtmc_batch_image_text_to_boxes_api_v1)},
    boxes_batch_run,
    boxes_batch_count,
    boxes_batch_item};
struct PointsStorage final : ResultStorage {
    explicit PointsStorage(internal::GroundedPointsResult result) : value(std::move(result)) {
        for (const auto& p : value.points)
            points.push_back({point_view(p.point), borrowed_string(p.label),
                              p.confidence.has_value(), p.confidence.value_or(0)});
        view = {points.data(), points.size(), borrowed_string(value.raw_response),
                value.parse_complete};
    }
    internal::GroundedPointsResult value;
    std::vector<trtmc_grounded_point_v1> points;
    trtmc_grounded_points_view_v1 view{};
};
struct RefinedPosesStorage final : ResultStorage {
    explicit RefinedPosesStorage(internal::RefinedPosesResult result) : value(std::move(result)) {
        valid_output(value.num_hypotheses > 0 &&
                         value.refined_poses.size() ==
                             static_cast<std::size_t>(value.num_hypotheses) * 16 &&
                         ((value.num_hypotheses == 1 && value.scores.empty()) ||
                          value.scores.size() == static_cast<std::size_t>(value.num_hypotheses)) &&
                         value.best_index >= 0 && value.best_index < value.num_hypotheses,
                     "pose result shape/selection mismatch");
        view = {{value.refined_poses.data(), value.refined_poses.size(),
                 static_cast<std::uint64_t>(value.num_hypotheses)},
                value.scores.data(),
                value.scores.size(),
                value.best_index,
                value.all_poses_rigid,
                value.refinement_ms,
                value.scoring_ms};
    }
    internal::RefinedPosesResult value;
    trtmc_refined_poses_view_v1 view{};
};
struct ObjectPoseStorage final : ResultStorage {
    explicit ObjectPoseStorage(internal::ObjectPoseResult result) : value(std::move(result)) {
        std::copy(value.object_to_camera.begin(), value.object_to_camera.end(),
                  view.object_to_camera);
        view.has_score = value.score.has_value();
        view.score = value.score.value_or(0);
        view.rigid = value.rigid;
    }
    internal::ObjectPoseResult value;
    trtmc_object_pose_view_v1 view{};
};

template <class Store, class View>
trtmc_status TRTMC_CALL result_view(const trtmc_result* result, View* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "perception result view is null");
        *out = require_result<Store>(result).view;
    });
}
template <class Interface, class Store, class Request, class Invoke>
trtmc_status dispatch(trtmc_model* model, const Request* request,
                      const trtmc_config_view_v1* config, trtmc_result** out, trtmc_error** error,
                      Invoke invoke) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request && out, "perception request or result output is null");
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        const ConvertedConfig options(config);
        validate_task_config(model_owner(model), internal::contract_key<Interface>(),
                             options.view());
        auto result = invoke(family, *request, options.view());
        if constexpr (std::is_same_v<Interface, internal::IImagePointsToMasks> ||
                      std::is_same_v<Interface, internal::IImageBoxToMasks> ||
                      std::is_same_v<Interface, internal::IImageMaskToMasks>)
            valid_output(result.predicted_iou.size() == result.count,
                         "point/box/prior masks require one predicted IoU per mask");
        else if constexpr (std::is_same_v<Interface, internal::IImageTextToInstanceMasks> ||
                           std::is_same_v<Interface, internal::IImageBoxExemplarsToInstanceMasks>)
            valid_output(result.confidence.size() == result.count &&
                             result.boxes.size() == result.count,
                         "instance masks require one confidence and box per mask");
        else if constexpr (std::is_same_v<Interface, internal::IImageToMaskProposals>)
            valid_output(
                result.proposals.size() == result.count &&
                    (result.predicted_iou.size() == result.count ||
                     result.confidence.size() == result.count ||
                     result.stability.size() == result.count),
                "automatic mask proposals require metadata and at least one score per mask");
        *out = make_result<Store>(std::move(result));
    });
}

internal::PoseCropBatch provide_crops(const trtmc_pose_refinement_request_v1& provider,
                                      const internal::PoseCropRequest& requested) {
    trtmc_pose_crop_request_v1 input{
        {requested.poses.values.data(), requested.poses.values.size(), requested.poses.count},
        static_cast<std::uint32_t>(requested.stage),
        requested.iteration};
    valid_output(input.stage == TRTMC_POSE_CROP_REFINEMENT ||
                     input.stage == TRTMC_POSE_CROP_SCORING,
                 "family requested an unknown pose crop stage");
    (void)poses(input.poses);
    trtmc_pose_crop_response_v1 output{};
    const auto status = provider.provide_crops(provider.context, &input, &output);
    // An invalid owner without a deleter cannot transfer ownership to the API.
    require((output.owner == nullptr) == (output.release == nullptr),
            "crop owner and release must be supplied together");
    std::shared_ptr<void> lease;
    if (output.owner) {
        std::unique_ptr<void, decltype(output.release)> guard(output.owner, output.release);
        lease = std::shared_ptr<void>(std::move(guard));
    }
    if (status != TRTMC_OK) {
        if (status == TRTMC_END || status == TRTMC_AGAIN || status < TRTMC_INVALID_ARGUMENT ||
            status > TRTMC_BUSY)
            throw OwnedApiFailure{TRTMC_INTERNAL_ERROR,
                                  "crop callback returned an invalid control status"};
        const auto message = string_view(output.error_message);
        throw OwnedApiFailure{status, message.empty() ? std::string("crop provider failed")
                                                      : std::string(message)};
    }
    const auto area = image_area(output.height, output.width);
    require(output.hypothesis_count == requested.poses.count && output.channels == 6,
            "crop batch must match hypotheses and RGB+XYZ channels");
    checked_size(output.hypothesis_count, area);
    const auto pixels = output.hypothesis_count * area;
    checked_size(pixels, 6);
    const auto count = pixels * 6;
    require(output.rendered_count == count && output.observed_count == count,
            "crop arrays must match N,H,W,6");
    return {checked_span(output.rendered, count),
            checked_span(output.observed, count),
            output.hypothesis_count,
            output.height,
            output.width,
            output.channels,
            std::move(lease)};
}

internal::TriangleMeshView mesh(const trtmc_triangle_mesh_v1& input) {
    const auto vertices = matrix_input(input.vertices);
    require(vertices.columns == 3, "mesh vertices must have XYZ columns");
    const auto triangles = checked_span(input.triangles, input.triangle_index_count);
    require(!triangles.empty() && triangles.size() % 3 == 0,
            "mesh requires triangle index triples");
    for (const auto index : triangles)
        require(index < vertices.rows, "triangle index is outside vertex array");
    const auto normals = matrix_input(input.vertex_normals, true);
    require(normals.values.empty() || (normals.rows == vertices.rows && normals.columns == 3),
            "vertex normal shape mismatch");
    internal::TriangleMeshView result;
    result.vertices = vertices;
    result.triangles = triangles;
    result.vertex_normals = normals;
    result.symmetries = poses(input.symmetries, true);
    if (input.appearance == TRTMC_MESH_NO_APPEARANCE) {
        require(input.vertex_rgb.count == 0 && input.uv.count == 0 && input.texture.byte_size == 0,
                "unselected mesh appearance inputs are not accepted");
    } else if (input.appearance == TRTMC_MESH_VERTEX_RGB) {
        result.appearance = internal::MeshAppearance::VertexRgb;
        result.vertex_rgb = matrix_input(input.vertex_rgb);
        require(result.vertex_rgb.rows == vertices.rows && result.vertex_rgb.columns == 3,
                "vertex color shape mismatch");
        require(input.uv.count == 0 && input.texture.byte_size == 0,
                "vertex-color mesh cannot also supply a texture");
    } else if (input.appearance == TRTMC_MESH_TEXTURE_UV) {
        result.appearance = internal::MeshAppearance::TextureUv;
        result.uv = matrix_input(input.uv);
        require(result.uv.rows == vertices.rows && result.uv.columns == 2,
                "mesh UV shape mismatch");
        require(input.vertex_rgb.count == 0, "textured mesh cannot also supply vertex colors");
        result.texture = image_input(input.texture);
        require(result.texture.channels == 3, "mesh texture must be RGB");
    } else
        throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unknown mesh appearance kind"};
    return result;
}

#define TRTMC_PERCEPTION_RUN(Function, Interface, Store, Request, Body)                            \
    trtmc_status TRTMC_CALL Function(trtmc_model* model, const Request* request,                   \
                                     const trtmc_config_view_v1* config, trtmc_result** out,       \
                                     trtmc_error** error) noexcept {                               \
        return dispatch<internal::Interface, Store>(model, request, config, out, error, [          \
        ](internal::Interface & family, const Request& in, internal::ConfigView options) Body);    \
    }
TRTMC_PERCEPTION_RUN(semantic, IImageToSemanticSegmentation, SemanticStorage,
                     trtmc_perception_image_request_v1,
                     { return family.run({image_input(in.image)}, options); })
TRTMC_PERCEPTION_RUN(point_masks, IImagePointsToMasks, MasksStorage, trtmc_image_points_request_v1,
                     {
                         auto supplied = points(in.points, in.point_count, true);
                         return family.run(
                             {image_input(in.image), {supplied.data(), supplied.size()}}, options);
                     })
TRTMC_PERCEPTION_RUN(box_masks, IImageBoxToMasks, MasksStorage, trtmc_image_box_request_v1,
                     { return family.run({image_input(in.image), box(in.box)}, options); })
TRTMC_PERCEPTION_RUN(prior_masks, IImageMaskToMasks, MasksStorage,
                     trtmc_image_prior_mask_request_v1, {
                         auto supplied = points(in.points, in.point_count, false);
                         require(in.has_box <= 1, "has_box must be zero or one");
                         std::optional<internal::PixelBox> supplied_box;
                         if (in.has_box)
                             supplied_box = box(in.box);
                         return family.run({image_input(in.image),
                                            {matrix_input(in.prior_logits)},
                                            {supplied.data(), supplied.size()},
                                            supplied_box},
                                           options);
                     })
TRTMC_PERCEPTION_RUN(proposals, IImageToMaskProposals, MasksStorage,
                     trtmc_perception_image_request_v1,
                     { return family.run({image_input(in.image)}, options); })
TRTMC_PERCEPTION_RUN(text_masks, IImageTextToInstanceMasks, MasksStorage,
                     trtmc_image_query_request_v1,
                     { return family.run({image_input(in.image), string_view(in.text)}, options); })
TRTMC_PERCEPTION_RUN(exemplar_masks, IImageBoxExemplarsToInstanceMasks, MasksStorage,
                     trtmc_image_exemplars_request_v1, {
                         const auto input = checked_span(in.exemplars, in.exemplar_count);
                         require(!input.empty(), "box exemplars are required");
                         std::vector<internal::BoxExemplar> exemplars;
                         exemplars.reserve(input.size());
                         for (const auto& value : input) {
                             require(value.positive <= 1, "exemplar polarity must be zero or one");
                             exemplars.push_back({box(value.box), value.positive != 0});
                         }
                         return family.run({image_input(in.image),
                                            {exemplars.data(), exemplars.size()},
                                            string_view(in.text_hint)},
                                           options);
                     })
TRTMC_PERCEPTION_RUN(stereo, IStereoImagesToDisparity, DisparityStorage,
                     trtmc_stereo_images_request_v1, {
                         auto left = image_input(in.left);
                         auto right = image_input(in.right);
                         require(left.height == right.height && left.width == right.width,
                                 "rectified stereo image dimensions must match");
                         return family.run({left, right}, options);
                     })
TRTMC_PERCEPTION_RUN(geometry, IImageToMetricGeometry, GeometryStorage,
                     trtmc_perception_image_request_v1,
                     { return family.run({image_input(in.image)}, options); })
TRTMC_PERCEPTION_RUN(detect_boxes, IImageToBoxes, DetectedBoxesStorage,
                     trtmc_perception_image_request_v1, {
                         const auto image = image_input(in.image);
                         auto result = family.run({image}, options);
                         valid_output(
                             result.image_height > 0 && result.image_width > 0 &&
                                 static_cast<std::uint32_t>(result.image_height) == image.height &&
                                 static_cast<std::uint32_t>(result.image_width) == image.width,
                             "detection dimensions must match the original image");
                         return result;
                     })
TRTMC_PERCEPTION_RUN(ground_boxes, IImageTextToBoxes, BoxesStorage, trtmc_image_query_request_v1,
                     { return family.run({image_input(in.image), string_view(in.text)}, options); })
TRTMC_PERCEPTION_RUN(ground_points, IImageTextToPoints, PointsStorage, trtmc_image_query_request_v1,
                     { return family.run({image_input(in.image), string_view(in.text)}, options); })
TRTMC_PERCEPTION_RUN(rgbd_pose, IRgbdMeshMaskToObjectPose, ObjectPoseStorage,
                     trtmc_rgbd_mesh_mask_request_v1, {
                         const auto rgb = image_input(in.rgb);
                         const auto depth = matrix_input(in.depth_meters);
                         require(rgb.channels == 3 && depth.rows == rgb.height &&
                                     depth.columns == rgb.width,
                                 "RGB-D inputs must be aligned RGB and HW depth");
                         auto mask = checked_span(in.object_mask, in.mask_count);
                         require(mask.size() == depth.values.size(), "object mask shape mismatch");
                         for (auto value : mask)
                             require(value <= 1, "object mask must be binary");
                         auto request = (internal::RgbdMeshMaskToObjectPoseRequest{
                             rgb, depth, {mask, rgb.height, rgb.width}, {}, mesh(in.mesh)});
                         request.pixel_intrinsics = pixel_intrinsics_input(in.pixel_intrinsics);
                         return family.run(request, options);
                     })
#undef TRTMC_PERCEPTION_RUN

trtmc_status TRTMC_CALL refine_pose(trtmc_model* model,
                                    const trtmc_pose_refinement_request_v1* request,
                                    const trtmc_config_view_v1* config, trtmc_result** out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request && out && request->provide_crops,
                "pose request, callback and result output are required");
        const auto input = pose_refinement_input(*request);
        const ConvertedConfig options(config);

        internal::IPoseHypothesesCropsToRefinedPoses* family;
        std::unique_ptr<ModelSession> session;
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            family = &require_interface<internal::IPoseHypothesesCropsToRefinedPoses>(
                model, internal::IPoseHypothesesCropsToRefinedPoses::kTask);
            validate_task_config(
                model_owner(model),
                internal::contract_key<internal::IPoseHypothesesCropsToRefinedPoses>(),
                options.view());
            session = std::make_unique<ModelSession>(model);
        }
        *out = make_result<RefinedPosesStorage>(family->run(input, options.view()));
    });
}

#define TRTMC_PERCEPTION_TABLE(Name, Api, Function, Store, View)                                   \
    const Api Name{{1, 0, sizeof(Api)}, Function, result_view<Store, View>};
TRTMC_PERCEPTION_TABLE(semantic_api, trtmc_image_to_semantic_segmentation_api_v1, semantic,
                       SemanticStorage, trtmc_semantic_segmentation_view_v1)
TRTMC_PERCEPTION_TABLE(points_api, trtmc_image_points_to_masks_api_v1, point_masks, MasksStorage,
                       trtmc_masks_view_v1)
TRTMC_PERCEPTION_TABLE(box_api, trtmc_image_box_to_masks_api_v1, box_masks, MasksStorage,
                       trtmc_masks_view_v1)
TRTMC_PERCEPTION_TABLE(prior_api, trtmc_image_mask_to_masks_api_v1, prior_masks, MasksStorage,
                       trtmc_masks_view_v1)
TRTMC_PERCEPTION_TABLE(proposals_api, trtmc_image_to_mask_proposals_api_v1, proposals, MasksStorage,
                       trtmc_masks_view_v1)
TRTMC_PERCEPTION_TABLE(text_masks_api, trtmc_image_text_to_instance_masks_api_v1, text_masks,
                       MasksStorage, trtmc_masks_view_v1)
TRTMC_PERCEPTION_TABLE(exemplars_api, trtmc_image_box_exemplars_to_instance_masks_api_v1,
                       exemplar_masks, MasksStorage, trtmc_masks_view_v1)
TRTMC_PERCEPTION_TABLE(stereo_api, trtmc_stereo_images_to_disparity_api_v1, stereo,
                       DisparityStorage, trtmc_disparity_view_v1)
TRTMC_PERCEPTION_TABLE(geometry_api, trtmc_image_to_metric_geometry_api_v1, geometry,
                       GeometryStorage, trtmc_metric_geometry_view_v1)
TRTMC_PERCEPTION_TABLE(detected_boxes_api, trtmc_image_to_boxes_api_v1, detect_boxes,
                       DetectedBoxesStorage, trtmc_detected_boxes_view_v1)
TRTMC_PERCEPTION_TABLE(boxes_api, trtmc_image_text_to_boxes_api_v1, ground_boxes, BoxesStorage,
                       trtmc_grounded_boxes_view_v1)
TRTMC_PERCEPTION_TABLE(ground_points_api, trtmc_image_text_to_points_api_v1, ground_points,
                       PointsStorage, trtmc_grounded_points_view_v1)
TRTMC_PERCEPTION_TABLE(refine_api, trtmc_pose_hypotheses_crops_to_refined_poses_api_v1, refine_pose,
                       RefinedPosesStorage, trtmc_refined_poses_view_v1)
TRTMC_PERCEPTION_TABLE(rgbd_api, trtmc_rgbd_mesh_mask_to_object_pose_api_v1, rgbd_pose,
                       ObjectPoseStorage, trtmc_object_pose_view_v1)
#undef TRTMC_PERCEPTION_TABLE

} // namespace
internal::PixelBox pixel_box_input(trtmc_pixel_box_v1 input) {
    return box(input);
}
std::vector<internal::PointPrompt> point_prompts_input(const trtmc_point_prompt_v1* data,
                                                       std::uint64_t count, bool required) {
    return points(data, count, required);
}
internal::PoseMatricesView pose_matrices_input(const trtmc_pose_matrices_v1& input, bool optional) {
    return poses(input, optional);
}
internal::TriangleMeshView triangle_mesh_input(const trtmc_triangle_mesh_v1& input) {
    return mesh(input);
}
internal::PoseCropBatch pose_crops_input(void* context, trtmc_pose_crop_callback_v1 callback,
                                         const internal::PoseCropRequest& requested) {
    require(callback, "pose crop callback is required");
    trtmc_pose_refinement_request_v1 provider{};
    provider.context = context;
    provider.provide_crops = callback;
    return provide_crops(provider, requested);
}
trtmc_result* make_masks_result(internal::MasksResult result) {
    return make_result<MasksStorage>(std::move(result));
}
internal::PoseHypothesesCropsToRefinedPosesRequest
pose_refinement_input(const trtmc_pose_refinement_request_v1& request) {
    require(request.provide_crops, "pose crop callback is required");
    require(std::isfinite(request.mesh_diameter_meters) && request.mesh_diameter_meters > 0,
            "mesh diameter must be positive meters");
    return {pose_matrices_input(request.candidates), request.mesh_diameter_meters,
            [context = request.context,
             callback = request.provide_crops](const internal::PoseCropRequest& query) {
                return pose_crops_input(context, callback, query);
            }};
}
std::array<float, 9> pixel_intrinsics_input(const float* values) {
    require(values, "pixel intrinsics are null");
    std::array<float, 9> result;
    std::copy(values, values + 9, result.begin());
    for (const auto value : result)
        require(std::isfinite(value), "pixel camera intrinsics must be finite");
    require(result[0] > 0 && result[4] > 0, "pixel focal lengths must be positive");
    return result;
}
trtmc_result* make_refined_poses_result(internal::RefinedPosesResult result) {
    return make_result<RefinedPosesStorage>(std::move(result));
}
trtmc_result* make_object_pose_result(internal::ObjectPoseResult result) {
    return make_result<ObjectPoseStorage>(std::move(result));
}
trtmc_status TRTMC_CALL masks_result_view(const trtmc_result* result, trtmc_masks_view_v1* out,
                                          trtmc_error** error) noexcept {
    return result_view<MasksStorage>(result, out, error);
}
trtmc_status TRTMC_CALL refined_poses_result_view(const trtmc_result* result,
                                                  trtmc_refined_poses_view_v1* out,
                                                  trtmc_error** error) noexcept {
    return result_view<RefinedPosesStorage>(result, out, error);
}
trtmc_status TRTMC_CALL object_pose_result_view(const trtmc_result* result,
                                                trtmc_object_pose_view_v1* out,
                                                trtmc_error** error) noexcept {
    return result_view<ObjectPoseStorage>(result, out, error);
}
Span<const TaskBinding> perception_task_bindings() noexcept {
#define B(Interface, Table)                                                                        \
    {                                                                                              \
        internal::Interface::kTask, 1, 0, &Table.header                                            \
    }
    static const TaskBinding bindings[] = {B(IBatchImageTextToBoxes, boxes_batch_api),
                                           B(IImageToSemanticSegmentation, semantic_api),
                                           B(IImagePointsToMasks, points_api),
                                           B(IImageBoxToMasks, box_api),
                                           B(IImageMaskToMasks, prior_api),
                                           B(IImageToMaskProposals, proposals_api),
                                           B(IImageTextToInstanceMasks, text_masks_api),
                                           B(IImageBoxExemplarsToInstanceMasks, exemplars_api),
                                           B(IStereoImagesToDisparity, stereo_api),
                                           B(IImageToMetricGeometry, geometry_api),
                                           B(IImageToBoxes, detected_boxes_api),
                                           B(IImageTextToBoxes, boxes_api),
                                           B(IImageTextToPoints, ground_points_api),
                                           B(IPoseHypothesesCropsToRefinedPoses, refine_api),
                                           B(IRgbdMeshMaskToObjectPose, rgbd_api)};
#undef B
    return {bindings};
}
} // namespace trtmc::api
