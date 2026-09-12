/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/core.hpp"
#include "trtmc/image.hpp"
#include "trtmc/matrix.hpp"
#include "trtmc/perception.h"

#include <algorithm>
#include <array>
#include <exception>
#include <functional>

namespace trtmc {
using PixelPoint = trtmc_pixel_point_v1;
using PixelBox = trtmc_pixel_box_v1;
struct PointPrompt {
    PixelPoint point;
    bool foreground{true};
};
struct BoxExemplar {
    PixelBox box;
    bool positive{true};
};
struct ImageToSemanticSegmentationRequest {
    ImageInput image;
};
struct ImagePointsToMasksRequest {
    ImageInput image;
    Span<const PointPrompt> points;
};
struct ImageBoxToMasksRequest {
    ImageInput image;
    PixelBox box;
};
struct ImageMaskToMasksRequest {
    ImageInput image;
    FloatMatrixView prior_logits;
    Span<const PointPrompt> points;
    std::optional<PixelBox> box;
};
struct ImageToMaskProposalsRequest {
    ImageInput image;
};
struct ImageTextToInstanceMasksRequest {
    ImageInput image;
    std::string text;
};
struct ImageBoxExemplarsToInstanceMasksRequest {
    ImageInput image;
    Span<const BoxExemplar> exemplars;
    std::string text_hint;
};
struct StereoImagesToDisparityRequest {
    ImageInput left, right;
};
struct ImageToMetricGeometryRequest {
    ImageInput image;
};
struct ImageToBoxesRequest {
    ImageInput image;
};
struct ImageTextToBoxesRequest {
    ImageInput image;
    std::string text;
};
struct ImageTextToPointsRequest {
    ImageInput image;
    std::string text;
};
struct PoseMatricesView {
    Span<const float> values;
    std::uint64_t count{0};
    trtmc_pose_matrices_v1 c_view() const noexcept { return {values.data(), values.size(), count}; }
};
enum class PoseCropPhase : std::uint32_t {
    Refinement = TRTMC_POSE_CROP_REFINEMENT,
    Scoring = TRTMC_POSE_CROP_SCORING
};
struct PoseCropQuery {
    PoseMatricesView poses;
    PoseCropPhase stage;
    std::uint64_t iteration;
};
struct PoseCrops {
    std::vector<float> rendered, observed;
    std::uint64_t count{0};
    std::uint32_t height{0}, width{0}, channels{6};
};
// v1 invokes providers serially during the synchronous run and never retains
// the provider/context after run returns. Returned arrays are owned by PoseCrops.
using PoseCropsProvider = std::function<PoseCrops(const PoseCropQuery&)>;
struct PoseHypothesesCropsToRefinedPosesRequest {
    PoseMatricesView candidates;
    float mesh_diameter_meters{0};
    PoseCropsProvider crops;
};
enum class MeshAppearance : std::uint32_t {
    None = TRTMC_MESH_NO_APPEARANCE,
    VertexRgb = TRTMC_MESH_VERTEX_RGB,
    TextureUv = TRTMC_MESH_TEXTURE_UV
};
struct TriangleMeshInput {
    FloatMatrixView vertices;
    Span<const std::uint32_t> triangles;
    FloatMatrixView vertex_normals;
    MeshAppearance appearance{MeshAppearance::None};
    FloatMatrixView vertex_rgb, uv;
    std::optional<ImageInput> texture;
    PoseMatricesView symmetries;
};
struct RgbdMeshMaskToObjectPoseRequest {
    ImageInput rgb;
    FloatMatrixView depth_meters;
    Span<const std::uint8_t> object_mask;
    std::array<float, 9> pixel_intrinsics{};
    TriangleMeshInput mesh;
};

using SemanticSegmentationResult = detail::ViewResult<trtmc_semantic_segmentation_view_v1>;
using MasksResult = detail::ViewResult<trtmc_masks_view_v1>;
using DisparityResult = detail::ViewResult<trtmc_disparity_view_v1>;
using MetricGeometryResult = detail::ViewResult<trtmc_metric_geometry_view_v1>;
using GroundedBoxesResult = detail::ViewResult<trtmc_grounded_boxes_view_v1>;
using DetectedBoxesResult = detail::ViewResult<trtmc_detected_boxes_view_v1>;
using GroundedPointsResult = detail::ViewResult<trtmc_grounded_points_view_v1>;
using RefinedPosesResult = detail::ViewResult<trtmc_refined_poses_view_v1>;
using ObjectPoseResult = detail::ViewResult<trtmc_object_pose_view_v1>;

namespace detail {
template <class Wire>
struct PerceptionRequest {
    explicit PerceptionRequest(Wire input) : wire(std::move(input)) {}
    PerceptionRequest(const PerceptionRequest&) = delete;
    PerceptionRequest& operator=(const PerceptionRequest&) = delete;
    PerceptionRequest(PerceptionRequest&&) noexcept = default;
    PerceptionRequest& operator=(PerceptionRequest&&) noexcept = default;
    Wire wire;
    std::vector<trtmc_point_prompt_v1> points;
    std::vector<trtmc_box_exemplar_v1> exemplars;
};
inline auto perception_request(const ImageToSemanticSegmentationRequest& input) {
    return PerceptionRequest<trtmc_perception_image_request_v1>({input.image.wire});
}
inline auto perception_request(const ImageToMaskProposalsRequest& input) {
    return PerceptionRequest<trtmc_perception_image_request_v1>({input.image.wire});
}
inline auto perception_request(const ImageToMetricGeometryRequest& input) {
    return PerceptionRequest<trtmc_perception_image_request_v1>({input.image.wire});
}
inline auto perception_request(const ImageToBoxesRequest& input) {
    return PerceptionRequest<trtmc_perception_image_request_v1>({input.image.wire});
}
inline auto perception_request(const ImagePointsToMasksRequest& input) {
    auto out = PerceptionRequest<trtmc_image_points_request_v1>({input.image.wire, nullptr, 0});
    for (const auto& p : input.points)
        out.points.push_back({p.point, p.foreground ? 1U : 0U});
    out.wire.points = out.points.data();
    out.wire.point_count = out.points.size();
    return out;
}
inline auto perception_request(const ImageBoxToMasksRequest& input) {
    return PerceptionRequest<trtmc_image_box_request_v1>({input.image.wire, input.box});
}
inline auto perception_request(const ImageMaskToMasksRequest& input) {
    auto out = PerceptionRequest<trtmc_image_prior_mask_request_v1>(
        {input.image.wire, input.prior_logits.c_view(), nullptr, 0, input.box.has_value() ? 1U : 0U,
         input.box.value_or(PixelBox{})});
    for (const auto& p : input.points)
        out.points.push_back({p.point, p.foreground ? 1U : 0U});
    out.wire.points = out.points.data();
    out.wire.point_count = out.points.size();
    return out;
}
inline auto perception_request(const ImageTextToInstanceMasksRequest& input) {
    return PerceptionRequest<trtmc_image_query_request_v1>(
        {input.image.wire, c_string(input.text)});
}
inline auto perception_request(const ImageTextToBoxesRequest& input) {
    return PerceptionRequest<trtmc_image_query_request_v1>(
        {input.image.wire, c_string(input.text)});
}
inline auto perception_request(const ImageTextToPointsRequest& input) {
    return PerceptionRequest<trtmc_image_query_request_v1>(
        {input.image.wire, c_string(input.text)});
}
inline auto perception_request(const ImageBoxExemplarsToInstanceMasksRequest& input) {
    auto out = PerceptionRequest<trtmc_image_exemplars_request_v1>(
        {input.image.wire, nullptr, 0, c_string(input.text_hint)});
    for (const auto& e : input.exemplars)
        out.exemplars.push_back({e.box, e.positive ? 1U : 0U});
    out.wire.exemplars = out.exemplars.data();
    out.wire.exemplar_count = out.exemplars.size();
    return out;
}
inline auto perception_request(const StereoImagesToDisparityRequest& input) {
    return PerceptionRequest<trtmc_stereo_images_request_v1>({input.left.wire, input.right.wire});
}
inline trtmc_triangle_mesh_v1 triangle_mesh_wire(const TriangleMeshInput& mesh) {
    return {mesh.vertices.c_view(),
            mesh.triangles.data(),
            mesh.triangles.size(),
            mesh.vertex_normals.c_view(),
            static_cast<std::uint32_t>(mesh.appearance),
            mesh.vertex_rgb.c_view(),
            mesh.uv.c_view(),
            mesh.texture ? mesh.texture->wire : trtmc_image_input_v1{},
            mesh.symmetries.c_view()};
}
inline auto perception_request(const RgbdMeshMaskToObjectPoseRequest& input) {
    trtmc_rgbd_mesh_mask_request_v1 wire{};
    wire.rgb = input.rgb.wire;
    wire.depth_meters = input.depth_meters.c_view();
    wire.object_mask = input.object_mask.data();
    wire.mask_count = input.object_mask.size();
    std::copy(input.pixel_intrinsics.begin(), input.pixel_intrinsics.end(), wire.pixel_intrinsics);
    wire.mesh = triangle_mesh_wire(input.mesh);
    return PerceptionRequest<trtmc_rgbd_mesh_mask_request_v1>(wire);
}
template <class Result, class Table, class Request>
Result perception_call(const std::shared_ptr<ModelState>& state, const Table* table,
                       const Request& input, const Config& config) {
    auto request = perception_request(input);
    auto entries = config.c_entries();
    auto options = entries.view();
    trtmc_result* raw = nullptr;
    trtmc_error* error = nullptr;
    const auto status = table->run(state->handle, &request.wire, &options, &raw, &error);
    ResultOwner owner(state, raw);
    check(state->api, status, error);
    return Result(std::move(owner), table->result_view);
}
struct PoseCallbackContext {
    const PoseCropsProvider& provider;
    std::exception_ptr exception;
};
inline void TRTMC_CALL release_pose_crops(void* owner) noexcept {
    delete static_cast<PoseCrops*>(owner);
}
inline trtmc_status TRTMC_CALL pose_crop_callback(void* context,
                                                  const trtmc_pose_crop_request_v1* request,
                                                  trtmc_pose_crop_response_v1* out) noexcept {
    if (!out)
        return TRTMC_INVALID_ARGUMENT;
    *out = {};
    if (!context || !request)
        return TRTMC_INVALID_ARGUMENT;
    auto& state = *static_cast<PoseCallbackContext*>(context);
    try {
        const PoseCropQuery input{
            {{request->poses.values, static_cast<std::size_t>(request->poses.value_count)},
             request->poses.count},
            static_cast<PoseCropPhase>(request->stage),
            request->iteration};
        auto owned = std::make_unique<PoseCrops>(state.provider(input));
        out->rendered = owned->rendered.data();
        out->rendered_count = owned->rendered.size();
        out->observed = owned->observed.data();
        out->observed_count = owned->observed.size();
        out->hypothesis_count = owned->count;
        out->height = owned->height;
        out->width = owned->width;
        out->channels = owned->channels;
        out->release = release_pose_crops;
        out->owner = owned.release();
        return TRTMC_OK;
    } catch (const Error& error) {
        state.exception = std::current_exception();
        out->error_message = c_string(error.what());
        return error.code();
    } catch (const std::bad_alloc&) {
        out->error_message = c_string("crop provider out of memory");
        return TRTMC_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        state.exception = std::current_exception();
        out->error_message = c_string(error.what());
        return TRTMC_INTERNAL_ERROR;
    } catch (...) {
        out->error_message = c_string("unknown crop provider exception");
        return TRTMC_INTERNAL_ERROR;
    }
}
} // namespace detail

#define TRTMC_PERCEPTION_WRAPPER(Name, Id, Api, Result)                                            \
    class Name {                                                                                   \
      public:                                                                                      \
        static constexpr std::string_view kTask = Id;                                              \
        static constexpr std::uint32_t kMajor = 1, kMinor = 0;                                     \
        std::vector<ConfigField> config_fields() const {                                           \
            return detail::config_fields(state_, kTask, kMajor, kMinor);                           \
        }                                                                                          \
        Result run(const Name##Request& input, const Config& config = {}) const {                  \
            return detail::perception_call<Result>(state_, api_, input, config);                   \
        }                                                                                          \
        static void validate_table(const trtmc_api_header* table) {                                \
            if (!table || table->major != 1 || table->minor != 0 ||                                \
                table->byte_size < sizeof(Api))                                                    \
                throw Error(TRTMC_VERSION_MISMATCH, "incompatible perception Task table");         \
        }                                                                                          \
                                                                                                   \
      private:                                                                                     \
        friend class Model;                                                                        \
        Name(std::shared_ptr<detail::ModelState> state, const trtmc_api_header* table) noexcept    \
            : state_(std::move(state)), api_(reinterpret_cast<const Api*>(table)) {}               \
        std::shared_ptr<detail::ModelState> state_;                                                \
        const Api* api_;                                                                           \
    };
TRTMC_PERCEPTION_WRAPPER(ImageToSemanticSegmentation, TRTMC_TASK_IMAGE_TO_SEMANTIC_SEGMENTATION,
                         trtmc_image_to_semantic_segmentation_api_v1, SemanticSegmentationResult)
TRTMC_PERCEPTION_WRAPPER(ImagePointsToMasks, TRTMC_TASK_IMAGE_POINTS_TO_MASKS,
                         trtmc_image_points_to_masks_api_v1, MasksResult)
TRTMC_PERCEPTION_WRAPPER(ImageBoxToMasks, TRTMC_TASK_IMAGE_BOX_TO_MASKS,
                         trtmc_image_box_to_masks_api_v1, MasksResult)
TRTMC_PERCEPTION_WRAPPER(ImageMaskToMasks, TRTMC_TASK_IMAGE_MASK_TO_MASKS,
                         trtmc_image_mask_to_masks_api_v1, MasksResult)
TRTMC_PERCEPTION_WRAPPER(ImageToMaskProposals, TRTMC_TASK_IMAGE_TO_MASK_PROPOSALS,
                         trtmc_image_to_mask_proposals_api_v1, MasksResult)
TRTMC_PERCEPTION_WRAPPER(ImageTextToInstanceMasks, TRTMC_TASK_IMAGE_TEXT_TO_INSTANCE_MASKS,
                         trtmc_image_text_to_instance_masks_api_v1, MasksResult)
TRTMC_PERCEPTION_WRAPPER(ImageBoxExemplarsToInstanceMasks,
                         TRTMC_TASK_IMAGE_BOX_EXEMPLARS_TO_INSTANCE_MASKS,
                         trtmc_image_box_exemplars_to_instance_masks_api_v1, MasksResult)
TRTMC_PERCEPTION_WRAPPER(StereoImagesToDisparity, TRTMC_TASK_STEREO_IMAGES_TO_DISPARITY,
                         trtmc_stereo_images_to_disparity_api_v1, DisparityResult)
TRTMC_PERCEPTION_WRAPPER(ImageToMetricGeometry, TRTMC_TASK_IMAGE_TO_METRIC_GEOMETRY,
                         trtmc_image_to_metric_geometry_api_v1, MetricGeometryResult)
TRTMC_PERCEPTION_WRAPPER(ImageToBoxes, TRTMC_TASK_IMAGE_TO_BOXES, trtmc_image_to_boxes_api_v1,
                         DetectedBoxesResult)
TRTMC_PERCEPTION_WRAPPER(ImageTextToBoxes, TRTMC_TASK_IMAGE_TEXT_TO_BOXES,
                         trtmc_image_text_to_boxes_api_v1, GroundedBoxesResult)
TRTMC_PERCEPTION_WRAPPER(ImageTextToPoints, TRTMC_TASK_IMAGE_TEXT_TO_POINTS,
                         trtmc_image_text_to_points_api_v1, GroundedPointsResult)
TRTMC_PERCEPTION_WRAPPER(RgbdMeshMaskToObjectPose, TRTMC_TASK_RGBD_MESH_MASK_TO_OBJECT_POSE,
                         trtmc_rgbd_mesh_mask_to_object_pose_api_v1, ObjectPoseResult)
#undef TRTMC_PERCEPTION_WRAPPER

class PoseHypothesesCropsToRefinedPoses {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_POSE_HYPOTHESES_CROPS_TO_REFINED_POSES;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    RefinedPosesResult run(const PoseHypothesesCropsToRefinedPosesRequest& input,
                           const Config& config = {}) const {
        detail::PoseCallbackContext context{input.crops, {}};
        const trtmc_pose_refinement_request_v1 request{
            input.candidates.c_view(), input.mesh_diameter_meters, &context,
            input.crops ? detail::pose_crop_callback : nullptr};
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(state_->handle, &request, &options, &raw, &error);
        detail::ResultOwner owner(state_, raw);
        detail::check(state_->api, status, error);
        return RefinedPosesResult(std::move(owner), api_->result_view);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_pose_hypotheses_crops_to_refined_poses_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible pose refinement API table");
    }

  private:
    friend class Model;
    PoseHypothesesCropsToRefinedPoses(std::shared_ptr<detail::ModelState> state,
                                      const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(
              reinterpret_cast<const trtmc_pose_hypotheses_crops_to_refined_poses_api_v1*>(table)) {
    }
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_pose_hypotheses_crops_to_refined_poses_api_v1* api_;
};
struct BatchImageTextToBoxesItem {
    ImageTextToBoxesRequest input;
    Config config{};
};
struct BatchImageTextToBoxesRequest {
    std::vector<BatchImageTextToBoxesItem> items;
};
class GroundedBoxesBatchResult {
  public:
    GroundedBoxesBatchResult(detail::ResultOwner owner,
                             const trtmc_batch_image_text_to_boxes_api_v1* api)
        : owner_(std::move(owner)), api_(api) {
        trtmc_error* error = nullptr;
        const auto status = api_->result_count(owner_.get(), &count_, &error);
        detail::check(owner_.api(), status, error);
    }
    GroundedBoxesBatchResult(const GroundedBoxesBatchResult&) = delete;
    GroundedBoxesBatchResult& operator=(const GroundedBoxesBatchResult&) = delete;
    GroundedBoxesBatchResult(GroundedBoxesBatchResult&& other) noexcept
        : owner_(std::move(other.owner_)), api_(other.api_),
          count_(std::exchange(other.count_, 0)) {}
    GroundedBoxesBatchResult& operator=(GroundedBoxesBatchResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            api_ = other.api_;
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    uint64_t size() const noexcept { return count_; }
    // Labels, raw response and coordinates borrow this result's lifetime.
    trtmc_grounded_boxes_view_v1 at(uint64_t index) const {
        trtmc_grounded_boxes_view_v1 out{};
        trtmc_error* error = nullptr;
        const auto status = api_->result_item_view(owner_.get(), index, &out, &error);
        detail::check(owner_.api(), status, error);
        return out;
    }
    trtmc_grounded_boxes_view_v1 operator[](uint64_t index) const { return at(index); }

  private:
    detail::ResultOwner owner_;
    const trtmc_batch_image_text_to_boxes_api_v1* api_;
    uint64_t count_{0};
};
class BatchImageTextToBoxes {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_TEXT_TO_BOXES;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = BatchImageTextToBoxesRequest;
    using Result = GroundedBoxesBatchResult;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_batch_image_text_to_boxes_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible boxes batch Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, 1, 0);
    }
    Result run(const Request& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_image_text_to_boxes_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({detail::perception_request(item.input).wire, configs.back().view()});
        }
        const trtmc_batch_image_text_to_boxes_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(model_->handle, &request, &raw, &error);
        detail::ResultOwner owner(model_, raw);
        detail::check(model_->api, status, error);
        return Result(std::move(owner), api_);
    }

  private:
    friend class Model;
    BatchImageTextToBoxes(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_batch_image_text_to_boxes_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_batch_image_text_to_boxes_api_v1* api_;
};
} // namespace trtmc
