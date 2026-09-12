/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/image.h"
#include "trtmc/internal/matrix.h"
#include "trtmc/internal/scores.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>

namespace trtmc::internal {

// Original-image pixels: x right, y down; boxes are continuous XYXY corners.
struct PixelPoint {
    float x{0}, y{0};
};
struct PixelBox {
    float x_min{0}, y_min{0}, x_max{0}, y_max{0};
};
struct PointPrompt {
    PixelPoint point;
    bool foreground{true};
};
struct BoxExemplar {
    PixelBox box;
    bool positive{true};
};

// A decoder-space prior is unthresholded low-resolution logits. It is not an
// image-space inpainting mask. The loaded family defines its decoder dimensions.
struct PriorMaskLogits {
    FloatMatrixView logits;
};
struct BinaryImageMask {
    Span<const std::uint8_t> values;
    std::uint32_t height{0}, width{0};
};

struct ImageToSemanticSegmentationRequest {
    ImageView image;
};
struct ImagePointsToMasksRequest {
    ImageView image;
    Span<const PointPrompt> points;
};
struct ImageBoxToMasksRequest {
    ImageView image;
    PixelBox box;
};
struct ImageMaskToMasksRequest {
    ImageView image;
    PriorMaskLogits prior;
    Span<const PointPrompt> points; // Optional correction prompts; mask is always required.
    std::optional<PixelBox> box;
};
struct ImageToMaskProposalsRequest {
    ImageView image;
};
struct ImageTextToInstanceMasksRequest {
    ImageView image;
    std::string_view text;
};
struct ImageBoxExemplarsToInstanceMasksRequest {
    ImageView image;
    Span<const BoxExemplar> exemplars;
    std::string_view text_hint; // Optional additional concept refinement.
};
struct StereoImagesToDisparityRequest {
    ImageView left;
    ImageView right;
};
struct ImageToMetricGeometryRequest {
    ImageView image;
};
struct ImageToBoxesRequest {
    ImageView image;
};
struct ImageTextToBoxesRequest {
    ImageView image;
    std::string_view text;
};
struct ImageTextToPointsRequest {
    ImageView image;
    std::string_view text;
};

struct SemanticSegmentationResult {
    std::vector<std::int32_t> labels; // [H,W], class IDs at original image resolution.
    std::uint32_t height{0}, width{0};
    std::vector<std::int32_t> class_ids;
    std::vector<std::string> class_names; // Empty or one name per class ID.
    std::string vocabulary_id;
    std::optional<std::int32_t> ignore_label, background_label;
    std::vector<float> class_scores; // Optional [class,score_height,score_width].
    std::uint32_t score_height{0}, score_width{0};
    ScoreKind score_kind{ScoreKind::Logit};
};
enum class MaskKind : std::uint32_t { Logits = 1, Probability = 2, Binary = 3 };
struct MaskProposalMetadata {
    std::uint64_t area{0};
    PixelBox crop_box;
    std::vector<PixelPoint> seed_points;
};
struct MasksResult {
    // Stateless contracts: point/box/prior require N predicted_iou values;
    // text/exemplar instance masks require N confidence values and N boxes;
    // automatic proposals require N proposals plus at least one N-sized score
    // vector (predicted_iou, confidence or stability). Empty detections are valid.
    // Other metadata remains optional, but any present per-mask vector is N-sized.
    std::vector<float> masks; // [mask,H,W], original image coordinates.
    std::uint64_t count{0};
    std::uint32_t height{0}, width{0};
    MaskKind kind{MaskKind::Logits};
    std::vector<float> predicted_iou; // Learned quality estimates, not calibrated probabilities.
    std::vector<float> confidence;    // Empty or N, detection confidence.
    std::vector<float> stability;     // Empty or N, proposal stability score.
    std::vector<PixelBox> boxes;      // Empty or N, original-image XYXY.
    std::vector<std::int64_t>
        object_ids; // Optional identities within this result, not tracking state.
    std::vector<MaskProposalMetadata> proposals; // Empty or N, automatic-mask metadata.
    std::vector<float>
        low_res_logits; // Optional [N,low_res_height,low_res_width], reusable priors.
    std::uint32_t low_res_height{0}, low_res_width{0};
};
struct DisparityResult {
    FloatMatrix disparity; // Left-image [H,W], d = x_left - x_right, in original-image pixels.
};
// OpenCV camera axes: x right, y down, z forward. Depth/points are meters.
// Intrinsics operate on normalized image coordinates (u/W,v/H), matching MoGe.
struct MetricGeometryResult {
    std::vector<float> points; // [H,W,3].
    std::vector<float> depth;  // [H,W].
    std::vector<std::uint8_t> valid;
    std::array<float, 9> normalized_intrinsics{};
    std::uint32_t height{0}, width{0};
};
struct GroundedBox {
    PixelBox box;
    std::string label;
    std::optional<float> confidence;
};
struct GroundedPoint {
    PixelPoint point;
    std::string label;
    std::optional<float> confidence;
};
struct GroundedBoxesResult {
    std::vector<GroundedBox> boxes;
    std::string raw_response;
    bool parse_complete{true}; // False retains malformed generation for inspection.
};
struct GroundedPointsResult {
    std::vector<GroundedPoint> points;
    std::string raw_response;
    bool parse_complete{true};
};

// Fixed-label detections, without a text query. Reuse the existing owned
// result: original-image XYXY pixels, numeric class IDs and detector scores.
// Families own filtering, suppression and all model-specific postprocessing.
using DetectedBoxesResult = trtmc::ObjectDetectionResult;

struct PoseMatricesView {
    Span<const float> values;
    std::uint64_t count{0};
}; // [N,4,4], row major.
enum class CropStage : std::uint32_t { Refinement = 1, Scoring = 2 };
struct PoseCropRequest {
    PoseMatricesView poses;
    CropStage stage;
    std::uint64_t iteration{0};
};
// Features are NHWC: RGB[0,1] then XYZ relative to candidate translation,
// normalized by half mesh diameter. Invalid/background XYZ is zero.
struct PoseCropBatch {
    Span<const float> rendered, observed;
    std::uint64_t count{0};
    std::uint32_t height{0}, width{0}, channels{0};
    std::shared_ptr<void>
        lease; // Keeps provider-owned arrays alive while the family consumes them.
};
// Invoke serially within run; neither callback nor crop leases may be retained
// by the family after run returns. A crop lease may outlive its callback call.
using PoseCropProvider = std::function<PoseCropBatch(const PoseCropRequest&)>;
struct PoseHypothesesCropsToRefinedPosesRequest {
    PoseMatricesView candidates; // Object-to-camera transforms, translation in meters.
    float mesh_diameter_meters{0};
    PoseCropProvider crops;
};
// Multiple hypotheses require one score per hypothesis. A single unscored
// hypothesis is valid with best_index=0; the family owns whether to score it.
using RefinedPosesResult = trtmc::PoseEstimationResult;

enum class MeshAppearance : std::uint32_t { None = 0, VertexRgb = 1, TextureUv = 2 };
struct TriangleMeshView {
    FloatMatrixView vertices;            // [vertex,3], original object coordinates in meters.
    Span<const std::uint32_t> triangles; // [triangle,3], zero-based vertex indices.
    FloatMatrixView vertex_normals;      // Optional [vertex,3].
    MeshAppearance appearance{MeshAppearance::None};
    FloatMatrixView vertex_rgb; // [vertex,3] in [0,1] for VertexRgb.
    FloatMatrixView uv;         // [vertex,2], U right/V up; shared triangle indexing.
    ImageView texture;          // RGB for TextureUv; image memory has top-left origin.
    PoseMatricesView
        symmetries; // Optional object-space rigid symmetries; empty means identity only.
};
struct RgbdMeshMaskToObjectPoseRequest {
    ImageView rgb;
    FloatMatrixView depth_meters; // Aligned [H,W]; zero means invalid depth.
    BinaryImageMask object_mask;
    std::array<float, 9> pixel_intrinsics{};
    TriangleMeshView mesh;
};
struct ObjectPoseResult {
    std::array<float, 16>
        object_to_camera{};     // ORIGINAL object frame, not internally centered mesh.
    std::optional<float> score; // Raw scorer output when provided, not a probability.
    bool rigid{false};
};

#define TRTMC_PERCEPTION_INTERFACE(Name, Id, Result)                                               \
    class I##Name {                                                                                \
      public:                                                                                      \
        using TaskInterface = I##Name;                                                             \
        static constexpr std::string_view kTask = Id;                                              \
        virtual ~I##Name() = default;                                                              \
        virtual Result run(const Name##Request&, ConfigView) = 0;                                  \
    };
TRTMC_PERCEPTION_INTERFACE(ImageToSemanticSegmentation, "image_to_semantic_segmentation",
                           SemanticSegmentationResult)
TRTMC_PERCEPTION_INTERFACE(ImagePointsToMasks, "image_points_to_masks", MasksResult)
TRTMC_PERCEPTION_INTERFACE(ImageBoxToMasks, "image_box_to_masks", MasksResult)
TRTMC_PERCEPTION_INTERFACE(ImageMaskToMasks, "image_mask_to_masks", MasksResult)
TRTMC_PERCEPTION_INTERFACE(ImageToMaskProposals, "image_to_mask_proposals", MasksResult)
TRTMC_PERCEPTION_INTERFACE(ImageTextToInstanceMasks, "image_text_to_instance_masks", MasksResult)
TRTMC_PERCEPTION_INTERFACE(ImageBoxExemplarsToInstanceMasks,
                           "image_box_exemplars_to_instance_masks", MasksResult)
TRTMC_PERCEPTION_INTERFACE(StereoImagesToDisparity, "stereo_images_to_disparity", DisparityResult)
TRTMC_PERCEPTION_INTERFACE(ImageToMetricGeometry, "image_to_metric_geometry", MetricGeometryResult)
TRTMC_PERCEPTION_INTERFACE(ImageToBoxes, "image_to_boxes", DetectedBoxesResult)
TRTMC_PERCEPTION_INTERFACE(ImageTextToBoxes, "image_text_to_boxes", GroundedBoxesResult)
TRTMC_PERCEPTION_INTERFACE(ImageTextToPoints, "image_text_to_points", GroundedPointsResult)
TRTMC_PERCEPTION_INTERFACE(PoseHypothesesCropsToRefinedPoses,
                           "pose_hypotheses_crops_to_refined_poses", RefinedPosesResult)
TRTMC_PERCEPTION_INTERFACE(RgbdMeshMaskToObjectPose, "rgbd_mesh_mask_to_object_pose",
                           ObjectPoseResult)
#undef TRTMC_PERCEPTION_INTERFACE

struct BatchImageTextToBoxesItem {
    ImageTextToBoxesRequest input;
    ConfigView config;
};
struct BatchImageTextToBoxesRequest {
    Span<const BatchImageTextToBoxesItem> items;
};
class IBatchImageTextToBoxes {
  public:
    using TaskInterface = IBatchImageTextToBoxes;
    static constexpr std::string_view kTask = "batch_image_text_to_boxes";
    virtual ~IBatchImageTextToBoxes() = default;
    // Family validates all complete items before one native batch generation;
    // success returns exactly N ordered results, including genuine empty detections.
    virtual std::vector<GroundedBoxesResult> run_batch(const BatchImageTextToBoxesRequest&) = 0;
};

} // namespace trtmc::internal
