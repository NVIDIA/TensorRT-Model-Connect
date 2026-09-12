/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <set>

namespace {
using namespace trtmc::internal;
std::atomic<int> boxes_batch_calls{0}, boxes_single_calls{0}, boxes_batch_items{0};

class PerceptionModel final : public IModel,
                              public IImageToSemanticSegmentation,
                              public IImagePointsToMasks,
                              public IImageBoxToMasks,
                              public IImageMaskToMasks,
                              public IImageToMaskProposals,
                              public IImageTextToInstanceMasks,
                              public IImageBoxExemplarsToInstanceMasks,
                              public IStereoImagesToDisparity,
                              public IImageToMetricGeometry,
                              public IImageTextToBoxes,
                              public IBatchImageTextToBoxes,
                              public IImageTextToPoints,
                              public IPoseHypothesesCropsToRefinedPoses,
                              public IRgbdMeshMaskToObjectPose {
  public:
    explicit PerceptionModel(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "example_pose_unsupported")
            return {};
        if (mode_.rfind("batch", 0) == 0)
            return {bind<IBatchImageTextToBoxes>(*this, fields_for(IBatchImageTextToBoxes::kTask))};
        if (mode_ == "pose_only")
            return {bind<IPoseHypothesesCropsToRefinedPoses>(
                *this, fields_for(IPoseHypothesesCropsToRefinedPoses::kTask))};
        return {
            bind<IImageToSemanticSegmentation>(*this,
                                               fields_for(IImageToSemanticSegmentation::kTask)),
            bind<IImagePointsToMasks>(*this, fields_for(IImagePointsToMasks::kTask)),
            bind<IImageBoxToMasks>(*this, fields_for(IImageBoxToMasks::kTask)),
            bind<IImageMaskToMasks>(*this, fields_for(IImageMaskToMasks::kTask)),
            bind<IImageToMaskProposals>(*this, fields_for(IImageToMaskProposals::kTask)),
            bind<IImageTextToInstanceMasks>(*this, fields_for(IImageTextToInstanceMasks::kTask)),
            bind<IImageBoxExemplarsToInstanceMasks>(
                *this, fields_for(IImageBoxExemplarsToInstanceMasks::kTask)),
            bind<IStereoImagesToDisparity>(*this, fields_for(IStereoImagesToDisparity::kTask)),
            bind<IImageToMetricGeometry>(*this, fields_for(IImageToMetricGeometry::kTask)),
            bind<IImageTextToBoxes>(*this, fields_for(IImageTextToBoxes::kTask)),
            bind<IImageTextToPoints>(*this, fields_for(IImageTextToPoints::kTask)),
            bind<IPoseHypothesesCropsToRefinedPoses>(
                *this, fields_for(IPoseHypothesesCropsToRefinedPoses::kTask)),
            bind<IRgbdMeshMaskToObjectPose>(*this, fields_for(IRgbdMeshMaskToObjectPose::kTask))};
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view id) const {
        if (mode_ == IImagePointsToMasks::kTask && id == IImagePointsToMasks::kTask) {
            static const ConfigField declared[] = {
                {"benchmark_masks", ConfigKind::String, ConfigValue{std::string_view{}},
                 "Explicit synthetic mask behavior for application regression tests"}};
            return declared;
        }
        if (id == IBatchImageTextToBoxes::kTask) {
            static const ConfigField declared[] = {
                {"score", ConfigKind::F64, ConfigValue{0.5}, "Synthetic score value"},
                {"include_scores", ConfigKind::Bool, ConfigValue{true}, "Return synthetic scores"}};
            return declared;
        }
        if (id == IPoseHypothesesCropsToRefinedPoses::kTask) {
            static const ConfigField declared[] = {
                {"refinement_iterations", ConfigKind::I64, ConfigValue{std::int64_t{2}},
                 "Refinement callback iterations"},
                {"score_hypotheses", ConfigKind::Bool, ConfigValue{true},
                 "Request final scoring crops"}};
            return declared;
        }
        if (id == IImageToMetricGeometry::kTask) {
            static const ConfigField declared[] = {
                {"fov_x", ConfigKind::F64, ConfigValue{60.0}, "Horizontal field of view, degrees"}};
            return declared;
        }
        return {};
    }
    SemanticSegmentationResult run(const ImageToSemanticSegmentationRequest& request,
                                   ConfigView config) override {
        (void)options(IImageToSemanticSegmentation::kTask, config);
        SemanticSegmentationResult result;
        result.height = request.image.height;
        result.width = request.image.width;
        const auto area = static_cast<std::size_t>(result.height) * result.width;
        result.labels.assign(area, 5);
        result.labels[0] = 255;
        result.class_ids = {0, 5};
        result.class_names = {"background", "object"};
        result.vocabulary_id = "fixture-semantics";
        result.ignore_label = 255;
        result.background_label = 0;
        result.class_scores.assign(area * 2, -2);
        result.score_height = result.height;
        result.score_width = result.width;
        if (mode_ == "broken_semantic")
            result.labels.pop_back();
        return result;
    }
    MasksResult run(const ImagePointsToMasksRequest& request, ConfigView config) override {
        const auto configured = options(IImagePointsToMasks::kTask, config);
        const auto behavior = configured.count("benchmark_masks")
                                  ? std::get<std::string_view>(configured.at("benchmark_masks"))
                                  : std::string_view{};
        if (mode_ == "empty_masks" || behavior == "empty") {
            MasksResult result;
            result.height = request.image.height;
            result.width = request.image.width;
            result.kind = MaskKind::Logits;
            return result; // A real empty result, never an invented all-zero mask.
        }
        auto result = masks(request.image, MaskKind::Logits);
        if (result.count == 0)
            return result;
        result.predicted_iou[0] = static_cast<float>(request.points.size());
        result.boxes[0].x_min = request.points[0].point.x;
        result.low_res_logits[0] = request.points[0].foreground ? 7 : -7;
        if (mode_ == "first_vs_best" || behavior == "first_vs_best") {
            result.predicted_iou = {0.1F, 0.9F};
            const auto area = static_cast<std::size_t>(request.image.height) * request.image.width;
            std::fill(result.masks.begin() + static_cast<std::ptrdiff_t>(area), result.masks.end(),
                      1.0F);
        }
        if (mode_ == "missing_point_iou")
            result.predicted_iou.clear();
        if (mode_ == "point_required_only") {
            result.boxes.clear();
            result.low_res_logits.clear();
            result.low_res_height = result.low_res_width = 0;
        }
        if (mode_ == "wrong_optional_count")
            result.confidence = {0.5F};
        return result;
    }
    MasksResult run(const ImageBoxToMasksRequest& request, ConfigView config) override {
        (void)options(IImageBoxToMasks::kTask, config);
        auto result = masks(request.image, MaskKind::Logits);
        if (result.count == 0)
            return result;
        result.boxes[0] = request.box;
        if (mode_ == "missing_box_iou")
            result.predicted_iou.clear();
        return result;
    }
    MasksResult run(const ImageMaskToMasksRequest& request, ConfigView config) override {
        (void)options(IImageMaskToMasks::kTask, config);
        auto result = masks(request.image, MaskKind::Logits);
        if (result.count == 0)
            return result;
        result.low_res_logits[0] = request.prior.logits.values[0];
        result.predicted_iou[0] = static_cast<float>(request.points.size());
        if (request.box)
            result.boxes[0] = *request.box;
        if (mode_ == "missing_prior_iou")
            result.predicted_iou.clear();
        return result;
    }
    MasksResult run(const ImageToMaskProposalsRequest& request, ConfigView config) override {
        (void)options(IImageToMaskProposals::kTask, config);
        auto result = masks(request.image, MaskKind::Binary);
        if (result.count == 0)
            return result;
        result.stability = {0.75F, 0.5F};
        result.proposals = {{3,
                             {0, 0, static_cast<float>(request.image.width),
                              static_cast<float>(request.image.height)},
                             {{0.5F, 1}, {1, 0.5F}}},
                            {2, {0, 0, 2, 1}, {{1, 0.5F}}}};
        if (mode_ == "missing_proposal_metadata")
            result.proposals.clear();
        if (mode_ == "missing_proposal_scores") {
            result.predicted_iou.clear();
            result.stability.clear();
        }
        if (mode_ == "proposal_iou_only")
            result.stability.clear();
        if (mode_ == "proposal_stability_only")
            result.predicted_iou.clear();
        if (mode_ == "proposal_confidence_only") {
            result.predicted_iou.clear();
            result.stability.clear();
            result.confidence = {0.4F, 0.6F};
        }
        return result;
    }
    MasksResult run(const ImageTextToInstanceMasksRequest& request, ConfigView config) override {
        (void)options(IImageTextToInstanceMasks::kTask, config);
        if (request.text.empty())
            throw ConfigError("concept text required");
        auto result = masks(request.image, MaskKind::Binary);
        if (result.count == 0)
            return result;
        result.predicted_iou.clear();
        result.confidence = {0.8F, 0.6F};
        result.object_ids = {100, 101};
        if (mode_ == "missing_instance_confidence")
            result.confidence.clear();
        if (mode_ == "missing_instance_boxes")
            result.boxes.clear();
        return result;
    }
    MasksResult run(const ImageBoxExemplarsToInstanceMasksRequest& request,
                    ConfigView config) override {
        (void)options(IImageBoxExemplarsToInstanceMasks::kTask, config);
        auto result = masks(request.image, MaskKind::Binary);
        if (result.count == 0)
            return result;
        result.predicted_iou.clear();
        result.confidence = {0.8F, 0.6F};
        for (const auto& item : request.exemplars)
            if (item.positive) {
                result.boxes[0] = item.box;
                break;
            }
        if (!request.text_hint.empty())
            result.confidence[0] = 0.9F;
        if (mode_ == "missing_instance_confidence")
            result.confidence.clear();
        if (mode_ == "missing_instance_boxes")
            result.boxes.clear();
        return result;
    }
    DisparityResult run(const StereoImagesToDisparityRequest& request, ConfigView config) override {
        (void)options(IStereoImagesToDisparity::kTask, config);
        const auto area = static_cast<std::size_t>(request.left.height) * request.left.width;
        return {{std::vector<float>(area, first_pixel(request.left) - first_pixel(request.right)),
                 request.left.height, request.left.width}};
    }
    MetricGeometryResult run(const ImageToMetricGeometryRequest& request,
                             ConfigView config) override {
        const auto value = options(IImageToMetricGeometry::kTask, config);
        const auto fov = config_value_as<double>(value.at("fov_x"));
        MetricGeometryResult result;
        result.height = request.image.height;
        result.width = request.image.width;
        const auto area = static_cast<std::size_t>(result.height) * result.width;
        result.points.assign(area * 3, 2);
        result.depth.assign(area, 2);
        result.valid.assign(area, 1);
        result.valid[0] = 0;
        result.depth[0] = std::numeric_limits<float>::infinity();
        for (int i = 0; i < 3; ++i)
            result.points[i] = std::numeric_limits<float>::infinity();
        result.normalized_intrinsics = {
            static_cast<float>(fov / 100), 0, 0.5F, 0, 1, 0.5F, 0, 0, 1};
        return result;
    }
    GroundedBoxesResult run(const ImageTextToBoxesRequest& request, ConfigView config) override {
        ++boxes_single_calls;
        (void)options(IImageTextToBoxes::kTask, config);
        if (request.text == "malformed")
            return {{}, "unparsed output", false};
        return {{{{0.25F, 0.5F, 2.75F, 1.75F}, std::string(request.text), std::nullopt}},
                "raw grounding text",
                true};
    }
    std::vector<GroundedBoxesResult>
    run_batch(const BatchImageTextToBoxesRequest& request) override {
        ++boxes_batch_calls;
        std::vector<std::map<std::string_view, ConfigValue>> resolved;
        for (const auto& item : request.items) {
            resolved.push_back(options(IBatchImageTextToBoxes::kTask, item.config));
            const auto score = config_value_as<double>(resolved.back().at("score"));
            if (score < 0 || score > 1)
                throw ConfigError("fixture score must be within zero and one");
            if (item.input.image.channels != 3)
                throw UnsupportedTask("fixture native batch requires RGB");
        }
        std::vector<GroundedBoxesResult> results;
        for (size_t i = 0; i < request.items.size(); ++i) {
            ++boxes_batch_items;
            const auto& input = request.items[i].input;
            GroundedBoxesResult result;
            result.raw_response = "batch:" + std::string(input.text);
            result.parse_complete = input.text != "malformed";
            if (result.parse_complete && input.text != "empty") {
                const auto score = config_value_as<bool>(resolved[i].at("include_scores"))
                                       ? std::optional<float>{static_cast<float>(
                                             config_value_as<double>(resolved[i].at("score")))}
                                       : std::nullopt;
                result.boxes.push_back(
                    {{float(input.image.width) * 0.1F, float(input.image.height) * 0.2F,
                      float(input.image.width) * 0.8F, float(input.image.height) * 0.9F},
                     std::string(input.text),
                     score});
                if (input.text == "two")
                    result.boxes.push_back({{0, 0, 1, 1}, "second", std::nullopt});
            }
            results.push_back(std::move(result));
        }
        if (mode_ == "batch_fail")
            throw std::runtime_error("fixture native boxes execution failed");
        if (mode_ == "batch_bad_count" && !results.empty())
            results.pop_back();
        return results;
    }
    GroundedPointsResult run(const ImageTextToPointsRequest& request, ConfigView config) override {
        (void)options(IImageTextToPoints::kTask, config);
        return {{{{2.5F, 1.5F}, std::string(request.text), 0.75F}}, "raw point text", true};
    }
    RefinedPosesResult run(const PoseHypothesesCropsToRefinedPosesRequest& request,
                           ConfigView config) override {
        if (mode_ == "example_pose_fail")
            throw std::runtime_error("preprocessed fixture provider failed");
        const auto values = options(IPoseHypothesesCropsToRefinedPoses::kTask, config);
        const auto iterations = config_value_as<std::int64_t>(values.at("refinement_iterations"));
        const auto score = config_value_as<bool>(values.at("score_hypotheses"));
        if (iterations < 1 || iterations > 3)
            throw ConfigError("invalid fixture iteration count");
        if (!score && request.candidates.count != 1)
            throw ConfigError("multiple hypotheses require scoring");
        RefinedPosesResult result;
        result.num_hypotheses = static_cast<std::int32_t>(request.candidates.count);
        result.refined_poses.assign(request.candidates.values.begin(),
                                    request.candidates.values.end());
        for (std::int64_t i = 0; i < iterations; ++i) {
            auto crops = request.crops({{{result.refined_poses.data(), result.refined_poses.size()},
                                         request.candidates.count},
                                        CropStage::Refinement,
                                        static_cast<std::uint64_t>(i)});
            const auto stride =
                static_cast<std::size_t>(crops.height) * crops.width * crops.channels;
            for (std::size_t n = 0; n < request.candidates.count; ++n) {
                result.refined_poses[n * 16 + 3] +=
                    crops.rendered[n * stride] * request.mesh_diameter_meters;
                result.refined_poses[n * 16 + 7] += crops.observed[n * stride];
            }
        }
        result.best_index = 0;
        if (score) {
            auto crops = request.crops({{{result.refined_poses.data(), result.refined_poses.size()},
                                         request.candidates.count},
                                        CropStage::Scoring,
                                        static_cast<std::uint64_t>(iterations)});
            const auto stride =
                static_cast<std::size_t>(crops.height) * crops.width * crops.channels;
            for (std::size_t n = 0; n < request.candidates.count; ++n)
                result.scores.push_back(crops.observed[n * stride] * 4);
            result.best_index = static_cast<std::int32_t>(
                std::max_element(result.scores.begin(), result.scores.end()) -
                result.scores.begin());
        }
        result.all_poses_rigid = true;
        result.refinement_ms = 1.5;
        result.scoring_ms = score ? 0.5 : 0;
        if (mode_ == "example_pose_missing_score")
            result.scores.clear();
        if (mode_ == "example_pose_bad_result")
            result.num_hypotheses = -1;
        return result;
    }
    ObjectPoseResult run(const RgbdMeshMaskToObjectPoseRequest& request,
                         ConfigView config) override {
        (void)options(IRgbdMeshMaskToObjectPose::kTask, config);
        ObjectPoseResult result;
        result.object_to_camera = {1, 0, 0, request.mesh.vertices.values[0],
                                   0, 1, 0, static_cast<float>(request.mesh.symmetries.count),
                                   0, 0, 1, request.depth_meters.values[1],
                                   0, 0, 0, 1};
        if (request.mesh.appearance == MeshAppearance::VertexRgb)
            result.score = request.mesh.vertex_rgb.values[0];
        if (request.mesh.appearance == MeshAppearance::TextureUv)
            result.score = request.mesh.uv.values[0] + first_pixel(request.mesh.texture);
        result.rigid = true;
        return result;
    }

  private:
    static float first_pixel(const ImageView& image) {
        return image.format == ImageFormat::Float32
                   ? static_cast<const float*>(image.data)[0]
                   : static_cast<float>(static_cast<const std::uint8_t*>(image.data)[0]);
    }
    MasksResult masks(const ImageView& image, MaskKind kind) const {
        MasksResult result;
        result.height = image.height;
        result.width = image.width;
        if (mode_ == "empty_contract_masks") {
            result.kind = kind;
            return result;
        }
        result.count = 2;
        result.kind = kind;
        const auto area = static_cast<std::size_t>(image.height) * image.width;
        result.masks.resize(area * 2);
        for (std::size_t i = 0; i < result.masks.size(); ++i)
            result.masks[i] =
                kind == MaskKind::Binary ? static_cast<float>(i % 2) : (i % 2 ? 1.0F : -2.0F);
        result.predicted_iou = {0.2F, 0.9F};
        result.boxes = {{0, 0, static_cast<float>(image.width), static_cast<float>(image.height)},
                        {0, 0, 1, 1}};
        result.low_res_logits = {-4, 3, -2, 1};
        result.low_res_height = 1;
        result.low_res_width = 2;
        if (mode_ == "broken_masks")
            result.masks.pop_back();
        return result;
    }
    std::map<std::string_view, ConfigValue> options(std::string_view id, ConfigView config) const {
        const auto fields = fields_for(id);
        std::map<std::string_view, ConfigValue> out;
        std::set<std::string_view> seen;
        for (const auto& field : fields)
            out.emplace(field.name, *field.default_value);
        for (const auto& entry : config) {
            const auto field = std::find_if(fields.begin(), fields.end(),
                                            [&](const auto& f) { return f.name == entry.name; });
            if (field == fields.end() || !seen.insert(entry.name).second ||
                field->kind != config_kind(entry.value))
                throw ConfigError("unknown, duplicate or wrongly typed config");
            out.at(entry.name) = entry.value;
        }
        return out;
    }
    std::string mode_;
};
// Old-path protocol fixture, independent of the SDK implementation above.
class ExistingPreprocessed final : public trtmc::IPoseHypothesisRefinement {
  public:
    trtmc::PoseEstimationResult
    estimate_pose_hypotheses(const trtmc::PoseEstimationRequest& input) override {
        if (input.num_hypotheses != 3 || input.candidate_poses.size() != 48 ||
            input.refinement_iterations != 2 || input.mesh_diameter != 0.18F ||
            !input.score_hypotheses || !input.crop_provider || input.use_tracked_pose)
            throw std::invalid_argument("old preprocessed fixture received the wrong request");
        trtmc::PoseEstimationResult result;
        result.num_hypotheses = 3;
        result.refined_poses = input.candidate_poses;
        for (int32_t iteration = 0; iteration < 2; ++iteration) {
            auto crops = input.crop_provider(result.refined_poses,
                                             trtmc::PoseCropStage::kRefinement, iteration);
            require_crops(crops);
            for (size_t index = 0; index < 3; ++index) {
                result.refined_poses[index * 16 + 3] +=
                    crops.rendered_features[index * 160 * 160 * 6] * 0.18F;
                result.refined_poses[index * 16 + 7] +=
                    crops.observed_features[index * 160 * 160 * 6];
            }
        }
        auto crops = input.crop_provider(result.refined_poses, trtmc::PoseCropStage::kScoring, 2);
        require_crops(crops);
        for (size_t index = 0; index < 3; ++index)
            result.scores.push_back(crops.observed_features[index * 160 * 160 * 6] * 4);
        result.best_index = static_cast<int32_t>(
            std::max_element(result.scores.begin(), result.scores.end()) - result.scores.begin());
        result.all_poses_rigid = true;
        return result;
    }
    void reset_pose_tracking() override {
        throw std::logic_error("one-shot example must not reset tracking");
    }

  private:
    static void require_crops(const trtmc::PoseCropBatch& crops) {
        if (crops.num_hypotheses != 3 || crops.height != 160 || crops.width != 160 ||
            crops.channels != 6 || crops.rendered_features.size() != 3 * 160 * 160 * 6 ||
            crops.observed_features.size() != 3 * 160 * 160 * 6)
            throw std::invalid_argument("old preprocessed crops changed shape");
    }
};
} // namespace
extern "C" int trtmc_test_boxes_batch_calls() {
    return boxes_batch_calls.load();
}
extern "C" int trtmc_test_boxes_single_calls() {
    return boxes_single_calls.load();
}
extern "C" int trtmc_test_boxes_batch_items() {
    return boxes_batch_items.load();
}
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().task == trtmc::IPoseHypothesisRefinement::kTask)
        return new ExistingPreprocessed;
    return new PerceptionModel(context.reader.info().task);
}
