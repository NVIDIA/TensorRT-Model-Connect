/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/perception.hpp"

#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures = 0;
void check(bool ok, const char* label) {
    if (!ok) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}
std::string_view text(trtmc_string_view value) {
    return trtmc::detail::string_view(value);
}
template <class Function>
void rejects(Function fn, trtmc_status status, const char* label) {
    bool rejected = false;
    try {
        fn();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == status;
    }
    check(rejected, label);
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"perception_fixture\",\"task\":\"" +
                               mode + "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.write(reinterpret_cast<const char*>(magic), 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
}
void batch_boxes(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    for (const char* mode : {"batch", "batch_fail", "batch_bad_count"})
        bundle(root / (std::string("perception-cpp-") + mode + ".bundle"), mode);
    auto load = [&](const char* mode) {
        return trtmc::Model::load(
            (root / (std::string("perception-cpp-") + mode + ".bundle")).string(), options);
    };
    auto model = load("batch");
    check(model.tasks().size() == 1 && model.supports<trtmc::BatchImageTextToBoxes>() &&
              !model.supports<trtmc::ImageTextToBoxes>(),
          "boxes batch support is independently declared");
    auto task = model.task<trtmc::BatchImageTextToBoxes>();
    check(task.config_fields().size() == 2, "batch Config declaration comes from family");
    float small[18]{}, large[60]{};
    const trtmc::ImageInput a({small}, 2, 3), b({large}, 4, 5);
    void* library =
        dlopen((root / "libtrtmc_model_perception_fixture.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library)
        throw std::runtime_error("boxes batch fixture not loaded");
    using Counter = int (*)();
    const auto batch_calls =
        reinterpret_cast<Counter>(dlsym(library, "trtmc_test_boxes_batch_calls"));
    const auto single_calls =
        reinterpret_cast<Counter>(dlsym(library, "trtmc_test_boxes_single_calls"));
    const auto executed = reinterpret_cast<Counter>(dlsym(library, "trtmc_test_boxes_batch_items"));
    if (!batch_calls || !single_calls || !executed)
        throw std::runtime_error("boxes batch counter missing");
    const auto before = batch_calls(), before_single = single_calls();
    const std::string label{"cup\0mug", 7};
    auto result =
        task.run({{{{a, label}, {{"score", 0.0}}}, {{b, "two"}, {{"include_scores", false}}}}});
    check(batch_calls() == before + 1 && single_calls() == before_single,
          "B2 calls one native batch virtual and zero single calls");
    const auto first = result.at(0), second = result.at(1);
    check(result.size() == 2 && first.count == 1 && second.count == 2 &&
              text(first.boxes[0].label) == label && first.boxes[0].has_confidence &&
              first.boxes[0].confidence == 0 && !second.boxes[0].has_confidence &&
              std::abs(first.boxes[0].box.x_min - 0.3F) < 1e-6 &&
              std::abs(second.boxes[0].box.x_max - 4.0F) < 1e-6 &&
              std::abs(second.boxes[0].box.y_max - 3.6F) < 1e-6 &&
              text(first.raw_response) == "batch:" + label,
          "batch preserves pixel coordinates for each image, variable boxes, NUL labels and "
          "zero/false Config");
    auto parsing = task.run({{{{a, "empty"}}, {{b, "malformed"}}}});
    check(parsing.at(0).count == 0 && parsing.at(0).parse_complete && parsing.at(1).count == 0 &&
              !parsing.at(1).parse_complete &&
              text(parsing.at(1).raw_response) == "batch:malformed",
          "empty detections and incomplete parsing remain distinct successful typed results");
    rejects([&] { (void)result.at(2); }, TRTMC_INVALID_ARGUMENT, "boxes batch item bounds");
    rejects([&] { (void)task.run({}); }, TRTMC_INVALID_ARGUMENT, "empty boxes batch rejected");
    const auto completed = executed();
    rejects([&] { (void)task.run({{{{a, "a"}}, {{b, "b"}, {{"unknown", true}}}}}); },
            TRTMC_INVALID_CONFIG, "late-item Config rejection is whole-call");
    rejects([&] { (void)task.run({{{{a, "a"}}, {{b, "b"}, {{"score", 2.0}}}}}); },
            TRTMC_INVALID_CONFIG, "family owns numeric Config range");
    check(executed() == completed, "native family preflights every item before generation");
    for (const char* mode : {"batch_fail", "batch_bad_count"})
        rejects(
            [&] {
                (void)load(mode).task<trtmc::BatchImageTextToBoxes>().run(
                    {{{{a, "a"}}, {{b, "b"}}}});
            },
            TRTMC_INTERNAL_ERROR,
            "native execution/count failure never becomes partial-success boxes");
    auto retained = [&] {
        auto temporary = load("batch");
        return temporary.task<trtmc::BatchImageTextToBoxes>().run({{{{a, "owned"}}}});
    }();
    auto moved = std::move(retained);
    check(retained.size() == 0 && moved.size() == 1 && text(moved.at(0).boxes[0].label) == "owned",
          "batch owner move retains nested strings/boxes beyond model/request scope");
    dlclose(library);
}
void required_mask_outputs(const std::filesystem::path& root, const trtmc::LoadOptions& options,
                           const trtmc::ImageInput& image) {
    auto load = [&](const char* mode) {
        const auto path = root / (std::string("perception-required-") + mode + ".bundle");
        bundle(path, mode);
        return trtmc::Model::load(path.string(), options);
    };
    const trtmc::PointPrompt point{{1, 1}, true};
    const trtmc::BoxExemplar exemplar{{0, 0, 1, 1}, true};
    const float prior_values[]{-1, 2};
    const trtmc::ImagePointsToMasksRequest points{image, {&point, 1}};
    const trtmc::ImageBoxToMasksRequest box{image, {0, 0, 1, 1}};
    const trtmc::ImageMaskToMasksRequest prior{image, {{prior_values, 2}, 1, 2}, {}, std::nullopt};
    const trtmc::ImageBoxExemplarsToInstanceMasksRequest exemplars{image, {&exemplar, 1}, "target"};
    rejects([&] { (void)load("missing_box_iou").task<trtmc::ImageBoxToMasks>().run(box); },
            TRTMC_INTERNAL_ERROR, "box masks cannot omit required predicted IoU");
    rejects([&] { (void)load("missing_prior_iou").task<trtmc::ImageMaskToMasks>().run(prior); },
            TRTMC_INTERNAL_ERROR, "prior masks cannot omit required predicted IoU");
    for (const char* mode : {"missing_instance_confidence", "missing_instance_boxes"}) {
        auto invalid = load(mode);
        rejects(
            [&] { (void)invalid.task<trtmc::ImageTextToInstanceMasks>().run({image, "object"}); },
            TRTMC_INTERNAL_ERROR, "text instance masks require both confidence and boxes");
        rejects(
            [&] { (void)invalid.task<trtmc::ImageBoxExemplarsToInstanceMasks>().run(exemplars); },
            TRTMC_INTERNAL_ERROR, "exemplar instance masks require both confidence and boxes");
    }
    for (const char* mode : {"missing_proposal_metadata", "missing_proposal_scores"})
        rejects([&] { (void)load(mode).task<trtmc::ImageToMaskProposals>().run({image}); },
                TRTMC_INTERNAL_ERROR,
                "automatic masks need proposal metadata and an actual typed score");
    for (const char* mode :
         {"proposal_iou_only", "proposal_confidence_only", "proposal_stability_only"}) {
        auto result = load(mode).task<trtmc::ImageToMaskProposals>().run({image});
        const auto view = result.view();
        check(view.mask_count == 2 && view.proposal_count == 2 &&
                  view.iou_count + view.confidence_count + view.stability_count == 2,
              "automatic proposals allow any one real score kind without invented alternatives");
    }
    auto required_only = load("point_required_only").task<trtmc::ImagePointsToMasks>().run(points);
    check(required_only.view().iou_count == 2 && required_only.view().box_count == 0 &&
              required_only.view().low_res_count == 0,
          "point quality is required but independent auxiliary boxes/priors remain optional");
    rejects(
        [&] { (void)load("wrong_optional_count").task<trtmc::ImagePointsToMasks>().run(points); },
        TRTMC_INTERNAL_ERROR, "nonempty optional metadata must still have N entries");
    auto empty = load("empty_contract_masks");
    check(
        empty.task<trtmc::ImagePointsToMasks>().run(points).view().mask_count == 0 &&
            empty.task<trtmc::ImageBoxToMasks>().run(box).view().mask_count == 0 &&
            empty.task<trtmc::ImageMaskToMasks>().run(prior).view().mask_count == 0 &&
            empty.task<trtmc::ImageToMaskProposals>().run({image}).view().mask_count == 0 &&
            empty.task<trtmc::ImageTextToInstanceMasks>().run({image, "none"}).view().mask_count ==
                0 &&
            empty.task<trtmc::ImageBoxExemplarsToInstanceMasks>()
                    .run(exemplars)
                    .view()
                    .mask_count == 0,
        "all six contracts keep genuine N=0 detections valid");
}

void masks_and_geometry(const trtmc::Model& model, const trtmc::ImageInput& image) {
    auto semantic = model.task<trtmc::ImageToSemanticSegmentation>().run({image});
    const auto s = semantic.view();
    check(s.height == 2 && s.width == 3 && s.pixel_count == 6 && s.labels[0] == 255 &&
              s.class_ids.data[1] == 5 && text(s.class_names.data[1]) == "object" &&
              s.has_ignore_label && s.ignore_label == 255 && s.has_background_label &&
              s.background_label == 0,
          "semantic class IDs, vocabulary and ignore/background labels are explicit");
    check(s.score_count == 12 && s.score_kind == TRTMC_SCORE_LOGIT && s.class_scores[0] == -2,
          "class logits are not silently normalized");
    const trtmc::PointPrompt points[] = {{{1.5F, 0.5F}, false}, {{2.5F, 1.5F}, true}};
    auto p = model.task<trtmc::ImagePointsToMasks>().run({image, {points}});
    const auto pv = p.view();
    check(pv.kind == TRTMC_MASK_LOGITS && pv.masks[0] == -2 && pv.mask_count == 2 &&
              pv.predicted_iou[0] == 2 && pv.boxes[0].x_min == 1.5F && pv.low_res_logits[0] == -7 &&
              pv.confidence_count == 0,
          "point polarity/pixel coordinates and unbounded quality/logits retain their roles");
    auto b = model.task<trtmc::ImageBoxToMasks>().run({image, {0.25F, 0.5F, 2.75F, 1.75F}});
    check(b.view().boxes[0].x_max == 2.75F, "box coordinates remain original-image XYXY");
    const float prior[] = {-10, 5};
    auto prior_result = model.task<trtmc::ImageMaskToMasks>().run(
        {image, {{prior}, 1, 2}, {points}, trtmc::PixelBox{0, 0, 2, 1}});
    check(prior_result.view().low_res_height == 1 && prior_result.view().low_res_width == 2 &&
              prior_result.view().low_res_logits[0] == -10 &&
              prior_result.view().predicted_iou[0] == 2 && prior_result.view().boxes[0].x_max == 2,
          "decoder prior logits use their own resolution and retain optional correction prompts");
    auto proposals = model.task<trtmc::ImageToMaskProposals>().run({image});
    const auto m = proposals.view();
    check(m.kind == TRTMC_MASK_BINARY && m.proposal_count == 2 && m.proposals[0].area == 3 &&
              m.proposals[0].seed_point_count == 2 && m.proposals[0].crop_box.x_max == 3 &&
              m.stability[0] == 0.75F,
          "automatic-mask proposal metadata is complete");
    auto instances = model.task<trtmc::ImageTextToInstanceMasks>().run({image, "person"});
    check(instances.view().kind == TRTMC_MASK_BINARY && instances.view().iou_count == 0 &&
              instances.view().confidence[0] == 0.8F && instances.view().object_ids.data[0] == 100,
          "instance confidence is distinct from predicted IoU");
    const trtmc::BoxExemplar exemplars[] = {{{0, 0, 1, 1}, false},
                                            {{0.25F, 0.5F, 2.75F, 1.75F}, true}};
    auto ex =
        model.task<trtmc::ImageBoxExemplarsToInstanceMasks>().run({image, {exemplars}, "target"});
    check(ex.view().boxes[0].x_min == 0.25F && ex.view().confidence[0] == 0.9F,
          "positive/negative exemplars and additional text are distinct typed inputs");
    float right_pixels[18];
    std::fill(std::begin(right_pixels), std::end(right_pixels), 0.25F);
    auto disparity = model.task<trtmc::StereoImagesToDisparity>().run(
        {image, trtmc::ImageInput({right_pixels}, 2, 3)});
    check(disparity.view().disparity.rows == 2 && disparity.view().disparity.columns == 3 &&
              disparity.view().disparity.data[0] == 0.5F,
          "stereo left/right order and original pixel grid are preserved");
    auto geometry = model.task<trtmc::ImageToMetricGeometry>().run({image}, {{"fov_x", 42.0}});
    const auto g = geometry.view();
    check(g.point_value_count == 18 && g.pixel_count == 6 && !g.valid[0] &&
              std::isinf(g.depth[0]) && g.depth[1] == 2 && g.normalized_intrinsics[0] == 0.42F,
          "metric geometry preserves validity, infinity and normalized K without shared math");
    auto boxes = model.task<trtmc::ImageTextToBoxes>().run({image, "cup"});
    check(boxes.view().boxes[0].box.x_max == 2.75F && text(boxes.view().boxes[0].label) == "cup" &&
              !boxes.view().boxes[0].has_confidence,
          "grounding boxes retain semantic labels without invented confidence");
    auto malformed = model.task<trtmc::ImageTextToBoxes>().run({image, "malformed"});
    check(!malformed.view().parse_complete && malformed.view().count == 0 &&
              text(malformed.view().raw_response) == "unparsed output",
          "malformed generation is distinguishable from a valid empty detection");
    auto located = model.task<trtmc::ImageTextToPoints>().run({image, "target"});
    check(located.view().points[0].point.x == 2.5F && located.view().points[0].confidence == 0.75F,
          "grounded points have explicit pixel coordinates");
    rejects([&] { (void)model.task<trtmc::ImagePointsToMasks>().run({image, {}}); },
            TRTMC_INVALID_ARGUMENT, "points route requires points");
    rejects([&] { (void)model.task<trtmc::ImageMaskToMasks>().run({image, {}, {}, {}}); },
            TRTMC_INVALID_ARGUMENT, "prior route requires decoder logits");
}
std::vector<float> identities(std::size_t count) {
    std::vector<float> result(count * 16, 0);
    for (std::size_t n = 0; n < count; ++n)
        for (std::size_t i = 0; i < 4; ++i)
            result[n * 16 + i * 5] = 1;
    return result;
}
void pose(const trtmc::Model& model) {
    auto task = model.task<trtmc::PoseHypothesesCropsToRefinedPoses>();
    auto candidates = identities(2);
    std::vector<std::pair<trtmc::PoseCropPhase, std::uint64_t>> calls;
    std::vector<float> translations;
    bool checked_busy = false;
    trtmc::PoseCropsProvider provider = [&](const trtmc::PoseCropQuery& query) {
        calls.emplace_back(query.stage, query.iteration);
        translations.push_back(query.poses.values[3]);
        check(model.info().family == "perception_fixture", "callback may query model metadata");
        if (!checked_busy) {
            checked_busy = true;
            rejects(
                [&] {
                    (void)task.run({{{candidates.data(), candidates.size()}, 2},
                                    2.0F,
                                    [](const auto&) { return trtmc::PoseCrops{}; }});
                },
                TRTMC_BUSY, "same-model callback reentry is BUSY instead of deadlocking");
        }
        trtmc::PoseCrops crops;
        crops.count = query.poses.count;
        crops.height = 1;
        crops.width = 1;
        crops.rendered.assign(crops.count * 6, static_cast<float>(query.iteration + 1) * 0.25F);
        crops.observed.assign(crops.count * 6, 0.25F);
        if (query.stage == trtmc::PoseCropPhase::Scoring)
            for (std::size_t n = 0; n < crops.count; ++n)
                crops.observed[n * 6] = static_cast<float>(n + 1) * 0.25F;
        return crops;
    };
    trtmc::PoseHypothesesCropsToRefinedPosesRequest request{
        {{candidates.data(), candidates.size()}, 2}, 2.0F, provider};
    auto result = task.run(request);
    const auto view = result.view();
    check(calls.size() == 3 && calls[0].second == 0 && calls[1].second == 1 &&
              calls[2].first == trtmc::PoseCropPhase::Scoring && calls[2].second == 2,
          "callbacks run in serial refinement/scoring order");
    check(translations == std::vector<float>({0, 0.5F, 1.5F}) &&
              view.refined_poses.values[3] == 1.5F && view.best_index == 1 && view.scores[1] == 2 &&
              view.refinement_ms == 1.5 && view.scoring_ms == 0.5,
          "updated poses and raw scorer values cross the callback boundary");
    request.crops = [](const auto&) -> trtmc::PoseCrops {
        throw trtmc::Error(TRTMC_INVALID_CONFIG, "caller crop failure");
    };
    rejects([&] { (void)task.run(request); }, TRTMC_INVALID_CONFIG,
            "callback exception preserves error status");
    request.crops = provider;
    check(task.run(request).view().best_index == 1,
          "failed callback releases model execution ownership");
    request.crops = {};
    rejects([&] { (void)task.run(request); }, TRTMC_INVALID_ARGUMENT,
            "missing crop provider is rejected");
    std::size_t refinement_calls = 0, scoring_calls = 0;
    trtmc::PoseCropsProvider single_provider = [&](const trtmc::PoseCropQuery& query) {
        query.stage == trtmc::PoseCropPhase::Scoring ? ++scoring_calls : ++refinement_calls;
        trtmc::PoseCrops crops;
        crops.count = query.poses.count;
        crops.height = crops.width = 1;
        crops.rendered.assign(crops.count * 6, 0.25F);
        crops.observed.assign(crops.count * 6, 0.25F);
        return crops;
    };
    auto unscored = task.run({{{candidates.data(), 16}, 1}, 2.0F, single_provider},
                             {{"score_hypotheses", false}});
    check(unscored.view().score_count == 0 && unscored.view().best_index == 0 &&
              refinement_calls == 2 && scoring_calls == 0 && unscored.view().scoring_ms == 0,
          "one unscored hypothesis is valid without shared forced or repeated scoring");
    rejects(
        [&] {
            (void)task.run({{{candidates.data(), candidates.size()}, 2}, 2.0F, single_provider},
                           {{"score_hypotheses", false}});
        },
        TRTMC_INVALID_CONFIG, "multi-hypothesis scoring request policy stays family-owned");
}
void rgbd(const trtmc::Model& model, const trtmc::ImageInput& image) {
    const float vertices[] = {0.25F, 0, 0, 1, 0, 0, 0, 1, 0}, depth[] = {0, 2, 2, 2, 2, 2},
                uv[] = {0.125F, 0, 1, 0, 0, 1}, colors[] = {0.5F, 0, 0, 1, 0, 0, 0, 1, 0};
    const std::uint32_t indices[] = {0, 1, 2};
    const std::uint8_t mask[] = {0, 1, 1, 1, 1, 1};
    auto symmetry = identities(1);
    trtmc::TriangleMeshInput mesh;
    mesh.vertices = {{vertices}, 3, 3};
    mesh.triangles = {indices};
    mesh.symmetries = {{symmetry.data(), symmetry.size()}, 1};
    trtmc::RgbdMeshMaskToObjectPoseRequest request{
        image, {{depth}, 2, 3}, {mask}, {10, 0, 1.5F, 0, 10, 1, 0, 0, 1}, mesh};
    auto task = model.task<trtmc::RgbdMeshMaskToObjectPose>();
    auto pose = task.run(request);
    check(pose.view().object_to_camera[3] == 0.25F && pose.view().object_to_camera[7] == 1 &&
              pose.view().object_to_camera[11] == 2 && !pose.view().has_score,
          "RGBD route retains original mesh frame, meter depth and explicit symmetries");
    request.mesh.appearance = trtmc::MeshAppearance::VertexRgb;
    request.mesh.vertex_rgb = {{colors}, 3, 3};
    check(task.run(request).view().score == 0.5F, "vertex appearance reaches the family");
    request.mesh.appearance = trtmc::MeshAppearance::TextureUv;
    request.mesh.vertex_rgb = {};
    request.mesh.uv = {{uv}, 3, 2};
    request.mesh.texture = image;
    check(task.run(request).view().score == 0.875F,
          "texture image and UV are preserved without shared rendering");
    request.mesh.vertex_rgb = {{colors}, 3, 3};
    rejects([&] { (void)task.run(request); }, TRTMC_INVALID_ARGUMENT,
            "conflicting appearance inputs are rejected");
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        const auto full = root / "perception-cpp-all.bundle",
                   restricted = root / "perception-cpp-pose.bundle",
                   broken = root / "perception-cpp-broken.bundle";
        bundle(full, "all");
        bundle(restricted, "pose_only");
        bundle(broken, "broken_masks");
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        batch_boxes(root, options);
        auto model = trtmc::Model::load(full.string(), options);
        float pixels[18];
        std::fill(std::begin(pixels), std::end(pixels), 0.75F);
        const trtmc::ImageInput image({pixels}, 2, 3);
        const auto missing_iou = root / "perception-cpp-missing-point-iou.bundle";
        bundle(missing_iou, "missing_point_iou");
        auto incomplete = trtmc::Model::load(missing_iou.string(), options);
        const trtmc::PointPrompt missing_point{{1, 1}, true};
        rejects(
            [&] {
                (void)incomplete.task<trtmc::ImagePointsToMasks>().run(
                    {image, {&missing_point, 1}});
            },
            TRTMC_INTERNAL_ERROR,
            "nonempty point masks cannot omit their required learned quality");
        check(model.tasks().size() == 13,
              "all thirteen perception contracts are explicitly advertised by fixture");
        masks_and_geometry(model, image);
        required_mask_outputs(root, options, image);
        pose(model);
        const auto missing_scores = root / "perception-cpp-missing-pose-scores.bundle";
        bundle(missing_scores, "example_pose_missing_score");
        auto incomplete_pose = trtmc::Model::load(missing_scores.string(), options);
        auto hypotheses = identities(2);
        rejects(
            [&] {
                (void)incomplete_pose.task<trtmc::PoseHypothesesCropsToRefinedPoses>().run(
                    {{{hypotheses.data(), hypotheses.size()}, 2},
                     1.0F,
                     [](const trtmc::PoseCropQuery& query) {
                         trtmc::PoseCrops crops;
                         crops.count = query.poses.count;
                         crops.height = crops.width = 1;
                         crops.rendered.assign(crops.count * 6, 0.25F);
                         crops.observed.assign(crops.count * 6, 0.25F);
                         return crops;
                     }});
            },
            TRTMC_INTERNAL_ERROR,
            "multiple returned hypotheses cannot omit scores despite a valid best index");
        rgbd(model, image);
        auto subset = trtmc::Model::load(restricted.string(), options);
        check(!subset.supports<trtmc::ImagePointsToMasks>() &&
                  subset.supports<trtmc::PoseHypothesesCropsToRefinedPoses>(),
              "loaded availability is narrower than C++ inheritance");
        rejects(
            [&] {
                (void)trtmc::Model::load(broken.string(), options)
                    .task<trtmc::ImageToMaskProposals>()
                    .run({image});
            },
            TRTMC_INTERNAL_ERROR, "incorrect family mask shape is rejected");
        auto retained = [&] {
            auto owner = trtmc::Model::load(full.string(), options);
            return owner.task<trtmc::ImageToMaskProposals>().run({image});
        }();
        auto moved = std::move(retained);
        check(retained.view().mask_count == 0 && moved.view().proposals[0].seed_points[0].x == 0.5F,
              "result move retains nested proposal metadata after model scope");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        ++failures;
    }
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
