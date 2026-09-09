/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/tracking.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures = 0;
void check(bool ok, const char* name) {
    if (!ok) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}
template <class F>
void rejects(F fn, trtmc_status expected, const char* name) {
    bool failed = false;
    try {
        fn();
    } catch (const trtmc::Error& e) {
        failed = e.code() == expected;
    }
    check(failed, name);
}
void bundle(const std::filesystem::path& path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"tracking_fixture\",\"task\":\"" +
                               std::string(mode) + "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(magic), 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        const auto all = root / "tracking-cpp-device.bundle",
                   host = root / "tracking-cpp-host.bundle";
        bundle(all, "device");
        bundle(host, "host_only");
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        auto model = trtmc::Model::load(all.string(), options);
        auto session = model.task<trtmc::FramesToDetectedMaskTracks>().create();
        float pixels[18]{};
        trtmc::VideoInput clip;
        for (int i = 0; i < 5; ++i)
            clip.frames.emplace_back(trtmc::Span<const float>{pixels}, 2, 3);
        auto result = session.segment(clip);
        const auto view = result.view();
        check(view.frame_count == 5 && view.frames[0].element_type == TRTMC_TRACK_UINT8 &&
                  view.frames[0].memory_kind == TRTMC_TRACK_HOST &&
                  view.frames[0].mask_byte_size == 6 && view.frames[0].object_ids.data[0] == 7 &&
                  view.frames[0].class_ids.data[0] == 2,
              "host masks preserve byte storage and separate object/class IDs");
        check(view.frames[1].box_count == 0 && view.detection_count == 1 &&
                  view.initial_detections[0].prompt_box.x_max == 3,
              "initial detection is not reported as a later tracked box");
        check(session.supports_device_masks(), "typed getter exposes device support");
        auto device = session.device_masks();
        auto borrowed = device.segment_device(clip);
        check(borrowed.view().frames[0].device_ordinal == 2 &&
                  borrowed.view().frames[0].memory_kind == TRTMC_TRACK_CUDA &&
                  borrowed.view().frames[0].masks != nullptr,
              "device descriptors are exposed without dereferencing them");
        check(session.supports_device_masks() && borrowed.view().frame_count == 5,
              "metadata reads do not invalidate device views");
        auto invalid = clip;
        invalid.frames.pop_back();
        rejects([&] { (void)session.segment(invalid); }, TRTMC_INVALID_ARGUMENT,
                "fixed-clip family rejects partial clips");
        rejects([&] { (void)borrowed.view(); }, TRTMC_INVALID_ARGUMENT,
                "delegated failing run invalidates prior device view");
        auto latest = device.segment_device(clip);
        session.close();
        rejects([&] { (void)latest.view(); }, TRTMC_INVALID_ARGUMENT,
                "close invalidates borrowed device output");
        check(result.view().frames[0].mask_byte_size == 6, "host result remains owned after close");
        auto other = trtmc::Model::load(host.string(), options)
                         .task<trtmc::FramesToDetectedMaskTracks>()
                         .create();
        check(!other.supports_device_masks(), "getter hides device mode for host-only session");
        rejects([&] { (void)other.device_masks(); }, TRTMC_UNSUPPORTED,
                "unsupported device path is not synthesized");
        auto text_session = model.task<trtmc::FramesTextToMaskTracks>().create();
        auto text_result = text_session.segment(clip, "bird");
        check(text_result.view().frame_count == 5 &&
                  text_result.view().frames[0].element_type == TRTMC_TRACK_FLOAT32 &&
                  text_result.view().frames[0].mask_kind == TRTMC_MASK_BINARY,
              "text clip preserves float binary masks");
        check(text_result.view().frames[0].removed_object_ids.data[0] == 19 &&
                  text_result.view().frames[0].suppressed_object_ids.data[0] == 23 &&
                  text_result.view().frames[0].detection_scores[0] == 0.875F &&
                  text_result.view().frames[0].tracker_scores[0] == 0.625F,
              "text tracking retains removal, suppression and distinct scores");
        rejects([&] { (void)text_session.segment(clip, "cat"); }, TRTMC_INVALID_ARGUMENT,
                "family validates text; shared layer does not replace it");
        text_session.close();
        auto prompt_session = model.task<trtmc::PromptFrameTextToMaskTracks>().create("bird");
        rejects([&] { (void)prompt_session.continue_borrowed(text_result, clip); },
                TRTMC_INVALID_ARGUMENT, "unrelated result cannot become prompt snapshot");
        auto prompt = prompt_session.accept_prompt_frame(clip.frames[0]);
        check(prompt.view().frame_count == 1 &&
                  static_cast<const float*>(prompt.view().frames[0].masks)[0] == 1.0F,
              "prompt snapshot is immediately human inspectable");
        auto continuation = prompt_session.continue_borrowed(prompt, clip);
        check(continuation.view().frame_count == 5 &&
                  static_cast<const float*>(continuation.view().frames[0].masks)[0] == 0.0F &&
                  static_cast<const float*>(prompt.view().frames[0].masks)[0] == 1.0F,
              "family consolidated frame zero replaces rather than prepends borrowed snapshot");
        rejects([&] { (void)prompt_session.accept_prompt_frame(clip.frames[0]); },
                TRTMC_INVALID_ARGUMENT, "family rejects a second prompt frame");
        rejects([&] { (void)prompt_session.continue_borrowed(prompt, clip); },
                TRTMC_INVALID_ARGUMENT, "family rejects a second continuation");
        prompt_session.close();
        auto new_prompt_session = model.task<trtmc::PromptFrameTextToMaskTracks>().create("bird");
        auto new_prompt = new_prompt_session.accept_prompt_frame(clip.frames[0]);
        rejects([&] { (void)new_prompt_session.continue_borrowed(prompt, clip); },
                TRTMC_INVALID_ARGUMENT, "prompt snapshot provenance prevents session mixing");
        check(new_prompt_session.continue_borrowed(new_prompt, clip).view().frame_count == 5,
              "provenance error does not call or consume family session");
        new_prompt_session.close();
        check(prompt.view().frame_count == 1 && continuation.view().frame_count == 5,
              "owned prompt and continuation remain valid after close");
        pixels[0] = 0.25F;
        auto context = model.task<trtmc::InteractiveImageMasks>().create(clip.frames[0]);
        pixels[0] = 0.75F;
        const trtmc::PointPrompt foreground[]{{{1, 1}, true}}, background[]{{{1, 1}, false}};
        auto first = context.points(trtmc::Span<const trtmc::PointPrompt>{foreground});
        auto second = context.points(trtmc::Span<const trtmc::PointPrompt>{background});
        check(first.view().masks[0] == 0.25F && second.view().masks[0] == -0.25F &&
                  first.view().predicted_iou[0] == 1 && second.view().predicted_iou[0] == 1,
              "family encodes once, owns retained input and receives point polarity");
        check(context.supports_points() && context.supports_box() && context.supports_prior(),
              "image editor support projects typed getters only");
        const float low_res[]{-3, 2, 1, 4};
        trtmc::ImagePriorPrompt correction{{trtmc::Span<const float>{low_res}, 2, 2}, {}, {}};
        auto corrected = context.prior(correction);
        check(corrected.view().masks[0] == -3 && context.box({0, 0, 3, 2}).view().masks[0] == 3,
              "decoder logits are not thresholded or confused with binary image masks");
        context.close();
        check(first.view().masks[0] == 0.25F, "image result snapshot survives context close");
        other.close();
        auto limited = trtmc::Model::load(host.string(), options)
                           .task<trtmc::InteractiveImageMasks>()
                           .create(clip.frames[0]);
        check(!limited.supports_prior(), "per-context typed getter hides unsupported prior");
        rejects([&] { (void)limited.prior(correction); }, TRTMC_UNSUPPORTED,
                "no synthesized image-prior capability");
        auto started = model.task<trtmc::FramesPointsToMaskTracks>().create(
            clip, {2, 41, trtmc::Span<const trtmc::PointPrompt>{foreground},
                   trtmc::PointUpdate::Replace});
        auto& tracker = started.session;
        check(started.initial.view().frames[0].frame_index == 2 &&
                  started.initial.view().frames[0].object_ids.data[0] == 41,
              "typed points factory preserves prompt frame and object identity");
        auto append = tracker.update_points(
            {2, 41, trtmc::Span<const trtmc::PointPrompt>{background}, trtmc::PointUpdate::Append});
        check(append.view().frames[0].tracker_scores[0] == 2 &&
                  static_cast<const float*>(append.view().frames[0].masks)[0] == 0,
              "append and point polarity reach family state");
        auto replaced =
            tracker.update_points({2, 41, trtmc::Span<const trtmc::PointPrompt>{foreground},
                                   trtmc::PointUpdate::Replace});
        check(replaced.view().frames[0].tracker_scores[0] == 1, "replace is distinct from append");
        auto boxed = tracker.update_box(
            {1, 41, {0, 0, 3, 2}, trtmc::Span<const trtmc::PointPrompt>{background}});
        check(boxed.view().frames[0].frame_index == 1 &&
                  boxed.view().frames[0].object_ids.data[0] == 41,
              "mixed box editing reuses one family session");
        std::uint8_t binary[]{0, 1, 1, 0, 1, 0};
        auto masked = tracker.update_mask({1, 41, trtmc::Span<const std::uint8_t>{binary}, 2, 3});
        check(static_cast<const float*>(masked.view().frames[0].masks)[0] == 0,
              "binary image prompt retains its distinct type");
        binary[0] = 255;
        rejects(
            [&] {
                (void)tracker.update_mask({1, 41, trtmc::Span<const std::uint8_t>{binary}, 2, 3});
            },
            TRTMC_INVALID_ARGUMENT, "binary mask transport rejects nonbinary values");
        binary[0] = 0;
        auto traversal = tracker.propagate({4, 3, trtmc::PropagationDirection::Backward});
        rejects([&] { tracker.reset(); }, TRTMC_BUSY, "live traversal excludes parent mutation");
        check(traversal.next()->view().frames[0].frame_index == 4 &&
                  traversal.next()->view().frames[0].frame_index == 3 &&
                  traversal.next()->view().frames[0].frame_index == 2 && !traversal.next(),
              "native reverse traversal preserves original frame IDs and END");
        traversal.close();
        auto removed = tracker.remove_object(41);
        check(removed.view().frames[0].object_ids.size == 0 &&
                  removed.view().frames[0].removed_object_ids.data[0] == 41,
              "object removal retains explicit removed identity and zero masks");
        tracker.reset();
        rejects([&] { (void)tracker.propagate({0, 2, trtmc::PropagationDirection::Forward}); },
                TRTMC_INVALID_ARGUMENT, "family requires re-prompting after reset");
        auto concept_result = tracker.replace_text({0, "bird"});
        auto exemplar = tracker.replace_exemplar({0, {{0, 0, 3, 2}, false}});
        check(concept_result.view().frames[0].object_ids.data[0] !=
                      exemplar.view().frames[0].object_ids.data[0] &&
                  static_cast<const float*>(exemplar.view().frames[0].masks)[0] == 0,
              "text/exemplar replacement resets native identity scope and retains polarity");
        auto cancelled = tracker.propagate({0, 3, trtmc::PropagationDirection::Forward});
        cancelled.cancel();
        check(!cancelled.next(), "cancel ends traversal without shared propagation");
        cancelled.close();
        auto retained = tracker.propagate({0, 2, trtmc::PropagationDirection::Forward});
        tracker.close();
        check(retained.next()->view().frames[0].frame_index == 0,
              "traversal retains native resources after parent handle close");
        retained.close();
        check(started.initial.view().frames[0].object_ids.data[0] == 41,
              "interactive initial snapshot survives edits/reset/release");
        auto box_start =
            model.task<trtmc::FramesBoxToMaskTracks>().create(clip, {1, 51, {0, 0, 3, 2}, {}});
        check(box_start.initial.view().frames[0].object_ids.data[0] == 51,
              "box factory required role");
        box_start.session.close();
        auto mask_start = model.task<trtmc::FramesMaskToMaskTracks>().create(
            clip, {2, 61, trtmc::Span<const std::uint8_t>{binary}, 2, 3});
        check(mask_start.initial.view().frames[0].frame_index == 2, "mask factory required role");
        mask_start.session.close();
        auto text_start =
            model.task<trtmc::InteractiveFramesTextToMaskTracks>().create(clip, {3, "bird"});
        check(text_start.initial.view().frames[0].frame_index == 3,
              "interactive text distinct from native fixed clip");
        text_start.session.close();
        auto exemplar_start = model.task<trtmc::FramesBoxExemplarToMaskTracks>().create(
            clip, {4, {{0, 0, 3, 2}, true}});
        check(exemplar_start.initial.view().frames[0].frame_index == 4,
              "singular exemplar factory required role");
        exemplar_start.session.close();
        limited.close();
        auto constrained =
            trtmc::Model::load(host.string(), options)
                .task<trtmc::FramesPointsToMaskTracks>()
                .create(clip, {0, 9, trtmc::Span<const trtmc::PointPrompt>{foreground},
                               trtmc::PointUpdate::Replace});
        check(constrained.session.supports_points() && !constrained.session.supports_text(),
              "per-session typed getter is sole editor availability");
        rejects([&] { (void)constrained.session.replace_text({0, "bird"}); }, TRTMC_UNSUPPORTED,
                "limited native session is not given invented text editing");
        auto pose_session = model.task<trtmc::CropPoseTracking>().create();
        int crop_calls = 0;
        trtmc::PoseCropsProvider crops = [&](const trtmc::PoseCropQuery& query) {
            ++crop_calls;
            rejects([&] { pose_session.reset(); }, TRTMC_BUSY,
                    "callback same-session reentry returns BUSY");
            rejects([&] { (void)model.task<trtmc::CropPoseTracking>().create(); }, TRTMC_BUSY,
                    "callback same-model reentry returns BUSY without model mutex deadlock");
            trtmc::PoseCrops result;
            result.count = query.poses.count;
            result.height = 1;
            result.width = 1;
            result.rendered.assign(static_cast<std::size_t>(result.count) * 6, 0.1F);
            result.observed.assign(static_cast<std::size_t>(result.count) * 6, 0.25F);
            return result;
        };
        rejects([&] { (void)pose_session.track(crops); }, TRTMC_INVALID_ARGUMENT,
                "crop pose track requires family initialization");
        std::array<float, 32> candidates{};
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 4; ++j)
                candidates[i * 16 + j * 5] = 1;
        candidates[3] = 0.5F;
        candidates[19] = 0.75F;
        trtmc::PoseHypothesesCropsToRefinedPosesRequest pose_request{
            {{candidates.data(), candidates.size()}, 2}, 2, crops};
        auto initialized = pose_session.initialize(pose_request);
        check(initialized.view().best_index == 1 && crop_calls == 2,
              "pose initializer retains family selection and both crop stages");
        auto tracked = pose_session.track(crops);
        check(tracked.view().refined_poses.count == 1 &&
                  std::fabs(tracked.view().refined_poses.values[3] - 1.15F) < 1e-6F,
              "family stores selected pose and mesh diameter for later tracking");
        const trtmc::PoseCropsProvider fail = [](const trtmc::PoseCropQuery&) -> trtmc::PoseCrops {
            throw trtmc::Error(TRTMC_INVALID_CONFIG, "per-call crop failure");
        };
        rejects([&] { (void)pose_session.track(fail); }, TRTMC_INVALID_CONFIG,
                "pose callback status survives the C boundary");
        check(pose_session.track(crops).view().refined_poses.count == 1,
              "failed callback does not strand session operation lock");
        pose_session.reset();
        rejects([&] { (void)pose_session.track(crops); }, TRTMC_INVALID_ARGUMENT,
                "pose reset clears native saved pose");
        pose_session.close();
        check(initialized.view().refined_poses.count == 2,
              "pose snapshot outlives reset and close");
        float vertices[]{2, 0, 0, 0, 1, 0, 0, 0, 1};
        const std::uint32_t triangles[]{0, 1, 2};
        trtmc::TriangleMeshInput mesh;
        mesh.vertices = {trtmc::Span<const float>{vertices}, 3, 3};
        mesh.triangles = trtmc::Span<const std::uint32_t>{triangles};
        std::array<float, 16> original_pose{};
        for (std::size_t i = 0; i < 4; ++i)
            original_pose[i * 5] = 1;
        original_pose[3] = 7;
        original_pose[11] = 3;
        auto rgbd =
            model.task<trtmc::RgbdInitializedPoseToTrackedPose>().create(mesh, original_pose);
        vertices[0] = 99;
        const float depth[]{0.5F, 1, 1, 1, 1, 1};
        trtmc::RgbdObservation observation{clip.frames[0],
                                           {trtmc::Span<const float>{depth}, 2, 3},
                                           {100, 0, 1, 0, 100, 1, 0, 0, 1}};
        auto object = rgbd.track(observation);
        check(object.view().object_to_camera[3] == 7 &&
                  object.view().object_to_camera[11] == 3.5F && object.view().score == 2,
              "RGBD family owns retained mesh, pixel intrinsics and original-object meter pose");
        original_pose[3] = 9;
        original_pose[11] = 4;
        rgbd.reset(original_pose);
        check(rgbd.track(observation).view().object_to_camera[11] == 4.5F,
              "RGBD reset accepts an explicit replacement pose");
        rgbd.close();
        check(object.view().object_to_camera[3] == 7, "owned object pose survives session reset");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        ++failures;
    }
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
