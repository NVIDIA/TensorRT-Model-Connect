/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/video.hpp"

#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures = 0;
bool near(double a, double b) {
    return std::abs(a - b) < 1e-6;
}
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
template <class Function>
void rejects(Function function, trtmc_status expected, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == expected;
    }
    check(rejected, message);
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header =
        "{\"format\":1,\"family\":\"video_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), header.size());
    out.write("PLAN", 4);
}
void marker(const trtmc::VideoGenerationResult& result, int expected, const char* label) {
    const auto frames = result.frames();
    check(!frames.empty() && frames[0].height == 1 && frames[0].width == 1 &&
              frames[0].channels == 3 && near(frames[0].pixels[0], expected / 100.0),
          label);
}
void av_marker(const trtmc::AudioVideoGenerationResult& result, int expected, const char* label) {
    check(!result.frames().empty() && near(result.frames()[0].pixels[0], expected / 100.0) &&
              result.audio().sample_rate == 10 && result.audio().channels == 2 &&
              result.audio().samples.size() == 8 && near(result.audio_start_seconds(), -0.05),
          label);
}

void exercise_batches(const trtmc::Model& model, const std::filesystem::path& root) {
    void* library =
        dlopen((root / "libtrtmc_model_video_fixture.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library)
        throw std::runtime_error("video fixture probes unavailable");
    using Counter = int (*)();
    auto batch_calls = reinterpret_cast<Counter>(dlsym(library, "trtmc_test_video_batch_calls"));
    auto single_calls = reinterpret_cast<Counter>(dlsym(library, "trtmc_test_video_single_calls"));
    auto executed =
        reinterpret_cast<Counter>(dlsym(library, "trtmc_test_video_batch_items_executed"));
    if (!batch_calls || !single_calls || !executed)
        throw std::runtime_error("video native batch counter missing");
    const auto before = batch_calls(), before_single = single_calls();
    const float pa[]{0.25F, 0, 0}, pb[]{0.5F, 0, 0}, pc[]{0.75F, 0, 0};
    const trtmc::ImageInput a({pa, 3}, 1, 1), b({pb, 3}, 1, 1), c({pc, 3}, 1, 1);
    const trtmc::VideoInput ca{{a, b}, {1, 1.2}}, cb{{b, a, c}, {2, 2.1, 2.5}};
    const trtmc::ActionSchema sa{"fixture.motion", {"x", "y"}, {"m", "m"}, "world", "raw"};
    const trtmc::ActionSchema sb{"other.motion", {"u", "v"}, {"rad", "rad"}, "robot", "normalized"};
    const float aa[]{0.25F, 0.5F, 0.75F, 1}, ab[]{0.75F, 0.1F, 0.2F, 0.3F};
    const trtmc::ActionSequenceView as{{{aa, 4}, 2, 2}, sa, {0.1, 0.3}, {{1, 2}, {2, 3}}};
    const trtmc::ActionSequenceView bs{{{ab, 4}, 2, 2}, sb, {0.2, 0.4}, {{1, 2}, {2, 3}}};
    const trtmc::ActionOutputSpec oa{sa, 2}, ob{sb, 2};
    auto initial = model.task<trtmc::BatchInitialImageTextToVideo>().run(
        {{{{a, "a"}}, {{b, "bb"}, {{"gain", 2.0}}}}});
    check(initial.size() == 2 && near(initial.at(0).frames[0].pixels[0], 0.22) &&
              near(initial.at(1).frames[0].pixels[0], 0.44) &&
              near(initial.at(0).frames[0].pixels[1], 0.26) &&
              near(initial.at(1).frames[0].pixels[1], 0.52),
          "batch initial image, prompt and per-row config stay distinct");
    auto future =
        model.task<trtmc::BatchVideoTextToFutureVideo>().run({{{{ca, "a"}}, {{cb, "bb"}}}});
    check(future.at(0).conditioned_prefix_frames == 1 &&
              near(future.at(0).frames[0].pixels[0], 0.5) &&
              near(future.at(1).frames[0].pixels[0], 0.75) &&
              near(future.at(0).timestamps_seconds.data[0], 1.2) &&
              near(future.at(1).timestamps_seconds.data[0], 2.5) &&
              near(future.at(1).frames[1].pixels[1], 0.032),
          "batch history retains input frame order/time and predicted suffix metadata");
    auto forward = model.task<trtmc::BatchImageActionToFutureVideo>().run(
        {{{{a, as}}, {{b, bs, std::string{}}}}});
    check(near(forward.at(0).frames[0].pixels[0], 0.25) &&
              near(forward.at(1).frames[1].pixels[1], 0.75 + sb.domain.size() / 1000.0 + 0.02),
          "batch forward action values/domain and absent-versus-present prompt survive");
    auto inverse = model.task<trtmc::BatchVideoToActionSequence>().run(
        {{{{ca, oa}}, {{cb, ob, std::string{}}, {{"gain", 2.0}}}}});
    const auto second = inverse.at(1);
    check(second.values.rows == 2 && second.values.columns == 2 &&
              near(second.values.data[0], 0.5) && near(second.values.data[1], 0.52) &&
              trtmc::detail::string_view(second.schema.domain) == "other.motion" &&
              trtmc::detail::string_view(second.schema.component_names.data[0]) == "u" &&
              trtmc::detail::string_view(second.schema.units.data[0]) == "rad" &&
              trtmc::detail::string_view(second.schema.coordinate_frame) == "robot" &&
              near(second.timestamps_seconds.data[1], 2.5) && second.frame_spans[1].end == 2,
          "batch inverse returns typed action axes/schema/input-frame association");
    auto joint = model.task<trtmc::BatchImageToActionAndVideo>().run(
        {{{{a, oa}, {{"gain", 0.0}}}, {{b, ob, std::string{}}}}});
    check(near(joint.at(0).actions.values.data[0], 0) &&
              near(joint.at(0).video.frames[1].pixels[0], 0) &&
              near(joint.at(1).actions.values.data[0], 0.26) &&
              near(joint.at(1).video.frames[1].pixels[0], 0.26) &&
              joint.at(1).actions.frame_spans[1].end == 3,
          "joint batch action/video is same-evaluation; explicit zero config survives");
    auto av =
        model.task<trtmc::BatchTextToAudioVideo>().run({{{{"a"}}, {{"bb"}, {{"gain", 2.0}}}}});
    check(near(av.at(1).video.frames[0].pixels[0], 0.54) &&
              av.at(0).audio.audio.sample_count == 8 && av.at(1).audio.audio.channels == 2 &&
              near(av.at(1).audio_start_seconds, -0.05),
          "text AV batch preserves paired PCM and actual A/V clock origin");
    auto iav =
        model.task<trtmc::BatchInitialImageTextToAudioVideo>().run({{{{a, "a"}}, {{b, "bbb"}}}});
    check(iav.at(0).audio.audio.sample_count == 2 && iav.at(1).audio.audio.sample_count == 6 &&
              near(iav.at(1).video.frames[0].pixels[1], 0.53),
          "image AV batch retains independent media and variable audio lengths");
    check(batch_calls() == before + 7 && single_calls() == before_single,
          "seven batch routes call seven native batch virtuals and zero single-model calls");
    const auto prior_execution = executed();
    rejects(
        [&] {
            (void)model.task<trtmc::BatchImageToActionAndVideo>().run(
                {{{{a, oa}}, {{b, ob}, {{"unknown", true}}}}});
        },
        TRTMC_INVALID_CONFIG, "native batch rejects invalid later item before any generation");
    check(executed() == prior_execution, "batch preflight has no partial item execution");
    auto moved = std::move(iav);
    check(iav.size() == 0 && moved.size() == 2 && moved.at(1).audio.audio.sample_count == 6,
          "batch owner move keeps nested buffers live");
    dlclose(library);
}

void exercise(const trtmc::Model& model) {
    const float first_pixels[] = {0.25F, 0, 0};
    const float last_pixels[] = {0.5F, 0, 0};
    const float reference_pixels[] = {0.75F, 0, 0};
    const trtmc::ImageInput first({first_pixels, 3}, 1, 1), last({last_pixels, 3}, 1, 1),
        reference({reference_pixels, 3}, 1, 1);
    const trtmc::VideoInput history{{first, last}, {0, 0.1}};
    const float mask_values[] = {0, 1};
    const trtmc::VideoMaskView mask{{mask_values, 2}, 2, 1, 1};
    const auto calibration_values = trtmc::pinhole_intrinsics(2, 3, 0.25F, 0.5F);
    const trtmc::FloatMatrixView calibration{
        {calibration_values.data(), calibration_values.size()}, 1, 9};
    const float poses[] = {1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                           1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const trtmc::CameraTrajectoryView camera{{{poses, 32}, 2, 16}, {0, 0.1}, "", ""};
    const float raw_actions[] = {0.25F, 0.5F, 0.75F, 1};
    const trtmc::ActionSchema schema{"fixture.motion", {"x", "y"}, {}, "", ""};
    const trtmc::ActionSequenceView actions{
        {{raw_actions, 4}, 2, 2}, schema, {0.1, 0.3}, {{1, 2}, {2, 3}}};
    const trtmc::ActionOutputSpec action_spec{schema, 2};

    check(model.tasks().size() == 20, "all 20 video input signatures are discoverable");
    auto text = model.task<trtmc::TextToVideo>().run({"p"});
    marker(text, 1, "text-to-video reaches its own family method");
    check(text.frames().size() == 3 && near(text.timestamps_seconds()[2], 0.3) &&
              near(text.setup_ms(), 0.5) && near(text.inference_ms(), 1.5),
          "ordered frame views, nonuniform timestamps and timings are preserved");
    auto initial = model.task<trtmc::InitialImageTextToVideo>().run({first, "p"});
    marker(initial, 2, "initial-image video Task is independent");
    check(near(initial.frames()[0].pixels[1], 0.25),
          "initial image is passed as its own required operand");
    auto boundary = model.task<trtmc::BoundaryFramesTextToVideo>().run({first, last, "p"});
    marker(boundary, 3, "boundary video Task is independent");
    check(near(boundary.frames()[0].pixels[1], 0.3),
          "first/last roles are not an unordered image list");
    auto timed = model.task<trtmc::TimedFramesTextToVideo>().run(
        {{{first, 2, 0.25}, {history, 0, std::nullopt}}, "p"});
    marker(timed, 4, "timed image and clip anchors use one complete signature");
    check(near(timed.frames()[0].pixels[1], 0.395),
          "anchor kind, start index, strength and default presence survive");
    auto edited = model.task<trtmc::VideoTextToVideoEdit>().run({history, "edit"});
    marker(edited, 5, "source-video editing stays distinct from future prediction");
    check(near(edited.frames()[0].pixels[1], 0.5), "source frame order survives conversion");
    auto masked = model.task<trtmc::MaskedVideoTextToVideo>().run({history, mask, "edit"});
    marker(masked, 6, "masked video edit has a typed aligned mask");
    check(near(masked.frames()[0].pixels[1], 0.3),
          "mask conditioning/generation polarity is not inverted");
    auto referenced = model.task<trtmc::MaskedVideoReferenceImagesTextToVideo>().run(
        {history, mask, {reference}, "edit"});
    marker(referenced, 7, "masked editing keeps references separate from source frames");
    check(near(referenced.frames()[0].pixels[1], 0.425),
          "reference image and mask both reach the family");
    auto scripted =
        model.task<trtmc::ImageTextActionToVideo>().run({first, "p", "w-2", calibration});
    marker(scripted, 8, "camera-motion DSL is a distinct typed input");
    check(near(scripted.frames()[0].pixels[1], 0.133),
          "dialect omission and input-pixel intrinsics survive");
    auto dialect = model.task<trtmc::ImageTextActionToVideo>().run(
        {first, "p", "w-2", calibration, std::string{"fixture.dialect"}}, {{"gain", 2.0}});
    check(near(dialect.frames()[0].pixels[0], 0.16) && near(dialect.frames()[0].pixels[1], 0.233),
          "explicit dialect and family config are independently transported");
    auto trajectory = model.task<trtmc::ImageTextCameraTrajectoryToVideo>().run(
        {first, "p", camera, calibration});
    marker(trajectory, 9, "camera-to-world trajectory is not the DSL or intrinsics");
    check(trajectory.frames().size() == 2 && near(trajectory.frames()[0].pixels[1], 0.24),
          "row-major poses and unspecified camera units remain explicit");
    const float replay_values[] = {0, -0.25F, 0.5F, -0.5F, 0.75F, 1};
    auto replayed = model.task<trtmc::TextToVideo>().run({"p", {replay_values}});
    check(near(replayed.frames()[0].pixels[1], 0.01) &&
              std::equal(replay_values, replay_values + 6, replayed.frames()[1].pixels),
          "video replay preserves every float and prompt without shared layout inference");
    auto replay_dsl = model.task<trtmc::ImageTextActionToVideo>().run(
        {first, "p", "w-2", calibration, std::nullopt, {replay_values}});
    auto replay_camera = model.task<trtmc::ImageTextCameraTrajectoryToVideo>().run(
        {first, "p", camera, calibration, {replay_values}});
    for (const auto* result : {&replay_dsl, &replay_camera}) {
        check(std::equal(first_pixels, first_pixels + 3, result->frames()[0].pixels) &&
                  near(result->frames()[1].pixels[0], -0.25) &&
                  near(result->frames()[1].pixels[1], -0.5) &&
                  near(result->frames()[1].pixels[2], 1),
              "SANA-style family overwrite preserves first-image conditioning during replay");
    }
    rejects([&] { (void)model.task<trtmc::TextToVideo>().run({"p", {replay_values, 3}}); },
            TRTMC_INVALID_ARGUMENT, "video family validates a different latent count than image");
    rejects(
        [&] {
            (void)model.task<trtmc::ImageTextActionToVideo>().run(
                {first, "p", "w-2", calibration, std::nullopt, {replay_values, 5}});
        },
        TRTMC_INVALID_ARGUMENT, "SANA family rejects a wrong replay layout");
    auto future = model.task<trtmc::VideoTextToFutureVideo>().run({history, "predict"});
    check(future.conditioned_prefix_frames() == 1 && near(future.frames()[0].pixels[0], 0.5) &&
              future.predicted_frames().size() == 2 &&
              near(future.predicted_frames()[0].pixels[0], 0.1) &&
              near(future.timestamps_seconds()[0], 0.1),
          "retained context is never reported as predicted future");
    auto forward_image = model.task<trtmc::ImageActionToFutureVideo>().run({first, actions});
    check(near(forward_image.predicted_frames()[0].pixels[0], 0.11) &&
              near(forward_image.predicted_frames()[0].pixels[1], 0.26),
          "image forward dynamics preserves action values and absent prompt");
    auto forward_video =
        model.task<trtmc::VideoActionToFutureVideo>().run({history, actions, std::string{}});
    check(near(forward_video.predicted_frames()[0].pixels[0], 0.12),
          "video forward dynamics is independent of image input");
    auto inverse = model.task<trtmc::VideoToActionSequence>().run({history, action_spec});
    check(inverse.values().rows == 2 && inverse.values().columns == 2 &&
              near(inverse.values().values[0], 0.13) && inverse.domain() == "fixture.motion" &&
              inverse.component_names()[1] == "y" && inverse.units().empty() &&
              inverse.coordinate_frame().empty() && inverse.frame_spans()[1].end == 2,
          "inverse dynamics retains action schema and unspecified physical units");
    auto joint_image = model.task<trtmc::ImageToActionAndVideo>().run({first, action_spec});
    check(near(joint_image.actions().values[0], 0.14) &&
              near(joint_image.predicted_frames()[0].pixels[0], 0.14) &&
              joint_image.action_view().frame_spans[0].begin == 1,
          "image joint action/video result is one atomic prediction with explicit association");
    auto joint_video = model.task<trtmc::VideoToActionAndVideo>().run({history, action_spec});
    check(near(joint_video.actions().values[0], 0.15) &&
              near(joint_video.predicted_frames()[0].pixels[0], 0.15),
          "history joint action/video result uses its own family interface");
    auto av = model.task<trtmc::TextToAudioVideo>().run({"p"});
    av_marker(av, 16, "text AV retains audio and shared clock origin");
    auto av_initial = model.task<trtmc::InitialImageTextToAudioVideo>().run({first, "p"});
    av_marker(av_initial, 17, "initial-image AV is independently callable");
    auto av_last = model.task<trtmc::LastImageTextToAudioVideo>().run({last, "p"});
    av_marker(av_last, 18, "last-image-only AV is not lost in first/last grouping");
    auto av_boundary = model.task<trtmc::BoundaryFramesTextToAudioVideo>().run({first, last, "p"});
    av_marker(av_boundary, 19, "boundary AV keeps both roles");
    const float sound[] = {0.1F, -0.1F};
    const trtmc::AudioView unknown_rate{{sound, 2}, std::nullopt, 1};
    const trtmc::AudioView known_rate{{sound, 2}, 16000, 1};
    const trtmc::VideoReference clip{history, unknown_rate, 0.02};
    auto av_refs = model.task<trtmc::ReferencesTextToAudioVideo>().run(
        {{clip, reference, known_rate}, "refs"});
    av_marker(av_refs, 20, "semantic references produce synchronized AV");
    check(near(av_refs.frames()[0].pixels[1], 0.23),
          "reference order, attached soundtrack, standalone audio and absent rate are preserved");

    auto bad_history = history;
    bad_history.timestamps_seconds = {0};
    rejects([&] { model.task<trtmc::VideoTextToVideoEdit>().run({bad_history, "x"}); },
            TRTMC_INVALID_ARGUMENT, "mismatched video timestamp count fails before family call");
    auto bad_mask = mask;
    bad_mask.frames = 1;
    rejects([&] { model.task<trtmc::MaskedVideoTextToVideo>().run({history, bad_mask, "x"}); },
            TRTMC_INVALID_ARGUMENT, "mask must align with every source frame");
    rejects([&] { model.task<trtmc::ReferencesTextToAudioVideo>().run({{known_rate}, "x"}); },
            TRTMC_INVALID_ARGUMENT,
            "audio-only input does not satisfy this visual-reference contract");
    rejects([&] { model.task<trtmc::TextToVideo>().run({"x"}, {{"gain", "bad"}}); },
            TRTMC_INVALID_CONFIG, "family remains responsible for config types");
    auto moved = std::move(text);
    check(text.frames().empty() && moved.frames().size() == 3,
          "moving video result clears old borrowed views");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        for (const std::string mode :
             {"all", "none", "bad_shape", "bad_future", "bad_av", "bad_actions", "unknown_time",
              "worker", "batch", "batch_bad_count", "batch_bad_shape", "batch_bad_future",
              "batch_bad_av", "batch_bad_actions", "batch_fail", "batch_uniform"})
            bundle(root / ("video_" + mode + ".bundle"), mode);
        auto load = [&](const char* mode) {
            return trtmc::Model::load((root / (std::string("video_") + mode + ".bundle")).string(),
                                      options);
        };
        auto model = load("all");
        exercise(model);
        auto batches = load("batch");
        check(batches.tasks().size() == 8 && batches.supports<trtmc::BatchTextToVideo>() &&
                  !batches.supports<trtmc::TextToVideo>(),
              "batch support is independently declared");
        auto batch =
            batches.task<trtmc::BatchTextToVideo>().run({{{{"a"}}, {{"bb"}, {{"gain", 2.0}}}}});
        check(batch.size() == 2 && batch.at(0).frame_count == 3 && batch.at(1).frame_count == 4 &&
                  near(batch.at(0).frames[0].pixels[0], 0.21) &&
                  near(batch.at(1).frames[0].pixels[0], 0.42) &&
                  near(batch.at(1).frames[0].pixels[1], 0.02),
              "native batch preserves independent prompts, Config and variable video lengths");
        exercise_batches(batches, root);
        rejects([&] { (void)batch.at(2); }, TRTMC_INVALID_ARGUMENT,
                "video batch bounds are checked without inventing an empty item");
        rejects([&] { (void)batches.task<trtmc::BatchTextToVideo>().run({}); },
                TRTMC_INVALID_ARGUMENT, "empty video batch is rejected");
        const float first_replay[]{0, -1, 2, -3, 4, -5};
        const float second_replay[]{1, -2, 3, -4, 5, -6};
        auto replay_batch = batches.task<trtmc::BatchTextToVideo>().run(
            {{{{"a", {first_replay}}}, {{"bb", {second_replay}}}}});
        check(std::equal(first_replay, first_replay + 6, replay_batch.at(0).frames[1].pixels) &&
                  std::equal(second_replay, second_replay + 6, replay_batch.at(1).frames[1].pixels),
              "batch reuses the complete existing request including per-item latent replay");
        rejects(
            [&] {
                (void)batches.task<trtmc::BatchTextToVideo>().run(
                    {{{{"a"}}, {{"bb"}, {{"unknown", true}}}}});
            },
            TRTMC_INVALID_CONFIG, "invalid later-item Config rejects the whole native batch");
        auto participant = load("worker").task<trtmc::TextToVideo>().run({"participate"});
        check(participant.frames().empty() && participant.frames().data() == nullptr &&
                  participant.timestamps_seconds().empty() &&
                  participant.conditioned_prefix_frames() == 0,
              "non-output distributed completion is not decoded media or an inference failure");
        auto none = load("none");
        check(none.tasks().empty() && !none.supports<trtmc::TextToVideo>(),
              "interface inheritance alone does not advertise video support");
        auto bad_shape = load("bad_shape");
        rejects([&] { bad_shape.task<trtmc::TextToVideo>().run({"p"}); }, TRTMC_INTERNAL_ERROR,
                "wrong-sized owned THWC output is rejected");
        const float pixels[] = {0.25F, 0, 0};
        const trtmc::ImageInput image({pixels, 3}, 1, 1);
        const trtmc::VideoInput history{{image, image}, {0, 0.1}};
        for (const char* mode : {"batch_bad_count", "batch_bad_shape", "batch_fail"})
            rejects(
                [&] {
                    (void)load(mode).task<trtmc::BatchTextToVideo>().run({{{{"a"}}, {{"bb"}}}});
                },
                TRTMC_INTERNAL_ERROR,
                "batch execution/count/shape failure returns no partial success");
        rejects(
            [&] {
                (void)load("batch_bad_future")
                    .task<trtmc::BatchVideoTextToFutureVideo>()
                    .run({{{{history, "p"}}, {{history, "p"}}}});
            },
            TRTMC_INTERNAL_ERROR, "batch future reuses complete single-request suffix validation");
        rejects(
            [&] {
                (void)load("batch_bad_av")
                    .task<trtmc::BatchTextToAudioVideo>()
                    .run({{{{"p"}}, {{"q"}}}});
            },
            TRTMC_INTERNAL_ERROR, "batch AV reuses actual PCM and timeline validation");
        rejects(
            [&] {
                (void)load("batch_bad_actions")
                    .task<trtmc::BatchVideoToActionSequence>()
                    .run({{{{history, {{"fixture.motion"}, 2}}},
                           {{history, {{"fixture.motion"}, 2}}}}});
            },
            TRTMC_INTERNAL_ERROR, "batch inverse cannot alter requested action domain");
        rejects(
            [&] {
                (void)load("batch_bad_actions")
                    .task<trtmc::BatchImageToActionAndVideo>()
                    .run(
                        {{{{image, {{"fixture.motion"}, 2}}}, {{image, {{"fixture.motion"}, 2}}}}});
            },
            TRTMC_INTERNAL_ERROR, "batch joint cannot detach action spans from its returned video");
        auto uniform = load("batch_uniform").task<trtmc::BatchTextToVideo>();
        check(uniform.run({{{{"a"}}, {{"b"}, {{"gain", 1.0}}}}}).size() == 2,
              "family batch compatibility compares resolved defaults, not raw Config entries");
        rejects([&] { (void)uniform.run({{{{"a"}}, {{"b"}, {{"gain", 2.0}}}}}); },
                TRTMC_UNSUPPORTED,
                "family rejects unsupported cross-item profile without shared regrouping");
        auto retained_batch = [&] {
            auto temporary = load("batch");
            return temporary.task<trtmc::BatchTextToAudioVideo>().run({{{{"a"}}, {{"bb"}}}});
        }();
        check(retained_batch.size() == 2 && retained_batch.at(1).audio.audio.sample_count == 8,
              "all nested batch results outlive model and request scope");
        auto bad_future = load("bad_future");
        rejects([&] { bad_future.task<trtmc::VideoTextToFutureVideo>().run({history, "p"}); },
                TRTMC_INTERNAL_ERROR, "context-only output cannot pass as a future prediction");
        auto bad_av = load("bad_av");
        rejects([&] { bad_av.task<trtmc::TextToAudioVideo>().run({"p"}); }, TRTMC_INTERNAL_ERROR,
                "synchronized AV cannot omit its audio clock origin");
        auto bad_actions = load("bad_actions");
        rejects(
            [&] {
                bad_actions.task<trtmc::VideoToActionSequence>().run(
                    {history, {{"fixture.motion"}, 2}});
            },
            TRTMC_INTERNAL_ERROR, "action result cannot silently change its requested domain");
        auto retained = [&] {
            auto value = load("unknown_time");
            return value.task<trtmc::TextToVideo>().run({"p"});
        }();
        check(retained.frames().size() == 3 && retained.timestamps_seconds().empty(),
              "owned video remains valid after model scope and no FPS is invented");
    } catch (const std::exception& error) {
        std::cerr << "unexpected: " << error.what() << '\n';
        return 1;
    }
    return failures ? 1 : 0;
}
