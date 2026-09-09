/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/video.h"

#include "api_internal.h"
#include "trtmc/video.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace trtmc::api {
internal::VideoView video_input(const trtmc_video_view_v1&, std::vector<internal::ImageView>&);
namespace {

void output_check(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
Span<const double> timeline(trtmc_f64_view source, size_t count) {
    const auto times = checked_span(source.data, source.size);
    require(times.empty() || times.size() == count, "timeline count differs from sample count");
    for (size_t i = 0; i < times.size(); ++i)
        require(std::isfinite(times[i]) && (i == 0 || times[i] >= times[i - 1]),
                "timeline must contain finite nondecreasing times");
    return times;
}
void output_timeline(const std::vector<double>& times, size_t count) {
    try {
        (void)timeline({times.data(), times.size()}, count);
    } catch (const ApiFailure&) {
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "invalid family timeline"};
    }
}
std::optional<std::string_view> optional_text(uint32_t present, trtmc_string_view value) {
    require(present <= 1, "text presence must be zero or one");
    return present ? std::optional<std::string_view>{string_view(value)} : std::nullopt;
}
std::optional<double> optional_number(uint32_t present, double value) {
    require(present <= 1, "number presence must be zero or one");
    require(!present || std::isfinite(value), "present number must be finite");
    return present ? std::optional<double>{value} : std::nullopt;
}

// Owns only converted arrays of borrowed views, until the synchronous family call ends.
class VideoInputs {
  public:
    Span<const internal::ImageView> images(const trtmc_image_input_v1* data, uint64_t count) {
        const auto source = checked_span(data, count);
        auto& result = image_lists_.emplace_back();
        result.reserve(source.size());
        for (const auto& image : source)
            result.push_back(image_input(image));
        return {result.data(), result.size()};
    }
    internal::VideoView video(const trtmc_video_view_v1& source) {
        return video_input(source, image_lists_.emplace_back());
    }
    internal::VideoMaskView mask(const trtmc_video_mask_view_v1& source,
                                 const internal::VideoView& clip) {
        require(source.frames == clip.frames.size() && source.height && source.width,
                "video mask must align with all source frames");
        checked_size(source.height, source.width);
        const auto plane = source.height * source.width;
        checked_size(source.frames, plane);
        const auto count = source.frames * plane;
        require(source.count == count, "video mask count does not match FHW shape");
        for (const auto& frame : clip.frames)
            require(frame.height == source.height && frame.width == source.width,
                    "video mask spatial dimensions differ from its source");
        return {checked_span(source.values, count), source.frames, source.height, source.width};
    }
    internal::FloatMatrixView intrinsics(const trtmc_f32_matrix_view_v1& input,
                                         uint64_t frames = 0) {
        auto result = matrix_input(input);
        require(result.columns == 9 && (!frames || result.rows == 1 || result.rows == frames),
                "intrinsics require one or one-per-frame flattened 3x3 matrix");
        return result;
    }
    internal::CameraTrajectoryView camera(const trtmc_camera_trajectory_view_v1& source) {
        auto poses = matrix_input(source.camera_to_world);
        require(poses.columns == 16, "camera-to-world matrices must have 16 row-major elements");
        return {poses, timeline(source.timestamps_seconds, poses.rows),
                string_view(source.coordinate_convention), string_view(source.translation_units)};
    }
    Span<const std::string_view> strings(trtmc_strings_view source) {
        auto input = checked_span(source.data, source.size);
        auto& values = string_lists_.emplace_back();
        values.reserve(input.size());
        for (const auto value : input)
            values.push_back(string_view(value));
        return {values.data(), values.size()};
    }
    internal::ActionSchemaView schema(const trtmc_action_schema_view_v1& source, uint64_t columns) {
        auto domain = string_view(source.domain);
        require(!domain.empty(), "action domain must be explicit");
        auto names = strings(source.component_names);
        auto units = strings(source.units);
        require((names.empty() || names.size() == columns) &&
                    (units.empty() || units.size() == columns),
                "action names or units differ from action dimension");
        return {domain, names, units, string_view(source.coordinate_frame),
                string_view(source.normalization)};
    }
    internal::ActionSequenceView actions(const trtmc_action_sequence_view_v1& source) {
        auto values = matrix_input(source.values);
        const auto ranges = checked_span(source.frame_spans, source.frame_span_count);
        require(ranges.empty() || ranges.size() == values.rows,
                "action frame-span count differs from step count");
        auto& converted = frame_spans_.emplace_back();
        converted.reserve(ranges.size());
        for (const auto& range : ranges) {
            require(range.begin < range.end, "action frame span must be nonempty and increasing");
            converted.push_back({range.begin, range.end});
        }
        return {values,
                schema(source.schema, values.columns),
                timeline(source.timestamps_seconds, values.rows),
                {converted.data(), converted.size()}};
    }
    internal::ActionOutputSpec action_spec(const trtmc_action_output_spec_v1& source) {
        require(source.dimensions > 0, "output action dimensions must be positive");
        return {schema(source.schema, source.dimensions), source.dimensions};
    }
    Span<const internal::TimedVideoAnchor> anchors(const trtmc_timed_video_anchor_v1* data,
                                                   uint64_t count) {
        const auto source = checked_span(data, count);
        require(!source.empty(), "timed conditioning requires at least one anchor");
        anchors_.reserve(source.size());
        for (const auto& anchor : source) {
            internal::TimedVideoAnchor item;
            switch (anchor.kind) {
            case TRTMC_VIDEO_ANCHOR_IMAGE:
                item.content = image_input(anchor.content.image);
                break;
            case TRTMC_VIDEO_ANCHOR_CLIP:
                item.content = video(anchor.content.clip);
                break;
            default:
                throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unknown temporal anchor kind"};
            }
            item.output_start_frame = anchor.output_start_frame;
            item.strength = optional_number(anchor.has_strength, anchor.strength);
            anchors_.push_back(std::move(item));
        }
        return {anchors_.data(), anchors_.size()};
    }
    Span<const internal::VideoReferenceItem> references(const trtmc_video_reference_item_v1* data,
                                                        uint64_t count) {
        const auto source = checked_span(data, count);
        require(!source.empty(), "reference generation requires references");
        references_.reserve(source.size());
        bool visual = false;
        for (const auto& reference : source) {
            switch (reference.kind) {
            case TRTMC_VIDEO_REFERENCE_IMAGE:
                references_.emplace_back(image_input(reference.content.image));
                visual = true;
                break;
            case TRTMC_VIDEO_REFERENCE_CLIP: {
                const auto& input = reference.content.clip;
                require(input.has_soundtrack <= 1, "soundtrack presence must be zero or one");
                auto start = optional_number(input.has_audio_start, input.audio_start_seconds);
                require(input.has_soundtrack || !start, "audio start requires a soundtrack");
                internal::VideoReference item{video(input.video), std::nullopt, start};
                if (input.has_soundtrack)
                    item.soundtrack = audio_view(input.soundtrack);
                references_.emplace_back(std::move(item));
                visual = true;
                break;
            }
            case TRTMC_VIDEO_REFERENCE_AUDIO:
                references_.emplace_back(audio_view(reference.content.audio));
                break;
            default:
                throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unknown semantic reference kind"};
            }
        }
        require(visual, "this reference Task requires at least one image or video reference");
        return {references_.data(), references_.size()};
    }

    internal::TextToVideoRequest convert(const trtmc_text_to_video_request_v1& input) {
        return {string_view(input.prompt),
                checked_span(input.initial_latents.data, input.initial_latents.size)};
    }
    internal::InitialImageTextToVideoRequest
    convert(const trtmc_initial_image_text_to_video_request_v1& input) {

        return {image_input(input.initial_image), string_view(input.prompt)};
    }
    internal::BoundaryFramesTextToVideoRequest
    convert(const trtmc_boundary_frames_text_to_video_request_v1& input) {

        return {image_input(input.first_frame), image_input(input.last_frame),
                string_view(input.prompt)};
    }
    internal::TimedFramesTextToVideoRequest
    convert(const trtmc_timed_frames_text_to_video_request_v1& input) {

        return {anchors(input.anchors, input.anchor_count), string_view(input.prompt)};
    }
    internal::VideoTextToVideoEditRequest
    convert(const trtmc_video_text_to_video_edit_request_v1& input) {

        return {video(input.source), string_view(input.prompt)};
    }
    internal::MaskedVideoTextToVideoRequest
    convert(const trtmc_masked_video_text_to_video_request_v1& input) {
        auto clip = video(input.source);
        return {clip, mask(input.mask, clip), string_view(input.prompt)};
    }
    internal::MaskedVideoReferenceImagesTextToVideoRequest
    convert(const trtmc_masked_video_reference_images_text_to_video_request_v1& input) {
        auto clip = video(input.source);
        auto refs = images(input.references, input.reference_count);
        require(!refs.empty(), "masked reference edit requires reference images");
        return {clip, mask(input.mask, clip), refs, string_view(input.prompt)};
    }
    internal::ImageTextActionToVideoRequest
    convert(const trtmc_image_text_action_to_video_request_v1& input) {
        auto dialect = optional_text(input.has_dialect, input.dialect);
        require(!dialect || !dialect->empty(), "present action dialect must be nonempty");
        return {image_input(input.initial_image),
                string_view(input.prompt),
                string_view(input.action_dsl),
                intrinsics(input.intrinsics),
                dialect,
                checked_span(input.initial_latents.data, input.initial_latents.size)};
    }
    internal::ImageTextCameraTrajectoryToVideoRequest
    convert(const trtmc_image_text_camera_trajectory_to_video_request_v1& input) {
        auto trajectory = camera(input.camera);
        return {image_input(input.initial_image), string_view(input.prompt), trajectory,
                intrinsics(input.intrinsics, trajectory.camera_to_world.rows),
                checked_span(input.initial_latents.data, input.initial_latents.size)};
    }
    internal::VideoTextToFutureVideoRequest
    convert(const trtmc_video_text_to_future_video_request_v1& input) {

        return {video(input.history), string_view(input.prompt)};
    }
    internal::ImageActionToFutureVideoRequest
    convert(const trtmc_image_action_to_future_video_request_v1& input) {

        return {image_input(input.observation), actions(input.actions),
                optional_text(input.has_prompt, input.prompt)};
    }
    internal::VideoActionToFutureVideoRequest
    convert(const trtmc_video_action_to_future_video_request_v1& input) {

        return {video(input.history), actions(input.actions),
                optional_text(input.has_prompt, input.prompt)};
    }
    internal::VideoToActionSequenceRequest
    convert(const trtmc_video_to_action_sequence_request_v1& input) {

        return {video(input.observations), action_spec(input.action_spec),
                optional_text(input.has_prompt, input.prompt)};
    }
    internal::ImageToActionAndVideoRequest
    convert(const trtmc_image_to_action_and_video_request_v1& input) {

        return {image_input(input.observation), action_spec(input.action_spec),
                optional_text(input.has_prompt, input.prompt)};
    }
    internal::VideoToActionAndVideoRequest
    convert(const trtmc_video_to_action_and_video_request_v1& input) {

        return {video(input.history), action_spec(input.action_spec),
                optional_text(input.has_prompt, input.prompt)};
    }
    internal::TextToAudioVideoRequest convert(const trtmc_text_to_audio_video_request_v1& input) {

        return {string_view(input.prompt)};
    }
    internal::InitialImageTextToAudioVideoRequest
    convert(const trtmc_initial_image_text_to_audio_video_request_v1& input) {

        return {image_input(input.initial_image), string_view(input.prompt)};
    }
    internal::LastImageTextToAudioVideoRequest
    convert(const trtmc_last_image_text_to_audio_video_request_v1& input) {

        return {image_input(input.last_image), string_view(input.prompt)};
    }
    internal::BoundaryFramesTextToAudioVideoRequest
    convert(const trtmc_boundary_frames_text_to_audio_video_request_v1& input) {

        return {image_input(input.first_frame), image_input(input.last_frame),
                string_view(input.prompt)};
    }
    internal::ReferencesTextToAudioVideoRequest
    convert(const trtmc_references_text_to_audio_video_request_v1& input) {

        return {references(input.references, input.reference_count), string_view(input.prompt)};
    }

  private:
    std::vector<std::vector<internal::ImageView>> image_lists_;
    std::vector<std::vector<std::string_view>> string_lists_;
    std::vector<std::vector<internal::ActionFrameSpan>> frame_spans_;
    std::vector<internal::TimedVideoAnchor> anchors_;
    std::vector<internal::VideoReferenceItem> references_;
};

struct VideoStorage final : ResultStorage {
    explicit VideoStorage(internal::VideoResult value) : result(std::move(value)) {
        const auto& pixels = result.frames;
        if (internal::is_worker_completion(pixels)) {
            output_check(result.conditioned_prefix_frames == 0 && result.timestamps_seconds.empty(),
                         "worker completion must not contain decoded-frame metadata");
            view = {nullptr, 0, {}, 0, result.setup_ms, result.inference_ms};
            return;
        }
        output_check(pixels.num_frames > 0 && pixels.height > 0 && pixels.width > 0 &&
                         pixels.channels > 0,
                     "family returned an invalid video shape");
        size_t frame_elements;
        try {
            checked_size(pixels.height, pixels.width);
            const auto plane = uint64_t(pixels.height) * pixels.width;
            checked_size(plane, pixels.channels);
            frame_elements = checked_size(plane * pixels.channels, sizeof(float));
            checked_size(pixels.num_frames, frame_elements);
            checked_size(uint64_t(pixels.num_frames) * frame_elements, sizeof(float));
        } catch (const ApiFailure&) {
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family video dimensions overflow"};
        }
        output_check(pixels.pixels.size() == uint64_t(pixels.num_frames) * frame_elements,
                     "family video storage does not match THWC shape");
        output_check(result.conditioned_prefix_frames <= uint64_t(pixels.num_frames),
                     "family context prefix exceeds video frame count");
        output_timeline(result.timestamps_seconds, pixels.num_frames);
        frames.reserve(pixels.num_frames);
        for (int32_t i = 0; i < pixels.num_frames; ++i)
            frames.push_back({pixels.pixels.data() + size_t(i) * frame_elements, frame_elements,
                              uint32_t(pixels.height), uint32_t(pixels.width),
                              uint32_t(pixels.channels)});
        view = {frames.data(),
                frames.size(),
                {result.timestamps_seconds.data(), result.timestamps_seconds.size()},
                result.conditioned_prefix_frames,
                result.setup_ms,
                result.inference_ms};
    }
    internal::VideoResult result;
    std::vector<trtmc_image_result_view_v1> frames;
    trtmc_video_result_view_v1 view{};
};
std::vector<trtmc_string_view> string_views(const std::vector<std::string>& values) {
    std::vector<trtmc_string_view> out;
    out.reserve(values.size());
    for (const auto& value : values)
        out.push_back(borrowed_string(value));
    return out;
}
} // namespace
ActionSequenceStorage::ActionSequenceStorage(internal::ActionSequenceResult value)
    : result(std::move(value)), names(string_views(result.schema.component_names)),
      units(string_views(result.schema.units)) {
    const auto matrix = matrix_result_view(result.values);
    output_check(!result.schema.domain.empty(), "family omitted action domain");
    output_check((names.empty() || names.size() == matrix.columns) &&
                     (units.empty() || units.size() == matrix.columns),
                 "family action schema dimension mismatch");
    output_timeline(result.timestamps_seconds, matrix.rows);
    output_check(result.frame_spans.empty() || result.frame_spans.size() == matrix.rows,
                 "family action frame-span count mismatch");
    for (const auto& range : result.frame_spans) {
        output_check(range.begin < range.end, "family action frame span is invalid");
        spans.push_back({range.begin, range.end});
    }
    view = {matrix,
            {borrowed_string(result.schema.domain),
             {names.data(), names.size()},
             {units.data(), units.size()},
             borrowed_string(result.schema.coordinate_frame),
             borrowed_string(result.schema.normalization)},
            {result.timestamps_seconds.data(), result.timestamps_seconds.size()},
            spans.data(),
            spans.size()};
}
namespace {
void frame_association(Span<const internal::ActionFrameSpan> spans, uint64_t frame_count) {
    for (const auto& span : spans)
        output_check(span.end <= frame_count, "action association exceeds the related video");
}
void future_suffix(const internal::VideoResult& video) {
    output_check(video.frames.num_frames > 0 &&
                     video.conditioned_prefix_frames < uint64_t(video.frames.num_frames),
                 "future Task must return a nonempty predicted suffix after context");
}
struct ActionVideoStorage final : ResultStorage {
    explicit ActionVideoStorage(internal::ActionVideoResult result)
        : actions(std::move(result.actions)), video(std::move(result.video)) {
        future_suffix(video.result);
        frame_association({actions.result.frame_spans.data(), actions.result.frame_spans.size()},
                          video.frames.size());
        view = {actions.view, video.view};
    }
    ActionSequenceStorage actions;
    VideoStorage video;
    trtmc_action_video_result_view_v1 view{};
};
struct AudioVideoStorage final : ResultStorage {
    explicit AudioVideoStorage(internal::AudioVideoResult result)
        : video(std::move(result.video)), audio(std::move(result.audio)) {
        output_check(
            result.audio_start_seconds && std::isfinite(*result.audio_start_seconds) &&
                !video.result.timestamps_seconds.empty() && !audio.result.samples.empty(),
            "synchronized AV requires video times, actual audio and an audio clock origin");
        view.video = video.view;
        fill_audio_result_view(audio, &view.audio);
        view.audio_start_seconds = *result.audio_start_seconds;
    }
    VideoStorage video;
    AudioResultStorage audio;
    trtmc_audio_video_result_view_v1 view{};
};

template <class Storage, class View>
trtmc_status TRTMC_CALL result_view(const trtmc_result* result, View* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "video-group result view output is null");
        *out = require_result<Storage>(result).view;
    });
}
template <class Request, class Result>
void validate_result(const Request&, const Result&) {}
void validate_result(const internal::VideoTextToFutureVideoRequest&,
                     const internal::VideoResult& result) {
    future_suffix(result);
}
void validate_result(const internal::ImageActionToFutureVideoRequest& input,
                     const internal::VideoResult& result) {
    future_suffix(result);
    frame_association(input.actions.frame_spans, uint64_t(result.frames.num_frames));
}
void validate_result(const internal::VideoActionToFutureVideoRequest& input,
                     const internal::VideoResult& result) {
    future_suffix(result);
    frame_association(input.actions.frame_spans, uint64_t(result.frames.num_frames));
}
void validate_result(const internal::ImageTextCameraTrajectoryToVideoRequest& input,
                     const internal::VideoResult& result) {
    output_check(result.frames.num_frames > 0 &&
                     uint64_t(result.frames.num_frames) == input.camera.camera_to_world.rows,
                 "camera trajectory must describe every output frame");
}
void action_spec_result(const internal::ActionOutputSpec& spec,
                        const internal::ActionSequenceResult& result) {
    output_check(result.values.columns == spec.dimensions &&
                     result.schema.domain == spec.schema.domain,
                 "family action result differs from the requested domain or dimension");
}
void validate_result(const internal::VideoToActionSequenceRequest& input,
                     const internal::ActionSequenceResult& result) {
    action_spec_result(input.action_spec, result);
    frame_association({result.frame_spans.data(), result.frame_spans.size()},
                      input.observations.frames.size());
}
void validate_result(const internal::ImageToActionAndVideoRequest& input,
                     const internal::ActionVideoResult& result) {
    action_spec_result(input.action_spec, result.actions);
}
void validate_result(const internal::VideoToActionAndVideoRequest& input,
                     const internal::ActionVideoResult& result) {
    action_spec_result(input.action_spec, result.actions);
}
template <class Interface, class Request, class Storage>
trtmc_status TRTMC_CALL run(trtmc_model* model, const Request* input,
                            const trtmc_config_view_v1* config, trtmc_result** out,
                            trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input && out, "video-group request or result output is null");
        VideoInputs storage;
        const auto request = storage.convert(*input);
        const ConvertedConfig options(config);
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        auto result = family.run(request, options.view());
        validate_result(request, result);
        *out = make_result<Storage>(std::move(result));
    });
}

const trtmc_text_to_video_api_v1 text_to_video_api = {
    {1, 0, sizeof(trtmc_text_to_video_api_v1)},
    run<internal::ITextToVideo, trtmc_text_to_video_request_v1, VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_text_to_video_api_v1, header) == 0);

const trtmc_initial_image_text_to_video_api_v1 initial_image_text_to_video_api = {
    {1, 0, sizeof(trtmc_initial_image_text_to_video_api_v1)},
    run<internal::IInitialImageTextToVideo, trtmc_initial_image_text_to_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_initial_image_text_to_video_api_v1, header) == 0);

const trtmc_boundary_frames_text_to_video_api_v1 boundary_frames_text_to_video_api = {
    {1, 0, sizeof(trtmc_boundary_frames_text_to_video_api_v1)},
    run<internal::IBoundaryFramesTextToVideo, trtmc_boundary_frames_text_to_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_boundary_frames_text_to_video_api_v1, header) == 0);

const trtmc_timed_frames_text_to_video_api_v1 timed_frames_text_to_video_api = {
    {1, 0, sizeof(trtmc_timed_frames_text_to_video_api_v1)},
    run<internal::ITimedFramesTextToVideo, trtmc_timed_frames_text_to_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_timed_frames_text_to_video_api_v1, header) == 0);

const trtmc_video_text_to_video_edit_api_v1 video_text_to_video_edit_api = {
    {1, 0, sizeof(trtmc_video_text_to_video_edit_api_v1)},
    run<internal::IVideoTextToVideoEdit, trtmc_video_text_to_video_edit_request_v1, VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_video_text_to_video_edit_api_v1, header) == 0);

const trtmc_masked_video_text_to_video_api_v1 masked_video_text_to_video_api = {
    {1, 0, sizeof(trtmc_masked_video_text_to_video_api_v1)},
    run<internal::IMaskedVideoTextToVideo, trtmc_masked_video_text_to_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_masked_video_text_to_video_api_v1, header) == 0);

const trtmc_masked_video_reference_images_text_to_video_api_v1
    masked_video_reference_images_text_to_video_api = {
        {1, 0, sizeof(trtmc_masked_video_reference_images_text_to_video_api_v1)},
        run<internal::IMaskedVideoReferenceImagesTextToVideo,
            trtmc_masked_video_reference_images_text_to_video_request_v1, VideoStorage>,
        result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_masked_video_reference_images_text_to_video_api_v1, header) == 0);

const trtmc_image_text_action_to_video_api_v1 image_text_action_to_video_api = {
    {1, 0, sizeof(trtmc_image_text_action_to_video_api_v1)},
    run<internal::IImageTextActionToVideo, trtmc_image_text_action_to_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_image_text_action_to_video_api_v1, header) == 0);

const trtmc_image_text_camera_trajectory_to_video_api_v1 image_text_camera_trajectory_to_video_api =
    {{1, 0, sizeof(trtmc_image_text_camera_trajectory_to_video_api_v1)},
     run<internal::IImageTextCameraTrajectoryToVideo,
         trtmc_image_text_camera_trajectory_to_video_request_v1, VideoStorage>,
     result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_image_text_camera_trajectory_to_video_api_v1, header) == 0);

const trtmc_video_text_to_future_video_api_v1 video_text_to_future_video_api = {
    {1, 0, sizeof(trtmc_video_text_to_future_video_api_v1)},
    run<internal::IVideoTextToFutureVideo, trtmc_video_text_to_future_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_video_text_to_future_video_api_v1, header) == 0);

const trtmc_image_action_to_future_video_api_v1 image_action_to_future_video_api = {
    {1, 0, sizeof(trtmc_image_action_to_future_video_api_v1)},
    run<internal::IImageActionToFutureVideo, trtmc_image_action_to_future_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_image_action_to_future_video_api_v1, header) == 0);

const trtmc_video_action_to_future_video_api_v1 video_action_to_future_video_api = {
    {1, 0, sizeof(trtmc_video_action_to_future_video_api_v1)},
    run<internal::IVideoActionToFutureVideo, trtmc_video_action_to_future_video_request_v1,
        VideoStorage>,
    result_view<VideoStorage, trtmc_video_result_view_v1>};
static_assert(offsetof(trtmc_video_action_to_future_video_api_v1, header) == 0);

const trtmc_video_to_action_sequence_api_v1 video_to_action_sequence_api = {
    {1, 0, sizeof(trtmc_video_to_action_sequence_api_v1)},
    run<internal::IVideoToActionSequence, trtmc_video_to_action_sequence_request_v1,
        ActionSequenceStorage>,
    result_view<ActionSequenceStorage, trtmc_action_sequence_view_v1>};
static_assert(offsetof(trtmc_video_to_action_sequence_api_v1, header) == 0);

const trtmc_image_to_action_and_video_api_v1 image_to_action_and_video_api = {
    {1, 0, sizeof(trtmc_image_to_action_and_video_api_v1)},
    run<internal::IImageToActionAndVideo, trtmc_image_to_action_and_video_request_v1,
        ActionVideoStorage>,
    result_view<ActionVideoStorage, trtmc_action_video_result_view_v1>};
static_assert(offsetof(trtmc_image_to_action_and_video_api_v1, header) == 0);

const trtmc_video_to_action_and_video_api_v1 video_to_action_and_video_api = {
    {1, 0, sizeof(trtmc_video_to_action_and_video_api_v1)},
    run<internal::IVideoToActionAndVideo, trtmc_video_to_action_and_video_request_v1,
        ActionVideoStorage>,
    result_view<ActionVideoStorage, trtmc_action_video_result_view_v1>};
static_assert(offsetof(trtmc_video_to_action_and_video_api_v1, header) == 0);

const trtmc_text_to_audio_video_api_v1 text_to_audio_video_api = {
    {1, 0, sizeof(trtmc_text_to_audio_video_api_v1)},
    run<internal::ITextToAudioVideo, trtmc_text_to_audio_video_request_v1, AudioVideoStorage>,
    result_view<AudioVideoStorage, trtmc_audio_video_result_view_v1>};
static_assert(offsetof(trtmc_text_to_audio_video_api_v1, header) == 0);

const trtmc_initial_image_text_to_audio_video_api_v1 initial_image_text_to_audio_video_api = {
    {1, 0, sizeof(trtmc_initial_image_text_to_audio_video_api_v1)},
    run<internal::IInitialImageTextToAudioVideo, trtmc_initial_image_text_to_audio_video_request_v1,
        AudioVideoStorage>,
    result_view<AudioVideoStorage, trtmc_audio_video_result_view_v1>};
static_assert(offsetof(trtmc_initial_image_text_to_audio_video_api_v1, header) == 0);

const trtmc_last_image_text_to_audio_video_api_v1 last_image_text_to_audio_video_api = {
    {1, 0, sizeof(trtmc_last_image_text_to_audio_video_api_v1)},
    run<internal::ILastImageTextToAudioVideo, trtmc_last_image_text_to_audio_video_request_v1,
        AudioVideoStorage>,
    result_view<AudioVideoStorage, trtmc_audio_video_result_view_v1>};
static_assert(offsetof(trtmc_last_image_text_to_audio_video_api_v1, header) == 0);

const trtmc_boundary_frames_text_to_audio_video_api_v1 boundary_frames_text_to_audio_video_api = {
    {1, 0, sizeof(trtmc_boundary_frames_text_to_audio_video_api_v1)},
    run<internal::IBoundaryFramesTextToAudioVideo,
        trtmc_boundary_frames_text_to_audio_video_request_v1, AudioVideoStorage>,
    result_view<AudioVideoStorage, trtmc_audio_video_result_view_v1>};
static_assert(offsetof(trtmc_boundary_frames_text_to_audio_video_api_v1, header) == 0);

const trtmc_references_text_to_audio_video_api_v1 references_text_to_audio_video_api = {
    {1, 0, sizeof(trtmc_references_text_to_audio_video_api_v1)},
    run<internal::IReferencesTextToAudioVideo, trtmc_references_text_to_audio_video_request_v1,
        AudioVideoStorage>,
    result_view<AudioVideoStorage, trtmc_audio_video_result_view_v1>};
static_assert(offsetof(trtmc_references_text_to_audio_video_api_v1, header) == 0);

template <class Storage>
struct VideoBatchStorage final : ResultStorage {
    explicit VideoBatchStorage(std::vector<std::unique_ptr<Storage>> values)
        : items(std::move(values)) {}
    std::vector<std::unique_ptr<Storage>> items;
};
template <class Storage>
trtmc_status TRTMC_CALL batch_count(const trtmc_result* input, uint64_t* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out, "video batch count output is null");
        *out = require_result<VideoBatchStorage<Storage>>(input).items.size();
    });
}
template <class Storage, class View>
trtmc_status TRTMC_CALL batch_item(const trtmc_result* input, uint64_t index, View* out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "video batch item output is null");
        const auto& items = require_result<VideoBatchStorage<Storage>>(input).items;
        require(index < items.size(), "video batch index is out of range");
        *out = items[static_cast<size_t>(index)]->view;
    });
}
template <class Interface, class Wire, class Storage>
trtmc_status TRTMC_CALL run_batch(trtmc_model* model, const Wire* input, trtmc_result** out,
                                  trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input && out, "video batch request or result is null");
        const auto supplied = checked_span(input->items, input->count);
        require(!supplied.empty(), "video batch must contain requests");
        std::vector<VideoInputs> conversions(supplied.size());
        std::vector<ConvertedConfig> configs;
        std::vector<typename Interface::Request::Item> items;
        configs.reserve(supplied.size());
        items.reserve(supplied.size());
        for (size_t i = 0; i < supplied.size(); ++i) {
            configs.emplace_back(&supplied[i].config);
            items.push_back({conversions[i].convert(supplied[i].input), configs.back().view()});
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        auto results = family.run_batch({{items.data(), items.size()}});
        output_check(results.size() == items.size(), "family changed video batch item count");
        std::vector<std::unique_ptr<Storage>> storage;
        for (size_t i = 0; i < items.size(); ++i) {
            validate_result(items[i].input, results[i]);
            storage.push_back(std::make_unique<Storage>(std::move(results[i])));
        }
        *out = make_result<VideoBatchStorage<Storage>>(std::move(storage));
    });
}
const trtmc_batch_text_to_video_api_v1 batch_text_to_video_api = {
    {1, 0, sizeof(trtmc_batch_text_to_video_api_v1)},
    run_batch<internal::IBatchTextToVideo, trtmc_batch_text_to_video_request_v1, VideoStorage>,
    batch_count<VideoStorage>,
    batch_item<VideoStorage, trtmc_video_result_view_v1>};

const trtmc_batch_initial_image_text_to_video_api_v1 batch_initial_image_text_to_video_api = {
    {1, 0, sizeof(trtmc_batch_initial_image_text_to_video_api_v1)},
    run_batch<internal::IBatchInitialImageTextToVideo,
              trtmc_batch_initial_image_text_to_video_request_v1, VideoStorage>,
    batch_count<VideoStorage>,
    batch_item<VideoStorage, trtmc_video_result_view_v1>};

const trtmc_batch_video_text_to_future_video_api_v1 batch_video_text_to_future_video_api = {
    {1, 0, sizeof(trtmc_batch_video_text_to_future_video_api_v1)},
    run_batch<internal::IBatchVideoTextToFutureVideo,
              trtmc_batch_video_text_to_future_video_request_v1, VideoStorage>,
    batch_count<VideoStorage>,
    batch_item<VideoStorage, trtmc_video_result_view_v1>};

const trtmc_batch_image_action_to_future_video_api_v1 batch_image_action_to_future_video_api = {
    {1, 0, sizeof(trtmc_batch_image_action_to_future_video_api_v1)},
    run_batch<internal::IBatchImageActionToFutureVideo,
              trtmc_batch_image_action_to_future_video_request_v1, VideoStorage>,
    batch_count<VideoStorage>,
    batch_item<VideoStorage, trtmc_video_result_view_v1>};

const trtmc_batch_video_to_action_sequence_api_v1 batch_video_to_action_sequence_api = {
    {1, 0, sizeof(trtmc_batch_video_to_action_sequence_api_v1)},
    run_batch<internal::IBatchVideoToActionSequence,
              trtmc_batch_video_to_action_sequence_request_v1, ActionSequenceStorage>,
    batch_count<ActionSequenceStorage>,
    batch_item<ActionSequenceStorage, trtmc_action_sequence_view_v1>};

const trtmc_batch_image_to_action_and_video_api_v1 batch_image_to_action_and_video_api = {
    {1, 0, sizeof(trtmc_batch_image_to_action_and_video_api_v1)},
    run_batch<internal::IBatchImageToActionAndVideo,
              trtmc_batch_image_to_action_and_video_request_v1, ActionVideoStorage>,
    batch_count<ActionVideoStorage>,
    batch_item<ActionVideoStorage, trtmc_action_video_result_view_v1>};

const trtmc_batch_text_to_audio_video_api_v1 batch_text_to_audio_video_api = {
    {1, 0, sizeof(trtmc_batch_text_to_audio_video_api_v1)},
    run_batch<internal::IBatchTextToAudioVideo, trtmc_batch_text_to_audio_video_request_v1,
              AudioVideoStorage>,
    batch_count<AudioVideoStorage>,
    batch_item<AudioVideoStorage, trtmc_audio_video_result_view_v1>};

const trtmc_batch_initial_image_text_to_audio_video_api_v1
    batch_initial_image_text_to_audio_video_api = {
        {1, 0, sizeof(trtmc_batch_initial_image_text_to_audio_video_api_v1)},
        run_batch<internal::IBatchInitialImageTextToAudioVideo,
                  trtmc_batch_initial_image_text_to_audio_video_request_v1, AudioVideoStorage>,
        batch_count<AudioVideoStorage>,
        batch_item<AudioVideoStorage, trtmc_audio_video_result_view_v1>};

const TaskBinding bindings[] = {
    {internal::IBatchInitialImageTextToVideo::kTask, 1, 0,
     &batch_initial_image_text_to_video_api.header,
     implements<internal::IBatchInitialImageTextToVideo>},
    {internal::IBatchVideoTextToFutureVideo::kTask, 1, 0,
     &batch_video_text_to_future_video_api.header,
     implements<internal::IBatchVideoTextToFutureVideo>},
    {internal::IBatchImageActionToFutureVideo::kTask, 1, 0,
     &batch_image_action_to_future_video_api.header,
     implements<internal::IBatchImageActionToFutureVideo>},
    {internal::IBatchVideoToActionSequence::kTask, 1, 0, &batch_video_to_action_sequence_api.header,
     implements<internal::IBatchVideoToActionSequence>},
    {internal::IBatchImageToActionAndVideo::kTask, 1, 0,
     &batch_image_to_action_and_video_api.header,
     implements<internal::IBatchImageToActionAndVideo>},
    {internal::IBatchTextToAudioVideo::kTask, 1, 0, &batch_text_to_audio_video_api.header,
     implements<internal::IBatchTextToAudioVideo>},
    {internal::IBatchInitialImageTextToAudioVideo::kTask, 1, 0,
     &batch_initial_image_text_to_audio_video_api.header,
     implements<internal::IBatchInitialImageTextToAudioVideo>},
    {internal::IBatchTextToVideo::kTask, 1, 0, &batch_text_to_video_api.header,
     implements<internal::IBatchTextToVideo>},
    {internal::ITextToVideo::kTask, 1, 0, &text_to_video_api.header,
     implements<internal::ITextToVideo>},
    {internal::IInitialImageTextToVideo::kTask, 1, 0, &initial_image_text_to_video_api.header,
     implements<internal::IInitialImageTextToVideo>},
    {internal::IBoundaryFramesTextToVideo::kTask, 1, 0, &boundary_frames_text_to_video_api.header,
     implements<internal::IBoundaryFramesTextToVideo>},
    {internal::ITimedFramesTextToVideo::kTask, 1, 0, &timed_frames_text_to_video_api.header,
     implements<internal::ITimedFramesTextToVideo>},
    {internal::IVideoTextToVideoEdit::kTask, 1, 0, &video_text_to_video_edit_api.header,
     implements<internal::IVideoTextToVideoEdit>},
    {internal::IMaskedVideoTextToVideo::kTask, 1, 0, &masked_video_text_to_video_api.header,
     implements<internal::IMaskedVideoTextToVideo>},
    {internal::IMaskedVideoReferenceImagesTextToVideo::kTask, 1, 0,
     &masked_video_reference_images_text_to_video_api.header,
     implements<internal::IMaskedVideoReferenceImagesTextToVideo>},
    {internal::IImageTextActionToVideo::kTask, 1, 0, &image_text_action_to_video_api.header,
     implements<internal::IImageTextActionToVideo>},
    {internal::IImageTextCameraTrajectoryToVideo::kTask, 1, 0,
     &image_text_camera_trajectory_to_video_api.header,
     implements<internal::IImageTextCameraTrajectoryToVideo>},
    {internal::IVideoTextToFutureVideo::kTask, 1, 0, &video_text_to_future_video_api.header,
     implements<internal::IVideoTextToFutureVideo>},
    {internal::IImageActionToFutureVideo::kTask, 1, 0, &image_action_to_future_video_api.header,
     implements<internal::IImageActionToFutureVideo>},
    {internal::IVideoActionToFutureVideo::kTask, 1, 0, &video_action_to_future_video_api.header,
     implements<internal::IVideoActionToFutureVideo>},
    {internal::IVideoToActionSequence::kTask, 1, 0, &video_to_action_sequence_api.header,
     implements<internal::IVideoToActionSequence>},
    {internal::IImageToActionAndVideo::kTask, 1, 0, &image_to_action_and_video_api.header,
     implements<internal::IImageToActionAndVideo>},
    {internal::IVideoToActionAndVideo::kTask, 1, 0, &video_to_action_and_video_api.header,
     implements<internal::IVideoToActionAndVideo>},
    {internal::ITextToAudioVideo::kTask, 1, 0, &text_to_audio_video_api.header,
     implements<internal::ITextToAudioVideo>},
    {internal::IInitialImageTextToAudioVideo::kTask, 1, 0,
     &initial_image_text_to_audio_video_api.header,
     implements<internal::IInitialImageTextToAudioVideo>},
    {internal::ILastImageTextToAudioVideo::kTask, 1, 0, &last_image_text_to_audio_video_api.header,
     implements<internal::ILastImageTextToAudioVideo>},
    {internal::IBoundaryFramesTextToAudioVideo::kTask, 1, 0,
     &boundary_frames_text_to_audio_video_api.header,
     implements<internal::IBoundaryFramesTextToAudioVideo>},
    {internal::IReferencesTextToAudioVideo::kTask, 1, 0, &references_text_to_audio_video_api.header,
     implements<internal::IReferencesTextToAudioVideo>},
};

} // namespace

internal::VideoView video_input(const trtmc_video_view_v1& source,
                                std::vector<internal::ImageView>& backing) {
    const auto input = checked_span(source.frames, source.frame_count);
    require(!input.empty(), "video input requires at least one frame");
    backing.clear();
    backing.reserve(input.size());
    for (const auto& frame : input)
        backing.push_back(image_input(frame));
    return {{backing.data(), backing.size()}, timeline(source.timestamps_seconds, backing.size())};
}

Span<const TaskBinding> video_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
