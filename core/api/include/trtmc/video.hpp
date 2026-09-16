/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/audio.hpp"
#include "trtmc/core.hpp"
#include "trtmc/image.hpp"
#include "trtmc/matrix.hpp"
#include "trtmc/video.h"

#include <array>

namespace trtmc {

// Convert the conventional (fx, fy, cx, cy) input into the canonical 3x3
// row-major representation. Keep the returned array alive while its view runs.
inline std::array<float, 9> pinhole_intrinsics(float fx, float fy, float cx, float cy) noexcept {
    return {fx, 0, cx, 0, fy, cy, 0, 0, 1};
}

// Pixel/audio data remain borrowed through run; these objects own only their
// small arrays, text and timing metadata.
struct VideoInput {
    std::vector<ImageInput> frames;
    std::vector<double> timestamps_seconds{};
};
struct VideoMaskView {
    Span<const float> values;
    uint64_t frames{0}, height{0}, width{0};
};
struct TimedVideoAnchor {
    std::variant<ImageInput, VideoInput> content;
    uint64_t output_start_frame{0};
    std::optional<double> strength{};
};
struct CameraTrajectoryView {
    FloatMatrixView camera_to_world;
    std::vector<double> timestamps_seconds{};
    std::string coordinate_convention{};
    std::string translation_units{};
};
using ActionFrameSpan = trtmc_action_frame_span_v1;
struct ActionSchema {
    std::string domain;
    std::vector<std::string> component_names{};
    std::vector<std::string> units{};
    std::string coordinate_frame{};
    std::string normalization{};
};
struct ActionSequenceView {
    FloatMatrixView values;
    ActionSchema schema;
    std::vector<double> timestamps_seconds{};
    std::vector<ActionFrameSpan> frame_spans{};
};
struct ActionOutputSpec {
    ActionSchema schema;
    uint64_t dimensions{0};
};
struct VideoReference {
    VideoInput video;
    std::optional<AudioView> soundtrack{};
    std::optional<double> audio_start_seconds{};
};
using VideoReferenceItem = std::variant<ImageInput, VideoReference, AudioView>;

struct TextToVideoRequest {
    std::string prompt;
    Span<const float> initial_latents{}; // Borrowed until run returns; family-defined layout.
};
struct InitialImageTextToVideoRequest {
    ImageInput initial_image;
    std::string prompt;
};
struct BoundaryFramesTextToVideoRequest {
    ImageInput first_frame;
    ImageInput last_frame;
    std::string prompt;
};
struct TimedFramesTextToVideoRequest {
    std::vector<TimedVideoAnchor> anchors;
    std::string prompt;
};
struct VideoTextToVideoEditRequest {
    VideoInput source;
    std::string prompt;
};
struct MaskedVideoTextToVideoRequest {
    VideoInput source;
    VideoMaskView mask;
    std::string prompt;
};
struct MaskedVideoReferenceImagesTextToVideoRequest {
    VideoInput source;
    VideoMaskView mask;
    std::vector<ImageInput> references;
    std::string prompt;
};
struct ImageTextActionToVideoRequest {
    ImageInput initial_image;
    std::string prompt;
    std::string action_dsl;
    FloatMatrixView intrinsics;
    std::optional<std::string> dialect{};
    Span<const float> initial_latents{};
};
struct ImageTextCameraTrajectoryToVideoRequest {
    ImageInput initial_image;
    std::string prompt;
    CameraTrajectoryView camera;
    FloatMatrixView intrinsics;
    Span<const float> initial_latents{};
};
struct VideoTextToFutureVideoRequest {
    VideoInput history;
    std::string prompt;
};
struct ImageActionToFutureVideoRequest {
    ImageInput observation;
    ActionSequenceView actions;
    std::optional<std::string> prompt{};
};
struct VideoActionToFutureVideoRequest {
    VideoInput history;
    ActionSequenceView actions;
    std::optional<std::string> prompt{};
};
struct VideoToActionSequenceRequest {
    VideoInput observations;
    ActionOutputSpec action_spec;
    std::optional<std::string> prompt{};
};
struct ImageToActionAndVideoRequest {
    ImageInput observation;
    ActionOutputSpec action_spec;
    std::optional<std::string> prompt{};
};
struct VideoToActionAndVideoRequest {
    VideoInput history;
    ActionOutputSpec action_spec;
    std::optional<std::string> prompt{};
};
struct TextToAudioVideoRequest {
    std::string prompt;
};
struct InitialImageTextToAudioVideoRequest {
    ImageInput initial_image;
    std::string prompt;
};
struct LastImageTextToAudioVideoRequest {
    ImageInput last_image;
    std::string prompt;
};
struct BoundaryFramesTextToAudioVideoRequest {
    ImageInput first_frame;
    ImageInput last_frame;
    std::string prompt;
};
struct ReferencesTextToAudioVideoRequest {
    std::vector<VideoReferenceItem> references;
    std::string prompt;
};

namespace detail {
template <class Wire>
class VideoResultOwner {
  public:
    VideoResultOwner(std::shared_ptr<ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    VideoResultOwner(const VideoResultOwner&) = delete;
    VideoResultOwner& operator=(const VideoResultOwner&) = delete;
    VideoResultOwner(VideoResultOwner&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    VideoResultOwner& operator=(VideoResultOwner&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    Wire& wire_view() noexcept { return view_; }
    const Wire& wire_view() const noexcept { return view_; }

  protected:
    ResultOwner owner_;
    Wire view_{};
};
inline Span<const trtmc_image_result_view_v1>
video_frames(const trtmc_video_result_view_v1& view) noexcept {
    return {view.frames, static_cast<size_t>(view.frame_count)};
}
inline Span<const trtmc_image_result_view_v1>
predicted_frames(const trtmc_video_result_view_v1& view) noexcept {
    if (view.conditioned_prefix_frames >= view.frame_count)
        return {};
    return {view.frames + view.conditioned_prefix_frames,
            static_cast<size_t>(view.frame_count - view.conditioned_prefix_frames)};
}
inline FloatMatrixView action_values(const trtmc_action_sequence_view_v1& view) noexcept {
    return {{view.values.data, static_cast<size_t>(view.values.count)},
            view.values.rows,
            view.values.columns};
}
inline std::vector<std::string_view> video_strings(trtmc_strings_view source) {
    std::vector<std::string_view> result;
    for (uint64_t i = 0; i < source.size; ++i)
        result.push_back(string_view(source.data[i]));
    return result;
}
} // namespace detail

class VideoGenerationResult : public detail::VideoResultOwner<trtmc_video_result_view_v1> {
  public:
    using VideoResultOwner::VideoResultOwner;
    Span<const trtmc_image_result_view_v1> frames() const noexcept {
        return detail::video_frames(view_);
    }
    Span<const double> timestamps_seconds() const noexcept {
        return {view_.timestamps_seconds.data, static_cast<size_t>(view_.timestamps_seconds.size)};
    }
    uint64_t conditioned_prefix_frames() const noexcept { return view_.conditioned_prefix_frames; }
    double setup_ms() const noexcept { return view_.setup_ms; }
    double inference_ms() const noexcept { return view_.inference_ms; }
};
class FutureVideoResult : public VideoGenerationResult {
  public:
    using VideoGenerationResult::VideoGenerationResult;
    Span<const trtmc_image_result_view_v1> predicted_frames() const noexcept {
        return detail::predicted_frames(view_);
    }
};
class ActionPredictionResult : public detail::VideoResultOwner<trtmc_action_sequence_view_v1> {
  public:
    using VideoResultOwner::VideoResultOwner;
    FloatMatrixView values() const noexcept { return detail::action_values(view_); }
    std::string_view domain() const { return detail::string_view(view_.schema.domain); }
    std::vector<std::string_view> component_names() const {
        return detail::video_strings(view_.schema.component_names);
    }
    std::vector<std::string_view> units() const {
        return detail::video_strings(view_.schema.units);
    }
    std::string_view coordinate_frame() const {
        return detail::string_view(view_.schema.coordinate_frame);
    }
    std::string_view normalization() const {
        return detail::string_view(view_.schema.normalization);
    }
    Span<const double> timestamps_seconds() const noexcept {
        return {view_.timestamps_seconds.data, static_cast<size_t>(view_.timestamps_seconds.size)};
    }
    Span<const ActionFrameSpan> frame_spans() const noexcept {
        return {view_.frame_spans, static_cast<size_t>(view_.frame_span_count)};
    }
};
class ActionVideoPredictionResult
    : public detail::VideoResultOwner<trtmc_action_video_result_view_v1> {
  public:
    using VideoResultOwner::VideoResultOwner;
    FloatMatrixView actions() const noexcept { return detail::action_values(view_.actions); }
    std::string_view action_domain() const {
        return detail::string_view(view_.actions.schema.domain);
    }
    Span<const trtmc_image_result_view_v1> frames() const noexcept {
        return detail::video_frames(view_.video);
    }
    Span<const trtmc_image_result_view_v1> predicted_frames() const noexcept {
        return detail::predicted_frames(view_.video);
    }
    const trtmc_action_sequence_view_v1& action_view() const noexcept { return view_.actions; }
    const trtmc_video_result_view_v1& video_view() const noexcept { return view_.video; }
};
class AudioVideoGenerationResult
    : public detail::VideoResultOwner<trtmc_audio_video_result_view_v1> {
  public:
    using VideoResultOwner::VideoResultOwner;
    Span<const trtmc_image_result_view_v1> frames() const noexcept {
        return detail::video_frames(view_.video);
    }
    Span<const double> timestamps_seconds() const noexcept {
        return {view_.video.timestamps_seconds.data,
                static_cast<size_t>(view_.video.timestamps_seconds.size)};
    }
    AudioView audio() const noexcept {
        return {{view_.audio.audio.samples, static_cast<size_t>(view_.audio.audio.sample_count)},
                view_.audio.audio.sample_rate,
                view_.audio.audio.channels};
    }
    double audio_start_seconds() const noexcept { return view_.audio_start_seconds; }
    const trtmc_video_result_view_v1& video_view() const noexcept { return view_.video; }
    const trtmc_audio_result_view_v1& audio_view() const noexcept { return view_.audio; }
};

namespace detail {
class VideoWireInputs {
  public:
    trtmc_video_view_v1 video(const VideoInput& input) {
        auto& frames = images_.emplace_back();
        frames.reserve(input.frames.size());
        for (const auto& image : input.frames)
            frames.push_back(image.wire);
        return {frames.data(),
                frames.size(),
                {input.timestamps_seconds.data(), input.timestamps_seconds.size()}};
    }
    trtmc_video_mask_view_v1 mask(const VideoMaskView& input) {
        return {input.values.data(), input.values.size(), input.frames, input.height, input.width};
    }
    trtmc_strings_view strings(const std::vector<std::string>& input) {
        auto& strings = strings_.emplace_back();
        strings.reserve(input.size());
        for (const auto& value : input)
            strings.push_back(c_string(value));
        return {strings.data(), strings.size()};
    }
    trtmc_action_schema_view_v1 schema(const ActionSchema& input) {
        return {c_string(input.domain), strings(input.component_names), strings(input.units),
                c_string(input.coordinate_frame), c_string(input.normalization)};
    }
    trtmc_action_sequence_view_v1 actions(const ActionSequenceView& input) {
        return {input.values.c_view(),
                schema(input.schema),
                {input.timestamps_seconds.data(), input.timestamps_seconds.size()},
                input.frame_spans.data(),
                input.frame_spans.size()};
    }
    trtmc_action_output_spec_v1 action_spec(const ActionOutputSpec& input) {
        return {schema(input.schema), input.dimensions};
    }
    trtmc_camera_trajectory_view_v1 camera(const CameraTrajectoryView& input) {
        return {input.camera_to_world.c_view(),
                {input.timestamps_seconds.data(), input.timestamps_seconds.size()},
                c_string(input.coordinate_convention),
                c_string(input.translation_units)};
    }
    const std::vector<trtmc_timed_video_anchor_v1>&
    anchors(const std::vector<TimedVideoAnchor>& input) {
        anchors_.reserve(input.size());
        for (const auto& item : input) {
            trtmc_timed_video_anchor_v1 anchor{};
            if (const auto* image = std::get_if<ImageInput>(&item.content)) {
                anchor.kind = TRTMC_VIDEO_ANCHOR_IMAGE;
                anchor.content.image = image->wire;
            } else {
                anchor.kind = TRTMC_VIDEO_ANCHOR_CLIP;
                anchor.content.clip = video(std::get<VideoInput>(item.content));
            }
            anchor.output_start_frame = item.output_start_frame;
            anchor.has_strength = item.strength ? 1U : 0U;
            anchor.strength = item.strength.value_or(0);
            anchors_.push_back(anchor);
        }
        return anchors_;
    }
    const std::vector<trtmc_video_reference_item_v1>&
    references(const std::vector<VideoReferenceItem>& input) {
        references_.reserve(input.size());
        for (const auto& item : input) {
            trtmc_video_reference_item_v1 reference{};
            if (const auto* image = std::get_if<ImageInput>(&item)) {
                reference.kind = TRTMC_VIDEO_REFERENCE_IMAGE;
                reference.content.image = image->wire;
            } else if (const auto* clip = std::get_if<VideoReference>(&item)) {
                reference.kind = TRTMC_VIDEO_REFERENCE_CLIP;
                reference.content.clip = {
                    video(clip->video), clip->soundtrack ? 1U : 0U,
                    clip->soundtrack ? c_audio(*clip->soundtrack) : trtmc_audio_view_v1{},
                    clip->audio_start_seconds ? 1U : 0U, clip->audio_start_seconds.value_or(0)};
            } else {
                reference.kind = TRTMC_VIDEO_REFERENCE_AUDIO;
                reference.content.audio = c_audio(std::get<AudioView>(item));
            }
            references_.push_back(reference);
        }
        return references_;
    }
    trtmc_text_to_video_request_v1 convert(const TextToVideoRequest& input) {
        return {c_string(input.prompt),
                {input.initial_latents.data(), input.initial_latents.size()}};
    }
    trtmc_initial_image_text_to_video_request_v1
    convert(const InitialImageTextToVideoRequest& input) {

        return {input.initial_image.wire, c_string(input.prompt)};
    }
    trtmc_boundary_frames_text_to_video_request_v1
    convert(const BoundaryFramesTextToVideoRequest& input) {

        return {input.first_frame.wire, input.last_frame.wire, c_string(input.prompt)};
    }
    trtmc_timed_frames_text_to_video_request_v1
    convert(const TimedFramesTextToVideoRequest& input) {
        const auto& values = anchors(input.anchors);
        return {values.data(), values.size(), c_string(input.prompt)};
    }
    trtmc_video_text_to_video_edit_request_v1 convert(const VideoTextToVideoEditRequest& input) {

        return {video(input.source), c_string(input.prompt)};
    }
    trtmc_masked_video_text_to_video_request_v1
    convert(const MaskedVideoTextToVideoRequest& input) {

        return {video(input.source), mask(input.mask), c_string(input.prompt)};
    }
    trtmc_masked_video_reference_images_text_to_video_request_v1
    convert(const MaskedVideoReferenceImagesTextToVideoRequest& input) {
        auto& refs = images_.emplace_back();
        for (const auto& image : input.references)
            refs.push_back(image.wire);
        const auto* data = refs.data();
        const auto count = refs.size();
        return {video(input.source), mask(input.mask), data, count, c_string(input.prompt)};
    }
    trtmc_image_text_action_to_video_request_v1
    convert(const ImageTextActionToVideoRequest& input) {

        return {input.initial_image.wire,
                c_string(input.prompt),
                c_string(input.action_dsl),
                input.intrinsics.c_view(),
                input.dialect ? 1U : 0U,
                input.dialect ? c_string(*input.dialect) : trtmc_string_view{},
                {input.initial_latents.data(), input.initial_latents.size()}};
    }
    trtmc_image_text_camera_trajectory_to_video_request_v1
    convert(const ImageTextCameraTrajectoryToVideoRequest& input) {

        return {input.initial_image.wire,
                c_string(input.prompt),
                camera(input.camera),
                input.intrinsics.c_view(),
                {input.initial_latents.data(), input.initial_latents.size()}};
    }
    trtmc_video_text_to_future_video_request_v1
    convert(const VideoTextToFutureVideoRequest& input) {

        return {video(input.history), c_string(input.prompt)};
    }
    trtmc_image_action_to_future_video_request_v1
    convert(const ImageActionToFutureVideoRequest& input) {

        return {input.observation.wire, actions(input.actions), input.prompt ? 1U : 0U,
                input.prompt ? c_string(*input.prompt) : trtmc_string_view{}};
    }
    trtmc_video_action_to_future_video_request_v1
    convert(const VideoActionToFutureVideoRequest& input) {

        return {video(input.history), actions(input.actions), input.prompt ? 1U : 0U,
                input.prompt ? c_string(*input.prompt) : trtmc_string_view{}};
    }
    trtmc_video_to_action_sequence_request_v1 convert(const VideoToActionSequenceRequest& input) {

        return {video(input.observations), action_spec(input.action_spec), input.prompt ? 1U : 0U,
                input.prompt ? c_string(*input.prompt) : trtmc_string_view{}};
    }
    trtmc_image_to_action_and_video_request_v1 convert(const ImageToActionAndVideoRequest& input) {

        return {input.observation.wire, action_spec(input.action_spec), input.prompt ? 1U : 0U,
                input.prompt ? c_string(*input.prompt) : trtmc_string_view{}};
    }
    trtmc_video_to_action_and_video_request_v1 convert(const VideoToActionAndVideoRequest& input) {

        return {video(input.history), action_spec(input.action_spec), input.prompt ? 1U : 0U,
                input.prompt ? c_string(*input.prompt) : trtmc_string_view{}};
    }
    trtmc_text_to_audio_video_request_v1 convert(const TextToAudioVideoRequest& input) {

        return {c_string(input.prompt)};
    }
    trtmc_initial_image_text_to_audio_video_request_v1
    convert(const InitialImageTextToAudioVideoRequest& input) {

        return {input.initial_image.wire, c_string(input.prompt)};
    }
    trtmc_last_image_text_to_audio_video_request_v1
    convert(const LastImageTextToAudioVideoRequest& input) {

        return {input.last_image.wire, c_string(input.prompt)};
    }
    trtmc_boundary_frames_text_to_audio_video_request_v1
    convert(const BoundaryFramesTextToAudioVideoRequest& input) {

        return {input.first_frame.wire, input.last_frame.wire, c_string(input.prompt)};
    }
    trtmc_references_text_to_audio_video_request_v1
    convert(const ReferencesTextToAudioVideoRequest& input) {
        const auto& refs = references(input.references);
        return {refs.data(), refs.size(), c_string(input.prompt)};
    }

  private:
    std::vector<std::vector<trtmc_image_input_v1>> images_;
    std::vector<std::vector<trtmc_string_view>> strings_;
    std::vector<trtmc_timed_video_anchor_v1> anchors_;
    std::vector<trtmc_video_reference_item_v1> references_;
};

template <class Traits>
class VideoTask {
  public:
    static constexpr std::string_view kTask = Traits::kTask;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Result = typename Traits::Result;
    using Table = typename Traits::Table;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible video Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, 1, 0);
    }
    Result run(const Request& input, const Config& config = {}) const {
        VideoWireInputs storage;
        const auto request = storage.convert(input);
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &request, &options, &raw, &error);
        Result result(state_, raw);
        check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.wire_view(), &error);
        check(state_->api, status, error);
        return result;
    }

  private:
    friend class ::trtmc::Model;
    VideoTask(std::shared_ptr<ModelState> state, const trtmc_api_header* table) noexcept
        : state_(std::move(state)), api_(reinterpret_cast<const Table*>(table)) {}
    std::shared_ptr<ModelState> state_;
    const Table* api_;
};

struct TextToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_VIDEO;
    using Request = TextToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_text_to_video_api_v1;
};

struct InitialImageTextToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_INITIAL_IMAGE_TEXT_TO_VIDEO;
    using Request = InitialImageTextToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_initial_image_text_to_video_api_v1;
};

struct BoundaryFramesTextToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BOUNDARY_FRAMES_TEXT_TO_VIDEO;
    using Request = BoundaryFramesTextToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_boundary_frames_text_to_video_api_v1;
};

struct TimedFramesTextToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TIMED_FRAMES_TEXT_TO_VIDEO;
    using Request = TimedFramesTextToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_timed_frames_text_to_video_api_v1;
};

struct VideoTextToVideoEditTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_VIDEO_TEXT_TO_VIDEO_EDIT;
    using Request = VideoTextToVideoEditRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_video_text_to_video_edit_api_v1;
};

struct MaskedVideoTextToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_MASKED_VIDEO_TEXT_TO_VIDEO;
    using Request = MaskedVideoTextToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_masked_video_text_to_video_api_v1;
};

struct MaskedVideoReferenceImagesTextToVideoTraits {
    static constexpr std::string_view kTask =
        TRTMC_TASK_MASKED_VIDEO_REFERENCE_IMAGES_TEXT_TO_VIDEO;
    using Request = MaskedVideoReferenceImagesTextToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_masked_video_reference_images_text_to_video_api_v1;
};

struct ImageTextActionToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TEXT_ACTION_TO_VIDEO;
    using Request = ImageTextActionToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_image_text_action_to_video_api_v1;
};

struct ImageTextCameraTrajectoryToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TEXT_CAMERA_TRAJECTORY_TO_VIDEO;
    using Request = ImageTextCameraTrajectoryToVideoRequest;
    using Result = VideoGenerationResult;
    using Table = trtmc_image_text_camera_trajectory_to_video_api_v1;
};

struct VideoTextToFutureVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_VIDEO_TEXT_TO_FUTURE_VIDEO;
    using Request = VideoTextToFutureVideoRequest;
    using Result = FutureVideoResult;
    using Table = trtmc_video_text_to_future_video_api_v1;
};

struct ImageActionToFutureVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_ACTION_TO_FUTURE_VIDEO;
    using Request = ImageActionToFutureVideoRequest;
    using Result = FutureVideoResult;
    using Table = trtmc_image_action_to_future_video_api_v1;
};

struct VideoActionToFutureVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_VIDEO_ACTION_TO_FUTURE_VIDEO;
    using Request = VideoActionToFutureVideoRequest;
    using Result = FutureVideoResult;
    using Table = trtmc_video_action_to_future_video_api_v1;
};

struct VideoToActionSequenceTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_VIDEO_TO_ACTION_SEQUENCE;
    using Request = VideoToActionSequenceRequest;
    using Result = ActionPredictionResult;
    using Table = trtmc_video_to_action_sequence_api_v1;
};

struct ImageToActionAndVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TO_ACTION_AND_VIDEO;
    using Request = ImageToActionAndVideoRequest;
    using Result = ActionVideoPredictionResult;
    using Table = trtmc_image_to_action_and_video_api_v1;
};

struct VideoToActionAndVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_VIDEO_TO_ACTION_AND_VIDEO;
    using Request = VideoToActionAndVideoRequest;
    using Result = ActionVideoPredictionResult;
    using Table = trtmc_video_to_action_and_video_api_v1;
};

struct TextToAudioVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_AUDIO_VIDEO;
    using Request = TextToAudioVideoRequest;
    using Result = AudioVideoGenerationResult;
    using Table = trtmc_text_to_audio_video_api_v1;
};

struct InitialImageTextToAudioVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_INITIAL_IMAGE_TEXT_TO_AUDIO_VIDEO;
    using Request = InitialImageTextToAudioVideoRequest;
    using Result = AudioVideoGenerationResult;
    using Table = trtmc_initial_image_text_to_audio_video_api_v1;
};

struct LastImageTextToAudioVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_LAST_IMAGE_TEXT_TO_AUDIO_VIDEO;
    using Request = LastImageTextToAudioVideoRequest;
    using Result = AudioVideoGenerationResult;
    using Table = trtmc_last_image_text_to_audio_video_api_v1;
};

struct BoundaryFramesTextToAudioVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BOUNDARY_FRAMES_TEXT_TO_AUDIO_VIDEO;
    using Request = BoundaryFramesTextToAudioVideoRequest;
    using Result = AudioVideoGenerationResult;
    using Table = trtmc_boundary_frames_text_to_audio_video_api_v1;
};

struct ReferencesTextToAudioVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_REFERENCES_TEXT_TO_AUDIO_VIDEO;
    using Request = ReferencesTextToAudioVideoRequest;
    using Result = AudioVideoGenerationResult;
    using Table = trtmc_references_text_to_audio_video_api_v1;
};

} // namespace detail

using TextToVideo = detail::VideoTask<detail::TextToVideoTraits>;
using InitialImageTextToVideo = detail::VideoTask<detail::InitialImageTextToVideoTraits>;
using BoundaryFramesTextToVideo = detail::VideoTask<detail::BoundaryFramesTextToVideoTraits>;
using TimedFramesTextToVideo = detail::VideoTask<detail::TimedFramesTextToVideoTraits>;
using VideoTextToVideoEdit = detail::VideoTask<detail::VideoTextToVideoEditTraits>;
using MaskedVideoTextToVideo = detail::VideoTask<detail::MaskedVideoTextToVideoTraits>;
using MaskedVideoReferenceImagesTextToVideo =
    detail::VideoTask<detail::MaskedVideoReferenceImagesTextToVideoTraits>;
using ImageTextActionToVideo = detail::VideoTask<detail::ImageTextActionToVideoTraits>;
using ImageTextCameraTrajectoryToVideo =
    detail::VideoTask<detail::ImageTextCameraTrajectoryToVideoTraits>;
using VideoTextToFutureVideo = detail::VideoTask<detail::VideoTextToFutureVideoTraits>;
using ImageActionToFutureVideo = detail::VideoTask<detail::ImageActionToFutureVideoTraits>;
using VideoActionToFutureVideo = detail::VideoTask<detail::VideoActionToFutureVideoTraits>;
using VideoToActionSequence = detail::VideoTask<detail::VideoToActionSequenceTraits>;
using ImageToActionAndVideo = detail::VideoTask<detail::ImageToActionAndVideoTraits>;
using VideoToActionAndVideo = detail::VideoTask<detail::VideoToActionAndVideoTraits>;
using TextToAudioVideo = detail::VideoTask<detail::TextToAudioVideoTraits>;
using InitialImageTextToAudioVideo = detail::VideoTask<detail::InitialImageTextToAudioVideoTraits>;
using LastImageTextToAudioVideo = detail::VideoTask<detail::LastImageTextToAudioVideoTraits>;
using BoundaryFramesTextToAudioVideo =
    detail::VideoTask<detail::BoundaryFramesTextToAudioVideoTraits>;
using ReferencesTextToAudioVideo = detail::VideoTask<detail::ReferencesTextToAudioVideoTraits>;

struct BatchTextToVideoItem {
    TextToVideoRequest input;
    Config config{};
};
struct BatchTextToVideoRequest {
    std::vector<BatchTextToVideoItem> items;
};
namespace detail {
template <class View>
class VideoBatchResult {
  public:
    using Count = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t*, trtmc_error**);
    using Item = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t, View*, trtmc_error**);
    VideoBatchResult(ResultOwner owner, Count count, Item item)
        : owner_(std::move(owner)), item_(item) {
        trtmc_error* error = nullptr;
        const auto status = count(owner_.get(), &count_, &error);
        check(owner_.api(), status, error);
    }
    VideoBatchResult(const VideoBatchResult&) = delete;
    VideoBatchResult& operator=(const VideoBatchResult&) = delete;
    VideoBatchResult(VideoBatchResult&& other) noexcept
        : owner_(std::move(other.owner_)), item_(other.item_),
          count_(std::exchange(other.count_, 0)) {}
    VideoBatchResult& operator=(VideoBatchResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            item_ = other.item_;
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    uint64_t size() const noexcept { return count_; }
    // Every nested frame/audio/action view borrows this result's lifetime.
    View at(uint64_t index) const {
        View out{};
        trtmc_error* error = nullptr;
        const auto status = item_(owner_.get(), index, &out, &error);
        check(owner_.api(), status, error);
        return out;
    }
    View operator[](uint64_t index) const { return at(index); }

  private:
    ResultOwner owner_;
    Item item_;
    uint64_t count_{0};
};
template <class Traits>
class VideoBatchTask {
  public:
    static constexpr std::string_view kTask = Traits::kTask;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Result = VideoBatchResult<typename Traits::View>;
    using Table = typename Traits::Table;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible video batch Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, 1, 0);
    }
    Result run(const Request& input) const {
        std::vector<VideoWireInputs> conversions(input.items.size());
        std::vector<Config::CEntries> configs;
        std::vector<typename Traits::WireItem> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (size_t i = 0; i < input.items.size(); ++i) {
            configs.push_back(input.items[i].config.c_entries());
            items.push_back({conversions[i].convert(input.items[i].input), configs.back().view()});
        }
        const typename Traits::WireRequest request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(model_->handle, &request, &raw, &error);
        ResultOwner owner(model_, raw);
        check(model_->api, status, error);
        return Result(std::move(owner), api_->result_count, api_->result_item_view);
    }

  private:
    friend class ::trtmc::Model;
    VideoBatchTask(std::shared_ptr<ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)), api_(reinterpret_cast<const Table*>(table)) {}
    std::shared_ptr<ModelState> model_;
    const Table* api_;
};
struct BatchTextToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_TO_VIDEO;
    using Request = BatchTextToVideoRequest;
    using Table = trtmc_batch_text_to_video_api_v1;
    using WireItem = trtmc_batch_text_to_video_item_v1;
    using WireRequest = trtmc_batch_text_to_video_request_v1;
    using View = trtmc_video_result_view_v1;
};
} // namespace detail
using BatchTextToVideo = detail::VideoBatchTask<detail::BatchTextToVideoTraits>;
struct BatchInitialImageTextToVideoItem {
    InitialImageTextToVideoRequest input;
    Config config{};
};
struct BatchInitialImageTextToVideoRequest {
    std::vector<BatchInitialImageTextToVideoItem> items;
};
namespace detail {
struct BatchInitialImageTextToVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_INITIAL_IMAGE_TEXT_TO_VIDEO;
    using Request = BatchInitialImageTextToVideoRequest;
    using Table = trtmc_batch_initial_image_text_to_video_api_v1;
    using WireItem = trtmc_batch_initial_image_text_to_video_item_v1;
    using WireRequest = trtmc_batch_initial_image_text_to_video_request_v1;
    using View = trtmc_video_result_view_v1;
};
} // namespace detail
using BatchInitialImageTextToVideo =
    detail::VideoBatchTask<detail::BatchInitialImageTextToVideoTraits>;

struct BatchVideoTextToFutureVideoItem {
    VideoTextToFutureVideoRequest input;
    Config config{};
};
struct BatchVideoTextToFutureVideoRequest {
    std::vector<BatchVideoTextToFutureVideoItem> items;
};
namespace detail {
struct BatchVideoTextToFutureVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_VIDEO_TEXT_TO_FUTURE_VIDEO;
    using Request = BatchVideoTextToFutureVideoRequest;
    using Table = trtmc_batch_video_text_to_future_video_api_v1;
    using WireItem = trtmc_batch_video_text_to_future_video_item_v1;
    using WireRequest = trtmc_batch_video_text_to_future_video_request_v1;
    using View = trtmc_video_result_view_v1;
};
} // namespace detail
using BatchVideoTextToFutureVideo =
    detail::VideoBatchTask<detail::BatchVideoTextToFutureVideoTraits>;

struct BatchImageActionToFutureVideoItem {
    ImageActionToFutureVideoRequest input;
    Config config{};
};
struct BatchImageActionToFutureVideoRequest {
    std::vector<BatchImageActionToFutureVideoItem> items;
};
namespace detail {
struct BatchImageActionToFutureVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_ACTION_TO_FUTURE_VIDEO;
    using Request = BatchImageActionToFutureVideoRequest;
    using Table = trtmc_batch_image_action_to_future_video_api_v1;
    using WireItem = trtmc_batch_image_action_to_future_video_item_v1;
    using WireRequest = trtmc_batch_image_action_to_future_video_request_v1;
    using View = trtmc_video_result_view_v1;
};
} // namespace detail
using BatchImageActionToFutureVideo =
    detail::VideoBatchTask<detail::BatchImageActionToFutureVideoTraits>;

struct BatchVideoToActionSequenceItem {
    VideoToActionSequenceRequest input;
    Config config{};
};
struct BatchVideoToActionSequenceRequest {
    std::vector<BatchVideoToActionSequenceItem> items;
};
namespace detail {
struct BatchVideoToActionSequenceTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_VIDEO_TO_ACTION_SEQUENCE;
    using Request = BatchVideoToActionSequenceRequest;
    using Table = trtmc_batch_video_to_action_sequence_api_v1;
    using WireItem = trtmc_batch_video_to_action_sequence_item_v1;
    using WireRequest = trtmc_batch_video_to_action_sequence_request_v1;
    using View = trtmc_action_sequence_view_v1;
};
} // namespace detail
using BatchVideoToActionSequence = detail::VideoBatchTask<detail::BatchVideoToActionSequenceTraits>;

struct BatchImageToActionAndVideoItem {
    ImageToActionAndVideoRequest input;
    Config config{};
};
struct BatchImageToActionAndVideoRequest {
    std::vector<BatchImageToActionAndVideoItem> items;
};
namespace detail {
struct BatchImageToActionAndVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_TO_ACTION_AND_VIDEO;
    using Request = BatchImageToActionAndVideoRequest;
    using Table = trtmc_batch_image_to_action_and_video_api_v1;
    using WireItem = trtmc_batch_image_to_action_and_video_item_v1;
    using WireRequest = trtmc_batch_image_to_action_and_video_request_v1;
    using View = trtmc_action_video_result_view_v1;
};
} // namespace detail
using BatchImageToActionAndVideo = detail::VideoBatchTask<detail::BatchImageToActionAndVideoTraits>;

struct BatchTextToAudioVideoItem {
    TextToAudioVideoRequest input;
    Config config{};
};
struct BatchTextToAudioVideoRequest {
    std::vector<BatchTextToAudioVideoItem> items;
};
namespace detail {
struct BatchTextToAudioVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_TO_AUDIO_VIDEO;
    using Request = BatchTextToAudioVideoRequest;
    using Table = trtmc_batch_text_to_audio_video_api_v1;
    using WireItem = trtmc_batch_text_to_audio_video_item_v1;
    using WireRequest = trtmc_batch_text_to_audio_video_request_v1;
    using View = trtmc_audio_video_result_view_v1;
};
} // namespace detail
using BatchTextToAudioVideo = detail::VideoBatchTask<detail::BatchTextToAudioVideoTraits>;

struct BatchInitialImageTextToAudioVideoItem {
    InitialImageTextToAudioVideoRequest input;
    Config config{};
};
struct BatchInitialImageTextToAudioVideoRequest {
    std::vector<BatchInitialImageTextToAudioVideoItem> items;
};
namespace detail {
struct BatchInitialImageTextToAudioVideoTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_INITIAL_IMAGE_TEXT_TO_AUDIO_VIDEO;
    using Request = BatchInitialImageTextToAudioVideoRequest;
    using Table = trtmc_batch_initial_image_text_to_audio_video_api_v1;
    using WireItem = trtmc_batch_initial_image_text_to_audio_video_item_v1;
    using WireRequest = trtmc_batch_initial_image_text_to_audio_video_request_v1;
    using View = trtmc_audio_video_result_view_v1;
};
} // namespace detail
using BatchInitialImageTextToAudioVideo =
    detail::VideoBatchTask<detail::BatchInitialImageTextToAudioVideoTraits>;

} // namespace trtmc
