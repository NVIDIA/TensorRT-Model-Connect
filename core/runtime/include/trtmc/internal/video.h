/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/audio.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/matrix.h"

#include <optional>
#include <variant>

namespace trtmc::internal {

// Ordered frames, not a request batch. Empty timestamps mean physical time is
// unspecified; otherwise one finite, nondecreasing presentation time per frame.
// Related media/action timestamps in one request/result use the same clock.
struct VideoView {
    Span<const ImageView> frames;
    Span<const double> timestamps_seconds;
};
struct VideoResult {
    ImageResult frames; // One owned contiguous THWC float32 buffer; num_frames=F.
    std::vector<double> timestamps_seconds;
    uint64_t conditioned_prefix_frames{0}; // [0,prefix) is context, not predicted future.
    double setup_ms{0}, inference_ms{0};
};
struct VideoMaskView {
    Span<const float> values; // Contiguous FHW; zero conditions, one requests generation.
    uint64_t frames{0}, height{0}, width{0};
};
struct TimedVideoAnchor {
    std::variant<ImageView, VideoView> content;
    uint64_t output_start_frame{0};
    std::optional<double> strength; // Absent uses only the family's declared default.
};
struct CameraTrajectoryView {
    FloatMatrixView camera_to_world; // [frame,16], flattened row-major 4x4 matrices.
    Span<const double> timestamps_seconds;
    std::string_view coordinate_convention; // Empty means unspecified, never guessed.
    std::string_view translation_units;
};
struct ActionFrameSpan {
    uint64_t begin{0}, end{0}; // Associated frame interval [begin,end), not a rate.
};
struct ActionSchemaView {
    std::string_view domain;
    Span<const std::string_view> component_names; // Empty or one per action column.
    Span<const std::string_view> units;           // Empty means unspecified.
    std::string_view coordinate_frame;
    std::string_view normalization;
};
struct ActionSchema {
    std::string domain;
    std::vector<std::string> component_names;
    std::vector<std::string> units;
    std::string coordinate_frame;
    std::string normalization;
};
struct ActionSequenceView {
    FloatMatrixView values; // [step,raw action dimension], interpretation from schema.
    ActionSchemaView schema;
    Span<const double> timestamps_seconds;   // Empty or one per step.
    Span<const ActionFrameSpan> frame_spans; // Empty means frame association unspecified.
};
struct ActionOutputSpec {
    ActionSchemaView schema;
    uint64_t dimensions{0};
};
struct ActionSequenceResult {
    FloatMatrix values;
    ActionSchema schema;
    std::vector<double> timestamps_seconds;
    std::vector<ActionFrameSpan> frame_spans;
};
struct ActionVideoResult {
    ActionSequenceResult actions;
    VideoResult video; // One paired future prediction, including any declared context.
};
struct AudioVideoResult {
    VideoResult video;
    AudioResult audio;
    // Required for this synchronized output; same clock as video timestamps.
    std::optional<double> audio_start_seconds;
};
struct VideoReference {
    VideoView video;
    std::optional<AudioView> soundtrack;       // Belongs to this video, unlike standalone audio.
    std::optional<double> audio_start_seconds; // Absent uses family-declared alignment only.
};
using VideoReferenceItem = std::variant<ImageView, VideoReference, AudioView>;

// Intrinsics in the SANA contracts are [1 or F,9] row-major 3x3 matrices in
// initial-image pixel coordinates. Dialect, frame count, FPS and sampling
// defaults belong to the family. A reference is not implicitly an edit source.
// For inverse dynamics, action frame spans refer to the observed input;
// for forward/joint dynamics they refer to the returned rollout.
struct TextToVideoRequest {
    std::string_view prompt;
    // Optional borrowed float32 in family-defined packed/CTHW layout. Family
    // validates count and initializes only when this operand is empty.
    Span<const float> initial_latents{};
};
struct InitialImageTextToVideoRequest {
    ImageView initial_image;
    std::string_view prompt;
};
struct BoundaryFramesTextToVideoRequest {
    ImageView first_frame;
    ImageView last_frame;
    std::string_view prompt;
};
struct TimedFramesTextToVideoRequest {
    Span<const TimedVideoAnchor> anchors;
    std::string_view prompt;
};
struct VideoTextToVideoEditRequest {
    VideoView source;
    std::string_view prompt;
};
struct MaskedVideoTextToVideoRequest {
    VideoView source;
    VideoMaskView mask;
    std::string_view prompt;
};
struct MaskedVideoReferenceImagesTextToVideoRequest {
    VideoView source;
    VideoMaskView mask;
    Span<const ImageView> references;
    std::string_view prompt;
};
struct ImageTextActionToVideoRequest {
    ImageView initial_image;
    std::string_view prompt;
    std::string_view action_dsl;
    FloatMatrixView intrinsics;
    std::optional<std::string_view> dialect{};
    // Replay does not remove the family's initial-image conditioning/overwrite.
    Span<const float> initial_latents{};
};
struct ImageTextCameraTrajectoryToVideoRequest {
    ImageView initial_image;
    std::string_view prompt;
    CameraTrajectoryView camera;
    FloatMatrixView intrinsics;
    Span<const float> initial_latents{};
};
struct VideoTextToFutureVideoRequest {
    VideoView history;
    std::string_view prompt;
};
struct ImageActionToFutureVideoRequest {
    ImageView observation;
    ActionSequenceView actions;
    std::optional<std::string_view> prompt{};
};
struct VideoActionToFutureVideoRequest {
    VideoView history;
    ActionSequenceView actions;
    std::optional<std::string_view> prompt{};
};
struct VideoToActionSequenceRequest {
    VideoView observations;
    ActionOutputSpec action_spec;
    std::optional<std::string_view> prompt{};
};
struct ImageToActionAndVideoRequest {
    ImageView observation;
    ActionOutputSpec action_spec;
    std::optional<std::string_view> prompt{};
};
struct VideoToActionAndVideoRequest {
    VideoView history;
    ActionOutputSpec action_spec;
    std::optional<std::string_view> prompt{};
};
struct TextToAudioVideoRequest {
    std::string_view prompt;
};
struct InitialImageTextToAudioVideoRequest {
    ImageView initial_image;
    std::string_view prompt;
};
struct LastImageTextToAudioVideoRequest {
    ImageView last_image;
    std::string_view prompt;
};
struct BoundaryFramesTextToAudioVideoRequest {
    ImageView first_frame;
    ImageView last_frame;
    std::string_view prompt;
};
struct ReferencesTextToAudioVideoRequest {
    Span<const VideoReferenceItem> references;
    std::string_view prompt;
};

class ITextToVideo {
  public:
    static constexpr std::string_view kTask = "text_to_video";
    virtual ~ITextToVideo() = default;
    virtual VideoResult run(const TextToVideoRequest&, ConfigView) = 0;
};

class IInitialImageTextToVideo {
  public:
    static constexpr std::string_view kTask = "initial_image_text_to_video";
    virtual ~IInitialImageTextToVideo() = default;
    virtual VideoResult run(const InitialImageTextToVideoRequest&, ConfigView) = 0;
};

class IBoundaryFramesTextToVideo {
  public:
    static constexpr std::string_view kTask = "boundary_frames_text_to_video";
    virtual ~IBoundaryFramesTextToVideo() = default;
    virtual VideoResult run(const BoundaryFramesTextToVideoRequest&, ConfigView) = 0;
};

class ITimedFramesTextToVideo {
  public:
    static constexpr std::string_view kTask = "timed_frames_text_to_video";
    virtual ~ITimedFramesTextToVideo() = default;
    virtual VideoResult run(const TimedFramesTextToVideoRequest&, ConfigView) = 0;
};

class IVideoTextToVideoEdit {
  public:
    static constexpr std::string_view kTask = "video_text_to_video_edit";
    virtual ~IVideoTextToVideoEdit() = default;
    virtual VideoResult run(const VideoTextToVideoEditRequest&, ConfigView) = 0;
};

class IMaskedVideoTextToVideo {
  public:
    static constexpr std::string_view kTask = "masked_video_text_to_video";
    virtual ~IMaskedVideoTextToVideo() = default;
    virtual VideoResult run(const MaskedVideoTextToVideoRequest&, ConfigView) = 0;
};

class IMaskedVideoReferenceImagesTextToVideo {
  public:
    static constexpr std::string_view kTask = "masked_video_reference_images_text_to_video";
    virtual ~IMaskedVideoReferenceImagesTextToVideo() = default;
    virtual VideoResult run(const MaskedVideoReferenceImagesTextToVideoRequest&, ConfigView) = 0;
};

class IImageTextActionToVideo {
  public:
    static constexpr std::string_view kTask = "image_text_action_to_video";
    virtual ~IImageTextActionToVideo() = default;
    virtual VideoResult run(const ImageTextActionToVideoRequest&, ConfigView) = 0;
};

class IImageTextCameraTrajectoryToVideo {
  public:
    static constexpr std::string_view kTask = "image_text_camera_trajectory_to_video";
    virtual ~IImageTextCameraTrajectoryToVideo() = default;
    virtual VideoResult run(const ImageTextCameraTrajectoryToVideoRequest&, ConfigView) = 0;
};

class IVideoTextToFutureVideo {
  public:
    static constexpr std::string_view kTask = "video_text_to_future_video";
    virtual ~IVideoTextToFutureVideo() = default;
    virtual VideoResult run(const VideoTextToFutureVideoRequest&, ConfigView) = 0;
};

class IImageActionToFutureVideo {
  public:
    static constexpr std::string_view kTask = "image_action_to_future_video";
    virtual ~IImageActionToFutureVideo() = default;
    virtual VideoResult run(const ImageActionToFutureVideoRequest&, ConfigView) = 0;
};

class IVideoActionToFutureVideo {
  public:
    static constexpr std::string_view kTask = "video_action_to_future_video";
    virtual ~IVideoActionToFutureVideo() = default;
    virtual VideoResult run(const VideoActionToFutureVideoRequest&, ConfigView) = 0;
};

class IVideoToActionSequence {
  public:
    static constexpr std::string_view kTask = "video_to_action_sequence";
    virtual ~IVideoToActionSequence() = default;
    virtual ActionSequenceResult run(const VideoToActionSequenceRequest&, ConfigView) = 0;
};

class IImageToActionAndVideo {
  public:
    static constexpr std::string_view kTask = "image_to_action_and_video";
    virtual ~IImageToActionAndVideo() = default;
    virtual ActionVideoResult run(const ImageToActionAndVideoRequest&, ConfigView) = 0;
};

class IVideoToActionAndVideo {
  public:
    static constexpr std::string_view kTask = "video_to_action_and_video";
    virtual ~IVideoToActionAndVideo() = default;
    virtual ActionVideoResult run(const VideoToActionAndVideoRequest&, ConfigView) = 0;
};

class ITextToAudioVideo {
  public:
    static constexpr std::string_view kTask = "text_to_audio_video";
    virtual ~ITextToAudioVideo() = default;
    virtual AudioVideoResult run(const TextToAudioVideoRequest&, ConfigView) = 0;
};

class IInitialImageTextToAudioVideo {
  public:
    static constexpr std::string_view kTask = "initial_image_text_to_audio_video";
    virtual ~IInitialImageTextToAudioVideo() = default;
    virtual AudioVideoResult run(const InitialImageTextToAudioVideoRequest&, ConfigView) = 0;
};

class ILastImageTextToAudioVideo {
  public:
    static constexpr std::string_view kTask = "last_image_text_to_audio_video";
    virtual ~ILastImageTextToAudioVideo() = default;
    virtual AudioVideoResult run(const LastImageTextToAudioVideoRequest&, ConfigView) = 0;
};

class IBoundaryFramesTextToAudioVideo {
  public:
    static constexpr std::string_view kTask = "boundary_frames_text_to_audio_video";
    virtual ~IBoundaryFramesTextToAudioVideo() = default;
    virtual AudioVideoResult run(const BoundaryFramesTextToAudioVideoRequest&, ConfigView) = 0;
};

class IReferencesTextToAudioVideo {
  public:
    static constexpr std::string_view kTask = "references_text_to_audio_video";
    virtual ~IReferencesTextToAudioVideo() = default;
    virtual AudioVideoResult run(const ReferencesTextToAudioVideoRequest&, ConfigView) = 0;
};

// Independent complete requests, not a single video's frame axis or multiple
// samples from one prompt. Family preflight covers all items and cross-item
// restrictions before one native batch invocation. No partial-success result.
struct BatchTextToVideoItem {
    TextToVideoRequest input;
    ConfigView config;
};
struct BatchTextToVideoRequest {
    using Item = BatchTextToVideoItem;
    Span<const Item> items;
};
class IBatchTextToVideo {
  public:
    static constexpr std::string_view kTask = "batch_text_to_video";
    using Request = BatchTextToVideoRequest;
    virtual ~IBatchTextToVideo() = default;
    virtual std::vector<VideoResult> run_batch(const Request&) = 0;
};

struct BatchInitialImageTextToVideoItem {
    InitialImageTextToVideoRequest input;
    ConfigView config;
};
struct BatchInitialImageTextToVideoRequest {
    using Item = BatchInitialImageTextToVideoItem;
    Span<const Item> items;
};
class IBatchInitialImageTextToVideo {
  public:
    static constexpr std::string_view kTask = "batch_initial_image_text_to_video";
    using Request = BatchInitialImageTextToVideoRequest;
    virtual ~IBatchInitialImageTextToVideo() = default;
    virtual std::vector<VideoResult> run_batch(const Request&) = 0;
};

struct BatchVideoTextToFutureVideoItem {
    VideoTextToFutureVideoRequest input;
    ConfigView config;
};
struct BatchVideoTextToFutureVideoRequest {
    using Item = BatchVideoTextToFutureVideoItem;
    Span<const Item> items;
};
class IBatchVideoTextToFutureVideo {
  public:
    static constexpr std::string_view kTask = "batch_video_text_to_future_video";
    using Request = BatchVideoTextToFutureVideoRequest;
    virtual ~IBatchVideoTextToFutureVideo() = default;
    virtual std::vector<VideoResult> run_batch(const Request&) = 0;
};

struct BatchImageActionToFutureVideoItem {
    ImageActionToFutureVideoRequest input;
    ConfigView config;
};
struct BatchImageActionToFutureVideoRequest {
    using Item = BatchImageActionToFutureVideoItem;
    Span<const Item> items;
};
class IBatchImageActionToFutureVideo {
  public:
    static constexpr std::string_view kTask = "batch_image_action_to_future_video";
    using Request = BatchImageActionToFutureVideoRequest;
    virtual ~IBatchImageActionToFutureVideo() = default;
    virtual std::vector<VideoResult> run_batch(const Request&) = 0;
};

struct BatchVideoToActionSequenceItem {
    VideoToActionSequenceRequest input;
    ConfigView config;
};
struct BatchVideoToActionSequenceRequest {
    using Item = BatchVideoToActionSequenceItem;
    Span<const Item> items;
};
class IBatchVideoToActionSequence {
  public:
    static constexpr std::string_view kTask = "batch_video_to_action_sequence";
    using Request = BatchVideoToActionSequenceRequest;
    virtual ~IBatchVideoToActionSequence() = default;
    virtual std::vector<ActionSequenceResult> run_batch(const Request&) = 0;
};

struct BatchImageToActionAndVideoItem {
    ImageToActionAndVideoRequest input;
    ConfigView config;
};
struct BatchImageToActionAndVideoRequest {
    using Item = BatchImageToActionAndVideoItem;
    Span<const Item> items;
};
class IBatchImageToActionAndVideo {
  public:
    static constexpr std::string_view kTask = "batch_image_to_action_and_video";
    using Request = BatchImageToActionAndVideoRequest;
    virtual ~IBatchImageToActionAndVideo() = default;
    virtual std::vector<ActionVideoResult> run_batch(const Request&) = 0;
};

struct BatchTextToAudioVideoItem {
    TextToAudioVideoRequest input;
    ConfigView config;
};
struct BatchTextToAudioVideoRequest {
    using Item = BatchTextToAudioVideoItem;
    Span<const Item> items;
};
class IBatchTextToAudioVideo {
  public:
    static constexpr std::string_view kTask = "batch_text_to_audio_video";
    using Request = BatchTextToAudioVideoRequest;
    virtual ~IBatchTextToAudioVideo() = default;
    virtual std::vector<AudioVideoResult> run_batch(const Request&) = 0;
};

struct BatchInitialImageTextToAudioVideoItem {
    InitialImageTextToAudioVideoRequest input;
    ConfigView config;
};
struct BatchInitialImageTextToAudioVideoRequest {
    using Item = BatchInitialImageTextToAudioVideoItem;
    Span<const Item> items;
};
class IBatchInitialImageTextToAudioVideo {
  public:
    static constexpr std::string_view kTask = "batch_initial_image_text_to_audio_video";
    using Request = BatchInitialImageTextToAudioVideoRequest;
    virtual ~IBatchInitialImageTextToAudioVideo() = default;
    virtual std::vector<AudioVideoResult> run_batch(const Request&) = 0;
};

} // namespace trtmc::internal
