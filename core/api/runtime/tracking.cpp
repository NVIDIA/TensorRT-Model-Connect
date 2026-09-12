/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/tracking.h"

#include "api_internal.h"
#include "trtmc/tracking.h"

#include <algorithm>
#include <cmath>

namespace trtmc::api {
namespace {
template <class Session>
struct NativeState {
    std::mutex operation;
    std::uint64_t generation{0};
    bool closed{false};
    std::shared_ptr<ModelState> owner;
    internal::TaskKey task_key{};
    std::unique_ptr<ModelSession> model;
    std::unique_ptr<Session> implementation;
    // Destruction order releases native GPU resources before ModelSession.
};
using DetectedState = NativeState<internal::IDetectedClipSession>;
using TextClipState = NativeState<internal::ITextClipSession>;
using PromptFrameState = NativeState<internal::ITextPromptFrameSession>;
using ImageMaskState = NativeState<internal::IImageMaskContext>;
using CropPoseState = NativeState<internal::ICropPoseSession>;
using RgbdPoseState = NativeState<internal::IRgbdPoseSession>;
using ClipGeometry = std::vector<std::pair<std::uint32_t, std::uint32_t>>;
struct InteractiveState : NativeState<internal::IMaskTrackSession> {
    ClipGeometry geometry;
    bool traversing{false};
};
void output_check(bool value, const char* message) {
    if (!value)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
trtmc_pixel_box_v1 box_view(const internal::PixelBox& b) {
    return {b.x_min, b.y_min, b.x_max, b.y_max};
}
trtmc_i64_view ids(const std::vector<std::int64_t>& value) {
    return {value.data(), value.size()};
}
struct TrackStorage final : ResultStorage {
    explicit TrackStorage(internal::TrackClipResult value, std::shared_ptr<void> prompt_source = {})
        : host(std::move(value)), prompt_origin(std::move(prompt_source)) {
        build_host();
    }
    TrackStorage(internal::BorrowedDeviceTrackClip value, std::shared_ptr<DetectedState> source)
        : device(std::move(value)), origin(std::move(source)), generation(origin->generation) {
        build_device();
    }
    void metadata(const internal::TrackFrameMetadata& m, trtmc_track_frame_view_v1& out) {
        output_check(m.height && m.width, "track frame has invalid dimensions");
        const auto n = m.object_ids.size();
        const auto optional = [n](std::size_t count) { return count == 0 || count == n; };
        output_check(optional(m.boxes.size()) && optional(m.detection_scores.size()) &&
                         optional(m.tracker_scores.size()) && optional(m.class_ids.size()),
                     "track metadata does not match object IDs");
        boxes.emplace_back();
        for (const auto& b : m.boxes)
            boxes.back().push_back(box_view(b));
        out.frame_index = m.frame_index;
        out.height = m.height;
        out.width = m.width;
        out.object_ids = ids(m.object_ids);
        out.boxes = boxes.back().data();
        out.box_count = boxes.back().size();
        out.detection_scores = m.detection_scores.data();
        out.detection_score_count = m.detection_scores.size();
        out.tracker_scores = m.tracker_scores.data();
        out.tracker_score_count = m.tracker_scores.size();
        out.class_ids = ids(m.class_ids);
        out.removed_object_ids = ids(m.removed_object_ids);
        out.suppressed_object_ids = ids(m.suppressed_object_ids);
    }
    void shape(trtmc_track_frame_view_v1& out) {
        output_check(out.element_type == TRTMC_TRACK_UINT8 ||
                         out.element_type == TRTMC_TRACK_FLOAT32,
                     "unknown tracking mask element type");
        output_check(out.mask_kind >= TRTMC_MASK_LOGITS && out.mask_kind <= TRTMC_MASK_BINARY,
                     "unknown tracking mask interpretation");
        output_check(out.element_type != TRTMC_TRACK_UINT8 || out.mask_kind == TRTMC_MASK_BINARY,
                     "uint8 tracking masks must be binary");
        const auto n = out.object_ids.size;
        const auto element_size = out.element_type == TRTMC_TRACK_UINT8 ? 1U : 4U;
        std::uint64_t expected;
        try {
            checked_size(out.height, out.width);
            const auto area = static_cast<std::uint64_t>(out.height) * out.width;
            checked_size(n, area);
            checked_size(n * area, element_size);
            expected = n * area * element_size;
        } catch (const ApiFailure&) {
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "tracking mask dimensions overflow"};
        }
        output_check(out.mask_byte_size == expected && (expected == 0 || out.masks),
                     "tracking mask bytes do not match object,H,W shape");
    }
    void detections(const std::vector<internal::InitialTrackDetection>& values) {
        for (const auto& d : values)
            initial.push_back(
                {d.frame_index, d.object_id, d.class_id, d.score, box_view(d.prompt_box)});
        view = {frames.data(), frames.size(), initial.data(), initial.size()};
    }
    void build_host() {
        frames.resize(host.frames.size());
        boxes.reserve(host.frames.size());
        for (std::size_t i = 0; i < frames.size(); ++i) {
            auto& out = frames[i];
            const auto& f = host.frames[i];
            metadata(f.metadata, out);
            out.memory_kind = TRTMC_TRACK_HOST;
            out.device_ordinal = -1;
            out.mask_kind = static_cast<std::uint32_t>(f.mask_kind);
            if (const auto* bytes = std::get_if<std::vector<std::uint8_t>>(&f.masks)) {
                out.element_type = TRTMC_TRACK_UINT8;
                out.masks = bytes->data();
                out.mask_byte_size = bytes->size();
            } else {
                const auto& floats = std::get<std::vector<float>>(f.masks);
                out.element_type = TRTMC_TRACK_FLOAT32;
                out.masks = floats.data();
                out.mask_byte_size = floats.size() * sizeof(float);
            }
            shape(out);
        }
        detections(host.initial_detections);
    }
    void build_device() {
        frames.resize(device.frames.size());
        boxes.reserve(device.frames.size());
        for (std::size_t i = 0; i < frames.size(); ++i) {
            auto& out = frames[i];
            const auto& f = device.frames[i];
            metadata(f.metadata, out);
            out.memory_kind = TRTMC_TRACK_CUDA;
            out.device_ordinal = f.device_ordinal;
            out.mask_kind = static_cast<std::uint32_t>(f.mask_kind);
            out.element_type = static_cast<std::uint32_t>(f.type);
            out.masks = f.address;
            out.mask_byte_size = f.byte_size;
            output_check(out.device_ordinal >= 0, "borrowed mask has no device ordinal");
            shape(out);
        }
        detections(device.initial_detections);
    }
    internal::TrackClipResult host;
    internal::BorrowedDeviceTrackClip device;
    std::shared_ptr<DetectedState> origin;
    std::shared_ptr<void> prompt_origin;
    std::uint64_t generation{0};
    std::vector<std::vector<trtmc_pixel_box_v1>> boxes;
    std::vector<trtmc_track_frame_view_v1> frames;
    std::vector<trtmc_initial_track_detection_v1> initial;
    trtmc_track_clip_view_v1 view{};
};
template <class Session>
std::unique_lock<std::mutex> operation(NativeState<Session>& state) {
    std::unique_lock<std::mutex> lock(state.operation, std::try_to_lock);
    if (!lock.owns_lock())
        throw ApiFailure{TRTMC_BUSY, "another tracking operation is active"};
    require(!state.closed && state.implementation, "tracking session is closed");
    return lock;
}
template <class Result>
void complete_clip(const Result& result, internal::VideoView input) {
    output_check(result.frames.size() == input.frames.size(),
                 "complete-clip tracking omitted frames");
    for (std::size_t i = 0; i < result.frames.size(); ++i) {
        const auto& frame = result.frames[i].metadata;
        output_check(frame.frame_index == i && frame.height == input.frames[i].height &&
                         frame.width == input.frames[i].width,
                     "track frame indices/geometry do not match the clip");
    }
    for (const auto& detection : result.initial_detections)
        output_check(detection.frame_index < result.frames.size(),
                     "initial detection frame is outside the clip");
}
} // namespace
} // namespace trtmc::api

struct trtmc_detected_mask_session {
    std::shared_ptr<trtmc::api::DetectedState> state;
};
struct trtmc_text_mask_clip_session {
    std::shared_ptr<trtmc::api::TextClipState> state;
};
struct trtmc_text_prompt_frame_session {
    std::shared_ptr<trtmc::api::PromptFrameState> state;
};
struct trtmc_image_mask_context {
    std::shared_ptr<trtmc::api::ImageMaskState> state;
};
struct trtmc_mask_track_session {
    std::shared_ptr<trtmc::api::InteractiveState> state;
};
struct trtmc_track_propagation {
    std::shared_ptr<trtmc::api::InteractiveState> state;
    std::unique_ptr<trtmc::internal::ITrackPropagation> implementation;
    bool ended{false};
};
struct trtmc_crop_pose_session {
    std::shared_ptr<trtmc::api::CropPoseState> state;
};
struct trtmc_rgbd_pose_session {
    std::shared_ptr<trtmc::api::RgbdPoseState> state;
};

namespace trtmc::api {
namespace {
trtmc_status TRTMC_CALL create_detected(trtmc_model* model, const trtmc_config_view_v1* config,
                                        trtmc_detected_mask_session** out,
                                        trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "session output is null");
        const ConvertedConfig options(config);
        auto state = std::make_shared<DetectedState>();
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::IFramesToDetectedMaskTracks>(
                model, internal::IFramesToDetectedMaskTracks::kTask);
            state->owner = model_owner(model);
            state->task_key = internal::contract_key<internal::IFramesToDetectedMaskTracks>();
            validate_task_config(state->owner, state->task_key, options.view());
            state->model = std::make_unique<ModelSession>(model);
            state->implementation = family.create_detected_session(options.view());
        }
        output_check(state->implementation != nullptr,
                     "family returned no detector tracking session");
        *out = new trtmc_detected_mask_session{std::move(state)};
    });
}
trtmc_status TRTMC_CALL segment_detected(trtmc_detected_mask_session* session,
                                         const trtmc_video_view_v1* clip,
                                         const trtmc_config_view_v1* config, trtmc_result** out,
                                         trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && clip && out, "session, clip and output are required");
        auto lock = operation(*session->state);
        std::vector<internal::ImageView> images;
        const auto input = video_input(*clip, images);
        const ConvertedConfig options(config);
        validate_task_config(session->state->owner, session->state->task_key, options.view());
        ++session->state->generation;
        auto result = session->state->implementation->segment(input, options.view());
        complete_clip(result, input);
        *out = make_result<TrackStorage>(std::move(result));
    });
}
trtmc_status TRTMC_CALL segment_device(trtmc_detected_mask_session* session,
                                       const trtmc_video_view_v1* clip,
                                       const trtmc_config_view_v1* config, trtmc_result** out,
                                       trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && clip && out, "session, clip and output are required");
        auto lock = operation(*session->state);
        std::vector<internal::ImageView> images;
        const auto input = video_input(*clip, images);
        const ConvertedConfig options(config);
        validate_task_config(session->state->owner, session->state->task_key, options.view());
        auto* device = session->state->implementation->device_masks();
        if (!device)
            throw ApiFailure{TRTMC_UNSUPPORTED, "device masks are unavailable"};
        ++session->state->generation;
        auto result = device->segment_device(input, options.view());
        complete_clip(result, input);
        *out = make_result<TrackStorage>(std::move(result), session->state);
    });
}
trtmc_status TRTMC_CALL track_view(const trtmc_result* result, trtmc_track_clip_view_v1* out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "track result output is null");
        const auto& storage = require_result<TrackStorage>(result);
        if (storage.origin) {
            auto lock = operation(*storage.origin);
            require(storage.generation == storage.origin->generation,
                    "borrowed mask view was invalidated");
            *out = storage.view;
        } else
            *out = storage.view;
    });
}
const trtmc_detected_device_masks_api_v1 device_api{
    {1, 0, sizeof(device_api)}, segment_device, track_view};
trtmc_status TRTMC_CALL get_device_api(trtmc_detected_mask_session* session, std::uint32_t major,
                                       std::uint32_t minor,
                                       const trtmc_detected_device_masks_api_v1** out,
                                       trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && out, "session and device table output are required");
        if (major != 1 || minor != 0)
            throw ApiFailure{TRTMC_VERSION_MISMATCH, "device mask API version unavailable"};
        auto lock = operation(*session->state);
        if (!session->state->implementation->device_masks())
            throw ApiFailure{TRTMC_UNSUPPORTED, "device masks are unavailable"};
        *out = &device_api;
    });
}
template <class Handle>
void TRTMC_CALL release_native(Handle* session) noexcept {
    if (!session)
        return;
    {
        const std::lock_guard<std::mutex> lock(session->state->operation);
        session->state->closed = true;
        ++session->state->generation;
        session->state->implementation.reset();
        session->state->model.reset();
        session->state->owner.reset();
    }
    delete session;
}
const trtmc_frames_to_detected_mask_tracks_api_v1 detected_api{
    {1, 0, sizeof(detected_api)},
    create_detected,
    segment_detected,
    track_view,
    get_device_api,
    release_native<trtmc_detected_mask_session>};

trtmc_status TRTMC_CALL create_text_clip(trtmc_model* model, const trtmc_config_view_v1* config,
                                         trtmc_text_mask_clip_session** out,
                                         trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "text clip session output is null");
        const ConvertedConfig options(config);
        auto state = std::make_shared<TextClipState>();
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::IFramesTextToMaskTracks>(
                model, internal::IFramesTextToMaskTracks::kTask);
            state->owner = model_owner(model);
            state->task_key = internal::contract_key<internal::IFramesTextToMaskTracks>();
            validate_task_config(state->owner, state->task_key, options.view());
            state->model = std::make_unique<ModelSession>(model);
            state->implementation = family.create_text_clip_session(options.view());
        }
        output_check(state->implementation != nullptr, "family returned no text clip session");
        *out = new trtmc_text_mask_clip_session{std::move(state)};
    });
}
trtmc_status TRTMC_CALL segment_text_clip(trtmc_text_mask_clip_session* session,
                                          const trtmc_video_view_v1* clip, trtmc_string_view text,
                                          const trtmc_config_view_v1* config, trtmc_result** out,
                                          trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && clip && out, "text session, clip and output are required");
        auto lock = operation(*session->state);
        std::vector<internal::ImageView> frames;
        const auto input = video_input(*clip, frames);
        const auto prompt = string_view(text);
        const ConvertedConfig options(config);
        validate_task_config(session->state->owner, session->state->task_key, options.view());
        auto result = session->state->implementation->segment(input, prompt, options.view());
        complete_clip(result, input);
        *out = make_result<TrackStorage>(std::move(result));
    });
}
trtmc_status TRTMC_CALL create_prompt_frame(trtmc_model* model, trtmc_string_view text,
                                            const trtmc_config_view_v1* config,
                                            trtmc_text_prompt_frame_session** out,
                                            trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "prompt session output is null");
        const auto prompt = string_view(text);
        const ConvertedConfig options(config);
        auto state = std::make_shared<PromptFrameState>();
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::IPromptFrameTextToMaskTracks>(
                model, internal::IPromptFrameTextToMaskTracks::kTask);
            state->owner = model_owner(model);
            state->task_key = internal::contract_key<internal::IPromptFrameTextToMaskTracks>();
            validate_task_config(state->owner, state->task_key, options.view());
            state->model = std::make_unique<ModelSession>(model);
            state->implementation = family.create_prompt_frame_session(prompt, options.view());
        }
        output_check(state->implementation != nullptr, "family returned no prompt-frame session");
        *out = new trtmc_text_prompt_frame_session{std::move(state)};
    });
}
trtmc_status TRTMC_CALL accept_prompt_frame(trtmc_text_prompt_frame_session* session,
                                            const trtmc_image_input_v1* image, trtmc_result** out,
                                            trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && image && out, "prompt session, image and output are required");
        auto lock = operation(*session->state);
        const auto input = image_input(*image);
        internal::TrackClipResult result;
        result.frames.push_back(session->state->implementation->accept_prompt_frame(input));
        complete_clip(result, {{&input, 1}, {}});
        *out = make_result<TrackStorage>(std::move(result), session->state);
    });
}
trtmc_status TRTMC_CALL continue_prompt_frame(trtmc_text_prompt_frame_session* session,
                                              const trtmc_result* prompt,
                                              const trtmc_video_view_v1* clip, trtmc_result** out,
                                              trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && prompt && clip && out,
                "prompt session, prompt result, clip and output are required");
        auto lock = operation(*session->state);
        const auto& snapshot = require_result<TrackStorage>(prompt);
        require(snapshot.prompt_origin.get() == session->state.get() &&
                    snapshot.host.frames.size() == 1,
                "prompt result must belong to this session");
        std::vector<internal::ImageView> frames;
        const auto input = video_input(*clip, frames);
        auto result =
            session->state->implementation->continue_borrowed(snapshot.host.frames.front(), input);
        complete_clip(result, input);
        *out = make_result<TrackStorage>(std::move(result));
    });
}
const trtmc_frames_text_to_mask_tracks_api_v1 text_clip_api{
    {1, 0, sizeof(text_clip_api)},
    create_text_clip,
    segment_text_clip,
    track_view,
    release_native<trtmc_text_mask_clip_session>};
const trtmc_prompt_frame_text_to_mask_tracks_api_v1 prompt_frame_api{
    {1, 0, sizeof(prompt_frame_api)},
    create_prompt_frame,
    accept_prompt_frame,
    continue_prompt_frame,
    track_view,
    release_native<trtmc_text_prompt_frame_session>};

template <class Editor>
Editor& require_editor(Editor* value) {
    if (!value)
        throw ApiFailure{TRTMC_UNSUPPORTED, "native session does not provide this editor"};
    return *value;
}
trtmc_status TRTMC_CALL create_image_context(trtmc_model* model, const trtmc_image_input_v1* image,
                                             const trtmc_config_view_v1* config,
                                             trtmc_image_mask_context** out,
                                             trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(image && out, "image and context output are required");
        const auto input = image_input(*image);
        const ConvertedConfig options(config);
        auto state = std::make_shared<ImageMaskState>();
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::IInteractiveImageMasks>(
                model, internal::IInteractiveImageMasks::kTask);
            state->owner = model_owner(model);
            state->task_key = internal::contract_key<internal::IInteractiveImageMasks>();
            validate_task_config(state->owner, state->task_key, options.view());
            state->model = std::make_unique<ModelSession>(model);
            state->implementation = family.create_image_context(input, options.view());
        }
        output_check(state->implementation != nullptr, "family returned no image context");
        *out = new trtmc_image_mask_context{std::move(state)};
    });
}
template <class Invoke>
trtmc_status image_edit(trtmc_image_mask_context* context, const trtmc_config_view_v1* config,
                        trtmc_result** out, trtmc_error** error, Invoke invoke) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(context && out, "image context and result output are required");
        auto lock = operation(*context->state);
        const ConvertedConfig options(config);
        validate_task_config(context->state->owner, context->state->task_key, options.view());
        *out = make_masks_result(invoke(*context->state->implementation, options.view()));
    });
}
trtmc_status TRTMC_CALL image_context_points(trtmc_image_mask_context* context,
                                             const trtmc_point_prompt_v1* supplied,
                                             std::uint64_t count,
                                             const trtmc_config_view_v1* config, trtmc_result** out,
                                             trtmc_error** error) noexcept {
    return image_edit(context, config, out, error, [&](auto& native, auto options) {
        auto& editor = require_editor(native.points());
        const auto input = point_prompts_input(supplied, count, true);
        return editor.masks_from_points({input.data(), input.size()}, options);
    });
}
trtmc_status TRTMC_CALL image_context_box(trtmc_image_mask_context* context,
                                          const trtmc_pixel_box_v1* supplied,
                                          const trtmc_config_view_v1* config, trtmc_result** out,
                                          trtmc_error** error) noexcept {
    return image_edit(context, config, out, error, [&](auto& native, auto options) {
        auto& editor = require_editor(native.boxes());
        require(supplied, "box prompt is null");
        return editor.masks_from_box(pixel_box_input(*supplied), options);
    });
}
trtmc_status TRTMC_CALL image_context_prior(trtmc_image_mask_context* context,
                                            const trtmc_image_prior_prompt_v1* supplied,
                                            const trtmc_config_view_v1* config, trtmc_result** out,
                                            trtmc_error** error) noexcept {
    return image_edit(context, config, out, error, [&](auto& native, auto options) {
        auto& editor = require_editor(native.priors());
        require(supplied, "prior prompt is null");
        require(supplied->has_box <= 1, "box presence must be zero or one");
        const auto points = point_prompts_input(supplied->points, supplied->point_count, false);
        internal::ImagePriorPrompt input{
            {matrix_input(supplied->prior_logits)}, {points.data(), points.size()}, {}};
        if (supplied->has_box)
            input.box = pixel_box_input(supplied->box);
        return editor.masks_from_prior(input, options);
    });
}
const trtmc_image_points_editor_api_v1 image_points_editor{{1, 0, sizeof(image_points_editor)},
                                                           image_context_points};
const trtmc_image_box_editor_api_v1 image_box_editor{{1, 0, sizeof(image_box_editor)},
                                                     image_context_box};
const trtmc_image_prior_editor_api_v1 image_prior_editor{{1, 0, sizeof(image_prior_editor)},
                                                         image_context_prior};
trtmc_status TRTMC_CALL image_context_editors(trtmc_image_mask_context* context,
                                              trtmc_image_mask_editors_v1* out,
                                              trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(context && out, "image context and editor output are required");
        auto lock = operation(*context->state);
        auto& native = *context->state->implementation;
        *out = {native.points() ? &image_points_editor : nullptr,
                native.boxes() ? &image_box_editor : nullptr,
                native.priors() ? &image_prior_editor : nullptr};
    });
}
const trtmc_interactive_image_masks_api_v1 image_context_api{
    {1, 0, sizeof(image_context_api)},
    create_image_context,
    image_context_editors,
    masks_result_view,
    release_native<trtmc_image_mask_context>};

std::unique_lock<std::mutex> interactive_operation(InteractiveState& state, bool mutation = true) {
    auto lock = operation(static_cast<NativeState<internal::IMaskTrackSession>&>(state));
    if (mutation && state.traversing)
        throw ApiFailure{TRTMC_BUSY, "tracking traversal owns this session"};
    return lock;
}
void valid_frame(std::uint64_t index, const ClipGeometry& geometry) {
    require(index < geometry.size(), "prompt frame index is outside the clip");
}
void valid_track_frame(const internal::TrackFrameResult& frame, const ClipGeometry& geometry) {
    const auto index = frame.metadata.frame_index;
    output_check(index < geometry.size(), "family tracking frame index is outside the clip");
    output_check(frame.metadata.height == geometry[index].first &&
                     frame.metadata.width == geometry[index].second,
                 "family tracking frame geometry differs from original clip");
}
template <class Input>
struct TrackingPrompt {
    Input input;
    std::vector<internal::PointPrompt> points;
    TrackingPrompt() = default;
    TrackingPrompt(const TrackingPrompt&) = delete;
    TrackingPrompt& operator=(const TrackingPrompt&) = delete;
    TrackingPrompt(TrackingPrompt&&) = default;
};
auto tracking_prompt(const trtmc_frame_object_points_v1& supplied, const ClipGeometry& geometry) {
    valid_frame(supplied.frame_index, geometry);
    require(supplied.update == TRTMC_POINTS_REPLACE || supplied.update == TRTMC_POINTS_APPEND,
            "unknown point update mode");
    TrackingPrompt<internal::FrameObjectPoints> result;
    result.points = point_prompts_input(supplied.points, supplied.point_count, true);
    result.input = {supplied.frame_index,
                    supplied.object_id,
                    {result.points.data(), result.points.size()},
                    static_cast<internal::PointUpdate>(supplied.update)};
    return result;
}
auto tracking_prompt(const trtmc_frame_object_box_v1& supplied, const ClipGeometry& geometry) {
    valid_frame(supplied.frame_index, geometry);
    TrackingPrompt<internal::FrameObjectBox> result;
    result.points =
        point_prompts_input(supplied.correction_points, supplied.correction_point_count, false);
    result.input = {supplied.frame_index,
                    supplied.object_id,
                    pixel_box_input(supplied.box),
                    {result.points.data(), result.points.size()}};
    return result;
}
auto tracking_prompt(const trtmc_frame_object_mask_v1& supplied, const ClipGeometry& geometry) {
    valid_frame(supplied.frame_index, geometry);
    require(supplied.height == geometry[supplied.frame_index].first &&
                supplied.width == geometry[supplied.frame_index].second,
            "binary prompt mask must match original frame geometry");
    checked_size(supplied.height, supplied.width);
    require(supplied.mask_count == static_cast<std::uint64_t>(supplied.height) * supplied.width,
            "binary prompt mask shape mismatch");
    const auto mask = checked_span(supplied.mask, supplied.mask_count);
    for (const auto value : mask)
        require(value <= 1, "binary prompt mask values must be zero or one");
    TrackingPrompt<internal::FrameObjectBinaryMask> result;
    result.input = {
        supplied.frame_index, supplied.object_id, {mask, supplied.height, supplied.width}};
    return result;
}
auto tracking_prompt(const trtmc_frame_text_v1& supplied, const ClipGeometry& geometry) {
    valid_frame(supplied.frame_index, geometry);
    TrackingPrompt<internal::FrameText> result;
    result.input = {supplied.frame_index, string_view(supplied.text)};
    return result;
}
auto tracking_prompt(const trtmc_frame_box_exemplar_v1& supplied, const ClipGeometry& geometry) {
    valid_frame(supplied.frame_index, geometry);
    require(supplied.exemplar.positive <= 1, "exemplar polarity must be zero or one");
    TrackingPrompt<internal::FrameBoxExemplar> result;
    result.input = {supplied.frame_index,
                    {pixel_box_input(supplied.exemplar.box), supplied.exemplar.positive != 0}};
    return result;
}
template <class Interface, class Input, class Invoke>
trtmc_status create_interactive(trtmc_model* model, const trtmc_video_view_v1* clip,
                                const Input* prompt, const trtmc_config_view_v1* config,
                                trtmc_mask_track_session** out, trtmc_result** initial,
                                trtmc_error** error, Invoke invoke) noexcept {
    if (out)
        *out = nullptr;
    if (initial)
        *initial = nullptr;
    return guarded(error, [&] {
        require(clip && prompt && out && initial,
                "clip, typed prompt, session and initial result outputs are required");
        std::vector<internal::ImageView> images;
        const auto input = video_input(*clip, images);
        auto state = std::make_shared<InteractiveState>();
        for (const auto& frame : images)
            state->geometry.emplace_back(frame.height, frame.width);
        const auto supplied = tracking_prompt(*prompt, state->geometry);
        const ConvertedConfig options(config);
        internal::MaskTrackSessionStart start;
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<Interface>(model, Interface::kTask);
            state->owner = model_owner(model);
            state->task_key = internal::contract_key<Interface>();
            validate_task_config(state->owner, state->task_key, options.view());
            state->model = std::make_unique<ModelSession>(model);
            start = invoke(family, input, supplied.input, options.view());
        }
        output_check(start.session != nullptr, "family returned no interactive tracking session");
        state->implementation = std::move(start.session);
        valid_track_frame(start.prompted_frame, state->geometry);
        output_check(start.prompted_frame.metadata.frame_index == supplied.input.frame_index,
                     "family initial result does not match prompt frame");
        internal::TrackClipResult result;
        result.frames.push_back(std::move(start.prompted_frame));
        std::unique_ptr<trtmc_result> snapshot(make_result<TrackStorage>(std::move(result)));
        *out = new trtmc_mask_track_session{std::move(state)};
        *initial = snapshot.release();
    });
}
#define TRACK_CREATE(Name, Interface, Input, Method)                                               \
    trtmc_status TRTMC_CALL Name(trtmc_model* model, const trtmc_video_view_v1* clip,              \
                                 const Input* prompt, const trtmc_config_view_v1* config,          \
                                 trtmc_mask_track_session** out, trtmc_result** initial,           \
                                 trtmc_error** error) noexcept {                                   \
        return create_interactive<internal::Interface>(                                            \
            model, clip, prompt, config, out, initial, error,                                      \
            [](auto& family, auto video, const auto& supplied, auto options) {                     \
                return family.Method(video, supplied, options);                                    \
            });                                                                                    \
    }
TRACK_CREATE(create_points_tracks, IFramesPointsToMaskTracks, trtmc_frame_object_points_v1,
             create_points_tracks)
TRACK_CREATE(create_box_tracks, IFramesBoxToMaskTracks, trtmc_frame_object_box_v1,
             create_box_tracks)
TRACK_CREATE(create_mask_tracks, IFramesMaskToMaskTracks, trtmc_frame_object_mask_v1,
             create_mask_tracks)
TRACK_CREATE(create_text_tracks, IInteractiveFramesTextToMaskTracks, trtmc_frame_text_v1,
             create_text_tracks)
TRACK_CREATE(create_exemplar_tracks, IFramesBoxExemplarToMaskTracks, trtmc_frame_box_exemplar_v1,
             create_exemplar_tracks)
#undef TRACK_CREATE
template <class Input, class Getter, class Invoke>
trtmc_status track_edit(trtmc_mask_track_session* session, const Input* supplied,
                        const trtmc_config_view_v1* config, trtmc_result** out, trtmc_error** error,
                        Getter getter, Invoke invoke) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && supplied && out, "session, typed prompt and result output are required");
        auto lock = interactive_operation(*session->state);
        const auto input = tracking_prompt(*supplied, session->state->geometry);
        const ConvertedConfig options(config);
        validate_task_config(session->state->owner, session->state->task_key, options.view());
        auto& editor = require_editor(getter(*session->state->implementation));
        internal::TrackClipResult result;
        result.frames.push_back(invoke(editor, input.input, options.view()));
        valid_track_frame(result.frames.front(), session->state->geometry);
        output_check(result.frames.front().metadata.frame_index == input.input.frame_index,
                     "editor returned wrong prompt frame");
        *out = make_result<TrackStorage>(std::move(result));
    });
}
#define TRACK_EDIT(Name, Input, Getter, Method)                                                    \
    trtmc_status TRTMC_CALL Name(trtmc_mask_track_session* session, const Input* supplied,         \
                                 const trtmc_config_view_v1* config, trtmc_result** out,           \
                                 trtmc_error** error) noexcept {                                   \
        return track_edit(                                                                         \
            session, supplied, config, out, error, [](auto& native) { return native.Getter(); },   \
            [](auto& editor, const auto& input, auto options) {                                    \
                return editor.Method(input, options);                                              \
            });                                                                                    \
    }
TRACK_EDIT(edit_track_points, trtmc_frame_object_points_v1, points, update_points)
TRACK_EDIT(edit_track_box, trtmc_frame_object_box_v1, boxes, update_box)
TRACK_EDIT(edit_track_mask, trtmc_frame_object_mask_v1, masks, update_mask)
TRACK_EDIT(edit_track_text, trtmc_frame_text_v1, text, replace_text)
TRACK_EDIT(edit_track_exemplar, trtmc_frame_box_exemplar_v1, exemplars, replace_exemplar)
#undef TRACK_EDIT
trtmc_status TRTMC_CALL remove_track_object(trtmc_mask_track_session* session, std::int64_t object,
                                            trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && out, "session and result output are required");
        auto lock = interactive_operation(*session->state);
        auto result =
            require_editor(session->state->implementation->objects()).remove_object(object);
        for (const auto& frame : result.frames)
            valid_track_frame(frame, session->state->geometry);
        *out = make_result<TrackStorage>(std::move(result));
    });
}
trtmc_status TRTMC_CALL reset_tracks(trtmc_mask_track_session* session,
                                     trtmc_error** error) noexcept {
    return guarded(error, [&] {
        require(session, "session is null");
        auto lock = interactive_operation(*session->state);
        require_editor(session->state->implementation->resetter()).reset();
    });
}
#define TRACK_EDITOR_TABLE(Name, Type, Method) const Type Name{{1, 0, sizeof(Name)}, Method};
TRACK_EDITOR_TABLE(track_points_editor, trtmc_points_track_editor_api_v1, edit_track_points)
TRACK_EDITOR_TABLE(track_box_editor, trtmc_box_track_editor_api_v1, edit_track_box)
TRACK_EDITOR_TABLE(track_mask_editor, trtmc_mask_track_editor_api_v1, edit_track_mask)
TRACK_EDITOR_TABLE(track_text_editor, trtmc_text_track_editor_api_v1, edit_track_text)
TRACK_EDITOR_TABLE(track_exemplar_editor, trtmc_exemplar_track_editor_api_v1, edit_track_exemplar)
TRACK_EDITOR_TABLE(track_object_removal, trtmc_track_object_removal_api_v1, remove_track_object)
TRACK_EDITOR_TABLE(track_reset, trtmc_track_reset_api_v1, reset_tracks)
#undef TRACK_EDITOR_TABLE
trtmc_status TRTMC_CALL get_track_editors(trtmc_mask_track_session* session,
                                          trtmc_mask_track_editors_v1* out,
                                          trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(session && out, "session and editor output are required");
        auto lock = interactive_operation(*session->state, false);
        auto& native = *session->state->implementation;
        *out = {native.points() ? &track_points_editor : nullptr,
                native.boxes() ? &track_box_editor : nullptr,
                native.masks() ? &track_mask_editor : nullptr,
                native.text() ? &track_text_editor : nullptr,
                native.exemplars() ? &track_exemplar_editor : nullptr,
                native.objects() ? &track_object_removal : nullptr,
                native.resetter() ? &track_reset : nullptr};
    });
}
trtmc_status TRTMC_CALL start_track_propagation(trtmc_mask_track_session* session,
                                                const trtmc_propagation_range_v1* range,
                                                trtmc_track_propagation** out,
                                                trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && range && out, "session, propagation range and output are required");
        auto lock = interactive_operation(*session->state);
        valid_frame(range->start_frame, session->state->geometry);
        require(range->frame_count > 0, "propagation requires a nonempty frame range");
        require(range->direction == TRTMC_PROPAGATE_FORWARD ||
                    range->direction == TRTMC_PROPAGATE_BACKWARD,
                "unknown propagation direction");
        const auto available = range->direction == TRTMC_PROPAGATE_FORWARD
                                   ? session->state->geometry.size() - range->start_frame
                                   : range->start_frame + 1;
        require(range->frame_count <= available, "propagation range extends outside the clip");
        auto native = session->state->implementation->start_propagation(
            {range->start_frame, range->frame_count,
             static_cast<internal::PropagationDirection>(range->direction)});
        output_check(native != nullptr, "family returned no tracking traversal");
        auto handle = std::make_unique<trtmc_track_propagation>();
        handle->state = session->state;
        handle->implementation = std::move(native);
        session->state->traversing = true;
        *out = handle.release();
    });
}
std::unique_lock<std::mutex> traversal_operation(trtmc_track_propagation* traversal) {
    require(traversal, "traversal is null");
    std::unique_lock<std::mutex> lock(traversal->state->operation, std::try_to_lock);
    if (!lock.owns_lock())
        throw ApiFailure{TRTMC_BUSY, "another tracking operation is active"};
    return lock;
}
trtmc_status TRTMC_CALL next_track(trtmc_track_propagation* traversal, trtmc_result** out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    bool ended = false;
    const auto status = guarded(error, [&] {
        require(out, "track result output is null");
        auto lock = traversal_operation(traversal);
        if (traversal->ended) {
            ended = true;
            return;
        }
        auto frame = traversal->implementation->next();
        if (!frame) {
            traversal->ended = true;
            ended = true;
            return;
        }
        valid_track_frame(*frame, traversal->state->geometry);
        internal::TrackClipResult result;
        result.frames.push_back(std::move(*frame));
        *out = make_result<TrackStorage>(std::move(result));
    });
    return status == TRTMC_OK && ended ? TRTMC_END : status;
}
trtmc_status TRTMC_CALL cancel_track(trtmc_track_propagation* traversal,
                                     trtmc_error** error) noexcept {
    return guarded(error, [&] {
        auto lock = traversal_operation(traversal);
        if (!traversal->ended) {
            traversal->implementation->cancel();
            traversal->ended = true;
        }
    });
}
void TRTMC_CALL release_track_propagation(trtmc_track_propagation* traversal) noexcept {
    if (!traversal)
        return;
    {
        const std::lock_guard<std::mutex> lock(traversal->state->operation);
        if (!traversal->ended)
            traversal->implementation->cancel();
        traversal->implementation.reset();
        traversal->state->traversing = false;
    }
    delete traversal;
}
void TRTMC_CALL release_interactive(trtmc_mask_track_session* session) noexcept {
    if (!session)
        return;
    {
        const std::lock_guard<std::mutex> lock(session->state->operation);
        session->state->closed = true;
    }
    // A live traversal retains native session/model resources, without a dangling parent handle.
    delete session;
}
const trtmc_mask_track_session_api_v1 interactive_session_api{
    {1, 0, sizeof(interactive_session_api)},
    get_track_editors,
    start_track_propagation,
    next_track,
    cancel_track,
    release_track_propagation,
    track_view,
    release_interactive};
#define TRACK_FACTORY_TABLE(Name, Type, Method)                                                    \
    const Type Name{{1, 0, sizeof(Name)}, Method, &interactive_session_api};
TRACK_FACTORY_TABLE(points_tracks_api, trtmc_frames_points_to_mask_tracks_api_v1,
                    create_points_tracks)
TRACK_FACTORY_TABLE(box_tracks_api, trtmc_frames_box_to_mask_tracks_api_v1, create_box_tracks)
TRACK_FACTORY_TABLE(mask_tracks_api, trtmc_frames_mask_to_mask_tracks_api_v1, create_mask_tracks)
TRACK_FACTORY_TABLE(interactive_text_tracks_api,
                    trtmc_interactive_frames_text_to_mask_tracks_api_v1, create_text_tracks)
TRACK_FACTORY_TABLE(exemplar_tracks_api, trtmc_frames_box_exemplar_to_mask_tracks_api_v1,
                    create_exemplar_tracks)
#undef TRACK_FACTORY_TABLE
trtmc_status TRTMC_CALL create_crop_pose(trtmc_model* model, const trtmc_config_view_v1* config,
                                         trtmc_crop_pose_session** out,
                                         trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "pose session output is null");
        const ConvertedConfig options(config);
        auto state = std::make_shared<CropPoseState>();
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::ICropPoseTracking>(
                model, internal::ICropPoseTracking::kTask);
            state->owner = model_owner(model);
            state->task_key = internal::contract_key<internal::ICropPoseTracking>();
            validate_task_config(state->owner, state->task_key, options.view());
            state->model = std::make_unique<ModelSession>(model);
            state->implementation = family.create_crop_pose_session(options.view());
        }
        output_check(state->implementation != nullptr, "family returned no crop pose session");
        *out = new trtmc_crop_pose_session{std::move(state)};
    });
}
trtmc_status TRTMC_CALL initialize_crop_pose(trtmc_crop_pose_session* session,
                                             const trtmc_pose_refinement_request_v1* request,
                                             const trtmc_config_view_v1* config, trtmc_result** out,
                                             trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && request && out, "pose session, request and result output are required");
        auto lock = operation(*session->state);
        const auto input = pose_refinement_input(*request);
        const ConvertedConfig options(config);
        validate_task_config(session->state->owner, session->state->task_key, options.view());
        *out = make_refined_poses_result(
            session->state->implementation->initialize(input, options.view()));
    });
}
trtmc_status TRTMC_CALL track_crop_pose(trtmc_crop_pose_session* session, void* context,
                                        trtmc_pose_crop_callback_v1 callback,
                                        const trtmc_config_view_v1* config, trtmc_result** out,
                                        trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && callback && out,
                "pose session, per-call callback and result output are required");
        auto lock = operation(*session->state);
        const ConvertedConfig options(config);
        validate_task_config(session->state->owner, session->state->task_key, options.view());
        const internal::PoseCropProvider provider =
            [context, callback](const internal::PoseCropRequest& query) {
                return pose_crops_input(context, callback, query);
            };
        *out = make_refined_poses_result(
            session->state->implementation->track(provider, options.view()));
    });
}
trtmc_status TRTMC_CALL reset_crop_pose(trtmc_crop_pose_session* session,
                                        trtmc_error** error) noexcept {
    return guarded(error, [&] {
        require(session, "pose session is null");
        auto lock = operation(*session->state);
        session->state->implementation->reset();
    });
}
std::array<float, 16> object_pose_input(const trtmc_object_pose_matrix_v1& input) {
    std::array<float, 16> result;
    std::copy(input.object_to_camera, input.object_to_camera + 16, result.begin());
    for (const auto value : result)
        require(std::isfinite(value), "object pose matrix must be finite");
    return result;
}
trtmc_status TRTMC_CALL create_rgbd_pose(trtmc_model* model, const trtmc_triangle_mesh_v1* mesh,
                                         const trtmc_object_pose_matrix_v1* pose,
                                         const trtmc_config_view_v1* config,
                                         trtmc_rgbd_pose_session** out,
                                         trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(mesh && pose && out,
                "mesh, initial original-object pose and session output are required");
        const auto mesh_input = triangle_mesh_input(*mesh);
        const auto initial = object_pose_input(*pose);
        const ConvertedConfig options(config);
        auto state = std::make_shared<RgbdPoseState>();
        {
            const std::lock_guard<std::mutex> lock(model_mutex(model));
            auto& family = require_interface<internal::IRgbdInitializedPoseToTrackedPose>(
                model, internal::IRgbdInitializedPoseToTrackedPose::kTask);
            state->owner = model_owner(model);
            state->task_key = internal::contract_key<internal::IRgbdInitializedPoseToTrackedPose>();
            validate_task_config(state->owner, state->task_key, options.view());
            state->model = std::make_unique<ModelSession>(model);
            state->implementation =
                family.create_rgbd_pose_session(mesh_input, initial, options.view());
        }
        output_check(state->implementation != nullptr, "family returned no RGBD pose session");
        *out = new trtmc_rgbd_pose_session{std::move(state)};
    });
}
trtmc_status TRTMC_CALL track_rgbd_pose(trtmc_rgbd_pose_session* session,
                                        const trtmc_rgbd_observation_v1* request,
                                        const trtmc_config_view_v1* config, trtmc_result** out,
                                        trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(session && request && out,
                "RGBD session, observation and result output are required");
        auto lock = operation(*session->state);
        const auto rgb = image_input(request->rgb);
        const auto depth = matrix_input(request->depth_meters);
        require(rgb.channels == 3 && depth.rows == rgb.height && depth.columns == rgb.width,
                "RGB-D inputs must be aligned RGB and HW depth");
        const internal::RgbdObservation input{rgb, depth,
                                              pixel_intrinsics_input(request->pixel_intrinsics)};
        const ConvertedConfig options(config);
        validate_task_config(session->state->owner, session->state->task_key, options.view());
        *out =
            make_object_pose_result(session->state->implementation->track(input, options.view()));
    });
}
trtmc_status TRTMC_CALL reset_rgbd_pose(trtmc_rgbd_pose_session* session,
                                        const trtmc_object_pose_matrix_v1* pose,
                                        trtmc_error** error) noexcept {
    return guarded(error, [&] {
        require(session && pose, "RGBD session and new original-object pose are required");
        auto lock = operation(*session->state);
        session->state->implementation->reset(object_pose_input(*pose));
    });
}
const trtmc_crop_pose_tracking_api_v1 crop_pose_api{{1, 0, sizeof(crop_pose_api)},
                                                    create_crop_pose,
                                                    initialize_crop_pose,
                                                    track_crop_pose,
                                                    reset_crop_pose,
                                                    refined_poses_result_view,
                                                    release_native<trtmc_crop_pose_session>};
const trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1 rgbd_pose_api{
    {1, 0, sizeof(rgbd_pose_api)},
    create_rgbd_pose,
    track_rgbd_pose,
    reset_rgbd_pose,
    object_pose_result_view,
    release_native<trtmc_rgbd_pose_session>};
} // namespace
Span<const TaskBinding> tracking_task_bindings() noexcept {
    static const TaskBinding bindings[] = {
        {internal::IFramesToDetectedMaskTracks::kTask, 1, 0, &detected_api.header},
        {internal::IFramesTextToMaskTracks::kTask, 1, 0, &text_clip_api.header},
        {internal::IPromptFrameTextToMaskTracks::kTask, 1, 0, &prompt_frame_api.header},
        {internal::IInteractiveImageMasks::kTask, 1, 0, &image_context_api.header},
        {internal::IFramesPointsToMaskTracks::kTask, 1, 0, &points_tracks_api.header},
        {internal::IFramesBoxToMaskTracks::kTask, 1, 0, &box_tracks_api.header},
        {internal::IFramesMaskToMaskTracks::kTask, 1, 0, &mask_tracks_api.header},
        {internal::IInteractiveFramesTextToMaskTracks::kTask, 1, 0,
         &interactive_text_tracks_api.header},
        {internal::IFramesBoxExemplarToMaskTracks::kTask, 1, 0, &exemplar_tracks_api.header},
        {internal::ICropPoseTracking::kTask, 1, 0, &crop_pose_api.header},
        {internal::IRgbdInitializedPoseToTrackedPose::kTask, 1, 0, &rgbd_pose_api.header}};
    return bindings;
}
} // namespace trtmc::api
