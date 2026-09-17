/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/core.hpp"
#include "trtmc/perception.hpp"
#include "trtmc/tracking.h"
#include "trtmc/video.hpp"

namespace trtmc {
using TrackClipResult = detail::ViewResult<trtmc_track_clip_view_v1>;

namespace detail {
template <class Handle, class Api>
struct TrackingSessionOwner {
    TrackingSessionOwner(std::shared_ptr<ModelState> model, const Api* api)
        : model(std::move(model)), api(api) {}
    ~TrackingSessionOwner() {
        if (handle)
            api->release(handle);
    }
    std::shared_ptr<ModelState> model;
    const Api* api;
    Handle* handle{nullptr};
};
using DetectedSessionOwner =
    TrackingSessionOwner<trtmc_detected_mask_session, trtmc_frames_to_detected_mask_tracks_api_v1>;
using TextClipOwner =
    TrackingSessionOwner<trtmc_text_mask_clip_session, trtmc_frames_text_to_mask_tracks_api_v1>;
using PromptFrameOwner = TrackingSessionOwner<trtmc_text_prompt_frame_session,
                                              trtmc_prompt_frame_text_to_mask_tracks_api_v1>;
using ImageMaskOwner =
    TrackingSessionOwner<trtmc_image_mask_context, trtmc_interactive_image_masks_api_v1>;
using MaskTrackOwner =
    TrackingSessionOwner<trtmc_mask_track_session, trtmc_mask_track_session_api_v1>;
using CropPoseOwner =
    TrackingSessionOwner<trtmc_crop_pose_session, trtmc_crop_pose_tracking_api_v1>;
using RgbdPoseOwner = TrackingSessionOwner<trtmc_rgbd_pose_session,
                                           trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1>;
template <class Api>
void validate_session_table(const Api* table) {
    if (table && (table->header.major != 1 || table->header.minor != 0 ||
                  table->header.byte_size < sizeof(Api)))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible typed session table");
}
template <class Function>
TrackClipResult tracking_call(const std::shared_ptr<DetectedSessionOwner>& owner, Function invoke,
                              const VideoInput& clip, const Config& config,
                              typename TrackClipResult::ReadView view) {
    VideoWireInputs inputs;
    const auto wire = inputs.video(clip);
    auto entries = config.c_entries();
    auto options = entries.view();
    trtmc_result* raw = nullptr;
    trtmc_error* error = nullptr;
    const auto status = invoke(owner->handle, &wire, &options, &raw, &error);
    ResultOwner result(owner->model, raw);
    check(owner->model->api, status, error);
    return TrackClipResult(std::move(result), view);
}
} // namespace detail

class DetectedDeviceMasks {
  public:
    // Result metadata is owned; device addresses expire on session reuse/close.
    TrackClipResult segment_device(const VideoInput& clip, const Config& config = {}) const {
        return detail::tracking_call(owner_, api_->segment_device, clip, config, api_->result_view);
    }

  private:
    friend class DetectedMaskSession;
    DetectedDeviceMasks(std::shared_ptr<detail::DetectedSessionOwner> owner,
                        const trtmc_detected_device_masks_api_v1* api)
        : owner_(std::move(owner)), api_(api) {}
    std::shared_ptr<detail::DetectedSessionOwner> owner_;
    const trtmc_detected_device_masks_api_v1* api_;
};
class DetectedMaskSession {
  public:
    TrackClipResult segment(const VideoInput& clip, const Config& config = {}) const {
        return detail::tracking_call(owner_, owner_->api->segment, clip, config,
                                     owner_->api->result_view);
    }
    bool supports_device_masks() const {
        const trtmc_detected_device_masks_api_v1* table = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->api->get_device_api(owner_->handle, 1, 0, &table, &error);
        if (status == TRTMC_UNSUPPORTED) {
            owner_->model->api.error_release(error);
            return false;
        }
        detail::check(owner_->model->api, status, error);
        return true;
    }
    DetectedDeviceMasks device_masks() const {
        const trtmc_detected_device_masks_api_v1* table = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->api->get_device_api(owner_->handle, 1, 0, &table, &error);
        detail::check(owner_->model->api, status, error);
        if (!table || table->header.major != 1 || table->header.minor != 0 ||
            table->header.byte_size < sizeof(*table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible device mask table");
        return DetectedDeviceMasks(owner_, table);
    }
    // Explicit close invalidates borrowed device views, including proxy copies.
    void close() {
        if (owner_->handle) {
            owner_->api->release(owner_->handle);
            owner_->handle = nullptr;
        }
    }

  private:
    friend class FramesToDetectedMaskTracks;
    explicit DetectedMaskSession(std::shared_ptr<detail::DetectedSessionOwner> owner)
        : owner_(std::move(owner)) {}
    std::shared_ptr<detail::DetectedSessionOwner> owner_;
};
class FramesToDetectedMaskTracks {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_FRAMES_TO_DETECTED_MASK_TRACKS;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    DetectedMaskSession create(const Config& config = {}) const {
        auto owner = std::make_shared<detail::DetectedSessionOwner>(model_, api_);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_error* error = nullptr;
        const auto status = api_->create(model_->handle, &options, &owner->handle, &error);
        detail::check(model_->api, status, error);
        return DetectedMaskSession(std::move(owner));
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_frames_to_detected_mask_tracks_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible detector tracking table");
    }

  private:
    friend class Model;
    FramesToDetectedMaskTracks(std::shared_ptr<detail::ModelState> model,
                               const trtmc_api_header* table)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_frames_to_detected_mask_tracks_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_frames_to_detected_mask_tracks_api_v1* api_;
};

class TextMaskClipSession {
  public:
    TrackClipResult segment(const VideoInput& clip, std::string_view text,
                            const Config& config = {}) const {
        detail::VideoWireInputs inputs;
        const auto wire = inputs.video(clip);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->api->segment(owner_->handle, &wire, detail::c_string(text),
                                                 &options, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return TrackClipResult(std::move(result), owner_->api->result_view);
    }
    void close() {
        if (owner_->handle) {
            owner_->api->release(owner_->handle);
            owner_->handle = nullptr;
        }
    }

  private:
    friend class FramesTextToMaskTracks;
    explicit TextMaskClipSession(std::shared_ptr<detail::TextClipOwner> owner)
        : owner_(std::move(owner)) {}
    std::shared_ptr<detail::TextClipOwner> owner_;
};
class FramesTextToMaskTracks {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_FRAMES_TEXT_TO_MASK_TRACKS;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    TextMaskClipSession create(const Config& config = {}) const {
        auto owner = std::make_shared<detail::TextClipOwner>(model_, api_);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_error* error = nullptr;
        const auto status = api_->create(model_->handle, &options, &owner->handle, &error);
        detail::check(model_->api, status, error);
        return TextMaskClipSession(std::move(owner));
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_frames_text_to_mask_tracks_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible text clip tracking table");
    }

  private:
    friend class Model;
    FramesTextToMaskTracks(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_frames_text_to_mask_tracks_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_frames_text_to_mask_tracks_api_v1* api_;
};
class TextPromptFrameSession {
  public:
    TrackClipResult accept_prompt_frame(const ImageInput& image) const {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status =
            owner_->api->accept_prompt_frame(owner_->handle, &image.wire, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return TrackClipResult(std::move(result), owner_->api->result_view);
    }
    TrackClipResult continue_borrowed(const TrackClipResult& prompt, const VideoInput& clip) const {
        detail::VideoWireInputs inputs;
        const auto wire = inputs.video(clip);
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->api->continue_borrowed(
            owner_->handle, detail::ViewResultAccess::get(prompt), &wire, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return TrackClipResult(std::move(result), owner_->api->result_view);
    }
    void close() {
        if (owner_->handle) {
            owner_->api->release(owner_->handle);
            owner_->handle = nullptr;
        }
    }

  private:
    friend class PromptFrameTextToMaskTracks;
    explicit TextPromptFrameSession(std::shared_ptr<detail::PromptFrameOwner> owner)
        : owner_(std::move(owner)) {}
    std::shared_ptr<detail::PromptFrameOwner> owner_;
};
class PromptFrameTextToMaskTracks {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_PROMPT_FRAME_TEXT_TO_MASK_TRACKS;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    TextPromptFrameSession create(std::string_view text, const Config& config = {}) const {
        auto owner = std::make_shared<detail::PromptFrameOwner>(model_, api_);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_error* error = nullptr;
        const auto status =
            api_->create(model_->handle, detail::c_string(text), &options, &owner->handle, &error);
        detail::check(model_->api, status, error);
        return TextPromptFrameSession(std::move(owner));
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_prompt_frame_text_to_mask_tracks_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible prompt-frame tracking table");
    }

  private:
    friend class Model;
    PromptFrameTextToMaskTracks(std::shared_ptr<detail::ModelState> model,
                                const trtmc_api_header* table)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_prompt_frame_text_to_mask_tracks_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_prompt_frame_text_to_mask_tracks_api_v1* api_;
};
struct ImagePriorPrompt {
    FloatMatrixView prior_logits;
    Span<const PointPrompt> points;
    std::optional<PixelBox> box;
};
class ImageMaskContext {
  public:
    bool supports_points() const { return editors().points != nullptr; }
    bool supports_box() const { return editors().box != nullptr; }
    bool supports_prior() const { return editors().prior != nullptr; }
    MasksResult points(Span<const PointPrompt> points, const Config& config = {}) const {
        const auto* api = editors().points;
        if (!api)
            throw Error(TRTMC_UNSUPPORTED, "image point editor unavailable");
        std::vector<trtmc_point_prompt_v1> input;
        for (const auto& point : points)
            input.push_back({point.point, point.foreground ? 1U : 0U});
        return invoke(config, [&](auto options, auto out, auto error) {
            return api->run(owner_->handle, input.data(), input.size(), options, out, error);
        });
    }
    MasksResult box(PixelBox box, const Config& config = {}) const {
        const auto* api = editors().box;
        if (!api)
            throw Error(TRTMC_UNSUPPORTED, "image box editor unavailable");
        return invoke(config, [&](auto options, auto out, auto error) {
            return api->run(owner_->handle, &box, options, out, error);
        });
    }
    MasksResult prior(const ImagePriorPrompt& prompt, const Config& config = {}) const {
        const auto* api = editors().prior;
        if (!api)
            throw Error(TRTMC_UNSUPPORTED, "image prior editor unavailable");
        std::vector<trtmc_point_prompt_v1> points;
        for (const auto& point : prompt.points)
            points.push_back({point.point, point.foreground ? 1U : 0U});
        const trtmc_image_prior_prompt_v1 input{prompt.prior_logits.c_view(), points.data(),
                                                points.size(), prompt.box.has_value() ? 1U : 0U,
                                                prompt.box.value_or(PixelBox{})};
        return invoke(config, [&](auto options, auto out, auto error) {
            return api->run(owner_->handle, &input, options, out, error);
        });
    }
    void close() {
        if (owner_->handle) {
            owner_->api->release(owner_->handle);
            owner_->handle = nullptr;
        }
    }

  private:
    friend class InteractiveImageMasks;
    explicit ImageMaskContext(std::shared_ptr<detail::ImageMaskOwner> owner)
        : owner_(std::move(owner)) {}
    trtmc_image_mask_editors_v1 editors() const {
        trtmc_image_mask_editors_v1 result{};
        trtmc_error* error = nullptr;
        const auto status = owner_->api->get_editors(owner_->handle, &result, &error);
        detail::check(owner_->model->api, status, error);
        detail::validate_session_table(result.points);
        detail::validate_session_table(result.box);
        detail::validate_session_table(result.prior);
        return result;
    }
    template <class Invoke>
    MasksResult invoke(const Config& config, Invoke callback) const {
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = callback(&options, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return MasksResult(std::move(result), owner_->api->result_view);
    }
    std::shared_ptr<detail::ImageMaskOwner> owner_;
};
class InteractiveImageMasks {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_INTERACTIVE_IMAGE_MASKS;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    ImageMaskContext create(const ImageInput& image, const Config& config = {}) const {
        auto owner = std::make_shared<detail::ImageMaskOwner>(model_, api_);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_error* error = nullptr;
        const auto status =
            api_->create(model_->handle, &image.wire, &options, &owner->handle, &error);
        detail::check(model_->api, status, error);
        return ImageMaskContext(std::move(owner));
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_interactive_image_masks_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible image context table");
    }

  private:
    friend class Model;
    InteractiveImageMasks(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_interactive_image_masks_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_interactive_image_masks_api_v1* api_;
};
enum class PointUpdate : std::uint32_t {
    Replace = TRTMC_POINTS_REPLACE,
    Append = TRTMC_POINTS_APPEND
};
struct FrameObjectPoints {
    std::uint64_t frame_index{0};
    std::int64_t object_id{0};
    Span<const PointPrompt> points;
    PointUpdate update{PointUpdate::Replace};
};
struct FrameObjectBox {
    std::uint64_t frame_index{0};
    std::int64_t object_id{0};
    PixelBox box;
    Span<const PointPrompt> correction_points;
};
struct FrameObjectBinaryMask {
    std::uint64_t frame_index{0};
    std::int64_t object_id{0};
    Span<const std::uint8_t> mask;
    std::uint32_t height{0}, width{0};
};
struct FrameText {
    std::uint64_t frame_index{0};
    std::string_view text;
};
struct FrameBoxExemplar {
    std::uint64_t frame_index{0};
    BoxExemplar exemplar;
};
enum class PropagationDirection : std::uint32_t {
    Forward = TRTMC_PROPAGATE_FORWARD,
    Backward = TRTMC_PROPAGATE_BACKWARD
};
struct PropagationRange {
    std::uint64_t start_frame{0}, frame_count{0};
    PropagationDirection direction{PropagationDirection::Forward};
};
namespace detail {
template <class Wire>
struct TrackingPromptWire {
    explicit TrackingPromptWire(Wire value) : wire(std::move(value)) {}
    Wire wire;
    std::vector<trtmc_point_prompt_v1> points;
    TrackingPromptWire(const TrackingPromptWire&) = delete;
    TrackingPromptWire(TrackingPromptWire&&) = default;
};
inline auto tracking_prompt_wire(const FrameObjectPoints& input) {
    TrackingPromptWire<trtmc_frame_object_points_v1> result(
        {input.frame_index, input.object_id, nullptr, 0, static_cast<std::uint32_t>(input.update)});
    for (const auto& point : input.points)
        result.points.push_back({point.point, point.foreground ? 1U : 0U});
    result.wire.points = result.points.data();
    result.wire.point_count = result.points.size();
    return result;
}
inline auto tracking_prompt_wire(const FrameObjectBox& input) {
    TrackingPromptWire<trtmc_frame_object_box_v1> result(
        {input.frame_index, input.object_id, input.box, nullptr, 0});
    for (const auto& point : input.correction_points)
        result.points.push_back({point.point, point.foreground ? 1U : 0U});
    result.wire.correction_points = result.points.data();
    result.wire.correction_point_count = result.points.size();
    return result;
}
inline auto tracking_prompt_wire(const FrameObjectBinaryMask& input) {
    return TrackingPromptWire<trtmc_frame_object_mask_v1>({input.frame_index, input.object_id,
                                                           input.mask.data(), input.mask.size(),
                                                           input.height, input.width});
}
inline auto tracking_prompt_wire(const FrameText& input) {
    return TrackingPromptWire<trtmc_frame_text_v1>({input.frame_index, c_string(input.text)});
}
inline auto tracking_prompt_wire(const FrameBoxExemplar& input) {
    return TrackingPromptWire<trtmc_frame_box_exemplar_v1>(
        {input.frame_index, {input.exemplar.box, input.exemplar.positive ? 1U : 0U}});
}
struct TrackPropagationOwner {
    explicit TrackPropagationOwner(std::shared_ptr<MaskTrackOwner> parent)
        : parent(std::move(parent)) {}
    ~TrackPropagationOwner() {
        if (handle)
            parent->api->release_propagation(handle);
    }
    std::shared_ptr<MaskTrackOwner> parent;
    trtmc_track_propagation* handle{nullptr};
};
} // namespace detail
class TrackPropagation {
  public:
    TrackPropagation(TrackPropagation&&) noexcept = default;
    TrackPropagation& operator=(TrackPropagation&&) noexcept = default;
    TrackPropagation(const TrackPropagation&) = delete;
    TrackPropagation& operator=(const TrackPropagation&) = delete;
    std::optional<TrackClipResult> next() const {
        if (!owner_)
            throw Error(TRTMC_INVALID_ARGUMENT, "tracking traversal is closed");
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->parent->api->next(owner_->handle, &raw, &error);
        detail::ResultOwner result(owner_->parent->model, raw);
        if (status == TRTMC_END) {
            owner_->parent->model->api.error_release(error);
            return std::nullopt;
        }
        detail::check(owner_->parent->model->api, status, error);
        return TrackClipResult(std::move(result), owner_->parent->api->result_view);
    }
    void cancel() const {
        if (!owner_)
            throw Error(TRTMC_INVALID_ARGUMENT, "tracking traversal is closed");
        trtmc_error* error = nullptr;
        const auto status = owner_->parent->api->cancel(owner_->handle, &error);
        detail::check(owner_->parent->model->api, status, error);
    }
    void close() { owner_.reset(); }

  private:
    friend class MaskTrackingSession;
    explicit TrackPropagation(std::unique_ptr<detail::TrackPropagationOwner> owner)
        : owner_(std::move(owner)) {}
    std::unique_ptr<detail::TrackPropagationOwner> owner_;
};
class MaskTrackingSession {
  public:
    bool supports_points() const { return editors().points != nullptr; }
    bool supports_box() const { return editors().box != nullptr; }
    bool supports_mask() const { return editors().mask != nullptr; }
    bool supports_text() const { return editors().text != nullptr; }
    bool supports_exemplar() const { return editors().exemplar != nullptr; }
    bool supports_removal() const { return editors().objects != nullptr; }
    bool supports_reset() const { return editors().reset != nullptr; }
    TrackClipResult update_points(const FrameObjectPoints& prompt,
                                  const Config& config = {}) const {
        return edit(editors().points, prompt, config);
    }
    TrackClipResult update_box(const FrameObjectBox& prompt, const Config& config = {}) const {
        return edit(editors().box, prompt, config);
    }
    TrackClipResult update_mask(const FrameObjectBinaryMask& prompt,
                                const Config& config = {}) const {
        return edit(editors().mask, prompt, config);
    }
    TrackClipResult replace_text(const FrameText& prompt, const Config& config = {}) const {
        return edit(editors().text, prompt, config);
    }
    TrackClipResult replace_exemplar(const FrameBoxExemplar& prompt,
                                     const Config& config = {}) const {
        return edit(editors().exemplar, prompt, config);
    }
    TrackClipResult remove_object(std::int64_t id) const {
        const auto* api = editors().objects;
        if (!api)
            throw Error(TRTMC_UNSUPPORTED, "track object removal unavailable");
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api->remove(owner_->handle, id, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return TrackClipResult(std::move(result), owner_->api->result_view);
    }
    void reset() const {
        const auto* api = editors().reset;
        if (!api)
            throw Error(TRTMC_UNSUPPORTED, "track reset unavailable");
        trtmc_error* error = nullptr;
        const auto status = api->reset(owner_->handle, &error);
        detail::check(owner_->model->api, status, error);
    }
    TrackPropagation propagate(PropagationRange range) const {
        const trtmc_propagation_range_v1 input{range.start_frame, range.frame_count,
                                               static_cast<std::uint32_t>(range.direction)};
        auto traversal = std::make_unique<detail::TrackPropagationOwner>(owner_);
        trtmc_error* error = nullptr;
        const auto status =
            owner_->api->start_propagation(owner_->handle, &input, &traversal->handle, &error);
        detail::check(owner_->model->api, status, error);
        return TrackPropagation(std::move(traversal));
    }
    void close() {
        if (owner_->handle) {
            owner_->api->release(owner_->handle);
            owner_->handle = nullptr;
        }
    }

  private:
    friend class FramesPointsToMaskTracks;
    friend class FramesBoxToMaskTracks;
    friend class FramesMaskToMaskTracks;
    friend class InteractiveFramesTextToMaskTracks;
    friend class FramesBoxExemplarToMaskTracks;
    explicit MaskTrackingSession(std::shared_ptr<detail::MaskTrackOwner> owner)
        : owner_(std::move(owner)) {}
    trtmc_mask_track_editors_v1 editors() const {
        trtmc_mask_track_editors_v1 result{};
        trtmc_error* error = nullptr;
        const auto status = owner_->api->get_editors(owner_->handle, &result, &error);
        detail::check(owner_->model->api, status, error);
        detail::validate_session_table(result.points);
        detail::validate_session_table(result.box);
        detail::validate_session_table(result.mask);
        detail::validate_session_table(result.text);
        detail::validate_session_table(result.exemplar);
        detail::validate_session_table(result.objects);
        detail::validate_session_table(result.reset);
        return result;
    }
    template <class Api, class Prompt>
    TrackClipResult edit(const Api* api, const Prompt& prompt, const Config& config) const {
        if (!api)
            throw Error(TRTMC_UNSUPPORTED, "typed track editor unavailable");
        const auto input = detail::tracking_prompt_wire(prompt);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api->run(owner_->handle, &input.wire, &options, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return TrackClipResult(std::move(result), owner_->api->result_view);
    }
    std::shared_ptr<detail::MaskTrackOwner> owner_;
};
struct MaskTrackingStart {
    MaskTrackingSession session;
    TrackClipResult initial;
};
#define TRTMC_TRACKING_FACTORY(Name, Id, Api, Prompt)                                              \
    class Name {                                                                                   \
      public:                                                                                      \
        static constexpr std::string_view kTask = Id;                                              \
        static constexpr std::uint32_t kMajor = 1, kMinor = 0;                                     \
        MaskTrackingStart create(const VideoInput& clip, const Prompt& prompt,                     \
                                 const Config& config = {}) const {                                \
            auto owner = std::make_shared<detail::MaskTrackOwner>(model_, api_->session_api);      \
            detail::VideoWireInputs inputs;                                                        \
            const auto wire = inputs.video(clip);                                                  \
            const auto supplied = detail::tracking_prompt_wire(prompt);                            \
            auto entries = config.c_entries();                                                     \
            auto options = entries.view();                                                         \
            trtmc_result* raw = nullptr;                                                           \
            trtmc_error* error = nullptr;                                                          \
            const auto status = api_->create(model_->handle, &wire, &supplied.wire, &options,      \
                                             &owner->handle, &raw, &error);                        \
            detail::ResultOwner initial(model_, raw);                                              \
            detail::check(model_->api, status, error);                                             \
            return {MaskTrackingSession(std::move(owner)),                                         \
                    TrackClipResult(std::move(initial), api_->session_api->result_view)};          \
        }                                                                                          \
        std::vector<ConfigField> config_fields() const {                                           \
            return detail::config_fields(model_, kTask, kMajor, kMinor);                           \
        }                                                                                          \
        static void validate_table(const trtmc_api_header* table) {                                \
            if (!table || table->major != 1 || table->minor != 0 ||                                \
                table->byte_size < sizeof(Api))                                                    \
                throw Error(TRTMC_VERSION_MISMATCH, "incompatible tracking factory table");        \
            const auto* session = reinterpret_cast<const Api*>(table)->session_api;                \
            if (!session)                                                                          \
                throw Error(TRTMC_VERSION_MISMATCH, "tracking session table missing");             \
            detail::validate_session_table(session);                                               \
        }                                                                                          \
                                                                                                   \
      private:                                                                                     \
        friend class Model;                                                                        \
        Name(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* table)             \
            : model_(std::move(model)), api_(reinterpret_cast<const Api*>(table)) {}               \
        std::shared_ptr<detail::ModelState> model_;                                                \
        const Api* api_;                                                                           \
    };
TRTMC_TRACKING_FACTORY(FramesPointsToMaskTracks, TRTMC_TASK_FRAMES_POINTS_TO_MASK_TRACKS,
                       trtmc_frames_points_to_mask_tracks_api_v1, FrameObjectPoints)
TRTMC_TRACKING_FACTORY(FramesBoxToMaskTracks, TRTMC_TASK_FRAMES_BOX_TO_MASK_TRACKS,
                       trtmc_frames_box_to_mask_tracks_api_v1, FrameObjectBox)
TRTMC_TRACKING_FACTORY(FramesMaskToMaskTracks, TRTMC_TASK_FRAMES_MASK_TO_MASK_TRACKS,
                       trtmc_frames_mask_to_mask_tracks_api_v1, FrameObjectBinaryMask)
TRTMC_TRACKING_FACTORY(InteractiveFramesTextToMaskTracks,
                       TRTMC_TASK_INTERACTIVE_FRAMES_TEXT_TO_MASK_TRACKS,
                       trtmc_interactive_frames_text_to_mask_tracks_api_v1, FrameText)
TRTMC_TRACKING_FACTORY(FramesBoxExemplarToMaskTracks, TRTMC_TASK_FRAMES_BOX_EXEMPLAR_TO_MASK_TRACKS,
                       trtmc_frames_box_exemplar_to_mask_tracks_api_v1, FrameBoxExemplar)
#undef TRTMC_TRACKING_FACTORY
class CropPoseSession {
  public:
    RefinedPosesResult initialize(const PoseHypothesesCropsToRefinedPosesRequest& input,
                                  const Config& config = {}) const {
        detail::PoseCallbackContext context{input.crops, {}};
        const trtmc_pose_refinement_request_v1 request{
            input.candidates.c_view(), input.mesh_diameter_meters, &context,
            input.crops ? detail::pose_crop_callback : nullptr};
        return call(config, [&](auto options, auto out, auto error) {
            return owner_->api->initialize(owner_->handle, &request, options, out, error);
        });
    }
    RefinedPosesResult track(const PoseCropsProvider& crops, const Config& config = {}) const {
        detail::PoseCallbackContext context{crops, {}};
        return call(config, [&](auto options, auto out, auto error) {
            return owner_->api->track(owner_->handle, &context,
                                      crops ? detail::pose_crop_callback : nullptr, options, out,
                                      error);
        });
    }
    void reset() const {
        trtmc_error* error = nullptr;
        const auto status = owner_->api->reset(owner_->handle, &error);
        detail::check(owner_->model->api, status, error);
    }
    void close() {
        if (owner_->handle) {
            owner_->api->release(owner_->handle);
            owner_->handle = nullptr;
        }
    }

  private:
    friend class CropPoseTracking;
    explicit CropPoseSession(std::shared_ptr<detail::CropPoseOwner> owner)
        : owner_(std::move(owner)) {}
    template <class Invoke>
    RefinedPosesResult call(const Config& config, Invoke invoke) const {
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = invoke(&options, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return RefinedPosesResult(std::move(result), owner_->api->result_view);
    }
    std::shared_ptr<detail::CropPoseOwner> owner_;
};
class CropPoseTracking {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_CROP_POSE_TRACKING;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    CropPoseSession create(const Config& config = {}) const {
        auto owner = std::make_shared<detail::CropPoseOwner>(model_, api_);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_error* error = nullptr;
        const auto status = api_->create(model_->handle, &options, &owner->handle, &error);
        detail::check(model_->api, status, error);
        return CropPoseSession(std::move(owner));
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table)
            throw Error(TRTMC_VERSION_MISMATCH, "pose session table missing");
        detail::validate_session_table(
            reinterpret_cast<const trtmc_crop_pose_tracking_api_v1*>(table));
    }

  private:
    friend class Model;
    CropPoseTracking(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_crop_pose_tracking_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_crop_pose_tracking_api_v1* api_;
};
struct RgbdObservation {
    ImageInput rgb;
    FloatMatrixView depth_meters;
    std::array<float, 9> pixel_intrinsics{};
};
class RgbdPoseTrackingSession {
  public:
    ObjectPoseResult track(const RgbdObservation& input, const Config& config = {}) const {
        trtmc_rgbd_observation_v1 request{input.rgb.wire, input.depth_meters.c_view(), {}};
        std::copy(input.pixel_intrinsics.begin(), input.pixel_intrinsics.end(),
                  request.pixel_intrinsics);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->api->track(owner_->handle, &request, &options, &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return ObjectPoseResult(std::move(result), owner_->api->result_view);
    }
    void reset(const std::array<float, 16>& original_object_to_camera) const {
        trtmc_object_pose_matrix_v1 pose{};
        std::copy(original_object_to_camera.begin(), original_object_to_camera.end(),
                  pose.object_to_camera);
        trtmc_error* error = nullptr;
        const auto status = owner_->api->reset(owner_->handle, &pose, &error);
        detail::check(owner_->model->api, status, error);
    }
    void close() {
        if (owner_->handle) {
            owner_->api->release(owner_->handle);
            owner_->handle = nullptr;
        }
    }

  private:
    friend class RgbdInitializedPoseToTrackedPose;
    explicit RgbdPoseTrackingSession(std::shared_ptr<detail::RgbdPoseOwner> owner)
        : owner_(std::move(owner)) {}
    std::shared_ptr<detail::RgbdPoseOwner> owner_;
};
class RgbdInitializedPoseToTrackedPose {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_RGBD_INITIALIZED_POSE_TO_TRACKED_POSE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    RgbdPoseTrackingSession create(const TriangleMeshInput& mesh,
                                   const std::array<float, 16>& original_object_to_camera,
                                   const Config& config = {}) const {
        const auto input = detail::triangle_mesh_wire(mesh);
        trtmc_object_pose_matrix_v1 pose{};
        std::copy(original_object_to_camera.begin(), original_object_to_camera.end(),
                  pose.object_to_camera);
        auto owner = std::make_shared<detail::RgbdPoseOwner>(model_, api_);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_error* error = nullptr;
        const auto status =
            api_->create(model_->handle, &input, &pose, &options, &owner->handle, &error);
        detail::check(model_->api, status, error);
        return RgbdPoseTrackingSession(std::move(owner));
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    static void validate_table(const trtmc_api_header* table) {
        if (!table)
            throw Error(TRTMC_VERSION_MISMATCH, "RGBD pose session table missing");
        detail::validate_session_table(
            reinterpret_cast<const trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1*>(table));
    }

  private:
    friend class Model;
    RgbdInitializedPoseToTrackedPose(std::shared_ptr<detail::ModelState> model,
                                     const trtmc_api_header* table)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1*>(table)) {
    }
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1* api_;
};
} // namespace trtmc
