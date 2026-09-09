/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/perception.h"
#include "trtmc/internal/video.h"

namespace trtmc::internal {

struct TrackFrameMetadata {
    std::uint64_t frame_index{0};
    std::uint32_t height{0}, width{0};
    std::vector<std::int64_t> object_ids;
    std::vector<PixelBox> boxes; // Actual per-frame tracked boxes, empty if unavailable.
    std::vector<float> detection_scores, tracker_scores;
    std::vector<std::int64_t> class_ids;
    std::vector<std::int64_t> removed_object_ids, suppressed_object_ids;
};
struct InitialTrackDetection {
    std::uint64_t frame_index{0};
    std::int64_t object_id{0}, class_id{0};
    float score{0};
    PixelBox prompt_box;
};
struct TrackFrameResult {
    TrackFrameMetadata metadata;
    std::variant<std::vector<std::uint8_t>, std::vector<float>> masks;
    MaskKind mask_kind{MaskKind::Binary};
};
struct TrackClipResult {
    std::vector<TrackFrameResult> frames;
    std::vector<InitialTrackDetection> initial_detections;
};
enum class TrackMaskType : std::uint32_t { UInt8 = 1, Float32 = 2 };
struct BorrowedDeviceTrackFrame {
    TrackFrameMetadata metadata;
    const void* address{nullptr};
    std::uint64_t byte_size{0};
    std::int32_t device_ordinal{-1};
    TrackMaskType type{TrackMaskType::UInt8};
    MaskKind mask_kind{MaskKind::Binary};
};
struct BorrowedDeviceTrackClip {
    std::vector<BorrowedDeviceTrackFrame> frames;
    std::vector<InitialTrackDetection> initial_detections;
};
class IDetectedDeviceClipSession {
  public:
    virtual ~IDetectedDeviceClipSession() = default;
    // Producer work is complete at return. Pointers expire at the next
    // segment/segment_device call (including failure) or session destruction.
    virtual BorrowedDeviceTrackClip segment_device(VideoView, ConfigView) = 0;
};
class IDetectedClipSession {
  public:
    virtual ~IDetectedClipSession() = default;
    virtual TrackClipResult segment(VideoView, ConfigView) = 0;
    virtual IDetectedDeviceClipSession* device_masks() noexcept { return nullptr; }
};
class IFramesToDetectedMaskTracks {
  public:
    static constexpr std::string_view kTask = "frames_to_detected_mask_tracks";
    virtual ~IFramesToDetectedMaskTracks() = default;
    virtual std::unique_ptr<IDetectedClipSession> create_detected_session(ConfigView) = 0;
};

class ITextClipSession {
  public:
    virtual ~ITextClipSession() = default;
    virtual TrackClipResult segment(VideoView, std::string_view text, ConfigView) = 0;
};
class IFramesTextToMaskTracks {
  public:
    static constexpr std::string_view kTask = "frames_text_to_mask_tracks";
    virtual ~IFramesTextToMaskTracks() = default;
    virtual std::unique_ptr<ITextClipSession> create_text_clip_session(ConfigView) = 0;
};
class ITextPromptFrameSession {
  public:
    virtual ~ITextPromptFrameSession() = default;
    virtual TrackFrameResult accept_prompt_frame(ImageView) = 0;
    // The complete clip includes frame zero. The prompt snapshot is borrowed,
    // not consumed; the family's returned frame zero may differ after consolidation.
    virtual TrackClipResult continue_borrowed(const TrackFrameResult&, VideoView) = 0;
};
class IPromptFrameTextToMaskTracks {
  public:
    static constexpr std::string_view kTask = "prompt_frame_text_to_mask_tracks";
    virtual ~IPromptFrameTextToMaskTracks() = default;
    virtual std::unique_ptr<ITextPromptFrameSession>
    create_prompt_frame_session(std::string_view text, ConfigView) = 0;
};

struct ImagePriorPrompt {
    PriorMaskLogits prior;
    Span<const PointPrompt> points;
    std::optional<PixelBox> box;
};
class IImagePointEditor {
  public:
    virtual ~IImagePointEditor() = default;
    virtual MasksResult masks_from_points(Span<const PointPrompt>, ConfigView) = 0;
};
class IImageBoxEditor {
  public:
    virtual ~IImageBoxEditor() = default;
    virtual MasksResult masks_from_box(PixelBox, ConfigView) = 0;
};
class IImagePriorEditor {
  public:
    virtual ~IImagePriorEditor() = default;
    virtual MasksResult masks_from_prior(const ImagePriorPrompt&, ConfigView) = 0;
};
class IImageMaskContext {
  public:
    virtual ~IImageMaskContext() = default;
    virtual IImagePointEditor* points() noexcept { return nullptr; }
    virtual IImageBoxEditor* boxes() noexcept { return nullptr; }
    virtual IImagePriorEditor* priors() noexcept { return nullptr; }
};
class IInteractiveImageMasks {
  public:
    static constexpr std::string_view kTask = "interactive_image_masks";
    virtual ~IInteractiveImageMasks() = default;
    // The family owns any retained image/features before this call returns.
    virtual std::unique_ptr<IImageMaskContext> create_image_context(ImageView, ConfigView) = 0;
};

enum class PointUpdate : std::uint32_t { Replace = 1, Append = 2 };
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
    BinaryImageMask mask; // Original-resolution binary mask, never decoder logits.
};
struct FrameText {
    std::uint64_t frame_index{0};
    std::string_view text;
};
struct FrameBoxExemplar {
    std::uint64_t frame_index{0};
    BoxExemplar exemplar; // One replacement concept exemplar, not accumulation.
};
enum class PropagationDirection : std::uint32_t { Forward = 1, Backward = 2 };
struct PropagationRange {
    std::uint64_t start_frame{0}, frame_count{0};
    PropagationDirection direction{PropagationDirection::Forward};
};
class ITrackPropagation {
  public:
    virtual ~ITrackPropagation() = default;
    virtual std::optional<TrackFrameResult> next() = 0;
    // Synchronous v1: cancel is called between next calls, not concurrently.
    virtual void cancel() noexcept = 0;
};
#define TRTMC_TRACK_EDITOR(Name, Method, Input)                                                    \
    class I##Name {                                                                                \
      public:                                                                                      \
        virtual ~I##Name() = default;                                                              \
        virtual TrackFrameResult Method(const Input&, ConfigView) = 0;                             \
    };
TRTMC_TRACK_EDITOR(PointsTrackEditor, update_points, FrameObjectPoints)
TRTMC_TRACK_EDITOR(BoxTrackEditor, update_box, FrameObjectBox)
TRTMC_TRACK_EDITOR(MaskTrackEditor, update_mask, FrameObjectBinaryMask)
TRTMC_TRACK_EDITOR(TextTrackEditor, replace_text, FrameText)
TRTMC_TRACK_EDITOR(ExemplarTrackEditor, replace_exemplar, FrameBoxExemplar)
#undef TRTMC_TRACK_EDITOR
class ITrackObjectRemoval {
  public:
    virtual ~ITrackObjectRemoval() = default;
    virtual TrackClipResult remove_object(std::int64_t object_id) = 0;
};
class ITrackReset {
  public:
    virtual ~ITrackReset() = default;
    virtual void reset() = 0;
};
class IMaskTrackSession {
  public:
    virtual ~IMaskTrackSession() = default;
    virtual IPointsTrackEditor* points() noexcept { return nullptr; }
    virtual IBoxTrackEditor* boxes() noexcept { return nullptr; }
    virtual IMaskTrackEditor* masks() noexcept { return nullptr; }
    virtual ITextTrackEditor* text() noexcept { return nullptr; }
    virtual IExemplarTrackEditor* exemplars() noexcept { return nullptr; }
    virtual ITrackObjectRemoval* objects() noexcept { return nullptr; }
    virtual ITrackReset* resetter() noexcept { return nullptr; }
    virtual std::unique_ptr<ITrackPropagation> start_propagation(PropagationRange) = 0;
};
struct MaskTrackSessionStart {
    std::unique_ptr<IMaskTrackSession> session;
    TrackFrameResult prompted_frame;
};
// Families retain their own clip/features before create returns. Shared code
// stores geometry only; encoding, edits, traversal and identities stay native.
#define TRTMC_TRACK_FACTORY(Name, Id, Method, Input)                                               \
    class I##Name {                                                                                \
      public:                                                                                      \
        static constexpr std::string_view kTask = Id;                                              \
        virtual ~I##Name() = default;                                                              \
        virtual MaskTrackSessionStart Method(VideoView, const Input&, ConfigView) = 0;             \
    };
TRTMC_TRACK_FACTORY(FramesPointsToMaskTracks, "frames_points_to_mask_tracks", create_points_tracks,
                    FrameObjectPoints)
TRTMC_TRACK_FACTORY(FramesBoxToMaskTracks, "frames_box_to_mask_tracks", create_box_tracks,
                    FrameObjectBox)
TRTMC_TRACK_FACTORY(FramesMaskToMaskTracks, "frames_mask_to_mask_tracks", create_mask_tracks,
                    FrameObjectBinaryMask)
TRTMC_TRACK_FACTORY(InteractiveFramesTextToMaskTracks, "interactive_frames_text_to_mask_tracks",
                    create_text_tracks, FrameText)
TRTMC_TRACK_FACTORY(FramesBoxExemplarToMaskTracks, "frames_box_exemplar_to_mask_tracks",
                    create_exemplar_tracks, FrameBoxExemplar)
#undef TRTMC_TRACK_FACTORY

class ICropPoseSession {
  public:
    virtual ~ICropPoseSession() = default;
    virtual RefinedPosesResult initialize(const PoseHypothesesCropsToRefinedPosesRequest&,
                                          ConfigView) = 0;
    // The family stores the selected pose and diameter; provider is per-call.
    virtual RefinedPosesResult track(const PoseCropProvider&, ConfigView) = 0;
    virtual void reset() = 0;
};
class ICropPoseTracking {
  public:
    static constexpr std::string_view kTask = "crop_pose_tracking";
    virtual ~ICropPoseTracking() = default;
    virtual std::unique_ptr<ICropPoseSession> create_crop_pose_session(ConfigView) = 0;
};
struct RgbdObservation {
    ImageView rgb;
    FloatMatrixView depth_meters; // Aligned H,W; zero means invalid.
    std::array<float, 9> pixel_intrinsics{};
};
class IRgbdPoseSession {
  public:
    virtual ~IRgbdPoseSession() = default;
    virtual ObjectPoseResult track(const RgbdObservation&, ConfigView) = 0;
    virtual void reset(const std::array<float, 16>& original_object_to_camera) = 0;
};
class IRgbdInitializedPoseToTrackedPose {
  public:
    static constexpr std::string_view kTask = "rgbd_initialized_pose_to_tracked_pose";
    virtual ~IRgbdInitializedPoseToTrackedPose() = default;
    // Family retains mesh/appearance before return and owns centered-frame conversions.
    // This upstream RGBD protocol is distinct from current native crop-only refinement.
    virtual std::unique_ptr<IRgbdPoseSession>
    create_rgbd_pose_session(const TriangleMeshView&,
                             const std::array<float, 16>& original_object_to_camera,
                             ConfigView) = 0;
};

} // namespace trtmc::internal
