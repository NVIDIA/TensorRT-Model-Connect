/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/model.h"
#include "trtmc/internal/tracking.h"
#include "trtmc/runtime/family_factory.h"

namespace {
using namespace trtmc::internal;
void empty(ConfigView config) {
    if (!config.empty())
        throw ConfigError("fixture accepts no config keys");
}
class DetectorSession final : public IDetectedClipSession, public IDetectedDeviceClipSession {
  public:
    explicit DetectorSession(bool device) : device_(device) {}
    IDetectedDeviceClipSession* device_masks() noexcept override {
        return device_ ? this : nullptr;
    }
    TrackClipResult segment(VideoView clip, ConfigView config) override {
        validate(clip, config);
        TrackClipResult result;
        for (std::size_t i = 0; i < clip.frames.size(); ++i) {
            TrackFrameResult frame;
            frame.metadata = metadata(i, clip.frames[i]);
            frame.masks = std::vector<std::uint8_t>(6, 1);
            result.frames.push_back(std::move(frame));
        }
        result.initial_detections = {{0, 7, 2, 0.75F, {0, 0, 3, 2}}};
        return result;
    }
    BorrowedDeviceTrackClip segment_device(VideoView clip, ConfigView config) override {
        validate(clip, config);
        BorrowedDeviceTrackClip result;
        for (std::size_t i = 0; i < clip.frames.size(); ++i) {
            BorrowedDeviceTrackFrame frame;
            frame.metadata = metadata(i, clip.frames[i]);
            // Protocol-only fake CUDA addresses. No CPU dereference or GPU qualification.
            frame.address = reinterpret_cast<const void*>(std::uintptr_t{0x10000} + i * 0x100);
            frame.byte_size = 6;
            frame.device_ordinal = 2;
            result.frames.push_back(std::move(frame));
        }
        result.initial_detections = {{0, 7, 2, 0.75F, {0, 0, 3, 2}}};
        return result;
    }

  private:
    void validate(VideoView clip, ConfigView config) {
        empty(config);
        if (clip.frames.size() != 5)
            throw std::invalid_argument("fixture requires exactly five complete frames");
        for (const auto& image : clip.frames)
            if (image.height != 2 || image.width != 3)
                throw std::invalid_argument("fixed fixture image dimensions");
    }
    static TrackFrameMetadata metadata(std::size_t index, const ImageView& image) {
        TrackFrameMetadata out;
        out.frame_index = index;
        out.height = image.height;
        out.width = image.width;
        out.object_ids = {7};
        out.class_ids = {2};
        return out;
    }
    bool device_;
};
TrackFrameResult text_frame(std::size_t index, const ImageView& image, float mask) {
    TrackFrameResult result;
    result.metadata.frame_index = index;
    result.metadata.height = image.height;
    result.metadata.width = image.width;
    result.metadata.object_ids = {13};
    result.metadata.boxes = {{0, 0, 3, 2}};
    result.metadata.detection_scores = {0.875F};
    result.metadata.tracker_scores = {0.625F};
    result.metadata.removed_object_ids = {19};
    result.metadata.suppressed_object_ids = {23};
    result.masks = std::vector<float>(static_cast<std::size_t>(image.height) * image.width, mask);
    return result;
}
class TextClipSession final : public ITextClipSession {
  public:
    TrackClipResult segment(VideoView clip, std::string_view text, ConfigView config) override {
        empty(config);
        if (text != "bird")
            throw std::invalid_argument("fixture expects bird text");
        TrackClipResult result;
        for (std::size_t i = 0; i < clip.frames.size(); ++i)
            result.frames.push_back(text_frame(i, clip.frames[i], 1.0F));
        return result;
    }
};
class PromptFrameSession final : public ITextPromptFrameSession {
  public:
    TrackFrameResult accept_prompt_frame(ImageView image) override {
        if (prompted_)
            throw std::invalid_argument("prompt frame was already accepted");
        prompted_ = true;
        return text_frame(0, image, 1.0F);
    }
    TrackClipResult continue_borrowed(const TrackFrameResult& prompt, VideoView clip) override {
        if (!prompted_ || continued_)
            throw std::invalid_argument("continuation requires one accepted prompt frame");
        if (clip.frames.empty() || clip.frames[0].height != prompt.metadata.height ||
            clip.frames[0].width != prompt.metadata.width)
            throw std::invalid_argument("continuation must include the prompt frame");
        continued_ = true;
        TrackClipResult result;
        for (std::size_t i = 0; i < clip.frames.size(); ++i)
            // Consolidation changes frame 0. The ABI must not prepend the old snapshot.
            result.frames.push_back(text_frame(i, clip.frames[i], i == 0 ? 0.0F : 1.0F));
        return result;
    }

  private:
    bool prompted_{false};
    bool continued_{false};
};
class ImageContext final : public IImageMaskContext,
                           public IImagePointEditor,
                           public IImageBoxEditor,
                           public IImagePriorEditor {
  public:
    ImageContext(ImageView image, bool prior, int encoding)
        : height_(image.height), width_(image.width), prior_(prior), encoding_(encoding) {
        if (image.format != ImageFormat::Float32)
            throw std::invalid_argument("fixture expects float image");
        encoded_ = static_cast<const float*>(image.data)[0];
    }
    IImagePointEditor* points() noexcept override { return this; }
    IImageBoxEditor* boxes() noexcept override { return this; }
    IImagePriorEditor* priors() noexcept override { return prior_ ? this : nullptr; }
    MasksResult masks_from_points(trtmc::Span<const PointPrompt> points,
                                  ConfigView config) override {
        empty(config);
        if (points.empty())
            throw std::invalid_argument("fixture needs point prompt");
        return output(points[0].foreground ? encoded_ : -encoded_);
    }
    MasksResult masks_from_box(PixelBox box, ConfigView config) override {
        empty(config);
        return output(box.x_max);
    }
    MasksResult masks_from_prior(const ImagePriorPrompt& input, ConfigView config) override {
        empty(config);
        if (input.prior.logits.rows != 2 || input.prior.logits.columns != 2)
            throw std::invalid_argument("fixture needs decoder-space 2x2 logits");
        return output(input.prior.logits.values[0]);
    }

  private:
    MasksResult output(float value) {
        MasksResult result;
        result.height = height_;
        result.width = width_;
        result.count = 1;
        result.masks = std::vector<float>(static_cast<std::size_t>(height_) * width_, value);
        result.predicted_iou = {static_cast<float>(encoding_)};
        return result;
    }
    std::uint32_t height_, width_;
    bool prior_;
    int encoding_;
    float encoded_{0};
};
class InteractiveSession;
class Propagation final : public ITrackPropagation {
  public:
    Propagation(InteractiveSession& session, PropagationRange range)
        : session_(session), range_(range) {}
    std::optional<TrackFrameResult> next() override;
    void cancel() noexcept override { cancelled_ = true; }

  private:
    InteractiveSession& session_;
    PropagationRange range_;
    std::uint64_t emitted_{0};
    bool cancelled_{false};
};
class InteractiveSession final : public IMaskTrackSession,
                                 public IPointsTrackEditor,
                                 public IBoxTrackEditor,
                                 public IMaskTrackEditor,
                                 public ITextTrackEditor,
                                 public IExemplarTrackEditor,
                                 public ITrackObjectRemoval,
                                 public ITrackReset {
  public:
    InteractiveSession(VideoView clip, bool limited) : limited_(limited) {
        for (const auto& frame : clip.frames)
            geometry_.emplace_back(frame.height, frame.width);
    }
    IPointsTrackEditor* points() noexcept override { return this; }
    IBoxTrackEditor* boxes() noexcept override { return limited_ ? nullptr : this; }
    IMaskTrackEditor* masks() noexcept override { return limited_ ? nullptr : this; }
    ITextTrackEditor* text() noexcept override { return limited_ ? nullptr : this; }
    IExemplarTrackEditor* exemplars() noexcept override { return limited_ ? nullptr : this; }
    ITrackObjectRemoval* objects() noexcept override { return limited_ ? nullptr : this; }
    ITrackReset* resetter() noexcept override { return this; }
    TrackFrameResult update_points(const FrameObjectPoints& input, ConfigView config) override {
        empty(config);
        object_ = input.object_id;
        initialized_ = true;
        if (input.update == PointUpdate::Replace)
            point_count_ = 0;
        point_count_ += input.points.size();
        value_ = input.points[0].foreground ? 1.0F : 0.0F;
        frame_ = input.frame_index;
        return snapshot(frame_);
    }
    TrackFrameResult update_box(const FrameObjectBox& input, ConfigView config) override {
        empty(config);
        object_ = input.object_id;
        initialized_ = true;
        value_ = 1;
        point_count_ = input.correction_points.size();
        frame_ = input.frame_index;
        return snapshot(frame_);
    }
    TrackFrameResult update_mask(const FrameObjectBinaryMask& input, ConfigView config) override {
        empty(config);
        object_ = input.object_id;
        initialized_ = true;
        value_ = input.mask.values[0];
        frame_ = input.frame_index;
        return snapshot(frame_);
    }
    TrackFrameResult replace_text(const FrameText& input, ConfigView config) override {
        empty(config);
        if (input.text.empty())
            throw std::invalid_argument("fixture text is required");
        object_ = 100 + ++epoch_;
        initialized_ = true;
        value_ = 1;
        point_count_ = 0;
        frame_ = input.frame_index;
        return snapshot(frame_);
    }
    TrackFrameResult replace_exemplar(const FrameBoxExemplar& input, ConfigView config) override {
        empty(config);
        object_ = 100 + ++epoch_;
        initialized_ = true;
        value_ = input.exemplar.positive ? 1.0F : 0.0F;
        point_count_ = 0;
        frame_ = input.frame_index;
        return snapshot(frame_);
    }
    TrackClipResult remove_object(std::int64_t object) override {
        if (!initialized_ || object != object_)
            throw std::invalid_argument("fixture object not found");
        auto frame = snapshot(frame_);
        frame.metadata.object_ids.clear();
        frame.metadata.boxes.clear();
        frame.metadata.detection_scores.clear();
        frame.metadata.tracker_scores.clear();
        frame.masks = std::vector<float>{};
        frame.metadata.removed_object_ids = {object};
        initialized_ = false;
        TrackClipResult result;
        result.frames.push_back(std::move(frame));
        return result;
    }
    void reset() override {
        initialized_ = false;
        point_count_ = 0;
        ++epoch_;
    }
    std::unique_ptr<ITrackPropagation> start_propagation(PropagationRange range) override {
        if (!initialized_)
            throw std::invalid_argument("fixture tracking must be prompted");
        return std::make_unique<Propagation>(*this, range);
    }
    TrackFrameResult snapshot(std::uint64_t index) const {
        ImageView geometry;
        geometry.height = geometry_.at(index).first;
        geometry.width = geometry_.at(index).second;
        auto result = text_frame(index, geometry, value_);
        result.metadata.object_ids = {object_};
        result.metadata.tracker_scores = {static_cast<float>(point_count_)};
        result.metadata.detection_scores = {static_cast<float>(epoch_)};
        return result;
    }

  private:
    std::vector<std::pair<std::uint32_t, std::uint32_t>> geometry_;
    bool limited_, initialized_{false};
    std::int64_t object_{0}, epoch_{0};
    std::uint64_t frame_{0};
    std::size_t point_count_{0};
    float value_{0};
};
std::optional<TrackFrameResult> Propagation::next() {
    if (cancelled_ || emitted_ == range_.frame_count)
        return std::nullopt;
    const auto index = range_.direction == PropagationDirection::Forward
                           ? range_.start_frame + emitted_
                           : range_.start_frame - emitted_;
    ++emitted_;
    return session_.snapshot(index);
}
class CropPoseSession final : public ICropPoseSession {
  public:
    RefinedPosesResult initialize(const PoseHypothesesCropsToRefinedPosesRequest& input,
                                  ConfigView config) override {
        empty(config);
        auto result = refine(input.candidates, input.mesh_diameter_meters, input.crops);
        diameter_ = input.mesh_diameter_meters;
        remember(result);
        return result;
    }
    RefinedPosesResult track(const PoseCropProvider& crops, ConfigView config) override {
        empty(config);
        if (pose_.empty())
            throw std::invalid_argument("fixture pose is uninitialized");
        auto result = refine({{pose_.data(), pose_.size()}, 1}, diameter_, crops);
        remember(result);
        return result;
    }
    void reset() override {
        pose_.clear();
        diameter_ = 0;
    }

  private:
    static RefinedPosesResult refine(PoseMatricesView candidates, float diameter,
                                     const PoseCropProvider& crops) {
        auto refinement = crops({candidates, CropStage::Refinement, 0});
        RefinedPosesResult result;
        result.num_hypotheses = static_cast<std::int32_t>(candidates.count);
        result.refined_poses.assign(candidates.values.begin(), candidates.values.end());
        for (std::size_t i = 0; i < candidates.count; ++i)
            result.refined_poses[i * 16 + 3] += refinement.rendered[i * 6] * diameter;
        auto scoring =
            crops({{{result.refined_poses.data(), result.refined_poses.size()}, candidates.count},
                   CropStage::Scoring,
                   0});
        for (std::size_t i = 0; i < candidates.count; ++i)
            result.scores.push_back(scoring.observed[i * 6] + static_cast<float>(i));
        result.best_index = result.num_hypotheses - 1;
        result.all_poses_rigid = true;
        result.refinement_ms = 2;
        result.scoring_ms = 1;
        return result;
    }
    void remember(const RefinedPosesResult& result) {
        const auto begin = result.refined_poses.begin() + result.best_index * 16;
        pose_.assign(begin, begin + 16);
    }
    std::vector<float> pose_;
    float diameter_{0};
};
class RgbdPoseSession final : public IRgbdPoseSession {
  public:
    RgbdPoseSession(const TriangleMeshView& mesh, const std::array<float, 16>& pose)
        : pose_(pose), mesh_marker_(mesh.vertices.values[0]) {}
    ObjectPoseResult track(const RgbdObservation& input, ConfigView config) override {
        empty(config);
        if (input.pixel_intrinsics[0] != 100 || input.depth_meters.values.empty())
            throw std::invalid_argument("fixture expects pixel intrinsics and meter depth");
        pose_[11] += input.depth_meters.values[0];
        return {pose_, mesh_marker_, true};
    }
    void reset(const std::array<float, 16>& pose) override { pose_ = pose; }

  private:
    std::array<float, 16> pose_;
    float mesh_marker_;
};
class Model final : public IModel,
                    public IFramesToDetectedMaskTracks,
                    public IFramesTextToMaskTracks,
                    public IPromptFrameTextToMaskTracks,
                    public IInteractiveImageMasks,
                    public IFramesPointsToMaskTracks,
                    public IFramesBoxToMaskTracks,
                    public IFramesMaskToMaskTracks,
                    public IInteractiveFramesTextToMaskTracks,
                    public IFramesBoxExemplarToMaskTracks,
                    public ICropPoseTracking,
                    public IRgbdInitializedPoseToTrackedPose {
  public:
    explicit Model(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<IFramesToDetectedMaskTracks>(*this),
                bind<IFramesTextToMaskTracks>(*this),
                bind<IPromptFrameTextToMaskTracks>(*this),
                bind<IInteractiveImageMasks>(*this),
                bind<IFramesPointsToMaskTracks>(*this),
                bind<IFramesBoxToMaskTracks>(*this),
                bind<IFramesMaskToMaskTracks>(*this),
                bind<IInteractiveFramesTextToMaskTracks>(*this),
                bind<IFramesBoxExemplarToMaskTracks>(*this),
                bind<ICropPoseTracking>(*this),
                bind<IRgbdInitializedPoseToTrackedPose>(*this)};
    }

    std::unique_ptr<IDetectedClipSession> create_detected_session(ConfigView config) override {
        empty(config);
        return std::make_unique<DetectorSession>(mode_ != "host_only");
    }
    std::unique_ptr<ITextClipSession> create_text_clip_session(ConfigView config) override {
        empty(config);
        return std::make_unique<TextClipSession>();
    }
    std::unique_ptr<ITextPromptFrameSession>
    create_prompt_frame_session(std::string_view text, ConfigView config) override {
        empty(config);
        if (text != "bird")
            throw std::invalid_argument("fixture expects bird text");
        return std::make_unique<PromptFrameSession>();
    }
    std::unique_ptr<IImageMaskContext> create_image_context(ImageView image,
                                                            ConfigView config) override {
        empty(config);
        return std::make_unique<ImageContext>(image, mode_ != "host_only", ++encodings_);
    }
#define FIXTURE_FACTORY(Method, Input, Editor)                                                     \
    MaskTrackSessionStart Method(VideoView clip, const Input& prompt, ConfigView config)           \
        override {                                                                                 \
        auto session = std::make_unique<InteractiveSession>(clip, mode_ == "host_only");           \
        auto initial = session->Editor(prompt, config);                                            \
        return {std::move(session), std::move(initial)};                                           \
    }
    FIXTURE_FACTORY(create_points_tracks, FrameObjectPoints, update_points)
    FIXTURE_FACTORY(create_box_tracks, FrameObjectBox, update_box)
    FIXTURE_FACTORY(create_mask_tracks, FrameObjectBinaryMask, update_mask)
    FIXTURE_FACTORY(create_text_tracks, FrameText, replace_text)
    FIXTURE_FACTORY(create_exemplar_tracks, FrameBoxExemplar, replace_exemplar)
#undef FIXTURE_FACTORY
    std::unique_ptr<ICropPoseSession> create_crop_pose_session(ConfigView config) override {
        empty(config);
        return std::make_unique<CropPoseSession>();
    }
    std::unique_ptr<IRgbdPoseSession> create_rgbd_pose_session(const TriangleMeshView& mesh,
                                                               const std::array<float, 16>& initial,
                                                               ConfigView config) override {
        empty(config);
        return std::make_unique<RgbdPoseSession>(mesh, initial);
    }

  private:
    std::string mode_;
    int encodings_{0};
};
} // namespace
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new Model(context.reader.info().task);
}
