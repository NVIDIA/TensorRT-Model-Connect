/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/family_factory.h"

#include <atomic>
#include <cmath>
#include <string>

namespace {
using namespace trtmc;
using namespace trtmc::internal;
std::atomic<int> batch_calls{0}, single_calls{0}, batch_items_executed{0};

// A protocol fixture: no TensorRT model, image-quality or timing-performance claim.
class VideoFixture final : public IModel,
                           public ITextToVideo,
                           public IBatchTextToVideo,
                           public IBatchInitialImageTextToVideo,
                           public IBatchVideoTextToFutureVideo,
                           public IBatchImageActionToFutureVideo,
                           public IBatchVideoToActionSequence,
                           public IBatchImageToActionAndVideo,
                           public IBatchTextToAudioVideo,
                           public IBatchInitialImageTextToAudioVideo,
                           public IInitialImageTextToVideo,
                           public IBoundaryFramesTextToVideo,
                           public ITimedFramesTextToVideo,
                           public IVideoTextToVideoEdit,
                           public IMaskedVideoTextToVideo,
                           public IMaskedVideoReferenceImagesTextToVideo,
                           public IImageTextActionToVideo,
                           public IImageTextCameraTrajectoryToVideo,
                           public IVideoTextToFutureVideo,
                           public IImageActionToFutureVideo,
                           public IVideoActionToFutureVideo,
                           public IVideoToActionSequence,
                           public IImageToActionAndVideo,
                           public IVideoToActionAndVideo,
                           public ITextToAudioVideo,
                           public IInitialImageTextToAudioVideo,
                           public ILastImageTextToAudioVideo,
                           public IBoundaryFramesTextToAudioVideo,
                           public IReferencesTextToAudioVideo {
  public:
    explicit VideoFixture(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "none")
            return {};
        if (mode_.rfind("batch", 0) == 0)
            return {bind<IBatchTextToVideo>(*this, fields_for(IBatchTextToVideo::kTask)),
                    bind<IBatchInitialImageTextToVideo>(
                        *this, fields_for(IBatchInitialImageTextToVideo::kTask)),
                    bind<IBatchVideoTextToFutureVideo>(
                        *this, fields_for(IBatchVideoTextToFutureVideo::kTask)),
                    bind<IBatchImageActionToFutureVideo>(
                        *this, fields_for(IBatchImageActionToFutureVideo::kTask)),
                    bind<IBatchVideoToActionSequence>(
                        *this, fields_for(IBatchVideoToActionSequence::kTask)),
                    bind<IBatchImageToActionAndVideo>(
                        *this, fields_for(IBatchImageToActionAndVideo::kTask)),
                    bind<IBatchTextToAudioVideo>(*this, fields_for(IBatchTextToAudioVideo::kTask)),
                    bind<IBatchInitialImageTextToAudioVideo>(
                        *this, fields_for(IBatchInitialImageTextToAudioVideo::kTask))};
        return {
            bind<ITextToVideo>(*this, fields_for(ITextToVideo::kTask)),
            bind<IInitialImageTextToVideo>(*this, fields_for(IInitialImageTextToVideo::kTask)),
            bind<IBoundaryFramesTextToVideo>(*this, fields_for(IBoundaryFramesTextToVideo::kTask)),
            bind<ITimedFramesTextToVideo>(*this, fields_for(ITimedFramesTextToVideo::kTask)),
            bind<IVideoTextToVideoEdit>(*this, fields_for(IVideoTextToVideoEdit::kTask)),
            bind<IMaskedVideoTextToVideo>(*this, fields_for(IMaskedVideoTextToVideo::kTask)),
            bind<IMaskedVideoReferenceImagesTextToVideo>(
                *this, fields_for(IMaskedVideoReferenceImagesTextToVideo::kTask)),
            bind<IImageTextActionToVideo>(*this, fields_for(IImageTextActionToVideo::kTask)),
            bind<IImageTextCameraTrajectoryToVideo>(
                *this, fields_for(IImageTextCameraTrajectoryToVideo::kTask)),
            bind<IVideoTextToFutureVideo>(*this, fields_for(IVideoTextToFutureVideo::kTask)),
            bind<IImageActionToFutureVideo>(*this, fields_for(IImageActionToFutureVideo::kTask)),
            bind<IVideoActionToFutureVideo>(*this, fields_for(IVideoActionToFutureVideo::kTask)),
            bind<IVideoToActionSequence>(*this, fields_for(IVideoToActionSequence::kTask)),
            bind<IImageToActionAndVideo>(*this, fields_for(IImageToActionAndVideo::kTask)),
            bind<IVideoToActionAndVideo>(*this, fields_for(IVideoToActionAndVideo::kTask)),
            bind<ITextToAudioVideo>(*this, fields_for(ITextToAudioVideo::kTask)),
            bind<IInitialImageTextToAudioVideo>(*this,
                                                fields_for(IInitialImageTextToAudioVideo::kTask)),
            bind<ILastImageTextToAudioVideo>(*this, fields_for(ILastImageTextToAudioVideo::kTask)),
            bind<IBoundaryFramesTextToAudioVideo>(
                *this, fields_for(IBoundaryFramesTextToAudioVideo::kTask)),
            bind<IReferencesTextToAudioVideo>(*this,
                                              fields_for(IReferencesTextToAudioVideo::kTask)),
        };
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view) const {
        static const ConfigField declared[] = {
            {"gain", ConfigKind::F64, ConfigValue{1.0}, "Synthetic video intensity gain."}};
        return declared;
    }

    VideoResult run(const TextToVideoRequest& input, ConfigView config) override {
        ++single_calls;
        if (mode_ == "worker" || input.prompt == "benchmark-worker") {
            (void)gain(config);
            VideoResult result;
            result.frames.num_frames = 0;
            return result;
        }
        auto result = video(1, float(input.prompt.size()) / 100, gain(config));
        validate_replay(input.initial_latents);
        if (!input.initial_latents.empty())
            std::copy(input.initial_latents.begin(), input.initial_latents.end(),
                      result.frames.pixels.begin() + 3);
        if (mode_ == "bad_shape")
            result.frames.pixels.pop_back();
        return result;
    }
    std::vector<VideoResult> run_batch(const BatchTextToVideoRequest& request) override {
        return batch<VideoResult>(request, [&](const auto& input, double intensity) {
            auto item =
                video(21, float(input.prompt.size()) / 100, intensity, input.prompt.size() + 2);
            if (!input.initial_latents.empty())
                std::copy(input.initial_latents.begin(), input.initial_latents.end(),
                          item.frames.pixels.begin() + 3);
            if (mode_ == "batch_bad_shape")
                item.frames.pixels.pop_back();
            return item;
        });
    }
    std::vector<VideoResult>
    run_batch(const BatchInitialImageTextToVideoRequest& request) override {
        return batch<VideoResult>(request, [&](const auto& input, double intensity) {
            return video(22, pixel(input.initial_image) + float(input.prompt.size()) / 100,
                         intensity);
        });
    }
    std::vector<VideoResult> run_batch(const BatchVideoTextToFutureVideoRequest& request) override {
        return batch<VideoResult>(request, [&](const auto& input, double intensity) {
            auto result = future(
                23, float(input.history.frames.size()) / 100 + float(input.prompt.size()) / 1000,
                intensity, pixel(input.history.frames[input.history.frames.size() - 1]),
                last_time(input.history));
            if (mode_ == "batch_bad_future")
                result.conditioned_prefix_frames = result.frames.num_frames;
            return result;
        });
    }
    std::vector<VideoResult>
    run_batch(const BatchImageActionToFutureVideoRequest& request) override {
        return batch<VideoResult>(request, [&](const auto& input, double intensity) {
            return future(24,
                          input.actions.values.values[0] +
                              float(input.actions.schema.domain.size()) / 1000 +
                              (input.prompt ? 0.02F : 0.01F),
                          intensity, pixel(input.observation), 0);
        });
    }
    std::vector<ActionSequenceResult>
    run_batch(const BatchVideoToActionSequenceRequest& request) override {
        return batch<ActionSequenceResult>(request, [&](const auto& input, double intensity) {
            auto result = actions(25, input.action_spec, intensity);
            result.values.values[1] =
                pixel(input.observations.frames[0]) + (input.prompt ? 0.02F : 0.01F);
            result.frame_spans = {{0, 1}, {1, 2}};
            result.timestamps_seconds = {input.observations.timestamps_seconds.empty()
                                             ? 0
                                             : input.observations.timestamps_seconds[0],
                                         last_time(input.observations)};
            if (mode_ == "batch_bad_actions")
                result.schema.domain = "wrong.domain";
            return result;
        });
    }
    std::vector<ActionVideoResult>
    run_batch(const BatchImageToActionAndVideoRequest& request) override {
        return batch<ActionVideoResult>(request, [&](const auto& input, double intensity) {
            auto result = ActionVideoResult{
                actions(26, input.action_spec, intensity),
                future(26, input.prompt ? 0.02F : 0.01F, intensity, pixel(input.observation), 0)};
            result.actions.frame_spans = {{1, 2}, {2, 3}};
            result.actions.timestamps_seconds = {result.video.timestamps_seconds[1],
                                                 result.video.timestamps_seconds[2]};
            if (mode_ == "batch_bad_actions")
                result.actions.frame_spans[1].end = 99;
            return result;
        });
    }
    std::vector<AudioVideoResult> run_batch(const BatchTextToAudioVideoRequest& request) override {
        return batch<AudioVideoResult>(request, [&](const auto& input, double intensity) {
            auto result = audio_video(27, float(input.prompt.size()) / 100, intensity);
            if (mode_ == "batch_bad_av")
                result.audio_start_seconds.reset();
            return result;
        });
    }
    std::vector<AudioVideoResult>
    run_batch(const BatchInitialImageTextToAudioVideoRequest& request) override {
        return batch<AudioVideoResult>(request, [&](const auto& input, double intensity) {
            auto result = audio_video(
                28, pixel(input.initial_image) + float(input.prompt.size()) / 100, intensity);
            result.audio.samples.resize(input.prompt.size() * 2);
            return result;
        });
    }
    VideoResult run(const InitialImageTextToVideoRequest& input, ConfigView config) override {
        ++single_calls;
        return video(2, pixel(input.initial_image), gain(config));
    }
    VideoResult run(const BoundaryFramesTextToVideoRequest& input, ConfigView config) override {
        ++single_calls;
        return video(3, pixel(input.first_frame) + pixel(input.last_frame) / 10, gain(config));
    }
    VideoResult run(const TimedFramesTextToVideoRequest& input, ConfigView config) override {
        ++single_calls;
        float probe = 0;
        for (const auto& anchor : input.anchors) {
            if (anchor.output_start_frame >= 3)
                throw std::invalid_argument("fixture anchor outside output");
            probe += std::holds_alternative<ImageView>(anchor.content) ? 0.1F : 0.2F;
            probe +=
                float(anchor.output_start_frame) / 100 + float(anchor.strength.value_or(0.5)) / 10;
        }
        return video(4, probe, gain(config));
    }
    VideoResult run(const VideoTextToVideoEditRequest& input, ConfigView config) override {
        ++single_calls;
        return video(5, pixel(input.source.frames[input.source.frames.size() - 1]), gain(config));
    }
    VideoResult run(const MaskedVideoTextToVideoRequest& input, ConfigView config) override {
        ++single_calls;
        return video(
            6, input.mask.values[0] * 0.2F + input.mask.values[input.mask.values.size() - 1] * 0.3F,
            gain(config));
    }
    VideoResult run(const MaskedVideoReferenceImagesTextToVideoRequest& input,
                    ConfigView config) override {
        ++single_calls;
        return video(7,
                     pixel(input.source.frames[0]) + pixel(input.references[0]) / 10 +
                         input.mask.values[input.mask.values.size() - 1] / 10,
                     gain(config));
    }
    VideoResult run(const ImageTextActionToVideoRequest& input, ConfigView config) override {
        ++single_calls;
        if (input.action_dsl.empty())
            throw std::invalid_argument("fixture action DSL is required");
        if (input.dialect && *input.dialect != "fixture.dialect")
            throw std::invalid_argument("unsupported fixture action dialect");
        auto result =
            video(8,
                  float(input.action_dsl.size()) / 1000 + (input.dialect ? 0.2F : 0.1F) +
                      float(input.intrinsics.rows) / 100 + input.intrinsics.values[0] / 100,
                  gain(config));
        return conditioned_replay(std::move(result), input.initial_image, input.initial_latents);
    }
    VideoResult run(const ImageTextCameraTrajectoryToVideoRequest& input,
                    ConfigView config) override {
        ++single_calls;
        auto result = video(9,
                            input.camera.camera_to_world.values[3] / 10 +
                                (input.camera.coordinate_convention.empty() ? 0.01F : 0.02F) +
                                (input.camera.translation_units.empty() ? 0.03F : 0.04F),
                            gain(config), input.camera.camera_to_world.rows);
        return conditioned_replay(std::move(result), input.initial_image, input.initial_latents);
    }
    VideoResult run(const VideoTextToFutureVideoRequest& input, ConfigView config) override {
        ++single_calls;
        auto result = future(10, float(input.history.frames.size()) / 100, gain(config),
                             pixel(input.history.frames[input.history.frames.size() - 1]),
                             last_time(input.history));
        if (mode_ == "bad_future")
            result.conditioned_prefix_frames = result.frames.num_frames;
        return result;
    }
    VideoResult run(const ImageActionToFutureVideoRequest& input, ConfigView config) override {
        ++single_calls;
        return future(11, input.actions.values.values[0] + (input.prompt ? 0.02F : 0.01F),
                      gain(config), pixel(input.observation), 0);
    }
    VideoResult run(const VideoActionToFutureVideoRequest& input, ConfigView config) override {
        ++single_calls;
        return future(12,
                      input.actions.values.values[0] / 10 +
                          pixel(input.history.frames[input.history.frames.size() - 1]) / 10,
                      gain(config), pixel(input.history.frames[input.history.frames.size() - 1]),
                      last_time(input.history));
    }
    ActionSequenceResult run(const VideoToActionSequenceRequest& input,
                             ConfigView config) override {
        ++single_calls;
        auto result = actions(13, input.action_spec, gain(config));
        result.frame_spans = {{0, 1}, {1, 2}};
        if (!input.observations.timestamps_seconds.empty())
            result.timestamps_seconds.assign(input.observations.timestamps_seconds.begin(),
                                             input.observations.timestamps_seconds.end());
        if (mode_ == "bad_actions")
            result.schema.domain = "wrong.domain";
        return result;
    }
    ActionVideoResult run(const ImageToActionAndVideoRequest& input, ConfigView config) override {
        ++single_calls;
        auto result = ActionVideoResult{actions(14, input.action_spec, gain(config)),
                                        future(14, 0.14F, 1, pixel(input.observation), 0)};
        result.actions.frame_spans = {{1, 2}, {2, 3}};
        result.actions.timestamps_seconds = {result.video.timestamps_seconds[1],
                                             result.video.timestamps_seconds[2]};
        return result;
    }
    ActionVideoResult run(const VideoToActionAndVideoRequest& input, ConfigView config) override {
        ++single_calls;
        auto result = ActionVideoResult{
            actions(15, input.action_spec, gain(config)),
            future(15, 0.15F, 1, pixel(input.history.frames[input.history.frames.size() - 1]),
                   last_time(input.history))};
        result.actions.frame_spans = {{1, 2}, {2, 3}};
        result.actions.timestamps_seconds = {result.video.timestamps_seconds[1],
                                             result.video.timestamps_seconds[2]};
        return result;
    }
    AudioVideoResult run(const TextToAudioVideoRequest& input, ConfigView config) override {
        ++single_calls;
        auto result = audio_video(16, float(input.prompt.size()) / 100, gain(config));
        if (mode_ == "bad_av")
            result.audio_start_seconds.reset();
        return result;
    }
    AudioVideoResult run(const InitialImageTextToAudioVideoRequest& input,
                         ConfigView config) override {
        ++single_calls;
        return audio_video(17, pixel(input.initial_image), gain(config));
    }
    AudioVideoResult run(const LastImageTextToAudioVideoRequest& input,
                         ConfigView config) override {
        ++single_calls;
        return audio_video(18, pixel(input.last_image), gain(config));
    }
    AudioVideoResult run(const BoundaryFramesTextToAudioVideoRequest& input,
                         ConfigView config) override {
        ++single_calls;
        return audio_video(19, pixel(input.first_frame) + pixel(input.last_frame) / 10,
                           gain(config));
    }
    AudioVideoResult run(const ReferencesTextToAudioVideoRequest& input,
                         ConfigView config) override {
        ++single_calls;
        float probe = 0;
        for (size_t i = 0; i < input.references.size(); ++i) {
            const auto& item = input.references[i];
            probe += float((i + 1) * (item.index() + 1)) / 100;
            if (const auto* clip = std::get_if<VideoReference>(&item)) {
                probe += float(clip->video.frames.size()) / 100;
                if (clip->soundtrack) {
                    probe += 0.05F;
                    probe += clip->soundtrack->sample_rate ? 0.02F : 0.01F;
                }
                probe += float(clip->audio_start_seconds.value_or(0));
            }
        }
        return audio_video(20, probe, gain(config));
    }

  private:
    template <class Input>
    static void preflight(const Input&) {}
    static void preflight(const TextToVideoRequest& input) {
        validate_replay(input.initial_latents);
        if ((!input.initial_latents.empty() && input.prompt.empty()) || input.prompt.size() > 6)
            throw UnsupportedTask("fixture batch prompt/replay exceeds native profile");
    }
    static void preflight(const VideoToActionSequenceRequest& input) {
        if (input.action_spec.dimensions != 2 || input.observations.frames.size() < 2)
            throw UnsupportedTask(
                "fixture inverse batch requires two channels and two observed frames");
    }
    static void preflight(const ImageToActionAndVideoRequest& input) {
        if (input.action_spec.dimensions != 2)
            throw UnsupportedTask("fixture joint batch requires two action channels");
    }
    template <class Result, class Request, class Make>
    std::vector<Result> batch(const Request& request, Make make) {
        ++batch_calls;
        std::vector<double> gains;
        for (const auto& item : request.items) {
            gains.push_back(gain(item.config));
            preflight(item.input);
        }
        if (mode_ == "batch_uniform")
            for (const auto value : gains)
                if (value != gains[0])
                    throw UnsupportedTask("native fixture requires equal resolved gain");
        std::vector<Result> results;
        for (size_t i = 0; i < request.items.size(); ++i) {
            ++batch_items_executed;
            results.push_back(make(request.items[i].input, gains[i]));
        }
        if (mode_ == "batch_fail")
            throw std::runtime_error("synthetic native batch execution failed");
        if (mode_ == "batch_bad_count" && !results.empty())
            results.pop_back();
        return results;
    }
    static void validate_replay(trtmc::Span<const float> values) {
        if (values.empty())
            return;
        if (values.size() != 6)
            throw std::invalid_argument("video fixture latent layout requires six floats");
        for (const auto value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument("video fixture latents must be finite");
    }
    static VideoResult conditioned_replay(VideoResult result, const ImageView& image,
                                          trtmc::Span<const float> values) {
        validate_replay(values);
        if (values.empty())
            return result;
        if (result.frames.num_frames < 2)
            throw std::invalid_argument("fixture conditioned replay needs two output frames");
        // Family-owned toy [C=3,T=2,H=1,W=1] layout. Like SANA, replay does
        // not eliminate the conditioning image's first-latent-frame overwrite.
        std::vector<float> latents(values.begin(), values.end());
        for (std::size_t c = 0; c < 3; ++c) {
            latents[2 * c] = image.format == ImageFormat::Float32
                                 ? static_cast<const float*>(image.data)[c]
                                 : static_cast<const std::uint8_t*>(image.data)[c] / 255.0F;
            result.frames.pixels[c] = latents[2 * c];
            result.frames.pixels[3 + c] = latents[2 * c + 1];
        }
        return result;
    }
    double gain(ConfigView config) const {
        const auto field = fields_for({})[0];
        double value = std::get<double>(*field.default_value);
        bool seen = false;
        for (const auto& entry : config) {
            if (entry.name != field.name || seen || config_kind(entry.value) != field.kind)
                throw ConfigError("invalid or duplicate video gain");
            seen = true;
            value = std::get<double>(entry.value);
        }
        if (!std::isfinite(value) || value < 0 || value > 2)
            throw ConfigError("gain must be in [0,2]");
        return value;
    }
    VideoResult video(int marker, float probe, double intensity, uint64_t count = 3) const {
        if (count == 0 || count > 8)
            throw std::invalid_argument("fixture supports one to eight frames");
        VideoResult result;
        result.frames.height = result.frames.width = 1;
        result.frames.channels = 3;
        result.frames.num_frames = int32_t(count);
        for (uint64_t i = 0; i < count; ++i) {
            result.frames.pixels.insert(result.frames.pixels.end(),
                                        {float(marker * intensity / 100), probe, float(i) / 10});
            if (mode_ != "unknown_time")
                result.timestamps_seconds.push_back(double(i * (i + 1)) / 20);
        }
        result.setup_ms = 0.5;
        result.inference_ms = 1.5;
        return result;
    }
    VideoResult future(int marker, float probe, double intensity, float context,
                       double origin) const {
        auto result = video(marker, probe, intensity);
        result.conditioned_prefix_frames = 1;
        result.frames.pixels[0] = context;
        for (auto& time : result.timestamps_seconds)
            time += origin;
        return result;
    }
    AudioVideoResult audio_video(int marker, float probe, double intensity) const {
        AudioVideoResult result;
        result.video = video(marker, probe, intensity);
        result.audio.sample_rate = 10;
        result.audio.channels = 2;
        for (size_t i = 0; i < 4; ++i)
            result.audio.samples.insert(result.audio.samples.end(),
                                        {float(marker) / 100, -float(marker) / 100});
        result.audio.inference_ms = 2;
        result.audio_start_seconds = -0.05;
        return result;
    }
    static ActionSequenceResult actions(int marker, const ActionOutputSpec& spec,
                                        double intensity) {
        if (spec.dimensions != 2)
            throw std::invalid_argument("fixture action dimension is two");
        ActionSequenceResult result;
        result.values = {{float(marker * intensity / 100), 0.25F, 0.5F, 0.75F}, 2, 2};
        result.schema.domain = spec.schema.domain;
        for (const auto value : spec.schema.component_names)
            result.schema.component_names.emplace_back(value);
        for (const auto value : spec.schema.units)
            result.schema.units.emplace_back(value);
        result.schema.coordinate_frame = spec.schema.coordinate_frame;
        result.schema.normalization = spec.schema.normalization;
        return result;
    }
    static float pixel(const ImageView& image) {
        return image.format == ImageFormat::Float32
                   ? static_cast<const float*>(image.data)[0]
                   : static_cast<const uint8_t*>(image.data)[0] / 255.0F;
    }
    static double last_time(const VideoView& video) {
        return video.timestamps_seconds.empty()
                   ? 0
                   : video.timestamps_seconds[video.timestamps_seconds.size() - 1];
    }
    std::string mode_;
};
} // namespace

extern "C" int trtmc_test_video_batch_calls() {
    return batch_calls.load();
}
extern "C" int trtmc_test_video_single_calls() {
    return single_calls.load();
}
extern "C" int trtmc_test_video_batch_items_executed() {
    return batch_items_executed.load();
}

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "video_fixture")
        throw std::runtime_error("wrong fixture family");
    return new VideoFixture(context.reader.info().task);
}
