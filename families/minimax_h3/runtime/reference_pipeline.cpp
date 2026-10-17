/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_pipeline.h"

#include "families/minimax_h3/runtime/audio_decoder.h"
#include "families/minimax_h3/runtime/video_decoder.h"

#include <chrono>
#include <limits>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {
const internal::ConfigField kFields[] = {
    {"seed", internal::ConfigKind::I64, internal::ConfigValue{int64_t{42}},
     "Nonnegative original-H3 request seed; reference posterior seeds remain 42."},
};

// Validate every borrowed operand and both packed sequence caps before image
// allocation or engine loading. A reference is not an initial frame. No video
// or audio operand is silently discarded, converted or treated as an image.
void preflight(const internal::ReferencesTextToAudioVideoRequest& request,
               const ITokenizer& tokenizer, const ReferencePipelineConfig& config) {
    if (request.references.empty())
        throw internal::UnsupportedTask("MiniMax-H3 image Ref2VA requires at least one image");
    std::vector<QwenImageGrid> grids;
    std::vector<ImageReferenceLatentShape> latent_shapes;
    for (const auto& item : request.references) {
        const auto* image = std::get_if<internal::ImageView>(&item);
        if (!image)
            throw internal::UnsupportedTask(
                "MiniMax-H3 native image Ref2VA does not support video or audio references");
        if (!image->data || image->height == 0 || image->width == 0 ||
            image->height > std::numeric_limits<int32_t>::max() ||
            image->width > std::numeric_limits<int32_t>::max() || image->channels != 3 ||
            image->format != internal::ImageFormat::UInt8 ||
            uint64_t{image->height} * image->width * 3 != image->byte_size)
            throw std::invalid_argument("MiniMax-H3 image references require contiguous RGB uint8");
        const auto shape = reference_image_shape(static_cast<int32_t>(image->height),
                                                 static_cast<int32_t>(image->width));
        grids.push_back({shape.height / 16, shape.width / 16});
        latent_shapes.push_back({shape.height / 16, shape.width / 16});
    }
    const auto presentation = make_qwen_image_presentation(tokenizer, std::string(request.prompt),
                                                           grids, config.max_text_rows);
    (void)make_image_reference_layout(presentation.h3_token_tags, latent_shapes,
                                      config.denoiser.target, config.denoiser.max_sequence_rows);
}
} // namespace

struct ImageReferencePipeline::State {
    ReferenceImageConditioner conditioner;
    ReferenceVisualEncoder visual;
    std::unique_ptr<ImageReferenceDenoiser> denoiser;
    VideoDecoder video;
    AudioDecoder audio;

    State(const ReferenceModuleLoader& loader, const ITokenizer& tokenizer,
          const ReferencePipelineConfig& config, cudaStream_t stream)
        : conditioner(loader, tokenizer, stream, config.max_text_rows,
                      config.memory == ReferenceMemoryPreset::Minimal
                          ? ConditionerResidency::Staged
                          : ConditionerResidency::Resident),
          visual(loader, config.normalization, stream,
                 config.memory == ReferenceMemoryPreset::Minimal
                     ? VisualReferenceResidency::Staged
                     : VisualReferenceResidency::Resident),
          denoiser(std::make_unique<ImageReferenceDenoiser>(
              loader, config.denoiser, stream,
              config.memory == ReferenceMemoryPreset::Minimal
                  ? ReferenceDenoiserResidency::Staged
                  : ReferenceDenoiserResidency::Resident)),
          video(loader,
                {config.denoiser.target.frames, config.denoiser.target.height,
                 config.denoiser.target.width,
                 config.memory == ReferenceMemoryPreset::Minimal ? 1 : 28, config.normalization},
                stream,
                config.memory == ReferenceMemoryPreset::Minimal ? VideoDecoderResidency::Staged
                                                                : VideoDecoderResidency::Resident),
          audio(loader, config.denoiser.target.audio_latents, stream,
                config.memory == ReferenceMemoryPreset::Minimal ? AudioDecoderResidency::Staged
                                                                : AudioDecoderResidency::Resident) {
    }
};

ImageReferencePipeline::ImageReferencePipeline(ReferenceModuleLoader loader,
                                               std::unique_ptr<ITokenizer> tokenizer,
                                               ReferencePipelineConfig config, cudaStream_t stream)
    : loader_(std::move(loader)), tokenizer_(std::move(tokenizer)), config_(std::move(config)),
      stream_(stream) {
    if (!loader_ || !tokenizer_ || config_.denoiser.layers != 50 ||
        config_.denoiser.hidden_size != 5376 || config_.denoiser.text_width != 5120 ||
        config_.denoiser.target.frames != 37 || config_.denoiser.target.audio_latents != 207 ||
        (config_.memory != ReferenceMemoryPreset::Minimal &&
         config_.memory != ReferenceMemoryPreset::Original))
        throw std::invalid_argument("MiniMax-H3 image Ref2VA requires the original full topology");
    // Constructors check normalization, geometry and explicit profile limits;
    // none loads a plan or performs a CUDA operation.
    state_ = std::make_unique<State>(loader_, *tokenizer_, config_, stream_);
}

ImageReferencePipeline::~ImageReferencePipeline() = default;

std::vector<internal::TaskInstance> ImageReferencePipeline::task_bindings() {
    return {internal::bind<internal::IReferencesTextToAudioVideo>(*this, kFields)};
}

internal::AudioVideoResult
ImageReferencePipeline::run(const internal::ReferencesTextToAudioVideoRequest& request,
                            internal::ConfigView config) {
    std::lock_guard<std::mutex> lock(mutex_);
    internal::validate_config(kFields, config);
    const auto seed = *internal::config_get<int64_t>(config, kFields, "seed");
    if (seed < 0)
        throw internal::ConfigError("MiniMax-H3 seed must be nonnegative");
    preflight(request, *tokenizer_, config_);
    const auto start = std::chrono::steady_clock::now();
    try {
        if (!state_)
            state_ = std::make_unique<State>(loader_, *tokenizer_, config_, stream_);
        // Release RGB references, Qwen embeddings and image latents before
        // allocating a decoded full-resolution video. Outputs are always owned.
        auto generated = [&] {
            std::vector<PreparedReferenceImage> prepared;
            std::vector<ResizedQwenImage> qwen_images;
            for (const auto& item : request.references) {
                const auto& image = std::get<internal::ImageView>(item);
                prepared.push_back(prepare_reference_image(
                    static_cast<const uint8_t*>(image.data), image.byte_size,
                    static_cast<int32_t>(image.height), static_cast<int32_t>(image.width)));
            }
            for (const auto& image : prepared)
                qwen_images.push_back(
                    {image.rgb.data(), image.rgb.size(), image.shape.height, image.shape.width});
            auto text = state_->conditioner.encode(qwen_images, std::string(request.prompt));
            std::vector<EncodedVisualReference> latents;
            for (auto& image : prepared) {
                latents.push_back(state_->visual.encode_image(image));
                std::vector<uint8_t>().swap(image.rgb);
            }
            if (!state_->denoiser)
                state_->denoiser = std::make_unique<ImageReferenceDenoiser>(
                    loader_, config_.denoiser, stream_, ReferenceDenoiserResidency::Staged);
            return state_->denoiser->run(text, latents, static_cast<uint64_t>(seed));
        }();
        // Minimal mode trades rebuilding deterministic AdaLN tables on the next
        // request for releasing their host storage before full video decoding.
        if (config_.memory == ReferenceMemoryPreset::Minimal)
            state_->denoiser.reset();
        if (generated.forwards != 49)
            throw std::runtime_error("MiniMax-H3 original Ref2VA requires 49 denoiser forwards");
        auto video = state_->video.run(generated.video_rows);
        std::vector<float>().swap(generated.video_rows);
        auto audio = state_->audio.run(generated.audio_rows);
        internal::AudioVideoResult result;
        result.video.frames = {std::move(video.rgb), video.height, video.width, 3, video.frames};
        result.video.timestamps_seconds.reserve(static_cast<std::size_t>(video.frames));
        for (int32_t frame = 0; frame < video.frames; ++frame)
            result.video.timestamps_seconds.push_back(static_cast<double>(frame) / 24.0);
        result.audio.samples = std::move(audio.interleaved_samples);
        result.audio.sample_rate = static_cast<uint32_t>(audio.sample_rate);
        result.audio.channels = static_cast<uint32_t>(audio.channels);
        // Original 124/24 video and 165600/32000 audio have different endpoints.
        // Preserve both native durations; no stretching, truncation or padding.
        result.audio_start_seconds = 0.0;
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        result.video.inference_ms = elapsed;
        result.audio.inference_ms = elapsed;
        return result;
    } catch (...) {
        // A downstream failure must not leave resident modules from an earlier
        // phase alive. Recovery recreates all native state on the next request.
        state_.reset();
        throw;
    }
}

} // namespace trtmc::minimax_h3
