/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_denoiser.h"

#include "families/minimax_h3/runtime/pipeline.h"
#include "families/minimax_h3/runtime/reference_continuation.h"
#include "families/minimax_h3/runtime/reference_vsa_geometry.h"
#include "families/minimax_h3/runtime/torch_cuda_normal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kTimeSlots = 4;
constexpr int32_t kVideoWidth = 96;
constexpr int32_t kAudioWidth = 32;

ReferenceVsaLayout vsa_layout(int32_t text_rows, const std::vector<EncodedVisualReference>& images,
                              ReferenceTargetLatentShape target,
                              std::optional<ReferenceContinuationGuideShape> guide,
                              int32_t max_tiles) {
    std::vector<ImageReferenceLatentShape> shapes;
    for (const auto& image : images)
        shapes.push_back({image.latent_height, image.latent_width});
    return make_reference_vsa_layout(text_rows, shapes, target, guide, max_tiles);
}

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

std::vector<std::byte> owned_bf16(const TensorMap& outputs, const std::string& name,
                                  const std::vector<int64_t>& shape) {
    const auto value = outputs.at(name);
    if (!value.data || value.dtype != DType::kBFloat16 || value.shape != shape)
        throw std::runtime_error("MiniMax-H3 Ref2VA AdaLN output ABI mismatch: " + name);
    std::vector<std::byte> result(value.nbytes());
    std::memcpy(result.data(), value.data, result.size());
    return result;
}

const float* velocity(const TensorMap& outputs, const char* name, int64_t rows, int64_t width) {
    const auto value = outputs.at(name);
    if (!value.data || value.dtype != DType::kFloat32 ||
        value.shape != std::vector<int64_t>({rows, width}))
        throw std::runtime_error("MiniMax-H3 Ref2VA generated-only velocity ABI mismatch");
    const auto* values = static_cast<const float*>(value.data);
    if (!std::all_of(values, values + value.numel(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("MiniMax-H3 Ref2VA returned a nonfinite velocity");
    return values;
}

void require_profile(ITrtModule& module, const char* name, const std::vector<int64_t>& shape,
                     DType dtype) {
    if (!module.has_input(name) || module.tensor_dtype(name) != dtype)
        throw std::runtime_error("MiniMax-H3 Ref2VA plan input ABI mismatch: " + std::string(name));
    if (!module.input_is_dynamic(name)) {
        if (module.tensor_shape(name) != shape)
            throw std::runtime_error("MiniMax-H3 Ref2VA static input shape mismatch");
        return;
    }
    for (const auto selector : {ProfileShapeSelector::kMin, ProfileShapeSelector::kMax}) {
        const auto bound = module.input_profile_shape(name, module.profile_idx(), selector);
        if (bound.size() != shape.size())
            throw std::runtime_error("MiniMax-H3 Ref2VA input rank mismatch");
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension)
            if (bound[dimension] <= 0 ||
                (selector == ProfileShapeSelector::kMin ? bound[dimension] > shape[dimension]
                                                        : bound[dimension] < shape[dimension]))
                throw std::length_error("MiniMax-H3 Ref2VA request exceeds its plan profile");
    }
}

// Match the separate FP32 PyTorch operations. The conditioning lives on CPU
// after posterior normalization; this host assembly introduces no GPU helper
// kernel or device-specific semantics. Products must not contract into an FMA.
void noise_condition(std::vector<float>& latents, const std::vector<float>& noise) {
    if (latents.size() != noise.size())
        throw std::invalid_argument("MiniMax-H3 Ref2VA condition noise shape mismatch");
    constexpr float time = 0.999F;
    constexpr float sigma = 1.0F - time;
    for (std::size_t index = 0; index < latents.size(); ++index) {
        volatile float clean = time * latents[index];
        volatile float random = sigma * noise[index];
        latents[index] = clean + random;
    }
}

} // namespace

struct ImageReferenceDenoiser::Modulations {
    struct Step {
        std::vector<std::vector<std::byte>> blocks;
        std::vector<std::byte> final;
        std::vector<float> unique_times;
    };
    std::vector<Step> steps;
    bool has_clean_audio_reference{false};
};

ImageReferenceDenoiser::ImageReferenceDenoiser(ReferenceModuleLoader loader,
                                               ReferenceDenoiserConfig config, cudaStream_t stream,
                                               ReferenceDenoiserResidency residency)
    : loader_(std::move(loader)), config_(config), stream_(stream), residency_(residency) {
    if (!loader_ || config.layers <= 0 || config.layers > 50 || config.hidden_size <= 0 ||
        config.hidden_size > 5376 || config.text_width <= 0 || config.text_width > 5120 ||
        config.max_sequence_rows <= 0 ||
        (residency != ReferenceDenoiserResidency::Staged &&
         residency != ReferenceDenoiserResidency::Resident))
        throw std::invalid_argument("MiniMax-H3 Ref2VA requires an explicit valid native profile");
    if ((config.recipe != ReferenceSamplingRecipe::OriginalH3 &&
         config.recipe != ReferenceSamplingRecipe::OmniRefPdd8) ||
        (config.recipe == ReferenceSamplingRecipe::OriginalH3 && config.max_vsa_tiles != 0) ||
        (config.recipe == ReferenceSamplingRecipe::OmniRefPdd8 &&
         (config.max_vsa_tiles < 2 || config.max_vsa_tiles > 8192 || config.max_vsa_tiles % 2)))
        throw std::invalid_argument("MiniMax-H3 requires an explicit compatible sampling recipe");
    // Reject invalid target geometry and arithmetic before any module load.
    (void)make_image_reference_layout({1}, {{2, 2}}, config.target, config.max_sequence_rows);
}

ImageReferenceDenoiser::~ImageReferenceDenoiser() = default;

void ImageReferenceDenoiser::precompute(const ImageReferenceLayout& layout,
                                        const std::vector<float>& video_times,
                                        const std::vector<float>& audio_times) {
    const bool has_audio_reference = layout.condition_audio_rows > 0;
    if (modulations_ && modulations_->has_clean_audio_reference == has_audio_reference)
        return;
    // Clean audio introduces a 1.0 time value. Reusing an image-only table
    // would bind the wrong AdaLN slot. Release a resident DiT before rebuilding
    // that table so the two weight sets do not overlap during this transition.
    denoiser_.reset();
    modulations_.reset();
    auto module = loader_("adaln.plan", stream_);
    if (!module || !module->ok())
        throw std::runtime_error("MiniMax-H3 failed to load Ref2VA AdaLN");
    require_profile(*module, "timesteps", {kTimeSlots}, DType::kFloat32);
    module->reset_execution_context();
    auto all = std::make_unique<Modulations>();
    all->has_clean_audio_reference = has_audio_reference;
    for (std::size_t index = 0; index < video_times.size(); ++index) {
        Modulations::Step step;
        step.unique_times =
            make_image_reference_timesteps(layout, video_times[index], audio_times[index])
                .timesteps;
        if (step.unique_times.size() > kTimeSlots)
            throw std::runtime_error("MiniMax-H3 Ref2VA exceeded its four distinct time slots");
        std::array<float, kTimeSlots> times{};
        std::copy(step.unique_times.begin(), step.unique_times.end(), times.begin());
        TensorMap inputs;
        inputs.emplace("timesteps", Tensor{times.data(), {kTimeSlots}, DType::kFloat32});
        const auto outputs = module->forward(inputs);
        for (int32_t layer = 0; layer < config_.layers; ++layer)
            step.blocks.push_back(owned_bf16(outputs, "block_modulation_" + std::to_string(layer),
                                             {kTimeSlots * 3, 6, config_.hidden_size}));
        step.final = owned_bf16(outputs, "final_modulation", {kTimeSlots, 2, config_.hidden_size});
        all->steps.push_back(std::move(step));
    }
    module->sync();
    // Module dies before run() can load the recurrent DiT. Publish only a
    // complete, owned table; no borrowed output survives the next forward.
    modulations_ = std::move(all);
}

ReferenceDenoiserResult
ImageReferenceDenoiser::run(const ReferenceImageConditioning& text,
                            const std::vector<EncodedVisualReference>& images, uint64_t seed) {
    std::lock_guard<std::mutex> guard(mutex_);
    return run_unlocked(text, images, seed, nullptr, 0.0, std::nullopt);
}

ReferenceDenoiserResult ImageReferenceDenoiser::run_continuation(
    const ReferenceImageConditioning& text, const std::vector<EncodedVisualReference>& images,
    const ReferenceLatentGuide& guide, uint64_t seed, double temporal_offset,
    std::optional<int32_t> media_time_origin) {
    std::lock_guard<std::mutex> guard(mutex_);
    return run_unlocked(text, images, seed, &guide, temporal_offset, media_time_origin);
}

ReferenceDenoiserResult ImageReferenceDenoiser::run_windows(
    const ReferenceImageConditioning& text, const std::vector<EncodedVisualReference>& images,
    uint64_t seed, int32_t output_frames, const ReferenceImageConditioning* continuation_text) {
    std::lock_guard<std::mutex> guard(mutex_);
    if ((output_frames != 124 && output_frames != 226) || config_.target.frames != 37 ||
        config_.target.audio_latents != 207 || (continuation_text && output_frames != 226))
        throw std::invalid_argument(
            "MiniMax-H3 window execution requires the explicit 124/226 profile");
    const auto windows = plan_reference_continuation(output_frames, 124, 22, 2);
    const auto& child_text = continuation_text ? *continuation_text : text;
    std::optional<int32_t> media_origin;
    if (continuation_text) {
        const auto rows = std::max(text.presentation.h3_token_tags.size(),
                                   child_text.presentation.h3_token_tags.size());
        if (rows > static_cast<std::size_t>(config_.max_sequence_rows))
            throw std::length_error("MiniMax-H3 window text exceeds its packed profile");
        media_origin = static_cast<int32_t>(rows);
    }
    (void)validate_conditioning(text, images, std::nullopt, 0.0, media_origin);
    if (windows.size() == 2) {
        // Validate the complete child payload, not just its packed row count,
        // before any first-window RNG draw, engine load or denoiser forward.
        const auto& child = windows[1];
        (void)validate_conditioning(child_text, images,
                                    ReferenceContinuationGuideShape{child.guide_video_latent_frames,
                                                                    child.guide_audio_latents},
                                    child.temporal_offset, media_origin);
    }
    auto result = run_unlocked(text, images, seed, nullptr, 0.0, media_origin);
    const int32_t expected_forwards =
        config_.recipe == ReferenceSamplingRecipe::OmniRefPdd8 ? 8 : 49;
    if (result.forwards != expected_forwards)
        throw std::runtime_error("MiniMax-H3 window forward count differs from its recipe");
    if (windows.size() == 2) {
        const auto guide = reference_latent_guide(result, windows[1]);
        // The fixed upstream reuses the request seed for each fresh window;
        // guide augmentation extends that window's own original RNG stream.
        auto child = run_unlocked(child_text, images, seed, &guide, windows[1].temporal_offset,
                                  media_origin);
        if (child.forwards != expected_forwards)
            throw std::runtime_error("MiniMax-H3 child forward count differs from its recipe");
        result = append_reference_window(result, child, windows[1]);
    }
    return result;
}

ImageReferenceLayout ImageReferenceDenoiser::validate_conditioning(
    const ReferenceImageConditioning& text, const std::vector<EncodedVisualReference>& images,
    std::optional<ReferenceContinuationGuideShape> guide, double temporal_offset,
    std::optional<int32_t> media_time_origin) const {
    const auto rows = text.presentation.h3_token_tags.size();
    if (rows == 0 || rows > static_cast<std::size_t>(config_.max_sequence_rows) ||
        text.embeddings.size() != rows * static_cast<std::size_t>(config_.text_width) ||
        text.presentation.input_ids.size() != rows || !finite(text.embeddings))
        throw std::invalid_argument("MiniMax-H3 Ref2VA received incompatible Qwen conditioning");
    std::vector<ImageReferenceLatentShape> shapes;
    for (const auto& image : images) {
        if (image.latent_height <= 0 || image.latent_width <= 0 ||
            image.latents.size() !=
                static_cast<std::size_t>(24) * image.latent_height * image.latent_width ||
            !finite(image.latents))
            throw std::invalid_argument(
                "MiniMax-H3 Ref2VA received incompatible reference latents");
        shapes.push_back({image.latent_height, image.latent_width});
    }
    auto layout =
        guide ? make_continuation_reference_layout(text.presentation.h3_token_tags, shapes,
                                                   config_.target, config_.max_sequence_rows,
                                                   *guide, temporal_offset, media_time_origin)
              : make_image_reference_layout(text.presentation.h3_token_tags, shapes, config_.target,
                                            config_.max_sequence_rows, media_time_origin);
    if (text.presentation.visual_rows.size() !=
        static_cast<std::size_t>(layout.condition_video_rows - layout.continuation_video_rows))
        throw std::invalid_argument(
            "MiniMax-H3 Qwen visual rows and encoded image geometry disagree");
    const auto& visual = text.presentation.visual_rows;
    if (!std::is_sorted(visual.begin(), visual.end()) ||
        std::adjacent_find(visual.begin(), visual.end()) != visual.end() ||
        !std::all_of(visual.begin(), visual.end(), [&](int32_t row) {
            return row >= 0 && static_cast<std::size_t>(row) < rows &&
                   text.presentation.h3_token_tags[static_cast<std::size_t>(row)] == 0;
        }))
        throw std::invalid_argument("MiniMax-H3 Ref2VA received incompatible Qwen visual indexes");
    if (config_.recipe == ReferenceSamplingRecipe::OmniRefPdd8) {
        const auto geometry = vsa_layout(static_cast<int32_t>(rows), images, config_.target, guide,
                                         config_.max_vsa_tiles);
        if (static_cast<std::size_t>(geometry.source_rows) != layout.token_tags.size())
            throw std::runtime_error("OmniRef VSA and packed Ref2VA rows disagree");
    }
    return layout;
}

ReferenceDenoiserResult ImageReferenceDenoiser::run_unlocked(
    const ReferenceImageConditioning& text, const std::vector<EncodedVisualReference>& images,
    uint64_t seed, const ReferenceLatentGuide* guide, double temporal_offset,
    std::optional<int32_t> media_time_origin) {
    if (guide && (guide->shape.height != config_.target.height ||
                  guide->shape.width != config_.target.width))
        throw std::invalid_argument("MiniMax-H3 continuation guide must retain target geometry");
    auto layout = validate_conditioning(
        text, images,
        guide ? std::optional<ReferenceContinuationGuideShape>{{guide->shape.frames,
                                                                guide->shape.audio_latents}}
              : std::nullopt,
        temporal_offset, media_time_origin);
    if (guide && (guide->video_rows.size() !=
                      static_cast<std::size_t>(layout.continuation_video_rows) * kVideoWidth ||
                  guide->audio_rows.size() !=
                      static_cast<std::size_t>(layout.condition_audio_rows) * kAudioWidth ||
                  !finite(guide->video_rows) || !finite(guide->audio_rows)))
        throw std::invalid_argument("MiniMax-H3 continuation guide has invalid latent payloads");
    const auto rows = text.presentation.h3_token_tags.size();
    const bool pdd = config_.recipe == ReferenceSamplingRecipe::OmniRefPdd8;
    const auto video_schedule =
        pdd ? make_minimax_h3_pdd_schedule(12.0F) : make_minimax_h3_schedule(50, 12.0F);
    const auto audio_schedule =
        pdd ? make_minimax_h3_pdd_schedule(3.0F) : make_minimax_h3_schedule(50, 3.0F);
    auto geometry =
        pdd ? vsa_layout(
                  static_cast<int32_t>(rows), images, config_.target,
                  guide
                      ? std::optional<ReferenceContinuationGuideShape>{{guide->shape.frames,
                                                                        guide->shape.audio_latents}}
                      : std::nullopt,
                  config_.max_vsa_tiles)
            : ReferenceVsaLayout{};
    if (video_schedule.timesteps.size() != audio_schedule.timesteps.size())
        throw std::runtime_error("MiniMax-H3 audio/video schedule lengths disagree");
    try {
        // Keep image-only draw order unchanged. Continuation adds one C/T/H/W
        // video-guide draw after the static images; clean audio draws no noise.
        // Target video and then target stereo audio retain their native layouts.
        uint64_t offset = 0;
        const auto draw = [&](std::size_t count) {
            auto noise = torch_cuda_normal(count, seed, offset);
            offset += torch_cuda_normal_consumed_offset(count);
            if (noise.size() != count || !finite(noise))
                throw std::runtime_error("MiniMax-H3 Ref2VA RNG returned invalid noise");
            return noise;
        };
        std::vector<float> video;
        video.reserve(layout.video_indices.size() * kVideoWidth);
        for (const auto& image : images) {
            auto condition = image.latents;
            noise_condition(condition, draw(condition.size()));
            auto packed =
                pack_reference_video_rows(condition, 1, image.latent_height, image.latent_width);
            video.insert(video.end(), packed.begin(), packed.end());
        }
        if (guide) {
            auto condition = guide->video_rows;
            auto noise = pack_reference_video_rows(draw(condition.size()), guide->shape.frames,
                                                   guide->shape.height, guide->shape.width);
            noise_condition(condition, noise);
            video.insert(video.end(), condition.begin(), condition.end());
        }
        const auto target_pixels = static_cast<std::size_t>(config_.target.frames) *
                                   config_.target.height * config_.target.width;
        auto video_noise = draw(24 * target_pixels);
        if (pdd)
            for (auto& value : video_noise)
                value *= video_schedule.sigmas.front();
        auto generated = pack_reference_video_rows(video_noise, config_.target.frames,
                                                   config_.target.height, config_.target.width);
        video.insert(video.end(), generated.begin(), generated.end());
        std::vector<float>().swap(video_noise);
        std::vector<float>().swap(generated);
        auto audio = draw(layout.target_audio_indices.size() * kAudioWidth);
        if (pdd)
            for (auto& value : audio)
                value *= audio_schedule.sigmas.front();
        if (guide)
            audio.insert(audio.begin(), guide->audio_rows.begin(), guide->audio_rows.end());
        precompute(layout, video_schedule.timesteps, audio_schedule.timesteps);
        if (!denoiser_)
            denoiser_ = loader_("denoiser.plan", stream_);
        if (!denoiser_ || !denoiser_->ok())
            throw std::runtime_error("MiniMax-H3 failed to load the Ref2VA DiT");
        auto& module = *denoiser_;
        module.reset_execution_context();
        const auto sequence = static_cast<int64_t>(layout.token_tags.size());
        std::vector<float> positions(layout.positions.begin(), layout.positions.end());
        TensorMap common;
        common.emplace("video_hidden_states",
                       Tensor{video.data(),
                              {static_cast<int64_t>(layout.video_indices.size()), kVideoWidth},
                              DType::kFloat32});
        common.emplace("audio_hidden_states",
                       Tensor{audio.data(),
                              {static_cast<int64_t>(layout.audio_indices.size()), kAudioWidth},
                              DType::kFloat32});
        common.emplace("encoder_hidden_states",
                       Tensor{const_cast<float*>(text.embeddings.data()),
                              {static_cast<int64_t>(rows), config_.text_width},
                              DType::kFloat32});
        common.emplace("position_ids", Tensor{positions.data(), {sequence, 3}, DType::kFloat32});
        common.emplace("packed_indices",
                       Tensor{layout.packed_indices.data(), {sequence}, DType::kInt32});
        common.emplace("target_video_indices",
                       Tensor{layout.target_video_indices.data(),
                              {static_cast<int64_t>(layout.target_video_indices.size())},
                              DType::kInt32});
        common.emplace("target_audio_indices",
                       Tensor{layout.target_audio_indices.data(),
                              {static_cast<int64_t>(layout.target_audio_indices.size())},
                              DType::kInt32});
        int32_t pdd_step = 0;
        if (pdd) {
            common.emplace("pdd_step", Tensor{&pdd_step, {1}, DType::kInt32});
            common.emplace("vsa_gather_indices",
                           Tensor{geometry.gather_indices.data(),
                                  {static_cast<int64_t>(geometry.gather_indices.size())},
                                  DType::kInt32});
            common.emplace("vsa_untile_indices",
                           Tensor{geometry.untile_indices.data(),
                                  {static_cast<int64_t>(geometry.untile_indices.size())},
                                  DType::kInt32});
            common.emplace("vsa_block_sizes",
                           Tensor{geometry.block_sizes.data(),
                                  {static_cast<int64_t>(geometry.block_sizes.size())},
                                  DType::kInt32});
            common.emplace("vsa_regions",
                           Tensor{geometry.regions.data(),
                                  {static_cast<int64_t>(geometry.regions.size() / 3), 3},
                                  DType::kInt32});
        }
        for (const auto& item : common)
            require_profile(module, item.first.c_str(), item.second.shape, item.second.dtype);
        const auto prefix = static_cast<std::size_t>(layout.condition_video_rows) * kVideoWidth;
        const auto audio_prefix =
            static_cast<std::size_t>(layout.condition_audio_rows) * kAudioWidth;
        int32_t forwards = 0;
        for (std::size_t index = 0; index < video_schedule.timesteps.size(); ++index) {
            pdd_step = static_cast<int32_t>(index);
            auto plan = make_image_reference_timesteps(layout, video_schedule.timesteps[index],
                                                       audio_schedule.timesteps[index]);
            auto& modulation = modulations_->steps.at(index);
            if (modulation.unique_times != plan.timesteps)
                throw std::runtime_error("MiniMax-H3 Ref2VA modulation cache timestep mismatch");
            auto inputs = common;
            inputs.emplace("timestep_indices",
                           Tensor{plan.timestep_indices.data(), {sequence}, DType::kInt32});
            inputs.emplace("adaln_indices",
                           Tensor{plan.adaln_indices.data(), {sequence}, DType::kInt32});
            for (int32_t layer = 0; layer < config_.layers; ++layer)
                inputs.emplace("block_modulation_" + std::to_string(layer),
                               Tensor{modulation.blocks[static_cast<std::size_t>(layer)].data(),
                                      {kTimeSlots * 3, 6, config_.hidden_size},
                                      DType::kBFloat16});
            inputs.emplace("final_modulation", Tensor{modulation.final.data(),
                                                      {kTimeSlots, 2, config_.hidden_size},
                                                      DType::kBFloat16});
            if (index == 0)
                for (const auto& item : inputs)
                    require_profile(module, item.first.c_str(), item.second.shape,
                                    item.second.dtype);
            const auto outputs = module.forward(inputs);
            ++forwards;
            const auto* video_velocity =
                velocity(outputs, "video_velocity",
                         static_cast<int64_t>(layout.target_video_indices.size()), kVideoWidth);
            const auto* audio_velocity =
                velocity(outputs, "audio_velocity",
                         static_cast<int64_t>(layout.target_audio_indices.size()), kAudioWidth);
            minimax_h3_scheduler_step(video.data() + prefix, video_velocity, video.size() - prefix,
                                      video_schedule.timesteps[index], video_schedule.sigmas[index],
                                      video_schedule.sigmas[index + 1]);
            minimax_h3_scheduler_step(audio.data() + audio_prefix, audio_velocity,
                                      audio.size() - audio_prefix, audio_schedule.timesteps[index],
                                      audio_schedule.sigmas[index],
                                      audio_schedule.sigmas[index + 1]);
        }
        module.sync();
        if (!finite(video) || !finite(audio))
            throw std::runtime_error("MiniMax-H3 Ref2VA denoising produced nonfinite samples");
        std::vector<float> target_video(video.begin() + static_cast<std::ptrdiff_t>(prefix),
                                        video.end());
        if (audio_prefix)
            audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(audio_prefix));
        if (residency_ == ReferenceDenoiserResidency::Staged) {
            denoiser_.reset();
            modulations_.reset();
        }
        return {config_.target, std::move(target_video), std::move(audio), forwards};
    } catch (...) {
        denoiser_.reset();
        modulations_.reset();
        throw;
    }
}

} // namespace trtmc::minimax_h3
