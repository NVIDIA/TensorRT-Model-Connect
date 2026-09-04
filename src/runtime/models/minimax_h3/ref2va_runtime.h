/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/minimax_h3/conditioning.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::minimax_h3 {

constexpr int32_t kRef2vaMaxImages = 9;
constexpr int32_t kRef2vaMaxVideos = 3;
constexpr int32_t kRef2vaMaxExplicitAudios = 3;
constexpr int32_t kRef2vaMaxReferences = 12;
constexpr int32_t kRef2vaMaxTextRows = 262144;
constexpr int32_t kRef2vaMaxVideoRows = 364608;
constexpr int32_t kRef2vaMaxAudioRows = 3558;
constexpr int32_t kRef2vaMaxPackedRows = 630310;

struct Ref2vaReferenceSummary {
    int32_t image_count{0};
    int32_t video_count{0};
    int32_t explicit_audio_count{0};
    int32_t audio_bearing_count{0};
    double total_video_seconds{0.0};
    double total_explicit_audio_seconds{0.0};
};

// Validates both the public API request and decoded media metadata. The
// returned references are normalized by conditioning.cpp in their original
// semantic order. A video soundtrack remains attached to its video and does
// not consume an explicit-audio/file slot.
struct Ref2vaPreparedRequest {
    std::vector<VideoReferenceInput> references;
    Ref2vaReferenceSummary summary;
};

Ref2vaPreparedRequest prepare_ref2va_request(const VideoGenerationRequest& request,
                                             int32_t output_frames);

struct Ref2vaVideoEncodeSchedule {
    int32_t snapped_frames{0};
    int32_t clip_count{0};
    int32_t repeated_tail_frames{0};
    int32_t raw_posterior_frames{0};
    int32_t dropped_tail_latents{0};
    int32_t output_latent_frames{0};
};

int32_t snap_ref2va_video_frames_down(int32_t frames);
Ref2vaVideoEncodeSchedule make_ref2va_video_encode_schedule(int32_t normalized_frames);
int32_t ref2va_audio_latent_frames(int32_t samples);

struct Ref2vaQwenVideoSample {
    std::vector<int32_t> frame_indices;
    std::vector<double> timestamp_seconds;
};

Ref2vaQwenVideoSample make_ref2va_qwen_video_sample(int32_t frames, double fps = 24.0,
                                                    double sample_fps = 2.0,
                                                    int32_t temporal_patch = 2);

enum class Ref2vaPresentationModality {
    kText,
    kImage,
    kVideo,
};

struct Ref2vaPresentationPiece {
    Ref2vaPresentationModality modality{Ref2vaPresentationModality::kText};
    std::string text;
    int32_t height{0};
    int32_t width{0};
};

// One entry for every image or video timestamp block in presentation order.
// The caller feeds the corresponding pixels to the shared Qwen vision plan in
// exactly this order, then concatenates all four feature streams.
struct Ref2vaVisionInvocation {
    std::size_t reference_index{0};
    int32_t first_frame{0};
    int32_t second_frame{0};
    int32_t height{0};
    int32_t width{0};
    bool is_image{false};
};

struct Ref2vaPresentationBlueprint {
    std::vector<Ref2vaPresentationPiece> pieces;
    std::vector<Ref2vaVisionInvocation> vision_invocations;
};

Ref2vaPresentationBlueprint
make_ref2va_presentation_blueprint(const std::string& prompt,
                                   const std::vector<VideoReferenceInput>& normalized_references);

struct Ref2vaMaterializedPresentation {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> qwen_token_types;
    std::vector<int32_t> h3_token_tags;
    std::vector<int32_t> vision_row_indices;
    // Axis-major [3, sequence_rows], matching text_encoder.plan.
    std::vector<int32_t> mrope_position_ids;
    int32_t vision_rows{0};
};

Ref2vaMaterializedPresentation
materialize_ref2va_presentation(const Ref2vaPresentationBlueprint& blueprint,
                                const ITokenizer& tokenizer);

struct Ref2vaVisionInputs {
    std::vector<float> pixel_values;          // [patch_rows, 1536]
    std::vector<int32_t> interp_indices;      // [patch_rows, 4]
    std::vector<float> interp_weights;        // [patch_rows, 4]
    std::vector<int32_t> vision_position_ids; // [patch_rows, 2]
    int32_t patch_rows{0};
};

struct Ref2vaVisionFeatures {
    std::vector<float> vision_embeds;
    std::vector<float> deepstack_0;
    std::vector<float> deepstack_1;
    std::vector<float> deepstack_2;
    int32_t rows{0};
};

// Qwen temporal_patch=2 input. Still images pass the same frame twice; video
// invocations pass the two sampled 2-fps frames selected by the blueprint.
Ref2vaVisionInputs make_ref2va_vision_inputs(const VideoImageInput& first,
                                             const VideoImageInput& second);
Ref2vaVisionFeatures run_ref2va_vision_encoder(ITrtModule& module,
                                               const Ref2vaVisionInputs& inputs);
Ref2vaVisionFeatures
run_ref2va_reference_vision_encoder(ITrtModule& module,
                                    const std::vector<VideoReferenceInput>& normalized_references,
                                    const Ref2vaPresentationBlueprint& blueprint);
std::vector<float> run_ref2va_text_encoder(ITrtModule& module,
                                           const Ref2vaMaterializedPresentation& presentation,
                                           const Ref2vaVisionFeatures& features);

struct Ref2vaEncodedReferenceGeometry {
    VideoReferenceKind kind{VideoReferenceKind::kImage};
    int32_t latent_frames{0};
    int32_t latent_height{0};
    int32_t latent_width{0};
    int32_t audio_latents{0};

    int32_t video_rows() const;
    int32_t audio_rows() const;
};

struct Ref2vaEncodedCondition {
    Ref2vaEncodedReferenceGeometry geometry;
    std::vector<float> video_hidden_states; // [video_rows, 96]
    std::vector<float> audio_hidden_states; // [audio_rows, 32]
};

// Native VAE plan runners. The image path reuses the FL2VA keyframe plan. The
// video path performs 17-frame clip padding, spatial tile stitch, one global
// three-latent drop, fresh seed-42 posterior sampling, FP16 round-trip, and
// transformer patchification. The audio path uses posterior mode and applies
// bundle-provided 32-channel normalization.
Ref2vaEncodedCondition run_ref2va_image_vae_encoder(ITrtModule& module,
                                                    const VideoImageInput& image);
Ref2vaEncodedCondition run_ref2va_video_vae_encoder(ITrtModule& module,
                                                    const VideoClipInput& video);
Ref2vaEncodedCondition run_ref2va_audio_vae_encoder(ITrtModule& module, const AudioResult& audio,
                                                    const std::array<float, 32>& latent_mean,
                                                    const std::array<float, 32>& latent_std);

struct Ref2vaPackedLayout {
    // Row-major [sequence_rows, 3].
    std::vector<float> position_ids;
    std::vector<int32_t> token_tags;
    std::vector<int32_t> video_indices;
    std::vector<int32_t> audio_indices;
    std::vector<int32_t> text_indices;
    int32_t condition_video_rows{0};
    int32_t condition_audio_rows{0};

    int32_t sequence_length() const;
};

Ref2vaPackedLayout
make_ref2va_packed_layout(const std::vector<int32_t>& text_token_tags,
                          const std::vector<Ref2vaEncodedReferenceGeometry>& references,
                          int32_t target_latent_frames, int32_t target_latent_height,
                          int32_t target_latent_width, int32_t target_audio_latents);

struct Ref2vaRowTimesteps {
    std::vector<float> unique_timesteps;
    std::vector<int32_t> timestep_indices;
    std::vector<int32_t> adaln_indices;
};

Ref2vaRowTimesteps make_ref2va_row_timesteps(const Ref2vaPackedLayout& layout, float video_timestep,
                                             float audio_timestep);

struct Ref2vaTimestepTable {
    std::array<float, 4> values{};
    int32_t live_count{0};
};

Ref2vaTimestepTable pad_ref2va_timesteps(const std::vector<float>& unique_timesteps);

enum class Ref2vaPlanKind {
    kVisionEncoder,
    kTextEncoder,
    kKeyframeVaeEncoder,
    kVideoVaeEncoder,
    kAudioVaeEncoder,
    kAdalnPrecompute,
    kDenoiser,
};

// Performs strict name/direction/dtype/profile validation. Unknown, legacy,
// undersized, or fallback-transformer plans fail closed before any enqueue.
void validate_ref2va_plan(ITrtModule& module, Ref2vaPlanKind kind);

struct Ref2vaOwnedTensor {
    DType dtype{DType::kFloat32};
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

struct Ref2vaModulations {
    std::array<Ref2vaOwnedTensor, 50> blocks;
    Ref2vaOwnedTensor final;
};

// Runs the fixed-four-row Ref2VA AdaLN plan. timestep_features are generated
// natively with the same sinusoidal embedding used by the H3 pipelines.
Ref2vaModulations run_ref2va_adaln_precompute(ITrtModule& module,
                                              const Ref2vaTimestepTable& timesteps);

struct Ref2vaDenoiserInputs {
    std::vector<float> video_hidden_states;   // [Nv, 96]
    std::vector<float> audio_hidden_states;   // [Na, 32]
    std::vector<float> encoder_hidden_states; // [Nt, 5120]
    Ref2vaPackedLayout layout;
    std::vector<int32_t> timestep_indices;
    std::vector<int32_t> adaln_indices;
};

struct Ref2vaVelocities {
    std::vector<float> video;
    std::vector<float> audio;
};

Ref2vaVelocities run_ref2va_denoiser(ITrtModule& module, Ref2vaDenoiserInputs& inputs,
                                     Ref2vaModulations& modulations);

} // namespace trtmc::minimax_h3
