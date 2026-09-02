/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/minimax_h3/conditioning.h"
#include "runtime/models/minimax_h3/pipeline.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::minimax_h3 {

// Exact Qwen3-VL presentation consumed by the unified MiniMax-H3 text plan.
// Visual features stay compact and vision_row_indices identifies the image_pad
// rows where the text plan scatters them.
struct Fl2vaTextPresentation {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> mrope_position_ids; // [3, text_rows]
    std::vector<float> vision_mask;          // [text_rows, 1]
    std::vector<int32_t> vision_row_indices;
    std::vector<int32_t> token_tags;
    int32_t keyframe_count{0};
    int32_t vision_rows_per_keyframe{0};
};

// One image's exact Qwen vision-tower inputs. Patch rows use the released
// merge-block-major ordering and each temporal patch duplicates the still
// image on Qwen's two-frame temporal axis.
struct Fl2vaVisionInputs {
    std::vector<float> pixel_values;          // [patch_rows, 1536]
    std::vector<int32_t> interp_indices;      // [patch_rows, 4]
    std::vector<float> interp_weights;        // [patch_rows, 4]
    std::vector<int32_t> vision_position_ids; // [patch_rows, 2]
    int32_t patch_rows{0};
};

struct Fl2vaVisionFeatures {
    std::vector<float> vision_embeds;
    std::vector<float> deepstack_0;
    std::vector<float> deepstack_1;
    std::vector<float> deepstack_2;
    int32_t rows{0};
};

enum class Fl2vaPlanKind {
    kVisionEncoder,
    kTextEncoder,
    kKeyframeVaeEncoder,
};

// Rejects unknown bindings, legacy profiles, and partially rebuilt shared
// Qwen plans before enqueue. The two accepted Qwen envelopes are the exact
// FL2VA-only profile and the authenticated Ref2VA superset profile; no
// intermediate or merely-larger profile is accepted.
void validate_fl2va_plan(ITrtModule& module, Fl2vaPlanKind kind);

using Fl2vaPlanLoader = std::function<std::unique_ptr<ITrtModule>(const std::string& section)>;

struct Fl2vaConditioningResult {
    MiniMaxH3PreparedKeyframes keyframes;
    std::vector<std::vector<float>> keyframe_latents;
    std::vector<float> text_embeddings;
    std::vector<int32_t> text_token_tags;
};

// Executes the complete native structured-request conditioning path through
// the keyframe VAE, shared Qwen vision plan, and unified multimodal text plan.
// The request overload owns endpoint validation and native resize/crop; the
// prepared overload is used by the pipeline after geometry resolution.
Fl2vaConditioningResult run_fl2va_conditioning(const VideoGenerationRequest& request,
                                               int32_t output_height, int32_t output_width,
                                               int32_t output_frames, ITokenizer& tokenizer,
                                               const Fl2vaPlanLoader& loader);
Fl2vaConditioningResult run_fl2va_conditioning(const std::string& prompt,
                                               const MiniMaxH3PreparedKeyframes& keyframes,
                                               ITokenizer& tokenizer,
                                               const Fl2vaPlanLoader& loader);

Fl2vaTextPresentation make_fl2va_text_presentation(const std::string& prompt,
                                                   int32_t keyframe_count, int32_t height,
                                                   int32_t width, const ITokenizer& tokenizer);
Fl2vaVisionInputs make_fl2va_vision_inputs(const VideoImageInput& image);
Fl2vaVisionFeatures run_fl2va_vision_encoder(ITrtModule& module, const Fl2vaVisionInputs& inputs);
std::vector<float> run_fl2va_text_encoder(ITrtModule& module,
                                          const Fl2vaTextPresentation& presentation,
                                          const Fl2vaVisionFeatures& features);

// Returns one normalized keyframe latent in contiguous [24, 1, H/16, W/16]
// order. The posterior is spatially stitched before sampling. Each call uses
// a fresh native Torch-compatible generator at seed 42, matching the released
// FL2VA recipe independently for first and last keyframes.
std::vector<float> run_fl2va_keyframe_vae_encoder(ITrtModule& module, const VideoImageInput& image);

// Pure helpers shared by runtime and mock-plan tests.
std::vector<float> stitch_fl2va_posterior_tiles(const std::vector<float>& tiles, int32_t height,
                                                int32_t width);
std::vector<float>
sample_and_normalize_fl2va_posterior(const std::vector<float>& posterior_parameters,
                                     int32_t latent_height, int32_t latent_width,
                                     const std::vector<float>& standard_normal);
std::vector<float> patchify_fl2va_keyframe_latent(const std::vector<float>& latent,
                                                  int32_t latent_height, int32_t latent_width);

} // namespace trtmc::minimax_h3
