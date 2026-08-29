/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host-side prompt planning for GLM-ASR. Kept free of CUDA and TensorRT so the
// token arithmetic can be tested on its own.

#include "runtime/models/glmasr/glmasr_config.h"

#include <cstdint>
#include <vector>

namespace trtmc {
namespace glmasr {

// Conv front-end geometry as (padding, kernel_size, stride) per stage. The
// encoder halves the frame count once, in the second stage.
struct ConvStage {
    int32_t padding;
    int32_t kernel_size;
    int32_t stride;
};

inline constexpr ConvStage kConvStages[] = {{1, 3, 1}, {1, 3, 2}};

// Frames the conv front-end emits for mel_frames input frames.
inline int32_t encoder_output_frames(int32_t mel_frames) {
    int32_t length = mel_frames;
    for (const auto& stage : kConvStages) {
        if (length <= 0)
            return 0;
        length = (length + 2 * stage.padding - (stage.kernel_size - 1) - 1) / stage.stride + 1;
    }
    return length > 0 ? length : 0;
}

// Audio embeddings the projector emits for mel_frames input frames. Mirrors
// GlmAsrProcessor::_get_audio_token_length, which decides how many audio
// placeholder tokens the prompt carries.
inline int32_t audio_embedding_count(int32_t mel_frames, int32_t merge_factor) {
    if (merge_factor <= 0)
        return 0;
    const int32_t frames = encoder_output_frames(mel_frames);
    if (frames < merge_factor)
        return 0;
    return (frames - merge_factor) / merge_factor + 1;
}

// Mel frames the front-end produces for a sample count at the bundle's hop.
inline int32_t mel_frames_for_samples(int32_t num_samples, int32_t hop_length) {
    if (num_samples <= 0 || hop_length <= 0)
        return 0;
    return num_samples / hop_length;
}

struct PromptPlan {
    std::vector<int32_t> input_ids;
    // Index of the first audio placeholder in input_ids.
    int32_t audio_offset{0};
    int32_t num_audio_embeddings{0};
};

// Build the transcription prompt.
//
// The checkpoint's chat template expands to:
//
//   <|user|> \n <|begin_of_audio|> <|pad|> * N <|end_of_audio|>
//   <|user|> \n {prompt tokens} <|assistant|> \n
//
// The doubled <|user|> is what the template emits, not a mistake: its audio
// macro closes with a second role marker before the instruction text.
inline PromptPlan build_prompt_plan(const GlmAsrConfig& config,
                                    const std::vector<int32_t>& prompt_tokens,
                                    int32_t num_audio_embeddings) {
    PromptPlan plan;
    plan.num_audio_embeddings = num_audio_embeddings > 0 ? num_audio_embeddings : 0;
    if (plan.num_audio_embeddings > config.max_audio_embeddings)
        plan.num_audio_embeddings = config.max_audio_embeddings;

    plan.input_ids.reserve(prompt_tokens.size() +
                           static_cast<std::size_t>(plan.num_audio_embeddings) + 8);
    plan.input_ids.push_back(config.user_token_id);
    plan.input_ids.push_back(config.newline_token_id);
    plan.input_ids.push_back(config.audio_start_token_id);
    plan.audio_offset = static_cast<int32_t>(plan.input_ids.size());
    plan.input_ids.insert(plan.input_ids.end(), static_cast<std::size_t>(plan.num_audio_embeddings),
                          config.audio_token_id);
    plan.input_ids.push_back(config.audio_end_token_id);
    plan.input_ids.push_back(config.user_token_id);
    plan.input_ids.push_back(config.newline_token_id);
    plan.input_ids.insert(plan.input_ids.end(), prompt_tokens.begin(), prompt_tokens.end());
    plan.input_ids.push_back(config.assistant_token_id);
    plan.input_ids.push_back(config.newline_token_id);
    return plan;
}

} // namespace glmasr
} // namespace trtmc
