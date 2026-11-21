/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host-side prompt planning for GLM-ASR. Kept free of CUDA and TensorRT so the
// token arithmetic can be tested on its own.

#include "families/glmasr/runtime/glmasr_config.h"

#include <algorithm>
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
// Mirrors extract_mel_spectrogram's centered/padded framing exactly (see
// glmasr_mel_spectrogram.cpp) rather than a plain floor division: the two
// disagree for some short clips (e.g. 761 samples at these defaults), which
// would otherwise under-count the audio placeholders the prompt needs.
inline int32_t mel_frames_for_samples(int32_t num_samples, int32_t hop_length, int32_t n_fft,
                                      int32_t chunk_length_s, int32_t sample_rate) {
    if (hop_length <= 0)
        return 0;
    const int32_t chunk_samples = chunk_length_s * sample_rate;
    const int32_t valid_audio = std::min(std::max(num_samples, 0), chunk_samples);
    if (valid_audio <= 0)
        return 0;
    return 1 + (n_fft / 2 + valid_audio - 1) / hop_length;
}

// Feature frames GlmAsrProcessor counts for one window: the Whisper feature
// extractor's sample mask taken every hop, so ceil(samples / hop). This, not
// the centered STFT frame count above, decides the placeholder count.
inline int32_t processor_frames_for_samples(int32_t num_samples, int32_t hop_length) {
    if (num_samples <= 0 || hop_length <= 0)
        return 0;
    return static_cast<int32_t>((static_cast<int64_t>(num_samples) + hop_length - 1) / hop_length);
}

// One chunk_length-second slice of the audio. GlmAsrProcessor encodes every
// window separately and splices the windows' valid embeddings in order.
struct AudioWindow {
    int32_t offset{0};
    int32_t samples{0};
    int32_t frames{0};     // processor frames of this window
    int32_t embeddings{0}; // valid embeddings the encoder emits for it
};

inline std::vector<AudioWindow> plan_audio_windows(int32_t num_samples, int32_t chunk_length_s,
                                                   int32_t sample_rate, int32_t hop_length,
                                                   int32_t merge_factor) {
    std::vector<AudioWindow> windows;
    const int32_t window_samples = chunk_length_s * sample_rate;
    if (num_samples <= 0 || window_samples <= 0)
        return windows;
    for (int32_t offset = 0; offset < num_samples; offset += window_samples) {
        AudioWindow window;
        window.offset = offset;
        window.samples = std::min(window_samples, num_samples - offset);
        window.frames = processor_frames_for_samples(window.samples, hop_length);
        window.embeddings = audio_embedding_count(window.frames, merge_factor);
        windows.push_back(window);
    }
    return windows;
}

// Placeholders GlmAsrProcessor inserts: one count over all windows' frames.
inline int32_t audio_placeholder_count(const std::vector<AudioWindow>& windows,
                                       int32_t merge_factor) {
    int32_t frames = 0;
    for (const auto& window : windows)
        frames += window.frames;
    return audio_embedding_count(frames, merge_factor);
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
