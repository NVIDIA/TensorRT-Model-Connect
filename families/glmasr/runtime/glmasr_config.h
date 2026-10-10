/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

// Runtime shape and token contract for a GLM-ASR bundle. Every field is read
// from the bundle's runtime.json; the defaults describe GLM-ASR-Nano-2512 so a
// bundle that predates a field still loads.
struct GlmAsrConfig {
    // Mel front-end, which follows the Whisper feature extractor.
    int32_t mel_num_bins{128};
    int32_t mel_n_fft{400};
    int32_t mel_hop_length{160};
    int32_t mel_chunk_length{30};
    int32_t mel_sampling_rate{16000};

    // Audio encoder geometry. The engine is compiled for one padded chunk of
    // mel_length frames and always emits max_audio_embeddings rows; only the
    // rows the real audio covers are spliced into the prompt.
    int32_t mel_length{3000};
    int32_t encoder_frames{1500};
    int32_t audio_merge_factor{4};
    int32_t max_audio_embeddings{375};
    int32_t audio_embedding_size{2048};

    // Prompt token ids.
    int32_t audio_token_id{59260};       // <|pad|>
    int32_t audio_start_token_id{59261}; // <|begin_of_audio|>
    int32_t audio_end_token_id{59262};   // <|end_of_audio|>
    int32_t user_token_id{59253};        // <|user|>
    int32_t assistant_token_id{59254};   // <|assistant|>
    int32_t newline_token_id{10};

    // KV cache capacity the engine was compiled for. The prompt grows with
    // the audio length, so this bounds how much audio a bundle can transcribe.
    int32_t max_cache_length{0};

    // Decoder vocabulary and stopping condition.
    int32_t vocab_size{59264};
    int32_t eos_token_id{59246};
    std::vector<int32_t> eos_token_ids;

    bool is_eos_token(int32_t token) const {
        // Older bundles carry only the scalar field. New bundles preserve
        // all stopping tokens, including role markers ending a transcript.
        if (eos_token_ids.empty())
            return token == eos_token_id;
        return std::find(eos_token_ids.begin(), eos_token_ids.end(), token) != eos_token_ids.end();
    }

    std::string transcription_prompt{"Please transcribe this audio into text"};
};

} // namespace trtmc
