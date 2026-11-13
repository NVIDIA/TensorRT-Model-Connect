/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/personaplex/runtime/speech_config.h"
#include "families/personaplex/runtime/speech_depth_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

namespace trtmc {

inline void add_speech_embedding_row(const std::vector<float>& table, std::size_t offset,
                                     int32_t hidden_size, float* out_embed) {
    if (offset + hidden_size > table.size()) {
        return;
    }
    const float* row = table.data() + offset;
    for (int32_t d = 0; d < hidden_size; ++d) {
        out_embed[d] += row[d];
    }
}

// Round an FP32 value to FP16 (round to nearest even) and back: the result of
// one PyTorch Half operation, which computes in FP32 and rounds once. Host-side
// cuda_fp16 conversion; no device code.
inline float personaplex_round_to_half(float value) {
    return __half2float(__float2half_rn(value));
}

// Accumulate one embedding row as a Half tensor add: the row is rounded to
// FP16 and the first row starts the sum.
inline void accumulate_half_embedding_row(const std::vector<float>& table, std::size_t offset,
                                          int32_t hidden_size, bool& started, float* out_embed) {
    if (offset + hidden_size > table.size()) {
        return;
    }
    const float* row = table.data() + offset;
    for (int32_t d = 0; d < hidden_size; ++d) {
        const float value = personaplex_round_to_half(row[d]);
        out_embed[d] = started ? personaplex_round_to_half(out_embed[d] + value) : value;
    }
    started = true;
}

inline std::size_t temporal_text_offset(const SpeechConfig& cfg, int32_t hidden_size,
                                        int32_t text_token) {
    const int32_t ttok = clamp_speech_depth_token(text_token, cfg.temporal_text_vocab);
    return static_cast<std::size_t>(ttok) * hidden_size;
}

inline std::size_t temporal_audio_offset(const SpeechConfig& cfg, int32_t hidden_size,
                                         int32_t emb_codebook_idx, int32_t token) {
    const int32_t tok = clamp_speech_depth_token(token, cfg.audio_vocab_size);
    return static_cast<std::size_t>(emb_codebook_idx) * cfg.audio_vocab_size * hidden_size +
           static_cast<std::size_t>(tok) * hidden_size;
}

// The reference's FP16 embed_codes: audio codebooks in order (model stream,
// then user stream), then text, each add a Half tensor add.
inline void compute_half_summed_embed(const SpeechConfig& cfg, int32_t hidden_size,
                                      const std::vector<int32_t>& audio_tokens, int32_t text_token,
                                      float* out_embed) {
    std::fill(out_embed, out_embed + hidden_size, 0.0F);
    bool started = false;
    if (cfg.audio_vocab_size > 0) {
        for (std::size_t cb = 0; cb < audio_tokens.size(); ++cb) {
            const auto offset =
                temporal_audio_offset(cfg, hidden_size, static_cast<int32_t>(cb), audio_tokens[cb]);
            accumulate_half_embedding_row(cfg.audio_embeddings, offset, hidden_size, started,
                                          out_embed);
        }
    }
    if (!cfg.temporal_text_embedding.empty() && cfg.temporal_text_vocab > 0) {
        accumulate_half_embedding_row(cfg.temporal_text_embedding,
                                      temporal_text_offset(cfg, hidden_size, text_token),
                                      hidden_size, started, out_embed);
    }
}

inline void add_temporal_text_embedding(const SpeechConfig& cfg, int32_t hidden_size,
                                        int32_t text_token, float* out_embed) {
    if (cfg.temporal_text_embedding.empty() || cfg.temporal_text_vocab <= 0) {
        return;
    }
    const int32_t ttok = clamp_speech_depth_token(text_token, cfg.temporal_text_vocab);
    const auto text_offset = static_cast<std::size_t>(ttok) * hidden_size;
    add_speech_embedding_row(cfg.temporal_text_embedding, text_offset, hidden_size, out_embed);
}

inline void add_temporal_audio_embedding(const SpeechConfig& cfg, int32_t hidden_size,
                                         int32_t emb_codebook_idx, int32_t token,
                                         float* out_embed) {
    const int32_t vocab = cfg.audio_vocab_size;
    if (vocab <= 0) {
        return;
    }
    const int32_t tok = clamp_speech_depth_token(token, vocab);
    const auto emb_stride_cb = static_cast<std::size_t>(vocab) * hidden_size;
    const auto emb_offset = static_cast<std::size_t>(emb_codebook_idx) * emb_stride_cb +
                            static_cast<std::size_t>(tok) * hidden_size;
    add_speech_embedding_row(cfg.audio_embeddings, emb_offset, hidden_size, out_embed);
}

inline void compute_dual_stream_summed_embed(const SpeechConfig& cfg, int32_t hidden_size,
                                             int32_t stream_cb, const int32_t* moshi_tokens,
                                             const int32_t* user_tokens, int32_t text_token,
                                             float* out_embed) {
    if (cfg.temporal_embed_half) {
        std::vector<int32_t> audio_tokens(moshi_tokens, moshi_tokens + stream_cb);
        audio_tokens.insert(audio_tokens.end(), user_tokens, user_tokens + stream_cb);
        compute_half_summed_embed(cfg, hidden_size, audio_tokens, text_token, out_embed);
        return;
    }
    std::fill(out_embed, out_embed + hidden_size, 0.0F);
    add_temporal_text_embedding(cfg, hidden_size, text_token, out_embed);
    for (int32_t cb = 0; cb < stream_cb; ++cb) {
        add_temporal_audio_embedding(cfg, hidden_size, cb, moshi_tokens[cb], out_embed);
    }
    for (int32_t cb = 0; cb < stream_cb; ++cb) {
        add_temporal_audio_embedding(cfg, hidden_size, cb + stream_cb, user_tokens[cb], out_embed);
    }
}

inline void append_hidden_from_logits(std::vector<float>& all_hidden,
                                      const std::vector<float>& logits, int32_t hidden_size) {
    const auto available = static_cast<int32_t>(logits.size());
    all_hidden.insert(all_hidden.end(), logits.begin(),
                      logits.begin() + std::min(available, hidden_size));
    if (available < hidden_size) {
        all_hidden.resize(all_hidden.size() + (hidden_size - available), 0.0F);
    }
}

inline void fill_hidden_from_logits(std::vector<float>& frame_hidden,
                                    const std::vector<float>& logits, int32_t hidden_size) {
    for (int32_t d = 0; d < hidden_size; ++d) {
        frame_hidden[static_cast<std::size_t>(d)] =
            (d < static_cast<int32_t>(logits.size())) ? logits[d] : 0.0F;
    }
}

} // namespace trtmc
