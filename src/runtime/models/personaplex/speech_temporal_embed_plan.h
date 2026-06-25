#pragma once

#include "runtime/models/personaplex/speech_config.h"
#include "runtime/models/personaplex/speech_depth_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
