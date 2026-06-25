#pragma once

#include "runtime/models/personaplex/speech_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

struct DepthProjectionView {
    bool has_projection{false};
    const float* temporal_hidden{nullptr};
    int32_t depth_hidden{0};
    int32_t temporal_hidden_dim{0};
    std::size_t proj_size_per_cb{0};
    const std::vector<float>* projection{nullptr};
};

inline int32_t clamp_speech_depth_token(int32_t token, int32_t vocab_size) {
    if (vocab_size <= 0) {
        return 0;
    }
    return std::max(0, std::min(token, vocab_size - 1));
}

inline void copy_speech_depth_embedding_row(const std::vector<float>& table, std::size_t offset,
                                            int32_t hidden_size, float* out_embed) {
    if (offset + hidden_size > table.size()) {
        return;
    }
    const float* row = table.data() + offset;
    for (int32_t dim = 0; dim < hidden_size; ++dim) {
        out_embed[dim] = row[dim];
    }
}

inline DepthProjectionView make_depth_projection_view(const SpeechConfig& cfg,
                                                      const float* temporal_hidden) {
    DepthProjectionView view;
    view.depth_hidden = cfg.depth_hidden_size;
    view.temporal_hidden_dim = cfg.temporal_hidden_size;
    view.proj_size_per_cb = static_cast<std::size_t>(view.depth_hidden) * view.temporal_hidden_dim;
    view.temporal_hidden = temporal_hidden;
    view.projection = &cfg.depth_projection;
    view.has_projection = !cfg.depth_projection.empty() && temporal_hidden != nullptr &&
                          view.temporal_hidden_dim > 0 && view.depth_hidden > 0;
    return view;
}

inline void apply_depth_projection(const DepthProjectionView& proj_view, int32_t proj_idx,
                                   float* out_embed) {
    if (!proj_view.has_projection || proj_view.projection == nullptr) {
        return;
    }
    const auto proj_offset = static_cast<std::size_t>(proj_idx) * proj_view.proj_size_per_cb;
    if (proj_offset + proj_view.proj_size_per_cb > proj_view.projection->size()) {
        return;
    }
    const float* proj = proj_view.projection->data() + proj_offset;
    for (int32_t row = 0; row < proj_view.depth_hidden; ++row) {
        float sum = 0.0F;
        const float* proj_row =
            proj + static_cast<std::size_t>(row) * proj_view.temporal_hidden_dim;
        for (int32_t col = 0; col < proj_view.temporal_hidden_dim; ++col) {
            sum += proj_row[col] * proj_view.temporal_hidden[col];
        }
        out_embed[row] += sum;
    }
}

inline void seed_depth_text_embedding(const SpeechConfig& cfg, int32_t text_token,
                                      int32_t depth_hidden, float* depth_embed) {
    if (cfg.depth_text_embedding.empty() || cfg.depth_text_vocab <= 0) {
        return;
    }
    const int32_t token = clamp_speech_depth_token(text_token, cfg.depth_text_vocab);
    const auto emb_offset = static_cast<std::size_t>(token) * depth_hidden;
    copy_speech_depth_embedding_row(cfg.depth_text_embedding, emb_offset, depth_hidden,
                                    depth_embed);
}

inline void seed_depth_audio_embedding(const SpeechConfig& cfg, int32_t codebook,
                                       int32_t prev_token, int32_t depth_hidden,
                                       float* depth_embed) {
    if (cfg.depth_audio_embeddings.empty() || cfg.num_depformer_emb <= 0) {
        return;
    }
    if ((codebook - 1) >= cfg.num_depformer_emb) {
        return;
    }
    const int32_t token = clamp_speech_depth_token(prev_token, cfg.audio_vocab_size);
    const auto stride = static_cast<std::size_t>(cfg.audio_vocab_size) * depth_hidden;
    const auto emb_offset = static_cast<std::size_t>(codebook - 1) * stride +
                            static_cast<std::size_t>(token) * depth_hidden;
    copy_speech_depth_embedding_row(cfg.depth_audio_embeddings, emb_offset, depth_hidden,
                                    depth_embed);
}

inline void build_depth_input_embedding(const SpeechConfig& cfg,
                                        const DepthProjectionView& proj_view, int32_t codebook,
                                        int32_t text_token, int32_t prev_token,
                                        int32_t depth_hidden, std::vector<float>& depth_embed) {
    std::fill(depth_embed.begin(), depth_embed.end(), 0.0F);
    if (codebook == 0) {
        seed_depth_text_embedding(cfg, text_token, depth_hidden, depth_embed.data());
    } else {
        seed_depth_audio_embedding(cfg, codebook, prev_token, depth_hidden, depth_embed.data());
    }
    apply_depth_projection(proj_view, codebook, depth_embed.data());
}

inline int32_t resolve_depth_prev_token(int32_t codebook, int32_t sampled_token,
                                        const SpeechConfig& cfg, const int32_t* forced_audio_tokens,
                                        const uint8_t* forced_audio_provided) {
    if (forced_audio_tokens == nullptr || forced_audio_provided == nullptr) {
        return sampled_token;
    }
    if (!forced_audio_provided[codebook]) {
        return sampled_token;
    }
    return clamp_speech_depth_token(forced_audio_tokens[codebook], cfg.audio_vocab_size);
}

} // namespace trtmc
